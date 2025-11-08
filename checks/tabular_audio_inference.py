"""
Inference script for tabular Random Forest model on audio files
Extracts acoustic features from audio and predicts using trained Random Forest
"""
import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call
import joblib
import json
import noisereduce as nr
from pathlib import Path

def extract_acoustic_features(audio_path):
    """
    Extract 26 acoustic features matching UCI dataset
    Maps Parselmouth/Praat features to UCI dataset column names
    """
    # Load audio
    y, sr = librosa.load(audio_path, sr=22050)
    
    # Denoise (CRITICAL - matches training data)
    y = nr.reduce_noise(y=y, sr=sr, stationary=False, prop_decrease=0.8)
    
    # Use Parselmouth for Praat-like analysis
    sound = parselmouth.Sound(y, sampling_frequency=sr)
    
    # Pitch analysis
    pitch = sound.to_pitch()
    pitch_values = pitch.selected_array['frequency']
    pitch_values = pitch_values[pitch_values > 0]  # Remove unvoiced frames
    
    # Point process for jitter
    point_process = call(sound, "To PointProcess (periodic, cc)", 75, 500)
    
    # Harmonicity
    harmonicity = sound.to_harmonicity()
    
    features = {}
    
    # Map to UCI dataset column names:
    # Jitter features
    try:
        features['Jitter (local)'] = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        features['Jitter (local, absolute)'] = call(point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
        features['Jitter (rap)'] = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
        features['Jitter (ppq5)'] = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
        features['Jitter (ddp)'] = features['Jitter (rap)'] * 3
    except Exception as e:
        print(f"⚠️  Jitter extraction failed: {e}")
        features['Jitter (local)'] = 0.0
        features['Jitter (local, absolute)'] = 0.0
        features['Jitter (rap)'] = 0.0
        features['Jitter (ppq5)'] = 0.0
        features['Jitter (ddp)'] = 0.0
    
    # Shimmer features
    try:
        features['Shimmer (local)'] = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        features['Shimmer (local, dB)'] = call([sound, point_process], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        features['Shimmer (apq3)'] = call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        features['Shimmer (apq5), '] = call([sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)  # Note trailing space
        features['Shimmer (apq11)'] = call([sound, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
        features['Shimmer (dda)'] = features['Shimmer (apq3)'] * 3
    except Exception as e:
        print(f"⚠️  Shimmer extraction failed: {e}")
        features['Shimmer (local)'] = 0.0
        features['Shimmer (local, dB)'] = 0.0
        features['Shimmer (apq3)'] = 0.0
        features['Shimmer (apq5), '] = 0.0
        features['Shimmer (apq11)'] = 0.0
        features['Shimmer (dda)'] = 0.0
    
    # Pitch statistics
    if len(pitch_values) > 0:
        features['Mean pitch'] = np.mean(pitch_values)
        features['Maximum pitch'] = np.max(pitch_values)
        features['Minimum pitch'] = np.min(pitch_values)
        features['Median pitch'] = np.median(pitch_values)
        features['Standard deviation'] = np.std(pitch_values)
        features['Mean period'] = 1.0 / np.mean(pitch_values) if np.mean(pitch_values) > 0 else 0.0
        features['Standard deviation of period'] = np.std(1.0 / pitch_values[pitch_values > 0]) if len(pitch_values[pitch_values > 0]) > 0 else 0.0
    else:
        features['Mean pitch'] = 0.0
        features['Maximum pitch'] = 0.0
        features['Minimum pitch'] = 0.0
        features['Median pitch'] = 0.0
        features['Standard deviation'] = 0.0
        features['Mean period'] = 0.0
        features['Standard deviation of period'] = 0.0
    
    # Voice breaks
    features['Number of voice breaks'] = 0.0
    features['Degree of voice breaks'] = 0.0
    
    # Pulses/periods
    try:
        features['Number of pulses'] = call(point_process, "Get number of points")
        features['Number of periods'] = features['Number of pulses'] - 1 if features['Number of pulses'] > 1 else 0
    except:
        features['Number of pulses'] = 0.0
        features['Number of periods'] = 0.0
    
    # Fraction unvoiced
    total_frames = len(pitch.selected_array['frequency'])
    voiced_frames = len(pitch_values)
    features['Fraction of locally unvoiced frames'] = 1.0 - (voiced_frames / total_frames) if total_frames > 0 else 0.0
    
    # Placeholders for unknown features
    features['AC'] = 0.0
    features['HTN'] = 0.0
    features['NTH'] = 0.0
    
    return features

def predict_from_audio(audio_path, model_path='models/uci_tabular_rf.pkl', 
                       scaler_path='models/uci_tabular_scaler.pkl',
                       features_path='models/uci_tabular_features.json'):
    """
    Predict Parkinson's from audio file using tabular Random Forest model
    """
    print(f"\n{'='*70}")
    print(f"🎤 TABULAR MODEL INFERENCE (Random Forest)")
    print(f"{'='*70}")
    print(f"\n📂 Audio: {Path(audio_path).name}")
    
    # Load model
    print(f"\n[1/4] Loading model...")
    model = joblib.load(model_path)
    
    scaler = joblib.load(scaler_path)
    
    with open(features_path, 'r') as f:
        feature_data = json.load(f)
        feature_names = feature_data.get('features', feature_data)  # Handle both formats
    
    print(f"   ✅ Model loaded ({len(feature_names)} features)")
    
    # Extract features
    print(f"\n[2/4] Extracting acoustic features...")
    features = extract_acoustic_features(audio_path)
    
    # Print key features
    print(f"   Key features:")
    print(f"      Jitter (local): {features.get('Jitter (local)', 0):.6f}")
    print(f"      Shimmer (local): {features.get('Shimmer (local)', 0):.6f}")
    print(f"      Mean pitch: {features.get('Mean pitch', 0):.2f} Hz")
    
    # Prepare feature vector (use only common features from training)
    X = []
    for feat_name in feature_names:
        X.append(features.get(feat_name, 0.0))
    X = np.array(X).reshape(1, -1)
    
    # Standardize
    print(f"\n[3/4] Standardizing features...")
    X_scaled = scaler.transform(X)
    
    # Predict
    print(f"\n[4/4] Predicting...")
    proba = model.predict_proba(X_scaled)[0]
    pred_class = model.predict(X_scaled)[0]
    
    # Results
    print(f"\n{'='*70}")
    print(f"📊 RESULTS")
    print(f"{'='*70}")
    print(f"\n   Parkinson's Probability: {proba[1]*100:.2f}%")
    print(f"   Healthy Probability: {proba[0]*100:.2f}%")
    pred_text = "🔴 PARKINSON'S DETECTED" if pred_class == 1 else "🟢 HEALTHY"
    print(f"   Prediction: {pred_text}")
    
    if proba[1] < 0.30:
        risk = "LOW"
        emoji = "🟢"
    elif proba[1] < 0.65:
        risk = "MODERATE"
        emoji = "🟡"
    else:
        risk = "HIGH"
        emoji = "🔴"
    
    print(f"   Risk Level: {emoji} {risk}")
    print(f"\n{'='*70}\n")
    
    return {
        'probability': float(proba[1]),
        'prediction': int(pred_class),
        'risk_level': risk,
        'features': features
    }

if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python tabular_audio_inference.py <audio_file>")
        print("\nExample:")
        print("  python tabular_audio_inference.py testy.wav")
        print("  python tabular_audio_inference.py healthy22.wav")
        sys.exit(1)
    
    audio_path = sys.argv[1]
    result = predict_from_audio(audio_path)
