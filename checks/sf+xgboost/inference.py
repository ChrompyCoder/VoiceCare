"""
Inference script for Parkinson's Detection using Custom Features + XGBoost
Load trained model and make predictions on new audio files
"""

import os
import sys
import json
import numpy as np
import pickle
import xgboost as xgb
import librosa
import noisereduce as nr
import parselmouth
from parselmouth.praat import call
import pandas as pd
import soundfile as sf
from pathlib import Path
import argparse

def get_audio_duration(audio_path):
    """Get duration of audio file in seconds."""
    try:
        info = sf.info(audio_path)
        return info.duration
    except Exception as e:
        print(f"Error reading {audio_path}: {e}")
        return 0

def extract_acoustic_features(audio_path):
    """
    Extract comprehensive acoustic features (must match training features).
    """
    try:
        # Load audio with librosa
        y, sr = librosa.load(audio_path, sr=22050)
        
        # 🔇 DENOISE AUDIO (Critical for matching training data)
        # Use gentler denoising to avoid over-processing
        y = nr.reduce_noise(y=y, sr=sr, stationary=False, prop_decrease=0.8)
        
        features = {}
        
        # MFCC Features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(13):
            features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i}_std'] = np.std(mfccs[i])
            features[f'mfcc_{i}_max'] = np.max(mfccs[i])
            features[f'mfcc_{i}_min'] = np.min(mfccs[i])
        
        # Spectral Features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroid)
        features['spectral_centroid_std'] = np.std(spectral_centroid)
        features['spectral_centroid_max'] = np.max(spectral_centroid)
        features['spectral_centroid_min'] = np.min(spectral_centroid)
        
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
        
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        for i in range(spectral_contrast.shape[0]):
            features[f'spectral_contrast_{i}_mean'] = np.mean(spectral_contrast[i])
        
        spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
        features['spectral_flatness_mean'] = np.mean(spectral_flatness)
        features['spectral_flatness_std'] = np.std(spectral_flatness)
        
        # Temporal Features
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        
        rms = librosa.feature.rms(y=y)[0]
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)
        features['rms_max'] = np.max(rms)
        features['rms_min'] = np.min(rms)
        
        # Chroma Features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features['chroma_mean'] = np.mean(chroma)
        features['chroma_std'] = np.std(chroma)
        
        # Prosodic Features using Parselmouth
        try:
            sound = parselmouth.Sound(audio_path)
            pitch = sound.to_pitch()
            pitch_values = pitch.selected_array['frequency']
            pitch_values = pitch_values[pitch_values > 0]
            
            if len(pitch_values) > 0:
                features['pitch_mean'] = np.mean(pitch_values)
                features['pitch_std'] = np.std(pitch_values)
                features['pitch_max'] = np.max(pitch_values)
                features['pitch_min'] = np.min(pitch_values)
                features['pitch_range'] = np.max(pitch_values) - np.min(pitch_values)
            else:
                for key in ['pitch_mean', 'pitch_std', 'pitch_max', 'pitch_min', 'pitch_range']:
                    features[key] = 0
            
            # Jitter
            try:
                point_process = call(sound, "To PointProcess (periodic, cc)", 75, 600)
                features['jitter_local'] = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
                features['jitter_rap'] = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
                features['jitter_ppq5'] = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
            except:
                for key in ['jitter_local', 'jitter_rap', 'jitter_ppq5']:
                    features[key] = 0
            
            # Shimmer
            try:
                shimmer_local = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
                shimmer_apq3 = call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
                shimmer_apq5 = call([sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
                features['shimmer_local'] = shimmer_local if not np.isnan(shimmer_local) else 0
                features['shimmer_apq3'] = shimmer_apq3 if not np.isnan(shimmer_apq3) else 0
                features['shimmer_apq5'] = shimmer_apq5 if not np.isnan(shimmer_apq5) else 0
            except:
                for key in ['shimmer_local', 'shimmer_apq3', 'shimmer_apq5']:
                    features[key] = 0
            
            # HNR
            try:
                harmonicity = sound.to_harmonicity()
                hnr_values = harmonicity.values[harmonicity.values != -200]
                if len(hnr_values) > 0:
                    features['hnr_mean'] = np.mean(hnr_values)
                    features['hnr_std'] = np.std(hnr_values)
                else:
                    features['hnr_mean'] = 0
                    features['hnr_std'] = 0
            except:
                features['hnr_mean'] = 0
                features['hnr_std'] = 0
            
            # Formants
            try:
                formant = sound.to_formant_burg()
                f1_values, f2_values, f3_values = [], [], []
                for t in np.linspace(sound.xmin, sound.xmax, 10):
                    f1 = formant.get_value_at_time(1, t)
                    f2 = formant.get_value_at_time(2, t)
                    f3 = formant.get_value_at_time(3, t)
                    if not np.isnan(f1): f1_values.append(f1)
                    if not np.isnan(f2): f2_values.append(f2)
                    if not np.isnan(f3): f3_values.append(f3)
                
                features['f1_mean'] = np.mean(f1_values) if f1_values else 0
                features['f2_mean'] = np.mean(f2_values) if f2_values else 0
                features['f3_mean'] = np.mean(f3_values) if f3_values else 0
                features['f1_std'] = np.std(f1_values) if f1_values else 0
                features['f2_std'] = np.std(f2_values) if f2_values else 0
                features['f3_std'] = np.std(f3_values) if f3_values else 0
            except:
                for key in ['f1_mean', 'f2_mean', 'f3_mean', 'f1_std', 'f2_std', 'f3_std']:
                    features[key] = 0
        except:
            # Set all prosodic features to 0 if Parselmouth fails
            for key in ['pitch_mean', 'pitch_std', 'pitch_max', 'pitch_min', 'pitch_range',
                       'jitter_local', 'jitter_rap', 'jitter_ppq5',
                       'shimmer_local', 'shimmer_apq3', 'shimmer_apq5',
                       'hnr_mean', 'hnr_std',
                       'f1_mean', 'f2_mean', 'f3_mean', 'f1_std', 'f2_std', 'f3_std']:
                if key not in features:
                    features[key] = 0
        
        feature_vector = np.array(list(features.values()))
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
        
        return feature_vector
            
    except Exception as e:
        print(f"Error extracting features from {audio_path}: {e}")
        return None

def load_model_and_scaler(model_path, scaler_path):
    """Load trained XGBoost model and scaler."""
    # Load XGBoost model
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    
    # Load scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    return model, scaler

def predict(audio_path, model, scaler, min_duration=4.0):
    """Make prediction on a single audio file."""
    # Check duration
    duration = get_audio_duration(audio_path)
    if duration < min_duration:
        return {
            'error': f'Audio duration ({duration:.2f}s) is less than minimum ({min_duration}s)',
            'prediction': None,
            'probability': None
        }
    
    # Extract features
    features = extract_acoustic_features(audio_path)
    if features is None:
        return {
            'error': 'Failed to extract features',
            'prediction': None,
            'probability': None
        }
    
    # Scale features
    features_scaled = scaler.transform(features.reshape(1, -1))
    
    # Make prediction
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]
    
    # Interpret result
    label = 'Parkinson\'s Disease' if prediction == 1 else 'Healthy'
    confidence = probability[1] if prediction == 1 else probability[0]
    
    return {
        'prediction': label,
        'prediction_class': int(prediction),
        'probability_healthy': float(probability[0]),
        'probability_parkinsons': float(probability[1]),
        'confidence': float(confidence),
        'duration': float(duration),
        'error': None
    }

def predict_batch(audio_files, model, scaler, min_duration=4.0):
    """Make predictions on multiple audio files."""
    results = {}
    
    print(f"\nProcessing {len(audio_files)} audio files...")
    print("=" * 70)
    
    for audio_path in audio_files:
        filename = os.path.basename(audio_path)
        print(f"\nProcessing: {filename}")
        
        result = predict(audio_path, model, scaler, min_duration)
        results[filename] = result
        
        if result['error']:
            print(f"  ❌ Error: {result['error']}")
        else:
            print(f"  ✓ Prediction: {result['prediction']}")
            print(f"  ✓ Confidence: {result['confidence']:.2%}")
            print(f"  ✓ P(Healthy): {result['probability_healthy']:.4f}")
            print(f"  ✓ P(Parkinson's): {result['probability_parkinsons']:.4f}")
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Parkinson\'s Detection Inference')
    parser.add_argument('--audio', type=str, help='Path to audio file or directory')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model (.json)')
    parser.add_argument('--scaler', type=str, required=True, help='Path to scaler (.pkl)')
    parser.add_argument('--min-duration', type=float, default=4.0, help='Minimum audio duration (seconds)')
    parser.add_argument('--output', type=str, help='Output JSON file for results')
    
    args = parser.parse_args()
    
    # Check if model and scaler exist
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)
    
    if not os.path.exists(args.scaler):
        print(f"Error: Scaler file not found: {args.scaler}")
        sys.exit(1)
    
    # Load model and scaler
    print("=" * 70)
    print("Loading model and scaler...")
    print("=" * 70)
    model, scaler = load_model_and_scaler(args.model, args.scaler)
    print(f"✓ Model loaded: {args.model}")
    print(f"✓ Scaler loaded: {args.scaler}")
    
    # Get audio files
    audio_files = []
    if args.audio:
        audio_path = Path(args.audio)
        if audio_path.is_file():
            audio_files = [str(audio_path)]
        elif audio_path.is_dir():
            audio_files = [str(f) for f in audio_path.glob('*.wav')]
        else:
            print(f"Error: Audio path not found: {args.audio}")
            sys.exit(1)
    else:
        print("Error: --audio argument is required")
        parser.print_help()
        sys.exit(1)
    
    if not audio_files:
        print("Error: No audio files found")
        sys.exit(1)
    
    # Make predictions
    results = predict_batch(audio_files, model, scaler, args.min_duration)
    
    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    total = len(results)
    errors = sum(1 for r in results.values() if r['error'])
    healthy = sum(1 for r in results.values() if r['prediction'] == 'Healthy')
    parkinsons = sum(1 for r in results.values() if r['prediction'] == 'Parkinson\'s Disease')
    
    print(f"Total files: {total}")
    print(f"Errors: {errors}")
    print(f"Healthy: {healthy}")
    print(f"Parkinson's: {parkinsons}")
    
    # Save results if output path specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\n✓ Results saved to: {args.output}")
    
    print("=" * 70)

if __name__ == "__main__":
    main()
