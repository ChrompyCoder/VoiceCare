"""
Inference script for Parkinson's Detection using OpenSMILE + XGBoost
Load trained model and make predictions on new audio files
"""

import os
import sys
import json
import numpy as np
import pickle
import xgboost as xgb
import opensmile
import librosa
import noisereduce as nr
import soundfile as sf
import argparse
from pathlib import Path
import tempfile

def denoise_audio(audio_path):
    """Denoise audio file and return path to denoised version."""
    try:
        # Load audio
        y, sr = librosa.load(audio_path, sr=None)
        
        # Apply denoising
        y_denoised = nr.reduce_noise(y=y, sr=sr, stationary=False, prop_decrease=0.8)
        
        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        sf.write(temp_file.name, y_denoised, sr)
        
        return temp_file.name
    except Exception as e:
        print(f"Warning: Denoising failed, using original audio: {e}")
        return audio_path

def extract_opensmile_features(audio_path, feature_set='ComParE_2016', denoise=True):
    """Extract features using OpenSMILE (must match training feature set)."""
    try:
        # Denoise audio first
        if denoise:
            audio_path_processed = denoise_audio(audio_path)
        else:
            audio_path_processed = audio_path
        
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet[feature_set],
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        
        features = smile.process_file(audio_path_processed)
        feature_vector = features.values.flatten()
        
        # Clean up temp file if denoising was used
        if denoise and audio_path_processed != audio_path:
            try:
                os.unlink(audio_path_processed)
            except:
                pass
        
        return feature_vector
    except Exception as e:
        print(f"Error extracting features from {audio_path}: {e}")
        return None

def load_model_and_scaler(model_path, scaler_path):
    """Load trained XGBoost model and scaler."""
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    return model, scaler

def predict(audio_path, model, scaler, feature_set='ComParE_2016'):
    """Make prediction on a single audio file."""
    # Extract features
    features = extract_opensmile_features(audio_path, feature_set)
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
        'error': None
    }

def predict_batch(audio_files, model, scaler, feature_set='ComParE_2016'):
    """Make predictions on multiple audio files."""
    results = {}
    
    print(f"\nProcessing {len(audio_files)} audio files...")
    print("=" * 70)
    
    for audio_path in audio_files:
        filename = os.path.basename(audio_path)
        print(f"\nProcessing: {filename}")
        
        result = predict(audio_path, model, scaler, feature_set)
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
    parser = argparse.ArgumentParser(description='Parkinson\'s Detection Inference (OpenSMILE + XGBoost)')
    parser.add_argument('--audio', type=str, required=True, help='Path to audio file or directory')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model (.json)')
    parser.add_argument('--scaler', type=str, required=True, help='Path to scaler (.pkl)')
    parser.add_argument('--feature-set', type=str, default='ComParE_2016', 
                       help='OpenSMILE feature set (default: ComParE_2016)')
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
    print(f"✓ Feature set: {args.feature_set}")
    
    # Get audio files
    audio_files = []
    audio_path = Path(args.audio)
    if audio_path.is_file():
        audio_files = [str(audio_path)]
    elif audio_path.is_dir():
        audio_files = [str(f) for f in audio_path.glob('*.wav')]
    else:
        print(f"Error: Audio path not found: {args.audio}")
        sys.exit(1)
    
    if not audio_files:
        print("Error: No audio files found")
        sys.exit(1)
    
    # Make predictions
    results = predict_batch(audio_files, model, scaler, args.feature_set)
    
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
