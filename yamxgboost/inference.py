"""
Inference script for Ensemble Parkinson's Detection
YAMNet + OpenSMILE + XGBoost
"""

import os
import sys
import json
import numpy as np
import pickle
import librosa
import opensmile
import tensorflow_hub as hub
import xgboost as xgb
import argparse
from pathlib import Path

def load_yamnet_model():
    """Load YAMNet model."""
    yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
    return yamnet_model

def extract_yamnet_embeddings(audio_path, yamnet_model):
    """Extract YAMNet embeddings."""
    try:
        waveform, sr = librosa.load(audio_path, sr=16000, mono=True)
        waveform = waveform.astype(np.float32)
        scores, embeddings, spectrogram = yamnet_model(waveform)
        embedding_mean = np.mean(embeddings.numpy(), axis=0)
        return embedding_mean
    except Exception as e:
        print(f"Error extracting YAMNet embeddings: {e}")
        return None

def extract_opensmile_features(audio_path, feature_set='eGeMAPSv02'):
    """Extract OpenSMILE features."""
    try:
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet[feature_set],
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        features = smile.process_file(audio_path)
        feature_vector = features.values.flatten()
        return feature_vector
    except Exception as e:
        print(f"Error extracting OpenSMILE features: {e}")
        return None

def load_model_and_scaler(model_path, scaler_path):
    """Load XGBoost model and scaler."""
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    return model, scaler

def predict(audio_path, model, scaler, yamnet_model, feature_set='eGeMAPSv02'):
    """Make prediction on a single audio file."""
    # Extract YAMNet embeddings
    yamnet_emb = extract_yamnet_embeddings(audio_path, yamnet_model)
    if yamnet_emb is None:
        return {'error': 'Failed to extract YAMNet embeddings', 'prediction': None}
    
    # Extract OpenSMILE features
    opensmile_feat = extract_opensmile_features(audio_path, feature_set)
    if opensmile_feat is None:
        return {'error': 'Failed to extract OpenSMILE features', 'prediction': None}
    
    # Combine features
    features_combined = np.concatenate([yamnet_emb, opensmile_feat])
    
    # Scale features
    features_scaled = scaler.transform(features_combined.reshape(1, -1))
    
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

def predict_batch(audio_files, model, scaler, yamnet_model, feature_set='eGeMAPSv02'):
    """Make predictions on multiple audio files."""
    results = {}
    
    print(f"\nProcessing {len(audio_files)} audio files...")
    print("=" * 70)
    
    for audio_path in audio_files:
        filename = os.path.basename(audio_path)
        print(f"\nProcessing: {filename}")
        
        result = predict(audio_path, model, scaler, yamnet_model, feature_set)
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
    parser = argparse.ArgumentParser(description='Ensemble Parkinson\'s Detection Inference')
    parser.add_argument('--audio', type=str, required=True, help='Path to audio file or directory')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model (.json)')
    parser.add_argument('--scaler', type=str, required=True, help='Path to scaler (.pkl)')
    parser.add_argument('--feature-set', type=str, default='eGeMAPSv02', 
                       help='OpenSMILE feature set (default: eGeMAPSv02)')
    parser.add_argument('--output', type=str, help='Output JSON file for results')
    
    args = parser.parse_args()
    
    # Check files
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        sys.exit(1)
    
    if not os.path.exists(args.scaler):
        print(f"Error: Scaler file not found: {args.scaler}")
        sys.exit(1)
    
    # Load models
    print("=" * 70)
    print("Loading models...")
    print("=" * 70)
    
    print("Loading YAMNet model...")
    yamnet_model = load_yamnet_model()
    print("✓ YAMNet model loaded")
    
    print("Loading XGBoost model and scaler...")
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
    results = predict_batch(audio_files, model, scaler, yamnet_model, args.feature_set)
    
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
    
    # Save results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\n✓ Results saved to: {args.output}")
    
    print("=" * 70)

if __name__ == "__main__":
    main()
