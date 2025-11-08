"""
Test Script - Parkinson's Detection
===================================
Tests all .wav files from ../../backend/test
using the trained OpenSMILE + XGBoost model (Temporal Features)

This script automatically finds the latest trained model.
"""

import os
import sys
import json
import pickle
import numpy as np
import xgboost as xgb
import opensmile
import librosa
import noisereduce as nr
import soundfile as sf
from pathlib import Path
import tempfile
from datetime import datetime
import glob

# Configuration
TEST_DIR = "../../backend/test"
MODELS_DIR = "models"
RESULTS_DIR = "results"
FEATURE_SET = "ComParE_2016"

def find_latest_model():
    """Find the most recently trained model and scaler."""
    model_files = glob.glob(f"{MODELS_DIR}/xgboost_model_*.json")
    scaler_files = glob.glob(f"{MODELS_DIR}/scaler_*.pkl")
    
    if not model_files:
        print("❌ No trained model found!")
        print("Please run train.py first to train a model.")
        sys.exit(1)
    
    if not scaler_files:
        print("❌ No scaler found!")
        print("Please run train.py first.")
        sys.exit(1)
    
    # Get most recent files
    latest_model = max(model_files, key=os.path.getctime)
    latest_scaler = max(scaler_files, key=os.path.getctime)
    
    return latest_model, latest_scaler

def denoise_audio(audio_path):
    """Denoise audio file."""
    try:
        y, sr = librosa.load(audio_path, sr=None)
        y_denoised = nr.reduce_noise(y=y, sr=sr, stationary=False, prop_decrease=0.8)
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        sf.write(temp_file.name, y_denoised, sr)
        return temp_file.name
    except Exception as e:
        print(f"Warning: Denoising failed: {e}")
        return audio_path

def extract_features(audio_path):
    """
    Extract OpenSMILE temporal features (Functionals).
    
    These are statistical summaries over time, not raw LLDs.
    """
    try:
        # Denoise
        audio_processed = denoise_audio(audio_path)
        
        # Extract temporal features (Functionals)
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet[FEATURE_SET],
            feature_level=opensmile.FeatureLevel.Functionals,  # Temporal features
        )
        features = smile.process_file(audio_processed)
        feature_vector = features.values.flatten()
        
        # Cleanup
        if audio_processed != audio_path:
            try:
                os.unlink(audio_processed)
            except:
                pass
        
        return feature_vector
    except Exception as e:
        print(f"Error: {e}")
        return None

def predict_file(audio_path, model, scaler):
    """Predict for a single audio file."""
    features = extract_features(audio_path)
    if features is None:
        return None
    
    features_scaled = scaler.transform(features.reshape(1, -1))
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]
    
    label = 'Parkinson\'s Disease' if prediction == 1 else 'Healthy'
    confidence = probability[1] if prediction == 1 else probability[0]
    
    return {
        'prediction': label,
        'class': int(prediction),
        'prob_healthy': float(probability[0]),
        'prob_parkinsons': float(probability[1]),
        'confidence': float(confidence)
    }

def main():
    print("=" * 80)
    print("🧪 Testing All WAV Files - Temporal Features Model")
    print("=" * 80)
    print(f"Test Directory: {TEST_DIR}")
    print(f"Feature Set: {FEATURE_SET} (Functionals - Temporal)")
    print("=" * 80)
    
    # Check test directory exists
    if not os.path.exists(TEST_DIR):
        print(f"\n❌ Test directory not found: {TEST_DIR}")
        print("Please ensure ../../backend/test exists with .wav files")
        sys.exit(1)
    
    # Find latest model
    print("\n📦 Loading trained model...")
    model_path, scaler_path = find_latest_model()
    print(f"✓ Model: {model_path}")
    print(f"✓ Scaler: {scaler_path}")
    
    # Load model
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    print("✓ Model and scaler loaded successfully")
    
    # Find all .wav files
    print(f"\n🔍 Searching for .wav files in {TEST_DIR}...")
    test_path = Path(TEST_DIR)
    wav_files = sorted(list(test_path.glob('*.wav')) + list(test_path.glob('*.WAV')))
    
    if not wav_files:
        print(f"❌ No .wav files found in {TEST_DIR}")
        sys.exit(1)
    
    print(f"✓ Found {len(wav_files)} .wav files")
    
    # List all files
    print("\n📋 Files to test:")
    for i, wav_file in enumerate(wav_files, 1):
        print(f"  {i}. {wav_file.name}")
    
    # Test each file
    print("\n" + "=" * 80)
    print("🧪 TESTING FILES (Temporal Features)")
    print("=" * 80)
    
    results = {}
    healthy = 0
    parkinsons = 0
    failed = 0
    
    for i, wav_file in enumerate(wav_files, 1):
        filename = wav_file.name
        print(f"\n[{i}/{len(wav_files)}] {filename}")
        print("-" * 80)
        
        result = predict_file(str(wav_file), model, scaler)
        
        if result:
            results[filename] = result
            print(f"  ✅ Prediction: {result['prediction']}")
            print(f"  📊 Confidence: {result['confidence']*100:.1f}%")
            print(f"  📈 Probabilities:")
            print(f"     • Healthy:       {result['prob_healthy']*100:.2f}%")
            print(f"     • Parkinson's:   {result['prob_parkinsons']*100:.2f}%")
            
            if result['prediction'] == 'Healthy':
                healthy += 1
            else:
                parkinsons += 1
        else:
            results[filename] = {'error': 'Failed to process'}
            failed += 1
            print(f"  ❌ Failed to process")
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    print(f"Total files tested: {len(wav_files)}")
    print(f"Successful: {len(wav_files) - failed}")
    print(f"Failed: {failed}")
    print(f"\nPrediction Results:")
    print(f"  🟢 Healthy:       {healthy} ({healthy/len(wav_files)*100:.1f}%)")
    print(f"  🔴 Parkinson's:   {parkinsons} ({parkinsons/len(wav_files)*100:.1f}%)")
    
    # Detailed table
    print("\n" + "=" * 80)
    print("📋 DETAILED RESULTS")
    print("=" * 80)
    print(f"{'Filename':<45} {'Prediction':<15} {'Confidence':<12} {'Status':<10}")
    print("-" * 80)
    
    for filename, result in results.items():
        if 'error' in result:
            print(f"{filename:<45} {'ERROR':<15} {'N/A':<12} {'❌':<10}")
        else:
            pred = 'PD' if result['class'] == 1 else 'Healthy'
            conf = f"{result['confidence']*100:.1f}%"
            status = '✅'
            print(f"{filename:<45} {pred:<15} {conf:<12} {status:<10}")
    
    # Summary
    print("=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)
    print(f"Total files: {len(wav_files)}")
    print(f"Healthy: {healthy}")
    print(f"Parkinson's: {parkinsons}")
    print()
    
    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    output_file = f"{RESULTS_DIR}/test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_data = {
        'test_timestamp': datetime.now().isoformat(),
        'test_directory': TEST_DIR,
        'model_used': model_path,
        'scaler_used': scaler_path,
        'feature_set': FEATURE_SET,
        'feature_type': 'Temporal (Functionals)',
        'summary': {
            'total_files': len(wav_files),
            'successful': len(wav_files) - failed,
            'failed': failed,
            'healthy': healthy,
            'parkinsons': parkinsons
        },
        'file_list': [f.name for f in wav_files],
        'results': results
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print("\n" + "=" * 80)
    print(f"💾 Results saved to: {output_file}")
    print("=" * 80)
    
    print("\n✅ Testing complete!")
    print("=" * 80)

if __name__ == "__main__":
    main()
