"""
Test Script - Parkinson's Detection
===================================
Tests all .wav files from ../../backend/test
using the trained OpenSMILE + XGBoost model

This script should be run AFTER train.py has completed training.
It will automatically find the latest trained model.
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
        print("Please run train.py first to train a model.")
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
        print(f"  Warning: Denoising failed: {e}")
        return audio_path

def extract_features(audio_path, denoise=True):
    """Extract OpenSMILE features from audio file."""
    try:
        # Denoise if requested
        if denoise:
            audio_processed = denoise_audio(audio_path)
        else:
            audio_processed = audio_path
        
        # Extract features
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet[FEATURE_SET],
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        features = smile.process_file(audio_processed)
        feature_vector = features.values.flatten()
        
        # Cleanup temp file
        if denoise and audio_processed != audio_path:
            try:
                os.unlink(audio_processed)
            except:
                pass
        
        return feature_vector
    except Exception as e:
        print(f"  ❌ Error extracting features: {e}")
        return None

def predict_file(audio_path, model, scaler, denoise=True):
    """Make prediction for a single audio file."""
    features = extract_features(audio_path, denoise=denoise)
    if features is None:
        return None
    
    # Scale features
    features_scaled = scaler.transform(features.reshape(1, -1))
    
    # Predict
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]
    
    # Prepare result
    label = "Parkinson's Disease" if prediction == 1 else "Healthy"
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
    print("🧪 Testing All WAV Files from Backend Test Directory")
    print("=" * 80)
    print(f"Test Directory: {TEST_DIR}")
    print(f"Feature Set: {FEATURE_SET}")
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
    
    # Load model and scaler
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    print("✓ Model and scaler loaded successfully")
    
    # Find all .wav files in test directory
    print(f"\n🔍 Searching for .wav files in {TEST_DIR}...")
    test_path = Path(TEST_DIR)
    wav_files = sorted(list(test_path.glob('*.wav')) + list(test_path.glob('*.WAV')))
    
    if not wav_files:
        print(f"❌ No .wav files found in {TEST_DIR}")
        sys.exit(1)
    
    print(f"✓ Found {len(wav_files)} .wav files")
    
    # List all files
    print("\nFiles to test:")
    for i, wav_file in enumerate(wav_files, 1):
        print(f"  {i}. {wav_file.name}")
    
    # Test each file
    print("\n" + "=" * 80)
    print("🧪 TESTING FILES")
    print("=" * 80)
    
    results = {}
    healthy_count = 0
    parkinsons_count = 0
    failed_count = 0
    
    for i, wav_file in enumerate(wav_files, 1):
        filename = wav_file.name
        print(f"\n[{i}/{len(wav_files)}] {filename}")
        print("-" * 80)
        
        result = predict_file(str(wav_file), model, scaler, denoise=True)
        
        if result:
            results[filename] = result
            
            # Display result
            print(f"  ✅ Prediction: {result['prediction']}")
            print(f"  📊 Confidence: {result['confidence']*100:.1f}%")
            print(f"  📈 Probabilities:")
            print(f"     • Healthy:       {result['prob_healthy']*100:.2f}%")
            print(f"     • Parkinson's:   {result['prob_parkinsons']*100:.2f}%")
            
            if result['prediction'] == 'Healthy':
                healthy_count += 1
            else:
                parkinsons_count += 1
        else:
            results[filename] = {'error': 'Failed to process'}
            failed_count += 1
            print(f"  ❌ Failed to process file")
    
    # Summary
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    print(f"Total files tested: {len(wav_files)}")
    print(f"Successful: {len(wav_files) - failed_count}")
    print(f"Failed: {failed_count}")
    print(f"\nPrediction Results:")
    print(f"  🟢 Healthy:       {healthy_count} ({healthy_count/len(wav_files)*100:.1f}%)")
    print(f"  🔴 Parkinson's:   {parkinsons_count} ({parkinsons_count/len(wav_files)*100:.1f}%)")
    
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
    
    # Save results
    output_file = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    output_data = {
        'test_timestamp': datetime.now().isoformat(),
        'test_directory': TEST_DIR,
        'model_used': model_path,
        'scaler_used': scaler_path,
        'feature_set': FEATURE_SET,
        'summary': {
            'total_files': len(wav_files),
            'successful': len(wav_files) - failed_count,
            'failed': failed_count,
            'healthy': healthy_count,
            'parkinsons': parkinsons_count
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
