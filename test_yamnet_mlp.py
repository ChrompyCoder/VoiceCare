"""
Inference script for YAMNet + MLP Parkinson's Detection
Tests trained model on audio files
"""
import os
import sys
import numpy as np
import librosa
import tensorflow_hub as hub
import tensorflow as tf
from pathlib import Path

# Load YAMNet model
def load_yamnet():
    """Load pre-trained YAMNet model"""
    return hub.load('https://tfhub.dev/google/yamnet/1')

# Load trained MLP model
def load_mlp_model(model_path):
    """Load trained MLP model"""
    return tf.keras.models.load_model(model_path)

def extract_yamnet_features(audio_path, yamnet_model):
    """Extract YAMNet embeddings from audio"""
    try:
        # Load audio
        waveform, sr = librosa.load(audio_path, sr=16000, mono=True)
        waveform = waveform.astype(np.float32)
        
        # Extract embeddings
        scores, embeddings, spectrogram = yamnet_model(waveform)
        
        # Average over time
        embedding_mean = np.mean(embeddings.numpy(), axis=0)
        
        return embedding_mean
    except Exception as e:
        print(f"  ❌ Error extracting features: {e}")
        return None

def predict(audio_path, yamnet_model, mlp_model):
    """Make prediction on audio file"""
    # Extract features
    features = extract_yamnet_features(audio_path, yamnet_model)
    if features is None:
        return None
    
    # Reshape for model
    features = features.reshape(1, -1)
    
    # Predict
    probability = mlp_model.predict(features, verbose=0)[0][0]
    
    return probability

def main():
    # Model paths
    MLP_MODEL = "yammlp/models/yamnet_mlp_20251108_163332.h5"
    TEST_FOLDER = Path("backend/test")
    
    # Check if model exists
    if not os.path.exists(MLP_MODEL):
        print(f"❌ Model not found: {MLP_MODEL}")
        sys.exit(1)
    
    # Get test files
    audio_files = sorted(list(TEST_FOLDER.glob("*.wav")))
    if not audio_files:
        print(f"❌ No audio files found in {TEST_FOLDER}")
        sys.exit(1)
    
    print("="*80)
    print(f"🎵 TESTING YAMNET + MLP MODEL ON {len(audio_files)} FILES")
    print("="*80)
    
    # Load models
    print("\n📦 Loading models...")
    print("  Loading YAMNet...")
    yamnet_model = load_yamnet()
    print("  ✅ YAMNet loaded")
    
    print("  Loading trained MLP...")
    mlp_model = load_mlp_model(MLP_MODEL)
    print("  ✅ MLP model loaded\n")
    
    # Test each file
    results = []
    for audio_file in audio_files:
        print("="*80)
        print(f"📂 File: {audio_file.name}")
        print("="*80)
        
        probability = predict(str(audio_file), yamnet_model, mlp_model)
        
        if probability is None:
            print("  ❌ Failed to process")
            continue
        
        # Interpret
        label = "🔴 PARKINSON'S DISEASE" if probability > 0.5 else "🟢 HEALTHY"
        confidence = probability if probability > 0.5 else (1 - probability)
        
        print(f"  ✅ Prediction: {label}")
        print(f"  📊 P(Parkinson's): {probability*100:.2f}%")
        print(f"  📊 P(Healthy): {(1-probability)*100:.2f}%")
        print(f"  📊 Confidence: {confidence*100:.2f}%")
        
        # Risk level
        if probability < 0.30:
            risk = "🟢 LOW"
        elif probability < 0.70:
            risk = "🟡 MODERATE"
        else:
            risk = "🔴 HIGH"
        print(f"  ⚠️  Risk Level: {risk}")
        
        results.append({
            'file': audio_file.name,
            'probability': probability,
            'label': label
        })
        print()
    
    # Summary
    print("\n" + "="*80)
    print("📊 SUMMARY")
    print("="*80)
    
    healthy = sum(1 for r in results if r['probability'] <= 0.5)
    parkinsons = sum(1 for r in results if r['probability'] > 0.5)
    
    print(f"Total files: {len(results)}")
    print(f"Healthy: {healthy}")
    print(f"Parkinson's: {parkinsons}")
    print("="*80)

if __name__ == "__main__":
    main()
