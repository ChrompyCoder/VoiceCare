"""
Inference script for Parkinson's Disease detection
Load a trained model and make predictions on new audio files
"""

import os
import sys
import numpy as np
import tensorflow as tf
from utils.data_loader import extract_features

def load_model(model_path):
    """Load a trained model from file"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    model = tf.keras.models.load_model(model_path)
    print(f"Model loaded from {model_path}")
    return model

def predict_single_file(model, file_path):
    """
    Predict Parkinson's disease from a single audio file
    
    Args:
        model: Trained Keras model
        file_path: Path to audio file
    
    Returns:
        label: "Healthy" or "Parkinson"
        confidence: Prediction confidence (0-1)
    """
    # Extract features
    mel = extract_features(file_path)
    
    if mel is None:
        print(f"Error: Could not process file {file_path}")
        return None, None
    
    # Reshape for model input
    mel = mel[np.newaxis, ..., np.newaxis]
    
    # Make prediction
    prediction = model.predict(mel, verbose=0)[0][0]
    label = "Parkinson" if prediction > 0.5 else "Healthy"
    
    return label, prediction

def main():
    if len(sys.argv) < 3:
        print("Usage: python inference.py <model_path> <audio_file>")
        print("Example: python inference.py models/best_model.h5 data/figdata/HC_AH/sample.wav")
        sys.exit(1)
    
    model_path = sys.argv[1]
    audio_file = sys.argv[2]
    
    # Load model
    print("Loading model...")
    model = load_model(model_path)
    
    # Make prediction
    print(f"\nAnalyzing audio file: {audio_file}")
    label, confidence = predict_single_file(model, audio_file)
    
    if label is not None:
        print(f"\nPrediction: {label}")
        print(f"Confidence: {confidence*100:.2f}%")
        
        if label == "Parkinson":
            print("\nThe model indicates potential Parkinson's disease markers in the voice sample.")
        else:
            print("\nThe model indicates a healthy voice pattern.")
    else:
        print("Failed to process the audio file.")

if __name__ == "__main__":
    main()
