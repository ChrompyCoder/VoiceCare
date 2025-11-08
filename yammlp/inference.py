"""
Parkinson's Disease Detection - Inference with Audio Denoising
=============================================================
This script performs inference on new audio files with:
  1. Audio denoising using noisereduce
  2. YAMNet feature extraction
  3. MLP classification

Usage:
    python inference.py --audio path/to/audio.wav --model path/to/model.h5 --scaler path/to/scaler.pkl
    
    or
    
    python inference.py --audio path/to/audio.wav  # uses latest model automatically
"""

import os
import sys
import argparse
import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
import pickle
import glob
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# YAMNet model URL
YAMNET_MODEL_URL = 'https://tfhub.dev/google/yamnet/1'

class ParkinsonDetector:
    """Parkinson's Disease Detection using YAMNet + MLP with audio denoising."""
    
    def __init__(self, model_path=None, scaler_path=None):
        """
        Initialize the detector.
        
        Args:
            model_path: Path to trained MLP model (.h5 file)
            scaler_path: Path to fitted scaler (.pkl file)
        """
        print("\n" + "=" * 70)
        print("🎯 Initializing Parkinson's Detector (YAMNet + MLP)")
        print("=" * 70)
        
        # Auto-detect latest model if not provided
        if model_path is None:
            model_path = self._find_latest_model()
        if scaler_path is None:
            scaler_path = self._find_latest_scaler()
        
        # Load YAMNet
        print("\n🤖 Loading YAMNet model...")
        self.yamnet_model = hub.load(YAMNET_MODEL_URL)
        print("  ✓ YAMNet loaded successfully")
        
        # Load MLP model
        print(f"\n🧠 Loading MLP model: {model_path}")
        self.model = keras.models.load_model(model_path)
        print("  ✓ MLP model loaded successfully")
        
        # Load scaler
        print(f"\n📊 Loading scaler: {scaler_path}")
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        print("  ✓ Scaler loaded successfully")
        
        print("\n" + "=" * 70)
        print("✅ Detector initialized and ready!")
        print("=" * 70)
    
    def _find_latest_model(self):
        """Find the latest trained model."""
        models_dir = "models"
        model_files = glob.glob(f"{models_dir}/yamnet_mlp_*.h5")
        if not model_files:
            raise FileNotFoundError(f"No model files found in {models_dir}/")
        latest_model = max(model_files, key=os.path.getctime)
        return latest_model
    
    def _find_latest_scaler(self):
        """Find the latest scaler."""
        models_dir = "models"
        scaler_files = glob.glob(f"{models_dir}/scaler_*.pkl")
        if not scaler_files:
            raise FileNotFoundError(f"No scaler files found in {models_dir}/")
        latest_scaler = max(scaler_files, key=os.path.getctime)
        return latest_scaler
    
    def denoise_audio(self, audio_path, output_path=None):
        """
        Denoise audio file using noisereduce library.
        
        Args:
            audio_path: Path to input audio file
            output_path: Optional path to save denoised audio
        
        Returns:
            Denoised audio waveform and sample rate
        """
        print(f"\n🔊 Denoising audio: {audio_path}")
        
        # Load audio
        waveform, sr = librosa.load(audio_path, sr=None, mono=True)
        
        print(f"  ✓ Loaded audio: duration={len(waveform)/sr:.2f}s, sr={sr}Hz")
        
        # Apply noise reduction
        # stationary=True assumes constant background noise
        # prop_decrease=1.0 is aggressive noise reduction
        denoised = nr.reduce_noise(
            y=waveform,
            sr=sr,
            stationary=True,
            prop_decrease=1.0
        )
        
        print(f"  ✓ Noise reduction applied")
        
        # Optionally save denoised audio
        if output_path:
            sf.write(output_path, denoised, sr)
            print(f"  ✓ Denoised audio saved: {output_path}")
        
        return denoised, sr
    
    def extract_yamnet_features(self, waveform, sr):
        """
        Extract YAMNet embeddings from audio waveform.
        
        Args:
            waveform: Audio waveform (numpy array)
            sr: Sample rate
        
        Returns:
            YAMNet embedding (1024-dimensional vector)
        """
        print(f"\n🎵 Extracting YAMNet features...")
        
        # Resample to 16kHz if necessary (YAMNet requirement)
        if sr != 16000:
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
            sr = 16000
            print(f"  ✓ Resampled to 16kHz")
        
        # Ensure float32 format
        waveform = waveform.astype(np.float32)
        
        # Get YAMNet embeddings
        scores, embeddings, spectrogram = self.yamnet_model(waveform)
        
        # Use mean pooling over time
        embedding_mean = np.mean(embeddings.numpy(), axis=0)
        
        print(f"  ✓ YAMNet embedding extracted: shape={embedding_mean.shape}")
        
        return embedding_mean
    
    def predict(self, audio_path, denoise=True, save_denoised=False):
        """
        Predict Parkinson's disease from audio file.
        
        Args:
            audio_path: Path to audio file
            denoise: Whether to denoise the audio (default: True)
            save_denoised: Whether to save denoised audio (default: False)
        
        Returns:
            Dictionary with prediction results
        """
        print("\n" + "=" * 70)
        print(f"🎤 Processing: {audio_path}")
        print("=" * 70)
        
        # Step 1: Denoise audio
        if denoise:
            denoised_path = None
            if save_denoised:
                audio_name = Path(audio_path).stem
                denoised_path = f"denoised_{audio_name}.wav"
            
            waveform, sr = self.denoise_audio(audio_path, output_path=denoised_path)
        else:
            print(f"\n🔊 Loading audio (no denoising): {audio_path}")
            waveform, sr = librosa.load(audio_path, sr=None, mono=True)
            print(f"  ✓ Loaded: duration={len(waveform)/sr:.2f}s, sr={sr}Hz")
        
        # Step 2: Extract YAMNet features
        embedding = self.extract_yamnet_features(waveform, sr)
        
        # Step 3: Scale features
        embedding_scaled = self.scaler.transform(embedding.reshape(1, -1))
        
        # Step 4: Make prediction
        print(f"\n🧠 Making prediction...")
        prediction_proba = self.model.predict(embedding_scaled, verbose=0)[0][0]
        prediction_class = int(prediction_proba > 0.5)
        
        # Step 5: Prepare results
        class_label = "Parkinson's Disease" if prediction_class == 1 else "Healthy"
        confidence = prediction_proba if prediction_class == 1 else (1 - prediction_proba)
        
        results = {
            'audio_file': audio_path,
            'denoised': denoise,
            'prediction': class_label,
            'confidence': float(confidence),
            'probability_parkinsons': float(prediction_proba),
            'probability_healthy': float(1 - prediction_proba)
        }
        
        # Print results
        print("\n" + "=" * 70)
        print("📊 PREDICTION RESULTS")
        print("=" * 70)
        print(f"  Audio File: {audio_path}")
        print(f"  Denoised: {'Yes' if denoise else 'No'}")
        print(f"\n  Prediction: {class_label}")
        print(f"  Confidence: {confidence * 100:.2f}%")
        print(f"\n  Probabilities:")
        print(f"    - Healthy:       {results['probability_healthy'] * 100:.2f}%")
        print(f"    - Parkinson's:   {results['probability_parkinsons'] * 100:.2f}%")
        print("=" * 70)
        
        return results
    
    def batch_predict(self, audio_files, denoise=True, save_denoised=False):
        """
        Predict on multiple audio files.
        
        Args:
            audio_files: List of audio file paths
            denoise: Whether to denoise the audio
            save_denoised: Whether to save denoised audio
        
        Returns:
            List of prediction results
        """
        results = []
        
        print("\n" + "=" * 70)
        print(f"📁 Batch Prediction: {len(audio_files)} files")
        print("=" * 70)
        
        for i, audio_file in enumerate(audio_files, 1):
            print(f"\n[{i}/{len(audio_files)}]")
            try:
                result = self.predict(audio_file, denoise=denoise, save_denoised=save_denoised)
                results.append(result)
            except Exception as e:
                print(f"  ❌ Error processing {audio_file}: {e}")
                results.append({
                    'audio_file': audio_file,
                    'error': str(e)
                })
        
        # Summary
        print("\n" + "=" * 70)
        print("📊 BATCH PREDICTION SUMMARY")
        print("=" * 70)
        
        successful = [r for r in results if 'prediction' in r]
        healthy_count = sum(1 for r in successful if r['prediction'] == 'Healthy')
        parkinsons_count = sum(1 for r in successful if r['prediction'] == "Parkinson's Disease")
        
        print(f"  Total files processed: {len(audio_files)}")
        print(f"  Successful predictions: {len(successful)}")
        print(f"  Failed: {len(audio_files) - len(successful)}")
        print(f"\n  Results:")
        print(f"    - Healthy: {healthy_count}")
        print(f"    - Parkinson's: {parkinsons_count}")
        print("=" * 70)
        
        return results


def main():
    parser = argparse.ArgumentParser(
        description='Parkinson\'s Disease Detection - Inference with Audio Denoising',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file prediction with denoising (default)
  python inference.py --audio sample.wav
  
  # Single file without denoising
  python inference.py --audio sample.wav --no-denoise
  
  # Save denoised audio
  python inference.py --audio sample.wav --save-denoised
  
  # Specify model and scaler
  python inference.py --audio sample.wav --model models/yamnet_mlp_20231108_120000.h5 --scaler models/scaler_20231108_120000.pkl
  
  # Batch prediction on directory
  python inference.py --audio-dir /path/to/audio/folder/
        """
    )
    
    parser.add_argument('--audio', type=str, help='Path to audio file')
    parser.add_argument('--audio-dir', type=str, help='Directory containing audio files')
    parser.add_argument('--model', type=str, help='Path to trained model (.h5 file)')
    parser.add_argument('--scaler', type=str, help='Path to scaler (.pkl file)')
    parser.add_argument('--no-denoise', action='store_true', help='Disable audio denoising')
    parser.add_argument('--save-denoised', action='store_true', help='Save denoised audio files')
    
    args = parser.parse_args()
    
    # Check arguments
    if not args.audio and not args.audio_dir:
        parser.error("Please provide either --audio or --audio-dir")
    
    # Initialize detector
    detector = ParkinsonDetector(model_path=args.model, scaler_path=args.scaler)
    
    # Denoise flag
    denoise = not args.no_denoise
    
    # Single file or batch prediction
    if args.audio:
        # Single file
        if not os.path.exists(args.audio):
            print(f"❌ Error: Audio file not found: {args.audio}")
            sys.exit(1)
        
        results = detector.predict(
            args.audio,
            denoise=denoise,
            save_denoised=args.save_denoised
        )
        
    elif args.audio_dir:
        # Batch prediction
        if not os.path.isdir(args.audio_dir):
            print(f"❌ Error: Directory not found: {args.audio_dir}")
            sys.exit(1)
        
        # Find all audio files
        audio_files = []
        for ext in ['*.wav', '*.WAV']:
            audio_files.extend(glob.glob(os.path.join(args.audio_dir, ext)))
        
        if not audio_files:
            print(f"❌ Error: No .wav files found in {args.audio_dir}")
            sys.exit(1)
        
        results = detector.batch_predict(
            audio_files,
            denoise=denoise,
            save_denoised=args.save_denoised
        )
        
        # Optionally save results to JSON
        import json
        results_file = "batch_predictions.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\n💾 Results saved to: {results_file}")


if __name__ == "__main__":
    main()
