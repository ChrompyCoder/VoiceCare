"""
Utility functions for loading and preprocessing audio data
for Parkinson's Disease detection
"""

import os
import librosa
import numpy as np

def load_audio_files_from_directory(directory, label):
    """
    Load all .wav files from a directory with the given label
    
    Args:
        directory: Path to directory containing .wav files
        label: Label to assign to all files (0 or 1)
    
    Returns:
        List of file paths and corresponding labels
    """
    file_paths = []
    labels = []
    
    if not os.path.exists(directory):
        print(f"Warning: Directory {directory} does not exist")
        return file_paths, labels
    
    for file in os.listdir(directory):
        if file.endswith(".wav"):
            file_path = os.path.join(directory, file)
            file_paths.append(file_path)
            labels.append(label)
    
    return file_paths, labels

def load_parkinson_dataset(base_path):
    """
    Load the complete Parkinson's disease dataset
    
    Directory structure:
    base_path/
        HC_AH/  -> Healthy Controls (label 0)
        PD_AH/  -> Parkinson's Disease (label 1)
    
    Returns:
        X: Array of file paths
        y: Array of labels (0=Healthy, 1=Parkinson's)
    """
    X, y = [], []
    
    # Load Healthy Controls
    hc_path = os.path.join(base_path, "HC_AH")
    hc_files, hc_labels = load_audio_files_from_directory(hc_path, label=0)
    X.extend(hc_files)
    y.extend(hc_labels)
    print(f"Loaded {len(hc_files)} healthy control files")
    
    # Load Parkinson's Disease patients
    pd_path = os.path.join(base_path, "PD_AH")
    pd_files, pd_labels = load_audio_files_from_directory(pd_path, label=1)
    X.extend(pd_files)
    y.extend(pd_labels)
    print(f"Loaded {len(pd_files)} Parkinson's patient files")
    
    return np.array(X), np.array(y)

def extract_features(file_path, sample_rate=22050, duration=5, n_mels=128, n_fft=2048, hop_length=512):
    """
    Extract Mel-spectrogram features from an audio file
    
    Args:
        file_path: Path to audio file
        sample_rate: Target sample rate
        duration: Duration to load (in seconds)
        n_mels: Number of Mel bands
        n_fft: FFT window size
        hop_length: Number of samples between frames
    
    Returns:
        Mel-spectrogram in dB scale
    """
    try:
        # Load audio
        y, sr = librosa.load(file_path, sr=sample_rate, duration=duration)
        
        # Pad or truncate to fixed length
        samples_per_file = sample_rate * duration
        if len(y) > samples_per_file:
            y = y[:samples_per_file]
        else:
            y = np.pad(y, (0, max(0, samples_per_file - len(y))), "constant")
        
        # Extract Mel-spectrogram
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels
        )
        
        # Convert to dB scale
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        return mel_db
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def extract_mfcc_features(file_path, sample_rate=22050, duration=5, n_mfcc=40):
    """
    Extract MFCC features from an audio file (alternative to Mel-spectrogram)
    
    Args:
        file_path: Path to audio file
        sample_rate: Target sample rate
        duration: Duration to load (in seconds)
        n_mfcc: Number of MFCC coefficients
    
    Returns:
        MFCC features
    """
    try:
        # Load audio
        y, sr = librosa.load(file_path, sr=sample_rate, duration=duration)
        
        # Pad or truncate to fixed length
        samples_per_file = sample_rate * duration
        if len(y) > samples_per_file:
            y = y[:samples_per_file]
        else:
            y = np.pad(y, (0, max(0, samples_per_file - len(y))), "constant")
        
        # Extract MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        
        return mfcc
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def get_audio_info(file_path):
    """
    Get information about an audio file
    
    Returns:
        Dictionary with duration, sample_rate, and number of samples
    """
    try:
        y, sr = librosa.load(file_path, sr=None)
        duration = librosa.get_duration(y=y, sr=sr)
        
        return {
            'duration': duration,
            'sample_rate': sr,
            'num_samples': len(y)
        }
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None
