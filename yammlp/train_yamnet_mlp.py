"""
Parkinson's Disease Detection using YAMNet + MLP
================================================
Feature extraction using YAMNet (Google's pre-trained audio model)
Classification using Multi-Layer Perceptron (MLP)

Dataset:
  - figdata: HC_AH=Healthy, PD_AH=Parkinson's
  - gitdata/original-speech-dataset: Parkinson's only
  - Duration filter: > 2 seconds
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys
import soundfile as sf
import librosa
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, 
                            classification_report, roc_curve)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# =============================
# 📋 CONFIGURATION
# =============================
MIN_DURATION = 2.0  # Minimum audio duration in seconds
DATA_DIR = Path(__file__).parent.parent / "data"  # Absolute path to data/
RESULTS_DIR = "results"
MODELS_DIR = "models"
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_FOLDS = 5
EPOCHS = 150
BATCH_SIZE = 16

# YAMNet model URL
YAMNET_MODEL_URL = 'https://tfhub.dev/google/yamnet/1'

# MLP architecture
MLP_LAYERS = [512, 256, 128, 64]
DROPOUT_RATE = 0.5
L2_REG = 0.001

# Create directories
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# =============================
# 🎵 HELPER FUNCTIONS
# =============================

def get_audio_duration(audio_path):
    """Get duration of audio file in seconds."""
    try:
        info = sf.info(audio_path)
        return info.duration
    except Exception as e:
        return 0

def load_gitdata_original():
    """Load audio files from gitdata/original-speech-dataset (Parkinson's only)."""
    print("\n" + "=" * 70)
    print("📂 Loading gitdata (original-speech-dataset - Parkinson's)...")
    print("=" * 70)
    
    audio_files = []
    labels = []
    
    original_path = Path(DATA_DIR) / "gitdata" / "original-speech-dataset"
    
    # Load from DL directory
    dl_path = original_path / "DL"
    if dl_path.exists():
        for txt_file in dl_path.glob("DL*.txt"):
            with open(txt_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        audio_path = dl_path / line
                        if audio_path.exists() and audio_path.suffix == '.wav':
                            duration = get_audio_duration(str(audio_path))
                            if duration >= MIN_DURATION:
                                audio_files.append(str(audio_path))
                                labels.append(1)  # Parkinson's
    
    # Load from LW directory
    lw_path = original_path / "LW"
    if lw_path.exists():
        for txt_file in lw_path.glob("LW*.txt"):
            with open(txt_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        audio_path = lw_path / line
                        if audio_path.exists() and audio_path.suffix == '.wav':
                            duration = get_audio_duration(str(audio_path))
                            if duration >= MIN_DURATION:
                                audio_files.append(str(audio_path))
                                labels.append(1)  # Parkinson's
    
    # Load from Faces directories
    faces_path = original_path / "Faces"
    if faces_path.exists():
        for speaker_dir in faces_path.glob("*_ori"):
            if speaker_dir.is_dir():
                for audio_file in speaker_dir.glob("*.wav"):
                    duration = get_audio_duration(str(audio_file))
                    if duration >= MIN_DURATION:
                        audio_files.append(str(audio_file))
                        labels.append(1)  # Parkinson's
    
    print(f"  ✓ Loaded {len(audio_files)} files (duration > {MIN_DURATION}s)")
    print(f"  ✓ All labeled as Parkinson's (label=1)")
    
    return audio_files, labels

def load_figdata():
    """Load audio files from figdata (HC_AH=Healthy, PD_AH=Parkinson's)."""
    print("\n" + "=" * 70)
    print("📂 Loading figdata (HC_AH=Healthy, PD_AH=Parkinson's)...")
    print("=" * 70)
    
    audio_files = []
    labels = []
    
    figdata_path = Path(DATA_DIR) / "figdata"
    
    # Load HC_AH (Healthy Controls)
    hc_path = figdata_path / "HC_AH"
    if hc_path.exists():
        hc_count = 0
        for audio_file in hc_path.glob("*.wav"):
            duration = get_audio_duration(str(audio_file))
            if duration >= MIN_DURATION:
                audio_files.append(str(audio_file))
                labels.append(0)  # Healthy
                hc_count += 1
        print(f"  ✓ Loaded {hc_count} Healthy files (HC_AH, duration > {MIN_DURATION}s)")
    
    # Load PD_AH (Parkinson's Disease)
    pd_path = figdata_path / "PD_AH"
    if pd_path.exists():
        pd_count = 0
        for audio_file in pd_path.glob("*.wav"):
            duration = get_audio_duration(str(audio_file))
            if duration >= MIN_DURATION:
                audio_files.append(str(audio_file))
                labels.append(1)  # Parkinson's
                pd_count += 1
        print(f"  ✓ Loaded {pd_count} Parkinson's files (PD_AH, duration > {MIN_DURATION}s)")
    
    print(f"  ✓ Total figdata: {len(audio_files)} files")
    
    return audio_files, labels

def load_yamnet_model():
    """Load pre-trained YAMNet model from TensorFlow Hub."""
    print("\n" + "=" * 70)
    print("🤖 Loading YAMNet model from TensorFlow Hub...")
    print("=" * 70)
    
    yamnet_model = hub.load(YAMNET_MODEL_URL)
    print("  ✓ YAMNet model loaded successfully")
    
    return yamnet_model

def extract_yamnet_embeddings(audio_path, yamnet_model):
    """Extract YAMNet embeddings from audio file."""
    try:
        # Load audio at 16kHz (YAMNet requirement)
        waveform, sr = librosa.load(audio_path, sr=16000, mono=True)
        
        # YAMNet expects float32 waveform in [-1.0, +1.0]
        waveform = waveform.astype(np.float32)
        
        # Get YAMNet embeddings
        scores, embeddings, spectrogram = yamnet_model(waveform)
        
        # Use mean pooling over time to get fixed-size representation
        embedding_mean = np.mean(embeddings.numpy(), axis=0)
        
        return embedding_mean
        
    except Exception as e:
        print(f"Error extracting YAMNet embeddings from {audio_path}: {e}")
        return None

def extract_features_from_dataset(audio_files, yamnet_model):
    """Extract YAMNet features from all audio files."""
    print("\n" + "=" * 70)
    print("🎵 Extracting YAMNet features...")
    print("=" * 70)
    print(f"  YAMNet: 1024-dimensional embeddings")
    print("=" * 70)
    
    features_list = []
    valid_files = []
    
    for audio_file in tqdm(audio_files, desc="Processing audio files"):
        embedding = extract_yamnet_embeddings(audio_file, yamnet_model)
        
        if embedding is not None:
            features_list.append(embedding)
            valid_files.append(audio_file)
    
    if not features_list:
        raise ValueError("No features extracted! Check audio files.")
    
    X = np.array(features_list)
    
    print(f"\n  ✓ Extracted features from {len(valid_files)} files")
    print(f"  ✓ Feature dimensions: {X.shape}")
    
    return X, valid_files

def build_mlp_model(input_dim):
    """
    Build Multi-Layer Perceptron (MLP) model.
    
    Architecture:
    - Input: YAMNet embeddings (1024 dimensions)
    - Hidden layers with batch normalization and dropout
    - Output: Binary classification (Healthy vs Parkinson's)
    """
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        
        # First hidden layer
        layers.Dense(MLP_LAYERS[0], kernel_regularizer=keras.regularizers.l2(L2_REG)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(DROPOUT_RATE),
        
        # Second hidden layer
        layers.Dense(MLP_LAYERS[1], kernel_regularizer=keras.regularizers.l2(L2_REG)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(DROPOUT_RATE),
        
        # Third hidden layer
        layers.Dense(MLP_LAYERS[2], kernel_regularizer=keras.regularizers.l2(L2_REG)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(DROPOUT_RATE),
        
        # Fourth hidden layer
        layers.Dense(MLP_LAYERS[3], kernel_regularizer=keras.regularizers.l2(L2_REG)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(DROPOUT_RATE),
        
        # Output layer
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')]
    )
    
    return model

# =============================
# 🎯 MAIN TRAINING PIPELINE
# =============================

def main():
    print("\n" + "=" * 70)
    print("🎯 Parkinson's Detection: YAMNet + MLP")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  - Minimum duration: {MIN_DURATION} seconds")
    print(f"  - YAMNet embeddings: 1024 dimensions")
    print(f"  - MLP layers: {MLP_LAYERS}")
    print(f"  - Dropout rate: {DROPOUT_RATE}")
    print(f"  - L2 regularization: {L2_REG}")
    print(f"  - Epochs: {EPOCHS}")
    print(f"  - Batch size: {BATCH_SIZE}")
    print(f"  - Test size: {TEST_SIZE * 100}%")
    print("=" * 70)
    
    # =============================
    # 📊 LOAD DATA
    # =============================
    
    # Load gitdata (original-speech-dataset)
    git_files, git_labels = load_gitdata_original()
    
    # Load figdata
    fig_files, fig_labels = load_figdata()
    
    # Combine datasets
    all_files = git_files + fig_files
    all_labels = git_labels + fig_labels
    
    print("\n" + "=" * 70)
    print(f"📊 Combined Dataset Summary")
    print("=" * 70)
    print(f"  Total files: {len(all_files)}")
    
    if len(all_files) == 0:
        print("  ❌ ERROR: No audio files loaded!")
        print("  Please check:")
        print("     - Data paths in DATA_DIR")
        print("     - Audio file durations (min 2.0 seconds)")
        sys.exit(1)
    
    print(f"  Healthy: {all_labels.count(0)} ({all_labels.count(0) / len(all_labels) * 100:.1f}%)")
    print(f"  Parkinson's: {all_labels.count(1)} ({all_labels.count(1) / len(all_labels) * 100:.1f}%)")
    
    # =============================
    # 🤖 LOAD YAMNET MODEL
    # =============================
    
    yamnet_model = load_yamnet_model()
    
    # =============================
    # 🎵 FEATURE EXTRACTION
    # =============================
    
    X, valid_files = extract_features_from_dataset(all_files, yamnet_model)
    
    # Update labels for valid files only
    y = np.array([all_labels[all_files.index(f)] for f in valid_files])
    
    print(f"\n  Dataset after feature extraction:")
    print(f"  ✓ Healthy samples: {np.sum(y == 0)} ({np.sum(y == 0) / len(y) * 100:.1f}%)")
    print(f"  ✓ Parkinson's samples: {np.sum(y == 1)} ({np.sum(y == 1) / len(y) * 100:.1f}%)")
    
    # =============================
    # 📊 FEATURE SCALING
    # =============================
    
    print("\n" + "=" * 70)
    print("📊 Scaling features...")
    print("=" * 70)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"  ✓ Features scaled using StandardScaler")
    
    # =============================
    # 🔀 TRAIN-TEST SPLIT
    # =============================
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print("\n" + "=" * 70)
    print("🔀 Train-Test Split")
    print("=" * 70)
    print(f"  Training samples: {len(X_train)}")
    print(f"    - Healthy: {np.sum(y_train == 0)}")
    print(f"    - Parkinson's: {np.sum(y_train == 1)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"    - Healthy: {np.sum(y_test == 0)}")
    print(f"    - Parkinson's: {np.sum(y_test == 1)}")
    
    # =============================
    # 🤖 BUILD AND TRAIN MLP MODEL
    # =============================
    
    print("\n" + "=" * 70)
    print("🤖 Building MLP model...")
    print("=" * 70)
    
    model = build_mlp_model(X_scaled.shape[1])
    
    print(model.summary())
    
    # Callbacks
    model_path = f"{MODELS_DIR}/yamnet_mlp_{timestamp}.h5"
    
    checkpoint = keras.callbacks.ModelCheckpoint(
        model_path,
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )
    
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
    )
    
    lr_schedule = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=1
    )
    
    print("\n" + "=" * 70)
    print("🚀 Training MLP model...")
    print("=" * 70)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[checkpoint, early_stopping, lr_schedule],
        verbose=1
    )
    
    # =============================
    # 📈 EVALUATION
    # =============================
    
    print("\n" + "=" * 70)
    print("📈 Evaluating on test set...")
    print("=" * 70)
    
    # Load best model
    model.load_weights(model_path)
    
    # Predictions
    y_pred_proba = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"\n  Test Results:")
    print(f"  {'=' * 50}")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  AUC-ROC:   {auc:.4f}")
    print(f"  {'=' * 50}")
    
    # Classification report
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Healthy', 'Parkinson\'s']))
    
    # =============================
    # 📊 VISUALIZATIONS
    # =============================
    
    print("\n" + "=" * 70)
    print("📊 Creating visualizations...")
    print("=" * 70)
    
    # Training history
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train', linewidth=2)
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Train', linewidth=2)
    axes[0, 1].plot(history.history['val_loss'], label='Validation', linewidth=2)
    axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Precision
    axes[1, 0].plot(history.history['precision'], label='Train', linewidth=2)
    axes[1, 0].plot(history.history['val_precision'], label='Validation', linewidth=2)
    axes[1, 0].set_title('Model Precision', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Recall
    axes[1, 1].plot(history.history['recall'], label='Train', linewidth=2)
    axes[1, 1].plot(history.history['val_recall'], label='Validation', linewidth=2)
    axes[1, 1].set_title('Model Recall', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    history_path = f"{RESULTS_DIR}/training_history_{timestamp}.png"
    plt.savefig(history_path, dpi=300)
    print(f"  ✓ Training history saved: {history_path}")
    plt.close()
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Healthy', 'Parkinson\'s'],
                yticklabels=['Healthy', 'Parkinson\'s'])
    plt.title('Confusion Matrix - YAMNet + MLP', fontsize=14, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    cm_path = f"{RESULTS_DIR}/confusion_matrix_{timestamp}.png"
    plt.savefig(cm_path, dpi=300)
    print(f"  ✓ Confusion matrix saved: {cm_path}")
    plt.close()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'YAMNet+MLP (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - YAMNet + MLP', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = f"{RESULTS_DIR}/roc_curve_{timestamp}.png"
    plt.savefig(roc_path, dpi=300)
    print(f"  ✓ ROC curve saved: {roc_path}")
    plt.close()
    
    # =============================
    # 💾 SAVE SCALER AND RESULTS
    # =============================
    
    print("\n" + "=" * 70)
    print("💾 Saving scaler and results...")
    print("=" * 70)
    
    # Save scaler
    import pickle
    scaler_path = f"{MODELS_DIR}/scaler_{timestamp}.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"  ✓ Scaler saved: {scaler_path}")
    
    # Save metrics
    metrics = {
        'timestamp': timestamp,
        'model_type': 'YAMNet + MLP',
        'min_duration': MIN_DURATION,
        'yamnet_features': 1024,
        'mlp_layers': MLP_LAYERS,
        'dropout_rate': DROPOUT_RATE,
        'l2_reg': L2_REG,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'test_size': TEST_SIZE,
        'total_samples': len(X),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'auc_roc': float(auc),
        'dataset_info': {
            'gitdata_original': len(git_files),
            'figdata_healthy': fig_labels.count(0),
            'figdata_parkinsons': fig_labels.count(1),
            'total_healthy': all_labels.count(0),
            'total_parkinsons': all_labels.count(1)
        }
    }
    
    metrics_path = f"{RESULTS_DIR}/metrics_{timestamp}.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"  ✓ Metrics saved: {metrics_path}")
    
    print("\n" + "=" * 70)
    print("✅ Training Complete!")
    print("=" * 70)
    print(f"\nBest Model: {model_path}")
    print(f"Scaler: {scaler_path}")
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"AUC-ROC: {auc:.4f}")
    print("=" * 70)

if __name__ == "__main__":
    main()
