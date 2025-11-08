"""
Project: Voice-Based Parkinson's Disease Detection using CNN + BiLSTM
Enhanced Version: Uses ALL available data from gitdata and figdata

Dataset Summary:
GITDATA (Parkinson's patients only):
  - denoised-speech-dataset/Faces: ~113 samples (BG_au, JC_au, MJ_au, SK_au, TP_au, TS_au)
  - original-speech-dataset/Faces: ~113 samples (BG_ori, JC_ori, MJ_ori, SK_ori, TP_ori, TS_ori)
  - Total: ~226 Parkinson's samples

FIGDATA (Mixed):
  - HC_AH: 41 Healthy Control samples
  - PD_AH: 40 Parkinson's Disease samples
  - Total: 81 samples

COMBINED TOTAL: ~443 audio samples (with augmentation: ~1300+ samples)

Training Strategy:
1. Phase 1: Load and prepare gitdata (all Parkinson's samples)
2. Phase 2: Load and prepare figdata (healthy + Parkinson's samples)
3. Phase 3: Combine all data and split into train/validation/test
4. Train CNN+BiLSTM model on combined dataset
5. Monitor for overfitting and apply regularization
6. Verify feature extraction quality

Features:
- Data augmentation (noise, time-stretch, pitch-shift)
- Advanced regularization (L2, dropout, batch normalization)
- Automatic overfitting detection
- Feature quality verification with visualizations
- Comprehensive metrics tracking

Author: Tushar (Hackathon Build)
"""

# =============================
# 📦 IMPORTS
# =============================
import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json

# =============================
# 📂 DATASET CONFIGURATION
# =============================
FIGDATA_PATH = "data/figdata"
GITDATA_PATH = "data/gitdata/denoised-speech-dataset"
SAMPLE_RATE = 22050
DURATION = 5  # seconds
SAMPLES_PER_FILE = SAMPLE_RATE * DURATION

# Training configuration
USE_REGULARIZATION = True
L2_REG = 0.001
DROPOUT_RATE = 0.5
DATA_AUGMENTATION = True

# =============================
# 🧹 DATA LOADER & FEATURE EXTRACTION
# =============================
def load_gitdata(base_path):
    """
    Load ALL gitdata - contains only Parkinson's patient data
    Loads from both:
    - denoised-speech-dataset/Faces/[BG_au, JC_au, MJ_au, SK_au, TP_au, TS_au] (113 files)
    - original-speech-dataset/Faces/[BG_ori, JC_ori, MJ_ori, SK_ori, TP_ori, TS_ori] (113 files)
    Total: ~226 Parkinson's samples
    """
    X, y = [], []
    
    # Load denoised-speech-dataset
    denoised_path = os.path.join(base_path, "denoised-speech-dataset", "Faces")
    if os.path.exists(denoised_path):
        print(f"Loading denoised-speech-dataset from {denoised_path}")
        for person_folder in os.listdir(denoised_path):
            person_path = os.path.join(denoised_path, person_folder)
            if os.path.isdir(person_path):
                wav_files = [f for f in os.listdir(person_path) if f.endswith(".wav")]
                for file in wav_files:
                    file_path = os.path.join(person_path, file)
                    X.append(file_path)
                    y.append(1)  # Parkinson's
                print(f"  {person_folder}: {len(wav_files)} files")
        print(f"  Total denoised: {len([f for f in X if 'denoised' in f])} samples")
    
    # Load original-speech-dataset
    original_path = os.path.join(base_path, "original-speech-dataset", "Faces")
    if os.path.exists(original_path):
        print(f"\nLoading original-speech-dataset from {original_path}")
        original_count_start = len(X)
        for person_folder in os.listdir(original_path):
            person_path = os.path.join(original_path, person_folder)
            if os.path.isdir(person_path):
                wav_files = [f for f in os.listdir(person_path) if f.endswith(".wav")]
                for file in wav_files:
                    file_path = os.path.join(person_path, file)
                    X.append(file_path)
                    y.append(1)  # Parkinson's
                print(f"  {person_folder}: {len(wav_files)} files")
        print(f"  Total original: {len(X) - original_count_start} samples")
    
    print(f"\n  TOTAL GITDATA: {len(X)} Parkinson's samples")
    return np.array(X), np.array(y)

def load_figdata(base_path):
    """
    Load figdata - contains both Healthy Controls and Parkinson's patients
    Structure:
        HC_AH/  -> Healthy Controls (label 0)
        PD_AH/  -> Parkinson's Disease (label 1)
    """
    X, y = [], []
    
    # Load Healthy Controls
    hc_path = os.path.join(base_path, "HC_AH")
    if os.path.exists(hc_path):
        print(f"Loading healthy controls from {hc_path}")
        for file in os.listdir(hc_path):
            if file.endswith(".wav"):
                file_path = os.path.join(hc_path, file)
                X.append(file_path)
                y.append(0)  # Healthy
        print(f"  Loaded {len([f for f in os.listdir(hc_path) if f.endswith('.wav')])} healthy control files")
    
    # Load Parkinson's Disease patients
    pd_path = os.path.join(base_path, "PD_AH")
    if os.path.exists(pd_path):
        print(f"Loading Parkinson's patients from {pd_path}")
        for file in os.listdir(pd_path):
            if file.endswith(".wav"):
                file_path = os.path.join(pd_path, file)
                X.append(file_path)
                y.append(1)  # Parkinson's
        print(f"  Loaded {len([f for f in os.listdir(pd_path) if f.endswith('.wav')])} Parkinson's patient files")
    
    return np.array(X), np.array(y)

def augment_audio(y, sr):
    """Apply data augmentation to audio signal"""
    augmentations = []
    
    # Original
    augmentations.append(y)
    
    if DATA_AUGMENTATION:
        # Add white noise
        noise = np.random.normal(0, 0.005, y.shape)
        augmentations.append(y + noise)
        
        # Time stretch
        try:
            y_stretched = librosa.effects.time_stretch(y, rate=0.9)
            if len(y_stretched) > len(y):
                y_stretched = y_stretched[:len(y)]
            else:
                y_stretched = np.pad(y_stretched, (0, len(y) - len(y_stretched)), 'constant')
            augmentations.append(y_stretched)
        except:
            pass
        
        # Pitch shift
        try:
            y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=2)
            augmentations.append(y_shifted)
        except:
            pass
    
    return augmentations

def extract_melspectrogram(file_path, n_mels=128, n_fft=2048, hop_length=512, augment=False):
    """Extracts a Mel-spectrogram from an audio file with optional augmentation"""
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        
        # Normalize audio
        if len(y) > 0:
            y = librosa.util.normalize(y)
        
        if len(y) > SAMPLES_PER_FILE:
            y = y[:SAMPLES_PER_FILE]
        else:
            y = np.pad(y, (0, max(0, SAMPLES_PER_FILE - len(y))), "constant")
        
        features = []
        
        # Get augmented versions if requested
        if augment:
            augmented_signals = augment_audio(y, sr)
        else:
            augmented_signals = [y]
        
        for signal in augmented_signals:
            mel = librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            
            # Normalize mel-spectrogram
            mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-6)
            
            features.append(mel_db)
        
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def prepare_dataset(X, y, augment=False):
    """Extracts features and converts to NumPy arrays suitable for CNN input."""
    features = []
    labels = []
    
    for i, path in enumerate(X):
        mel_features = extract_melspectrogram(path, augment=augment)
        if mel_features is not None:
            for mel in mel_features:
                features.append(mel)
                labels.append(y[i])
        
        if (i + 1) % 20 == 0:
            print(f"Processed {i + 1}/{len(X)} files...")
    
    X_out = np.array(features)
    X_out = X_out[..., np.newaxis]  # add channel dimension
    print(f"Feature shape: {X_out.shape}")
    print(f"Total samples after augmentation: {len(X_out)}")
    
    return X_out, np.array(labels)

def verify_features(X, y, dataset_name="Dataset"):
    """Verify extracted features for quality"""
    print(f"\n{'='*60}")
    print(f"Feature Verification for {dataset_name}")
    print(f"{'='*60}")
    
    print(f"Shape: {X.shape}")
    print(f"Data type: {X.dtype}")
    print(f"Value range: [{X.min():.4f}, {X.max():.4f}]")
    print(f"Mean: {X.mean():.4f}, Std: {X.std():.4f}")
    print(f"NaN count: {np.isnan(X).sum()}")
    print(f"Inf count: {np.isinf(X).sum()}")
    
    # Check label distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"\nLabel distribution:")
    for label, count in zip(unique, counts):
        label_name = "Healthy" if label == 0 else "Parkinson's"
        print(f"  {label_name} (label {label}): {count} samples ({count/len(y)*100:.1f}%)")
    
    # Visualize sample spectrograms
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Sample from each class
    for idx, (label, title) in enumerate([(0, "Healthy"), (1, "Parkinson's")]):
        class_indices = np.where(y == label)[0]
        if len(class_indices) > 0:
            sample_idx = class_indices[0]
            axes[idx].imshow(X[sample_idx, :, :, 0], aspect='auto', origin='lower', cmap='viridis')
            axes[idx].set_title(f'{title} - Sample Mel-spectrogram')
            axes[idx].set_xlabel('Time')
            axes[idx].set_ylabel('Mel Frequency')
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("results", exist_ok=True)
    plt.savefig(f'results/feature_verification_{dataset_name}_{timestamp}.png')
    print(f"\nFeature visualization saved to results/feature_verification_{dataset_name}_{timestamp}.png")
    plt.close()

# =============================
# 🧠 MODEL DEFINITION
# =============================
def build_cnn_bilstm(input_shape, use_regularization=True, dropout_rate=0.5):
    """
    CNN + BiLSTM hybrid architecture with optional regularization
    """
    l2_reg = regularizers.l2(L2_REG) if use_regularization else None
    
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', padding='same', 
                     kernel_regularizer=l2_reg, input_shape=input_shape),
        layers.MaxPooling2D((2,2)),
        layers.BatchNormalization(),
        layers.Dropout(0.3 if use_regularization else 0.2),

        layers.Conv2D(64, (3,3), activation='relu', padding='same',
                     kernel_regularizer=l2_reg),
        layers.MaxPooling2D((2,2)),
        layers.BatchNormalization(),
        layers.Dropout(0.3 if use_regularization else 0.2),

        layers.Conv2D(128, (3,3), activation='relu', padding='same',
                     kernel_regularizer=l2_reg),
        layers.MaxPooling2D((2,2)),
        layers.BatchNormalization(),
        layers.Dropout(0.4 if use_regularization else 0.3),

        layers.Reshape((-1, 128)),
        layers.Bidirectional(layers.LSTM(128, return_sequences=False,
                                        kernel_regularizer=l2_reg)),
        layers.Dropout(dropout_rate),
        layers.Dense(64, activation='relu', kernel_regularizer=l2_reg),
        layers.Dropout(0.3 if use_regularization else 0.2),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics.AUC()]
    )
    return model

def detect_overfitting(history):
    """Detect if model is overfitting based on training history"""
    train_acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    # Check last 5 epochs
    if len(train_acc) >= 5:
        recent_train_acc = np.mean(train_acc[-5:])
        recent_val_acc = np.mean(val_acc[-5:])
        recent_train_loss = np.mean(train_loss[-5:])
        recent_val_loss = np.mean(val_loss[-5:])
        
        acc_gap = recent_train_acc - recent_val_acc
        loss_gap = recent_val_loss - recent_train_loss
        
        print(f"\n{'='*60}")
        print(f"Overfitting Detection")
        print(f"{'='*60}")
        print(f"Recent Train Accuracy: {recent_train_acc:.4f}")
        print(f"Recent Val Accuracy: {recent_val_acc:.4f}")
        print(f"Accuracy Gap: {acc_gap:.4f}")
        print(f"Recent Train Loss: {recent_train_loss:.4f}")
        print(f"Recent Val Loss: {recent_val_loss:.4f}")
        print(f"Loss Gap: {loss_gap:.4f}")
        
        if acc_gap > 0.15 or loss_gap > 0.3:
            print("⚠️  WARNING: Potential overfitting detected!")
            print("   Suggestions:")
            print("   - Increase dropout rate")
            print("   - Add more L2 regularization")
            print("   - Use data augmentation")
            print("   - Reduce model complexity")
            return True
        else:
            print("✓ No significant overfitting detected")
            return False
    
    return False

# =============================
# ⚙️ TRAINING PIPELINE
# =============================
def main():
    print("=" * 60)
    print("Voice-Based Parkinson's Disease Detection")
    print("CNN + BiLSTM Model Training (Combined Dataset)")
    print("=" * 60)
    print()
    
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # =============================
    # PHASE 1: Load and prepare GITDATA
    # =============================
    print("\n" + "=" * 60)
    print("PHASE 1: Loading GITDATA (Parkinson's patients only)")
    print("=" * 60)
    
    X_git_paths, y_git = load_gitdata(GITDATA_PATH)
    print(f"\nTotal gitdata samples: {len(X_git_paths)}")
    
    X_git = None
    y_git_processed = None
    
    if len(X_git_paths) > 0:
        print("\nExtracting features from gitdata...")
        X_git, y_git_processed = prepare_dataset(X_git_paths, y_git, augment=DATA_AUGMENTATION)
        verify_features(X_git, y_git_processed, "GITDATA")
        
        print("\n" + "=" * 60)
        print(f"✓ GITDATA prepared: {len(X_git)} samples (after augmentation)")
        print("  This data will be combined with FIGDATA for training")
        print("=" * 60)
    else:
        print("No samples found in gitdata. Proceeding to figdata only.")
    
    # =============================
    # PHASE 2: Load and prepare FIGDATA
    # =============================
    print("\n" + "=" * 60)
    print("PHASE 2: Loading FIGDATA (Healthy + Parkinson's)")
    print("=" * 60)
    
    X_fig_paths, y_fig = load_figdata(FIGDATA_PATH)
    
    if len(X_fig_paths) == 0:
        print("No audio files found! Please check the data directory.")
        return
    
    print("\nExtracting features from figdata...")
    X_fig, y_fig_processed = prepare_dataset(X_fig_paths, y_fig, augment=DATA_AUGMENTATION)
    verify_features(X_fig, y_fig_processed, "FIGDATA")
    
    print("\n" + "=" * 60)
    print(f"✓ FIGDATA prepared: {len(X_fig)} samples (after augmentation)")
    print("=" * 60)
    
    # =============================
    # PHASE 3: Combine datasets and split
    # =============================
    print("\n" + "=" * 60)
    print("PHASE 3: Combining datasets for training")
    print("=" * 60)
    
    if X_git is not None and len(X_git) > 0:
        # Combine gitdata (all Parkinson's) with figdata (mixed)
        print(f"Combining {len(X_git)} gitdata samples with {len(X_fig)} figdata samples")
        X_combined = np.concatenate([X_git, X_fig], axis=0)
        y_combined = np.concatenate([y_git_processed, y_fig_processed], axis=0)
        print(f"Total combined samples: {len(X_combined)}")
    else:
        # Use only figdata
        print("Using figdata only (gitdata not available)")
        X_combined = X_fig
        y_combined = y_fig_processed
    
    # Check label distribution
    unique, counts = np.unique(y_combined, return_counts=True)
    print(f"\nCombined dataset distribution:")
    for label, count in zip(unique, counts):
        label_name = "Healthy" if label == 0 else "Parkinson's"
        print(f"  {label_name}: {count} samples ({count/len(y_combined)*100:.1f}%)")
    
    # Split combined dataset: 70% train, 15% validation (from train), 15% test
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y_combined, test_size=0.15, stratify=y_combined, random_state=42
    )
    
    print(f"\nFinal dataset split:")
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Validation: ~{int(len(X_train) * 0.18)} samples (from training set)")
    
    # Build model
    print("\n" + "=" * 60)
    print("Building CNN + BiLSTM model with regularization...")
    print("=" * 60)
    input_shape = X_train.shape[1:]
    print(f"Input shape: {input_shape}")
    print(f"Total training samples: {len(X_train)}")
    print(f"Total test samples: {len(X_test)}")
    print(f"Using regularization: {USE_REGULARIZATION}")
    print(f"L2 regularization: {L2_REG}")
    print(f"Dropout rate: {DROPOUT_RATE}")
    print(f"Data augmentation: {DATA_AUGMENTATION}")
    
    model = build_cnn_bilstm(input_shape, USE_REGULARIZATION, DROPOUT_RATE)
    model.summary()
    
    # Setup callbacks
    model_path = f"models/best_model_combined_{timestamp}.h5"
    
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        model_path, 
        monitor="val_accuracy", 
        save_best_only=True, 
        mode="max",
        verbose=1
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", 
        patience=20, 
        restore_best_weights=True,
        verbose=1
    )
    lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", 
        factor=0.5, 
        patience=10, 
        min_lr=1e-7,
        verbose=1
    )
    
    # Train model
    print("\n" + "=" * 60)
    print("Training model on COMBINED dataset (gitdata + figdata)...")
    print("=" * 60)
    
    history = model.fit(
        X_train, y_train,
        validation_split=0.18,  # ~15% of total data for validation
        epochs=150,
        batch_size=8,  # Smaller batch for better generalization
        callbacks=[checkpoint, early_stopping, lr_schedule],
        verbose=1
    )
    
    # Detect overfitting
    is_overfitting = detect_overfitting(history)
    
    # Save training history
    print("\n" + "=" * 60)
    print("Saving training history...")
    
    # Plot training history
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0, 0].plot(history.history['val_accuracy'], label='Val Accuracy', linewidth=2)
    axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[0, 1].plot(history.history['val_loss'], label='Val Loss', linewidth=2)
    axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Precision
    axes[1, 0].plot(history.history['precision'], label='Train Precision', linewidth=2)
    axes[1, 0].plot(history.history['val_precision'], label='Val Precision', linewidth=2)
    axes[1, 0].set_title('Model Precision', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Recall
    axes[1, 1].plot(history.history['recall'], label='Train Recall', linewidth=2)
    axes[1, 1].plot(history.history['val_recall'], label='Val Recall', linewidth=2)
    axes[1, 1].set_title('Model Recall', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'results/training_history_combined_{timestamp}.png', dpi=300)
    print(f"Training history plot saved to results/training_history_combined_{timestamp}.png")
    plt.close()
    
    # =============================
    # 📈 EVALUATION
    # =============================
    print("\n" + "=" * 60)
    print("Evaluating on test set...")
    print("=" * 60)
    
    # Load best model
    model.load_weights(model_path)
    
    # Make predictions
    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype("int32")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Healthy', 'Parkinson']))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Healthy', 'Parkinson'], 
                yticklabels=['Healthy', 'Parkinson'],
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('Actual', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'results/confusion_matrix_combined_{timestamp}.png', dpi=300)
    print(f"Confusion matrix saved to results/confusion_matrix_combined_{timestamp}.png")
    plt.close()
    
    # ROC Curve
    if len(np.unique(y_test)) == 2:
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        auc_score = roc_auc_score(y_test, y_pred_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve', fontsize=16, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'results/roc_curve_combined_{timestamp}.png', dpi=300)
        print(f"ROC curve saved to results/roc_curve_combined_{timestamp}.png")
        plt.close()
    
    # Test metrics
    test_loss, test_acc, test_precision, test_recall, test_auc = model.evaluate(X_test, y_test, verbose=0)
    f1_score = 2 * (test_precision * test_recall) / (test_precision + test_recall + 1e-7)
    
    print(f"\n{'='*60}")
    print(f"Final Test Results:")
    print(f"{'='*60}")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall: {test_recall:.4f}")
    print(f"  F1-Score: {f1_score:.4f}")
    print(f"  AUC: {test_auc:.4f}")
    
    # Save metrics to JSON
    metrics = {
        'timestamp': timestamp,
        'test_loss': float(test_loss),
        'test_accuracy': float(test_acc),
        'test_precision': float(test_precision),
        'test_recall': float(test_recall),
        'test_f1_score': float(f1_score),
        'test_auc': float(test_auc),
        'overfitting_detected': is_overfitting,
        'configuration': {
            'use_regularization': USE_REGULARIZATION,
            'l2_reg': L2_REG,
            'dropout_rate': DROPOUT_RATE,
            'data_augmentation': DATA_AUGMENTATION,
            'sample_rate': SAMPLE_RATE,
            'duration': DURATION
        }
    }
    
    with open(f'results/metrics_combined_{timestamp}.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"\nMetrics saved to results/metrics_combined_{timestamp}.json")
    
    print("\n" + "=" * 60)
    print(f"Training complete! Best model saved to: {model_path}")
    if is_overfitting:
        print("⚠️  Note: Overfitting was detected. Consider retraining with more regularization.")
    print("=" * 60)

if __name__ == "__main__":
    main()
