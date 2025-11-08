"""
Project: Voice-Based Parkinson's Disease Detection using CNN + BiLSTM
Enhanced Version: Uses denoised-speech-dataset + figdata with K-Fold CV

Dataset Clarification:
GITDATA (Parkinson's patients only):
  - denoised-speech-dataset/Faces: ~113 Parkinson's samples
  - NOTE: original-speech-dataset is the same data, so we use ONLY denoised version
  
FIGDATA (Mixed - Healthy + Parkinson's):
  - HC_AH: 41 samples (HEALTHY CONTROLS - NOT Parkinson's)
  - PD_AH: 40 samples (PARKINSON'S DISEASE patients)
  - Total: 81 samples

COMBINED TOTAL: ~194 audio samples (after filtering duration > 3 seconds)

Training Strategy:
1. Phase 1: Load gitdata (denoised-speech-dataset only - Parkinson's samples)
2. Phase 2: Load figdata (HC_AH=Healthy + PD_AH=Parkinson's)
3. Phase 3: Filter audio clips with duration > 3 seconds
4. Phase 4: Combine all data and apply K-Fold Cross Validation (5 folds)
5. Train CNN+BiLSTM model with K-Fold validation
6. Monitor for overfitting and apply regularization
7. Verify feature extraction quality

Features:
- K-Fold Cross Validation (5 folds for robust evaluation)
- Audio duration filtering (>3 seconds only)
- Data augmentation (noise, time-stretch, pitch-shift)
- Advanced regularization (L2, dropout, batch normalization)
- Automatic overfitting detection
- Feature quality verification with visualizations
- Comprehensive metrics tracking per fold

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
GITDATA_PATH = "data/gitdata"
SAMPLE_RATE = 22050
MIN_DURATION = 3.0  # Minimum duration in seconds - only use clips > 3 seconds
DURATION = 5  # seconds for feature extraction
SAMPLES_PER_FILE = SAMPLE_RATE * DURATION

# K-Fold Configuration
N_FOLDS = 5  # Number of folds for cross-validation

# Training configuration
USE_REGULARIZATION = True
L2_REG = 0.001
DROPOUT_RATE = 0.5
DATA_AUGMENTATION = True

# =============================
# 🧹 DATA LOADER & FEATURE EXTRACTION
# =============================
# =============================
# 🧹 DATA LOADER & FEATURE EXTRACTION
# =============================
def get_audio_duration(file_path):
    """Get duration of audio file in seconds"""
    try:
        duration = librosa.get_duration(path=file_path)
        return duration
    except Exception as e:
        print(f"Error getting duration for {file_path}: {e}")
        return 0.0

def load_gitdata(base_path):
    """
    Load gitdata - ONLY denoised-speech-dataset (Parkinson's patients only)
    NOTE: We use ONLY denoised dataset because original-speech-dataset is the same data
    
    Filter: Only audio clips with duration > MIN_DURATION seconds
    
    Structure: denoised-speech-dataset/Faces/[BG_au, JC_au, MJ_au, SK_au, TP_au, TS_au]
    Label: All samples are Parkinson's patients (label = 1)
    """
    X, y = [], []
    skipped_short = 0
    
    # Load ONLY denoised-speech-dataset (NOT original, as it's duplicate data)
    denoised_path = os.path.join(base_path, "denoised-speech-dataset", "Faces")
    
    if os.path.exists(denoised_path):
        print(f"Loading gitdata (PARKINSON'S PATIENTS) from {denoised_path}")
        print(f"NOTE: Using ONLY denoised dataset (original is duplicate)")
        print(f"Filtering: Only clips with duration > {MIN_DURATION} seconds\n")
        
        for person_folder in os.listdir(denoised_path):
            person_path = os.path.join(denoised_path, person_folder)
            if os.path.isdir(person_path):
                wav_files = [f for f in os.listdir(person_path) if f.endswith(".wav")]
                valid_count = 0
                
                for file in wav_files:
                    file_path = os.path.join(person_path, file)
                    duration = get_audio_duration(file_path)
                    
                    if duration > MIN_DURATION:
                        X.append(file_path)
                        y.append(1)  # Parkinson's
                        valid_count += 1
                    else:
                        skipped_short += 1
                
                print(f"  {person_folder}: {valid_count}/{len(wav_files)} files (>{MIN_DURATION}s)")
        
        print(f"\n  ✓ Total gitdata: {len(X)} Parkinson's samples")
        print(f"  ✗ Skipped: {skipped_short} files (duration ≤ {MIN_DURATION}s)")
    else:
        print(f"Warning: Path not found: {denoised_path}")
    
    return np.array(X), np.array(y)

def load_figdata(base_path):
    """
    Load figdata - contains BOTH Healthy Controls and Parkinson's patients
    
    IMPORTANT CLARIFICATION:
    - HC_AH: HEALTHY CONTROLS (label 0) - NOT Parkinson's
    - PD_AH: PARKINSON'S DISEASE patients (label 1)
    
    Filter: Only audio clips with duration > MIN_DURATION seconds
    """
    X, y = [], []
    skipped_short = 0
    
    print(f"Filtering: Only clips with duration > {MIN_DURATION} seconds\n")
    
    # Load Healthy Controls (HC_AH = HEALTHY PATIENTS, NOT Parkinson's)
    hc_path = os.path.join(base_path, "HC_AH")
    hc_count = 0
    if os.path.exists(hc_path):
        print(f"Loading HEALTHY CONTROLS (HC_AH) from {hc_path}")
        total_files = 0
        
        for file in os.listdir(hc_path):
            if file.endswith(".wav"):
                total_files += 1
                file_path = os.path.join(hc_path, file)
                duration = get_audio_duration(file_path)
                
                if duration > MIN_DURATION:
                    X.append(file_path)
                    y.append(0)  # Healthy (NOT Parkinson's)
                    hc_count += 1
                else:
                    skipped_short += 1
        
        print(f"  ✓ Loaded {hc_count}/{total_files} HEALTHY files (>{MIN_DURATION}s)")
    else:
        print(f"Warning: Path not found: {hc_path}")
    
    # Load Parkinson's Disease patients (PD_AH = PARKINSON'S PATIENTS)
    pd_path = os.path.join(base_path, "PD_AH")
    pd_count = 0
    if os.path.exists(pd_path):
        print(f"Loading PARKINSON'S PATIENTS (PD_AH) from {pd_path}")
        total_files = 0
        
        for file in os.listdir(pd_path):
            if file.endswith(".wav"):
                total_files += 1
                file_path = os.path.join(pd_path, file)
                duration = get_audio_duration(file_path)
                
                if duration > MIN_DURATION:
                    X.append(file_path)
                    y.append(1)  # Parkinson's
                    pd_count += 1
                else:
                    skipped_short += 1
        
        print(f"  ✓ Loaded {pd_count}/{total_files} PARKINSON'S files (>{MIN_DURATION}s)")
    else:
        print(f"Warning: Path not found: {pd_path}")
    
    print(f"\n  ✓ Total figdata: {len(X)} samples ({hc_count} healthy + {pd_count} Parkinson's)")
    print(f"  ✗ Skipped: {skipped_short} files (duration ≤ {MIN_DURATION}s)")
    
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
# ⚙️ TRAINING PIPELINE WITH K-FOLD CV
# =============================
def main():
    print("=" * 70)
    print("Voice-Based Parkinson's Disease Detection")
    print("CNN + BiLSTM Model Training with K-Fold Cross Validation")
    print("=" * 70)
    print()
    
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # =============================
    # PHASE 1: Load GITDATA (Parkinson's only)
    # =============================
    print("\n" + "=" * 70)
    print("PHASE 1: Loading GITDATA (Parkinson's patients - denoised only)")
    print("=" * 70)
    
    X_git_paths, y_git = load_gitdata(GITDATA_PATH)
    
    # =============================
    # PHASE 2: Load FIGDATA (Healthy + Parkinson's)
    # =============================
    print("\n" + "=" * 70)
    print("PHASE 2: Loading FIGDATA (HC_AH=Healthy + PD_AH=Parkinson's)")
    print("=" * 70)
    
    X_fig_paths, y_fig = load_figdata(FIGDATA_PATH)
    
    if len(X_fig_paths) == 0 and len(X_git_paths) == 0:
        print("\n❌ ERROR: No audio files found! Please check the data directory.")
        return
    
    # =============================
    # PHASE 3: Combine ALL data
    # =============================
    print("\n" + "=" * 70)
    print("PHASE 3: Combining ALL datasets")
    print("=" * 70)
    
    # Combine paths
    if len(X_git_paths) > 0 and len(X_fig_paths) > 0:
        X_all_paths = np.concatenate([X_git_paths, X_fig_paths])
        y_all = np.concatenate([y_git, y_fig])
        print(f"✓ Combined {len(X_git_paths)} gitdata + {len(X_fig_paths)} figdata")
    elif len(X_git_paths) > 0:
        X_all_paths = X_git_paths
        y_all = y_git
        print(f"✓ Using gitdata only: {len(X_git_paths)} samples")
    else:
        X_all_paths = X_fig_paths
        y_all = y_fig
        print(f"✓ Using figdata only: {len(X_fig_paths)} samples")
    
    print(f"\nTotal audio files (>3s duration): {len(X_all_paths)}")
    
    # Check label distribution
    unique, counts = np.unique(y_all, return_counts=True)
    print(f"\nCombined dataset distribution:")
    for label, count in zip(unique, counts):
        label_name = "Healthy" if label == 0 else "Parkinson's"
        print(f"  {label_name}: {count} samples ({count/len(y_all)*100:.1f}%)")
    
    # =============================
    # PHASE 4: Extract Features
    # =============================
    print("\n" + "=" * 70)
    print("PHASE 4: Extracting Features from ALL audio files")
    print("=" * 70)
    
    print("\nExtracting features with augmentation...")
    X_all, y_all_processed = prepare_dataset(X_all_paths, y_all, augment=DATA_AUGMENTATION)
    verify_features(X_all, y_all_processed, "COMBINED_DATASET")
    
    # =============================
    # PHASE 5: K-Fold Cross Validation
    # =============================
    print("\n" + "=" * 70)
    print(f"PHASE 5: K-Fold Cross Validation ({N_FOLDS} folds)")
    print("=" * 70)
    
    # Initialize K-Fold
    kfold = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    
    # Store results for each fold
    fold_results = []
    all_histories = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_all, y_all_processed), 1):
        print(f"\n{'='*70}")
        print(f"FOLD {fold_idx}/{N_FOLDS}")
        print(f"{'='*70}")
        
        # Split data for this fold
        X_train_fold = X_all[train_idx]
        y_train_fold = y_all_processed[train_idx]
        X_val_fold = X_all[val_idx]
        y_val_fold = y_all_processed[val_idx]
        
        print(f"Training samples: {len(X_train_fold)}")
        print(f"Validation samples: {len(X_val_fold)}")
        
        # Check distribution
        unique_train, counts_train = np.unique(y_train_fold, return_counts=True)
        unique_val, counts_val = np.unique(y_val_fold, return_counts=True)
        print(f"\nTrain distribution:")
        for label, count in zip(unique_train, counts_train):
            print(f"  {'Healthy' if label == 0 else 'Parkinson'}: {count}")
        print(f"Validation distribution:")
        for label, count in zip(unique_val, counts_val):
            print(f"  {'Healthy' if label == 0 else 'Parkinson'}: {count}")
        
        # Build model for this fold
        input_shape = X_train_fold.shape[1:]
        model = build_cnn_bilstm(input_shape, USE_REGULARIZATION, DROPOUT_RATE)
        
        # Setup callbacks for this fold
        model_path = f"models/fold{fold_idx}_model_{timestamp}.h5"
        
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            model_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=0
        )
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=20,
            restore_best_weights=True,
            verbose=0
        )
        lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=10,
            min_lr=1e-7,
            verbose=0
        )
        
        # Train model for this fold
        print(f"\nTraining Fold {fold_idx}...")
        history = model.fit(
            X_train_fold, y_train_fold,
            validation_data=(X_val_fold, y_val_fold),
            epochs=150,
            batch_size=8,
            callbacks=[checkpoint, early_stopping, lr_schedule],
            verbose=2  # Less verbose output
        )
        
        all_histories.append(history)
        
        # Detect overfitting for this fold
        is_overfitting = detect_overfitting(history)
        
        # Evaluate on validation set
        model.load_weights(model_path)
        val_loss, val_acc, val_precision, val_recall, val_auc = model.evaluate(
            X_val_fold, y_val_fold, verbose=0
        )
        val_f1 = 2 * (val_precision * val_recall) / (val_precision + val_recall + 1e-7)
        
        # Make predictions
        y_val_pred_prob = model.predict(X_val_fold, verbose=0)
        y_val_pred = (y_val_pred_prob > 0.5).astype("int32")
        
        # Store results
        fold_result = {
            'fold': fold_idx,
            'val_loss': float(val_loss),
            'val_accuracy': float(val_acc),
            'val_precision': float(val_precision),
            'val_recall': float(val_recall),
            'val_f1_score': float(val_f1),
            'val_auc': float(val_auc),
            'overfitting_detected': is_overfitting,
            'model_path': model_path
        }
        fold_results.append(fold_result)
        
        print(f"\n{'='*70}")
        print(f"Fold {fold_idx} Results:")
        print(f"{'='*70}")
        print(f"  Validation Loss: {val_loss:.4f}")
        print(f"  Validation Accuracy: {val_acc:.4f} ({val_acc*100:.2f}%)")
        print(f"  Validation Precision: {val_precision:.4f}")
        print(f"  Validation Recall: {val_recall:.4f}")
        print(f"  Validation F1-Score: {val_f1:.4f}")
        print(f"  Validation AUC: {val_auc:.4f}")
        print(f"  Overfitting: {'⚠️ Yes' if is_overfitting else '✓ No'}")
        print(f"  Model saved: {model_path}")
        
        # Classification report for this fold
        print(f"\nClassification Report (Fold {fold_idx}):")
        print(classification_report(y_val_fold, y_val_pred, 
                                   target_names=['Healthy', 'Parkinson']))
    
    # =============================
    # PHASE 6: Aggregate K-Fold Results
    # =============================
    print("\n" + "=" * 70)
    print("PHASE 6: K-Fold Cross Validation Summary")
    print("=" * 70)
    
    # Calculate mean and std for each metric
    metrics = ['val_accuracy', 'val_precision', 'val_recall', 'val_f1_score', 'val_auc', 'val_loss']
    metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC', 'Loss']
    
    print("\nCross-Validation Results:")
    print("-" * 70)
    for metric, name in zip(metrics, metric_names):
        values = [r[metric] for r in fold_results]
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"  {name:12s}: {mean_val:.4f} ± {std_val:.4f}")
    
    print("\nPer-Fold Results:")
    print("-" * 70)
    for result in fold_results:
        print(f"  Fold {result['fold']}: Acc={result['val_accuracy']:.4f}, " +
              f"Prec={result['val_precision']:.4f}, Rec={result['val_recall']:.4f}, " +
              f"F1={result['val_f1_score']:.4f}, AUC={result['val_auc']:.4f}")
    
    # Save K-Fold results to JSON
    kfold_summary = {
        'timestamp': timestamp,
        'n_folds': N_FOLDS,
        'total_samples': len(X_all),
        'min_duration': MIN_DURATION,
        'fold_results': fold_results,
        'mean_metrics': {
            metric: float(np.mean([r[metric] for r in fold_results]))
            for metric in metrics
        },
        'std_metrics': {
            metric: float(np.std([r[metric] for r in fold_results]))
            for metric in metrics
        },
        'configuration': {
            'use_regularization': USE_REGULARIZATION,
            'l2_reg': L2_REG,
            'dropout_rate': DROPOUT_RATE,
            'data_augmentation': DATA_AUGMENTATION,
            'sample_rate': SAMPLE_RATE,
            'duration': DURATION,
            'min_duration': MIN_DURATION
        }
    }
    
    results_file = f'results/kfold_results_{timestamp}.json'
    with open(results_file, 'w') as f:
        json.dump(kfold_summary, f, indent=4)
    print(f"\n✓ K-Fold results saved to: {results_file}")
    
    # =============================
    # PHASE 7: Visualize K-Fold Results
    # =============================
    print("\n" + "=" * 70)
    print("PHASE 7: Generating Visualizations")
    print("=" * 70)
    
    # Plot metrics across folds
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'{N_FOLDS}-Fold Cross Validation Results', fontsize=16, fontweight='bold')
    
    metrics_to_plot = [
        ('val_accuracy', 'Accuracy'),
        ('val_precision', 'Precision'),
        ('val_recall', 'Recall'),
        ('val_f1_score', 'F1-Score'),
        ('val_auc', 'AUC'),
        ('val_loss', 'Loss')
    ]
    
    for idx, (metric, name) in enumerate(metrics_to_plot):
        ax = axes[idx // 3, idx % 3]
        values = [r[metric] for r in fold_results]
        folds = [r['fold'] for r in fold_results]
        
        ax.bar(folds, values, color='steelblue', alpha=0.7)
        ax.axhline(y=np.mean(values), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(values):.4f}')
        ax.set_xlabel('Fold')
        ax.set_ylabel(name)
        ax.set_title(f'{name} per Fold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(folds)
    
    plt.tight_layout()
    viz_file = f'results/kfold_metrics_{timestamp}.png'
    plt.savefig(viz_file, dpi=300, bbox_inches='tight')
    print(f"✓ K-Fold metrics visualization saved to: {viz_file}")
    plt.close()
    
    print("\n" + "=" * 70)
    print("✅ K-Fold Cross Validation Complete!")
    print("=" * 70)
    print(f"\nBest Fold: Fold {max(fold_results, key=lambda x: x['val_accuracy'])['fold']}")
    print(f"Best Validation Accuracy: {max(r['val_accuracy'] for r in fold_results):.4f}")
    print(f"\nAll fold models saved in models/ directory")
    print(f"Results and visualizations saved in results/ directory")
    print("=" * 70)

if __name__ == "__main__":
    main()
