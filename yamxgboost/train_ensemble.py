"""
Ensemble Parkinson's Disease Detection: YAMNet + OpenSMILE + XGBoost
====================================================================
Feature extraction using:
  1. YAMNet (Google's pre-trained audio model) - Deep learning embeddings
  2. OpenSMILE - Traditional acoustic features
  3. XGBoost - Gradient boosted trees for classification

Dataset: 
  - figdata (HC_AH=Healthy, PD_AH=Parkinson's)
  - gitdata/original-speech-dataset (Parkinson's only)
  - Duration filter: > 3 seconds only
  - NO denoised-speech-dataset
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import soundfile as sf
import librosa
import opensmile
import tensorflow as tf
import tensorflow_hub as hub
import xgboost as xgb
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
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
MIN_DURATION = 3.0  # Minimum audio duration in seconds
DATA_DIR = "../data"
RESULTS_DIR = "results"
MODELS_DIR = "models"
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_FOLDS = 5

# XGBoost hyperparameters
XGB_PARAMS = {
    'max_depth': 7,
    'learning_rate': 0.05,
    'n_estimators': 300,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': RANDOM_STATE,
    'use_label_encoder': False,
    'tree_method': 'hist'
}

# OpenSMILE feature set
OPENSMILE_FEATURE_SET = 'eGeMAPSv02'  # 88 features (balanced)

# YAMNet model URL
YAMNET_MODEL_URL = 'https://tfhub.dev/google/yamnet/1'

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

def extract_opensmile_features(audio_path, feature_set='eGeMAPSv02'):
    """Extract OpenSMILE acoustic features."""
    try:
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet[feature_set],
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        
        features = smile.process_file(audio_path)
        feature_vector = features.values.flatten()
        
        return feature_vector
        
    except Exception as e:
        print(f"Error extracting OpenSMILE features from {audio_path}: {e}")
        return None

def extract_ensemble_features(audio_files, yamnet_model):
    """
    Extract combined features from YAMNet + OpenSMILE.
    Creates a rich feature set combining deep learning and traditional features.
    """
    print("\n" + "=" * 70)
    print("🎵 Extracting Ensemble Features (YAMNet + OpenSMILE)...")
    print("=" * 70)
    print(f"  - YAMNet: Deep learning embeddings (1024 dimensions)")
    print(f"  - OpenSMILE: Acoustic features ({OPENSMILE_FEATURE_SET})")
    print("=" * 70)
    
    yamnet_features_list = []
    opensmile_features_list = []
    valid_files = []
    
    for audio_file in tqdm(audio_files, desc="Processing audio files"):
        # Extract YAMNet embeddings
        yamnet_emb = extract_yamnet_embeddings(audio_file, yamnet_model)
        
        # Extract OpenSMILE features
        opensmile_feat = extract_opensmile_features(audio_file, OPENSMILE_FEATURE_SET)
        
        # Only include if both succeeded
        if yamnet_emb is not None and opensmile_feat is not None:
            yamnet_features_list.append(yamnet_emb)
            opensmile_features_list.append(opensmile_feat)
            valid_files.append(audio_file)
    
    if not yamnet_features_list:
        raise ValueError("No features extracted! Check audio files and model installation.")
    
    # Convert to numpy arrays
    yamnet_features = np.array(yamnet_features_list)
    opensmile_features = np.array(opensmile_features_list)
    
    # Concatenate features
    X_combined = np.concatenate([yamnet_features, opensmile_features], axis=1)
    
    print(f"\n  ✓ Extracted features from {len(valid_files)} files")
    print(f"  ✓ YAMNet dimensions: {yamnet_features.shape[1]}")
    print(f"  ✓ OpenSMILE dimensions: {opensmile_features.shape[1]}")
    print(f"  ✓ Combined feature dimensions: {X_combined.shape[1]}")
    
    return X_combined, valid_files, yamnet_features, opensmile_features

# =============================
# 🎯 MAIN TRAINING PIPELINE
# =============================

def main():
    print("\n" + "=" * 70)
    print("🎯 Ensemble Parkinson's Detection")
    print("YAMNet + OpenSMILE + XGBoost")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  - Minimum duration: {MIN_DURATION} seconds")
    print(f"  - YAMNet embeddings: 1024 dimensions")
    print(f"  - OpenSMILE feature set: {OPENSMILE_FEATURE_SET}")
    print(f"  - Test size: {TEST_SIZE * 100}%")
    print(f"  - K-Folds: {N_FOLDS}")
    print(f"  - Using ORIGINAL speech dataset (not denoised)")
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
    print(f"  Healthy: {all_labels.count(0)} ({all_labels.count(0) / len(all_labels) * 100:.1f}%)")
    print(f"  Parkinson's: {all_labels.count(1)} ({all_labels.count(1) / len(all_labels) * 100:.1f}%)")
    
    # =============================
    # 🤖 LOAD YAMNET MODEL
    # =============================
    
    yamnet_model = load_yamnet_model()
    
    # =============================
    # 🎵 FEATURE EXTRACTION
    # =============================
    
    X_combined, valid_files, yamnet_features, opensmile_features = extract_ensemble_features(
        all_files, yamnet_model
    )
    
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
    X_scaled = scaler.fit_transform(X_combined)
    
    print(f"  ✓ Features scaled using StandardScaler")
    print(f"  ✓ Feature mean: {X_scaled.mean():.6f}")
    print(f"  ✓ Feature std: {X_scaled.std():.6f}")
    
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
    # 🤖 TRAIN XGBOOST MODEL
    # =============================
    
    print("\n" + "=" * 70)
    print("🤖 Training XGBoost on ensemble features...")
    print("=" * 70)
    
    model = xgb.XGBClassifier(**XGB_PARAMS)
    
    # Train with evaluation set
    eval_set = [(X_train, y_train), (X_test, y_test)]
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=False
    )
    
    print(f"  ✓ Model trained successfully")
    
    # =============================
    # 🔍 K-FOLD CROSS VALIDATION
    # =============================
    
    print("\n" + "=" * 70)
    print(f"🔍 {N_FOLDS}-Fold Cross Validation...")
    print("=" * 70)
    
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    
    print(f"  Cross-validation scores: {cv_scores}")
    print(f"  Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # =============================
    # 📈 EVALUATION
    # =============================
    
    print("\n" + "=" * 70)
    print("📈 Evaluating on test set...")
    print("=" * 70)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
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
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Healthy', 'Parkinson\'s'],
                yticklabels=['Healthy', 'Parkinson\'s'])
    plt.title('Confusion Matrix - Ensemble\n(YAMNet + OpenSMILE + XGBoost)', 
              fontsize=14, fontweight='bold')
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
    plt.plot(fpr, tpr, linewidth=2, label=f'Ensemble (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Ensemble\n(YAMNet + OpenSMILE + XGBoost)', 
              fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = f"{RESULTS_DIR}/roc_curve_{timestamp}.png"
    plt.savefig(roc_path, dpi=300)
    print(f"  ✓ ROC curve saved: {roc_path}")
    plt.close()
    
    # Feature Importance (top 30)
    feature_importance = model.feature_importances_
    top_n = 30
    top_indices = np.argsort(feature_importance)[-top_n:]
    
    # Create feature names
    yamnet_dim = yamnet_features.shape[1]
    feature_names = []
    for i in top_indices:
        if i < yamnet_dim:
            feature_names.append(f'YAMNet_{i}')
        else:
            feature_names.append(f'OpenSMILE_{i-yamnet_dim}')
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), feature_importance[top_indices])
    plt.yticks(range(top_n), feature_names)
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Feature Importances\n(YAMNet + OpenSMILE)', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    fi_path = f"{RESULTS_DIR}/feature_importance_{timestamp}.png"
    plt.savefig(fi_path, dpi=300)
    print(f"  ✓ Feature importance plot saved: {fi_path}")
    plt.close()
    
    # =============================
    # 💾 SAVE MODEL AND RESULTS
    # =============================
    
    print("\n" + "=" * 70)
    print("💾 Saving model and results...")
    print("=" * 70)
    
    # Save model
    model_path = f"{MODELS_DIR}/ensemble_yamnet_opensmile_{timestamp}.json"
    model.save_model(model_path)
    print(f"  ✓ Model saved: {model_path}")
    
    # Save scaler
    import pickle
    scaler_path = f"{MODELS_DIR}/scaler_{timestamp}.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"  ✓ Scaler saved: {scaler_path}")
    
    # Save metrics
    metrics = {
        'timestamp': timestamp,
        'model_type': 'Ensemble: YAMNet + OpenSMILE + XGBoost',
        'min_duration': MIN_DURATION,
        'yamnet_features': yamnet_dim,
        'opensmile_feature_set': OPENSMILE_FEATURE_SET,
        'opensmile_features': opensmile_features.shape[1],
        'total_features': X_combined.shape[1],
        'test_size': TEST_SIZE,
        'n_folds': N_FOLDS,
        'total_samples': len(X_combined),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'auc_roc': float(auc),
        'cv_mean': float(cv_scores.mean()),
        'cv_std': float(cv_scores.std()),
        'xgb_params': XGB_PARAMS,
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
    print("✅ Ensemble Training Complete!")
    print("=" * 70)
    print(f"\nBest Model: {model_path}")
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"AUC-ROC: {auc:.4f}")
    print(f"Total Features: {X_combined.shape[1]} (YAMNet: {yamnet_dim}, OpenSMILE: {opensmile_features.shape[1]})")
    print("=" * 70)

if __name__ == "__main__":
    main()
