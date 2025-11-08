"""
Parkinson's Disease Detection using OpenSMILE + XGBoost
=======================================================
Feature extraction using OpenSMILE library
Classification using XGBoost
Uses ORIGINAL speech dataset only (not denoised)
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import soundfile as sf
import opensmile
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
DATA_DIR = "../data"
RESULTS_DIR = "results"
MODELS_DIR = "models"
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_FOLDS = 5

# XGBoost hyperparameters
XGB_PARAMS = {
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 200,
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
# Options: 'ComParE_2016', 'GeMAPSv01b', 'eGeMAPSv02', 'emobase'
FEATURE_SET = 'ComParE_2016'  # Comprehensive feature set with 6373 features

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
        print(f"Error reading {audio_path}: {e}")
        return 0

def check_all_files():
    """Check and display all available audio files in the dataset."""
    print("\n" + "=" * 70)
    print("📂 Checking all audio files in dataset...")
    print("=" * 70)
    
    data_path = Path(DATA_DIR)
    
    # Check gitdata/original-speech-dataset
    print("\n1. GitData (original-speech-dataset):")
    print("-" * 70)
    
    original_path = data_path / "gitdata" / "original-speech-dataset"
    gitdata_count = 0
    
    if original_path.exists():
        # DL directory
        dl_path = original_path / "DL"
        if dl_path.exists():
            dl_files = list(dl_path.glob("*.wav"))
            gitdata_count += len(dl_files)
            print(f"  DL/: {len(dl_files)} .wav files")
        
        # LW directory
        lw_path = original_path / "LW"
        if lw_path.exists():
            lw_files = list(lw_path.glob("*.wav"))
            gitdata_count += len(lw_files)
            print(f"  LW/: {len(lw_files)} .wav files")
        
        # Faces directories
        faces_path = original_path / "Faces"
        if faces_path.exists():
            for speaker_dir in faces_path.glob("*_ori"):
                if speaker_dir.is_dir():
                    speaker_files = list(speaker_dir.glob("*.wav"))
                    gitdata_count += len(speaker_files)
                    print(f"  Faces/{speaker_dir.name}/: {len(speaker_files)} .wav files")
        
        print(f"  Total GitData: {gitdata_count} files (all Parkinson's)")
    else:
        print("  ⚠️  original-speech-dataset not found!")
    
    # Check figdata
    print("\n2. FigData:")
    print("-" * 70)
    
    figdata_path = data_path / "figdata"
    hc_count = 0
    pd_count = 0
    
    if figdata_path.exists():
        # HC_AH (Healthy)
        hc_path = figdata_path / "HC_AH"
        if hc_path.exists():
            hc_files = list(hc_path.glob("*.wav"))
            hc_count = len(hc_files)
            print(f"  HC_AH/: {hc_count} .wav files (Healthy)")
        
        # PD_AH (Parkinson's)
        pd_path = figdata_path / "PD_AH"
        if pd_path.exists():
            pd_files = list(pd_path.glob("*.wav"))
            pd_count = len(pd_files)
            print(f"  PD_AH/: {pd_count} .wav files (Parkinson's)")
        
        print(f"  Total FigData: {hc_count + pd_count} files")
    else:
        print("  ⚠️  figdata not found!")
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 Dataset Summary:")
    print("=" * 70)
    total_files = gitdata_count + hc_count + pd_count
    total_healthy = hc_count
    total_parkinsons = gitdata_count + pd_count
    
    print(f"  Total files: {total_files}")
    print(f"  Healthy: {total_healthy} ({total_healthy/total_files*100:.1f}%)")
    print(f"  Parkinson's: {total_parkinsons} ({total_parkinsons/total_files*100:.1f}%)")
    print("=" * 70)
    
    return {
        'gitdata_count': gitdata_count,
        'hc_count': hc_count,
        'pd_count': pd_count,
        'total_files': total_files
    }

def load_gitdata_original():
    """Load audio files from gitdata/original-speech-dataset (all Parkinson's)."""
    print("\n" + "=" * 70)
    print("📂 Loading gitdata (original-speech-dataset - Parkinson's patients)...")
    print("=" * 70)
    
    audio_files = []
    labels = []
    
    # Path to original-speech-dataset (all Parkinson's)
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
                            audio_files.append(str(audio_path))
                            labels.append(1)  # Parkinson's
    
    # Load from Faces directories
    faces_path = original_path / "Faces"
    if faces_path.exists():
        for speaker_dir in faces_path.glob("*_ori"):
            if speaker_dir.is_dir():
                for audio_file in speaker_dir.glob("*.wav"):
                    audio_files.append(str(audio_file))
                    labels.append(1)  # Parkinson's
    
    print(f"  ✓ Loaded {len(audio_files)} audio files from original-speech-dataset")
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
            audio_files.append(str(audio_file))
            labels.append(0)  # Healthy
            hc_count += 1
        print(f"  ✓ Loaded {hc_count} Healthy Control files (HC_AH, label=0)")
    
    # Load PD_AH (Parkinson's Disease)
    pd_path = figdata_path / "PD_AH"
    if pd_path.exists():
        pd_count = 0
        for audio_file in pd_path.glob("*.wav"):
            audio_files.append(str(audio_file))
            labels.append(1)  # Parkinson's
            pd_count += 1
        print(f"  ✓ Loaded {pd_count} Parkinson's files (PD_AH, label=1)")
    
    print(f"  ✓ Total figdata: {len(audio_files)} files")
    
    return audio_files, labels

def extract_opensmile_features(audio_files, feature_set='ComParE_2016'):
    """
    Extract features using OpenSMILE library.
    
    Feature sets:
    - ComParE_2016: 6373 features (Computational Paralinguistics Challenge)
    - GeMAPSv01b: 62 features (Geneva Minimalistic Acoustic Parameter Set)
    - eGeMAPSv02: 88 features (extended GeMAPS)
    - emobase: 988 features (Emotion base)
    """
    print("\n" + "=" * 70)
    print(f"🎵 Extracting OpenSMILE features (feature set: {feature_set})...")
    print("=" * 70)
    
    # Initialize OpenSMILE
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet[feature_set],
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    
    features_list = []
    valid_files = []
    
    for audio_file in tqdm(audio_files, desc="Processing audio files"):
        try:
            # Extract features
            features = smile.process_file(audio_file)
            
            # Convert to numpy array
            feature_vector = features.values.flatten()
            
            features_list.append(feature_vector)
            valid_files.append(audio_file)
            
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            continue
    
    if not features_list:
        raise ValueError("No features extracted! Check audio files and OpenSMILE installation.")
    
    # Convert to numpy array
    X = np.array(features_list)
    
    print(f"\n  ✓ Extracted features from {len(features_list)} files")
    print(f"  ✓ Feature dimensions: {X.shape}")
    print(f"  ✓ Feature set: {feature_set}")
    
    return X, valid_files

# =============================
# 🎯 MAIN TRAINING PIPELINE
# =============================

def main():
    print("\n" + "=" * 70)
    print("🎯 Parkinson's Detection: OpenSMILE + XGBoost")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  - Feature set: {FEATURE_SET}")
    print(f"  - Test size: {TEST_SIZE * 100}%")
    print(f"  - K-Folds: {N_FOLDS}")
    print(f"  - Using ORIGINAL speech dataset (not denoised)")
    print("=" * 70)
    
    # =============================
    # 📊 CHECK ALL FILES
    # =============================
    
    file_stats = check_all_files()
    
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
    # 🎵 FEATURE EXTRACTION
    # =============================
    
    X, valid_files = extract_opensmile_features(all_files, FEATURE_SET)
    
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
    print("🤖 Training XGBoost model...")
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
    plt.title(f'Confusion Matrix - OpenSMILE + XGBoost\n({FEATURE_SET})', 
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
    plt.plot(fpr, tpr, linewidth=2, label=f'XGBoost (AUC = {auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - OpenSMILE + XGBoost\n({FEATURE_SET})', 
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
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), feature_importance[top_indices])
    plt.yticks(range(top_n), [f'Feature {i}' for i in top_indices])
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Feature Importances\n({FEATURE_SET})', 
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
    model_path = f"{MODELS_DIR}/xgboost_opensmile_{timestamp}.json"
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
        'feature_set': FEATURE_SET,
        'test_size': TEST_SIZE,
        'n_folds': N_FOLDS,
        'total_samples': len(X),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'feature_dim': X.shape[1],
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
            'figdata_healthy': all_labels.count(0) - len(git_labels),
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
    print(f"Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"AUC-ROC: {auc:.4f}")
    print(f"Feature Set: {FEATURE_SET} ({X.shape[1]} features)")
    print("=" * 70)

if __name__ == "__main__":
    main()
