"""
Parkinson's Disease Detection using Custom Features + XGBoost
=============================================================
Feature extraction using Librosa, Parselmouth, and custom methods
Classification using XGBoost
Filters audio files with duration > 4 seconds
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import librosa
import soundfile as sf
import parselmouth
from parselmouth.praat import call
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
MIN_DURATION = 4.0  # Minimum audio duration in seconds
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

def load_gitdata():
    """Load audio files from gitdata (denoised-speech-dataset)."""
    print("\n" + "=" * 70)
    print("📂 Loading gitdata (Parkinson's patients)...")
    print("=" * 70)
    
    audio_files = []
    labels = []
    
    # Path to denoised-speech-dataset (all Parkinson's)
    denoised_path = Path(DATA_DIR) / "gitdata" / "denoised-speech-dataset"
    
    # Load from DL directory
    dl_path = denoised_path / "DL"
    if dl_path.exists():
        for txt_file in dl_path.glob("DL*.txt"):
            with open(txt_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        audio_path = dl_path / line
                        if audio_path.exists():
                            duration = get_audio_duration(str(audio_path))
                            if duration >= MIN_DURATION:
                                audio_files.append(str(audio_path))
                                labels.append(1)  # Parkinson's
    
    # Load from LW directory
    lw_path = denoised_path / "LW"
    if lw_path.exists():
        for txt_file in lw_path.glob("LW*.txt"):
            with open(txt_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        audio_path = lw_path / line
                        if audio_path.exists():
                            duration = get_audio_duration(str(audio_path))
                            if duration >= MIN_DURATION:
                                audio_files.append(str(audio_path))
                                labels.append(1)  # Parkinson's
    
    # Load from Faces directories
    faces_path = denoised_path / "Faces"
    if faces_path.exists():
        for speaker_dir in faces_path.glob("*_au"):
            if speaker_dir.is_dir():
                for audio_file in speaker_dir.glob("*.wav"):
                    duration = get_audio_duration(str(audio_file))
                    if duration >= MIN_DURATION:
                        audio_files.append(str(audio_file))
                        labels.append(1)  # Parkinson's
    
    print(f"  ✓ Loaded {len(audio_files)} audio files from gitdata (duration > {MIN_DURATION}s)")
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
        print(f"  ✓ Loaded {hc_count} Healthy Control files (HC_AH, label=0)")
    
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
        print(f"  ✓ Loaded {pd_count} Parkinson's files (PD_AH, label=1)")
    
    print(f"  ✓ Total figdata: {len(audio_files)} files (duration > {MIN_DURATION}s)")
    
    return audio_files, labels

def extract_acoustic_features(audio_path):
    """
    Extract comprehensive acoustic features using Librosa and Parselmouth.
    Features include: MFCC, spectral, prosodic, voice quality metrics.
    """
    try:
        # Load audio with librosa
        y, sr = librosa.load(audio_path, sr=22050)
        
        # Initialize feature dictionary
        features = {}
        
        # ========== MFCC Features (13 coefficients) ==========
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(13):
            features[f'mfcc_{i}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i}_std'] = np.std(mfccs[i])
            features[f'mfcc_{i}_max'] = np.max(mfccs[i])
            features[f'mfcc_{i}_min'] = np.min(mfccs[i])
        
        # ========== Spectral Features ==========
        # Spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroid)
        features['spectral_centroid_std'] = np.std(spectral_centroid)
        features['spectral_centroid_max'] = np.max(spectral_centroid)
        features['spectral_centroid_min'] = np.min(spectral_centroid)
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
        
        # Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        for i in range(spectral_contrast.shape[0]):
            features[f'spectral_contrast_{i}_mean'] = np.mean(spectral_contrast[i])
        
        # Spectral flatness
        spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
        features['spectral_flatness_mean'] = np.mean(spectral_flatness)
        features['spectral_flatness_std'] = np.std(spectral_flatness)
        
        # ========== Temporal Features ==========
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        
        # RMS energy
        rms = librosa.feature.rms(y=y)[0]
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)
        features['rms_max'] = np.max(rms)
        features['rms_min'] = np.min(rms)
        
        # ========== Chroma Features ==========
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features['chroma_mean'] = np.mean(chroma)
        features['chroma_std'] = np.std(chroma)
        
        # ========== Prosodic Features using Parselmouth ==========
        try:
            sound = parselmouth.Sound(audio_path)
            
            # Pitch (F0) features
            pitch = sound.to_pitch()
            pitch_values = pitch.selected_array['frequency']
            pitch_values = pitch_values[pitch_values > 0]  # Remove unvoiced frames
            
            if len(pitch_values) > 0:
                features['pitch_mean'] = np.mean(pitch_values)
                features['pitch_std'] = np.std(pitch_values)
                features['pitch_max'] = np.max(pitch_values)
                features['pitch_min'] = np.min(pitch_values)
                features['pitch_range'] = np.max(pitch_values) - np.min(pitch_values)
            else:
                features['pitch_mean'] = 0
                features['pitch_std'] = 0
                features['pitch_max'] = 0
                features['pitch_min'] = 0
                features['pitch_range'] = 0
            
            # Jitter (pitch variability)
            try:
                point_process = call(sound, "To PointProcess (periodic, cc)", 75, 600)
                jitter_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
                jitter_rap = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
                jitter_ppq5 = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
                
                features['jitter_local'] = jitter_local if not np.isnan(jitter_local) else 0
                features['jitter_rap'] = jitter_rap if not np.isnan(jitter_rap) else 0
                features['jitter_ppq5'] = jitter_ppq5 if not np.isnan(jitter_ppq5) else 0
            except:
                features['jitter_local'] = 0
                features['jitter_rap'] = 0
                features['jitter_ppq5'] = 0
            
            # Shimmer (amplitude variability)
            try:
                shimmer_local = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
                shimmer_apq3 = call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
                shimmer_apq5 = call([sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
                
                features['shimmer_local'] = shimmer_local if not np.isnan(shimmer_local) else 0
                features['shimmer_apq3'] = shimmer_apq3 if not np.isnan(shimmer_apq3) else 0
                features['shimmer_apq5'] = shimmer_apq5 if not np.isnan(shimmer_apq5) else 0
            except:
                features['shimmer_local'] = 0
                features['shimmer_apq3'] = 0
                features['shimmer_apq5'] = 0
            
            # Harmonicity (HNR - Harmonics-to-Noise Ratio)
            try:
                harmonicity = sound.to_harmonicity()
                hnr_values = harmonicity.values[harmonicity.values != -200]  # Remove undefined values
                if len(hnr_values) > 0:
                    features['hnr_mean'] = np.mean(hnr_values)
                    features['hnr_std'] = np.std(hnr_values)
                else:
                    features['hnr_mean'] = 0
                    features['hnr_std'] = 0
            except:
                features['hnr_mean'] = 0
                features['hnr_std'] = 0
            
            # Formants
            try:
                formant = sound.to_formant_burg()
                f1_values = []
                f2_values = []
                f3_values = []
                
                for t in np.linspace(sound.xmin, sound.xmax, 10):
                    f1 = formant.get_value_at_time(1, t)
                    f2 = formant.get_value_at_time(2, t)
                    f3 = formant.get_value_at_time(3, t)
                    
                    if not np.isnan(f1): f1_values.append(f1)
                    if not np.isnan(f2): f2_values.append(f2)
                    if not np.isnan(f3): f3_values.append(f3)
                
                features['f1_mean'] = np.mean(f1_values) if f1_values else 0
                features['f2_mean'] = np.mean(f2_values) if f2_values else 0
                features['f3_mean'] = np.mean(f3_values) if f3_values else 0
                features['f1_std'] = np.std(f1_values) if f1_values else 0
                features['f2_std'] = np.std(f2_values) if f2_values else 0
                features['f3_std'] = np.std(f3_values) if f3_values else 0
            except:
                features['f1_mean'] = 0
                features['f2_mean'] = 0
                features['f3_mean'] = 0
                features['f1_std'] = 0
                features['f2_std'] = 0
                features['f3_std'] = 0
                
        except Exception as e:
            # If Parselmouth fails, set prosodic features to 0
            for key in ['pitch_mean', 'pitch_std', 'pitch_max', 'pitch_min', 'pitch_range',
                       'jitter_local', 'jitter_rap', 'jitter_ppq5',
                       'shimmer_local', 'shimmer_apq3', 'shimmer_apq5',
                       'hnr_mean', 'hnr_std',
                       'f1_mean', 'f2_mean', 'f3_mean', 'f1_std', 'f2_std', 'f3_std']:
                if key not in features:
                    features[key] = 0
        
        # Convert to numpy array
        feature_vector = np.array(list(features.values()))
        
        # Check for any NaN or Inf values
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
        
        return feature_vector
            
    except Exception as e:
        print(f"Error extracting features from {audio_path}: {e}")
        return None

def extract_all_features(audio_files, labels):
    """Extract features from all audio files."""
    print("\n" + "=" * 70)
    print("🎵 Extracting acoustic features (MFCC, Spectral, Prosodic, Voice Quality)...")
    print("=" * 70)
    
    features_list = []
    valid_labels = []
    valid_files = []
    
    for i, audio_file in enumerate(tqdm(audio_files, desc="Processing audio files")):
        features = extract_acoustic_features(audio_file)
        
        if features is not None and len(features) > 0:
            features_list.append(features)
            valid_labels.append(labels[i])
            valid_files.append(audio_file)
    
    if not features_list:
        raise ValueError("No features extracted! Check audio files and Surfboard installation.")
    
    # Convert to numpy array
    X = np.array(features_list)
    y = np.array(valid_labels)
    
    print(f"\n  ✓ Extracted features from {len(features_list)} files")
    print(f"  ✓ Feature dimensions: {X.shape}")
    print(f"  ✓ Healthy samples: {np.sum(y == 0)} ({np.sum(y == 0) / len(y) * 100:.1f}%)")
    print(f"  ✓ Parkinson's samples: {np.sum(y == 1)} ({np.sum(y == 1) / len(y) * 100:.1f}%)")
    
    return X, y, valid_files

# =============================
# 🎯 MAIN TRAINING PIPELINE
# =============================

def main():
    print("\n" + "=" * 70)
    print("🎯 Parkinson's Detection: Surfboard + XGBoost")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  - Minimum duration: {MIN_DURATION} seconds")
    print(f"  - Test size: {TEST_SIZE * 100}%")
    print(f"  - K-Folds: {N_FOLDS}")
    print(f"  - Features: MFCC, Spectral, Prosodic, Voice Quality (Jitter, Shimmer, HNR)")
    print("=" * 70)
    
    # =============================
    # 📊 LOAD DATA
    # =============================
    
    # Load gitdata
    git_files, git_labels = load_gitdata()
    
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
    
    X, y, valid_files = extract_all_features(all_files, all_labels)
    
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
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
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
    plt.title('ROC Curve', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    roc_path = f"{RESULTS_DIR}/roc_curve_{timestamp}.png"
    plt.savefig(roc_path, dpi=300)
    print(f"  ✓ ROC curve saved: {roc_path}")
    plt.close()
    
    # Feature Importance
    feature_importance = model.feature_importances_
    top_n = 20
    top_indices = np.argsort(feature_importance)[-top_n:]
    
    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), feature_importance[top_indices])
    plt.yticks(range(top_n), [f'Feature {i}' for i in top_indices])
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
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
    model_path = f"{MODELS_DIR}/xgboost_model_{timestamp}.json"
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
        'min_duration': MIN_DURATION,
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
        'feature_types': ['mfcc', 'spectral', 'prosodic', 'voice_quality', 'temporal', 'chroma']
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
    print("=" * 70)

if __name__ == "__main__":
    main()
