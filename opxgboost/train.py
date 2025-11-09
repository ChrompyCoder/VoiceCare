"""
Parkinson's Disease Detection - Training Script
===============================================
Uses OpenSMILE + XGBoost

FOCUS: Temporal Features (Functionals) from OpenSMILE
- Statistical functionals over time (mean, std, max, min, etc.)
- NOT raw Low-Level Descriptors (LLDs)
- Functionals capture temporal dynamics of speech

Dataset:
  - ../data/figdata/HC_AH/ → Healthy patients
  - ../data/figdata/PD_AH/ → Parkinson's patients
  - ../data/gitdata/ → All gitdata (Parkinson's patients)
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, 
                            classification_report, roc_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pickle
import warnings
warnings.filterwarnings('ignore')

# Configuration
# Use absolute path resolution to work from any directory
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data"
RESULTS_DIR = SCRIPT_DIR / "results"
MODELS_DIR = SCRIPT_DIR / "models"
TEST_SIZE = 0.2
RANDOM_STATE = 42
FEATURE_SET = 'ComParE_2016'  # Uses FUNCTIONALS (temporal features), not LLDs
FEATURE_LEVEL = 'Functionals'  # Explicitly use Functionals (temporal)
MIN_DURATION = 0  # No duration filter

# XGBoost parameters
XGB_PARAMS = {
    'max_depth': 15,
    'learning_rate': 0.1,
    'n_estimators': 300,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': RANDOM_STATE,
    'use_label_encoder': False,
    'tree_method': 'hist'
}

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

def get_audio_duration(audio_path):
    """Get audio duration."""
    try:
        info = sf.info(audio_path)
        return info.duration
    except:
        return 0

def load_figdata():
    """Load figdata: HC_AH (Healthy) and PD_AH (Parkinson's)."""
    print("\n" + "=" * 70)
    print("📂 Loading figdata...")
    print("=" * 70)
    
    audio_files = []
    labels = []
    
    figdata_path = Path(DATA_DIR) / "figdata"
    
    # HC_AH - Healthy
    hc_path = figdata_path / "HC_AH"
    if hc_path.exists():
        hc_files = list(hc_path.glob("*.wav"))
        audio_files.extend([str(f) for f in hc_files])
        labels.extend([0] * len(hc_files))
        print(f"✓ HC_AH (Healthy): {len(hc_files)} files")
    
    # PD_AH - Parkinson's
    pd_path = figdata_path / "PD_AH"
    if pd_path.exists():
        pd_files = list(pd_path.glob("*.wav"))
        audio_files.extend([str(f) for f in pd_files])
        labels.extend([1] * len(pd_files))
        print(f"✓ PD_AH (Parkinson's): {len(pd_files)} files")
    
    print(f"✓ Total figdata: {len(audio_files)} files")
    return audio_files, labels

def load_gitdata():
    """Load all gitdata (Parkinson's patients)."""
    print("\n" + "=" * 70)
    print("📂 Loading gitdata (all directories)...")
    print("=" * 70)
    
    audio_files = []
    labels = []
    
    gitdata_path = Path(DATA_DIR) / "gitdata"
    
    # Find all .wav files in gitdata recursively
    if gitdata_path.exists():
        for wav_file in gitdata_path.rglob("*.wav"):
            audio_files.append(str(wav_file))
            labels.append(1)  # All gitdata are Parkinson's
    
    print(f"✓ Total gitdata: {len(audio_files)} files (all Parkinson's)")
    return audio_files, labels

def load_healthy_voice():
    """Load healthy voice directory (Healthy patients only)."""
    print("\n" + "=" * 70)
    print("📂 Loading healthy voice...")
    print("=" * 70)
    
    audio_files = []
    labels = []
    
    # healthy_voice is inside figdata directory
    healthy_path = Path(DATA_DIR) / "figdata" / "healthy_voice"
    
    # Load all .wav files from healthy voice directory
    if healthy_path.exists():
        healthy_files = list(healthy_path.glob("*.wav"))
        audio_files.extend([str(f) for f in healthy_files])
        labels.extend([0] * len(healthy_files))  # All are Healthy
        print(f"✓ Healthy voice: {len(healthy_files)} files (all Healthy)")
    else:
        print(f"⚠️  Healthy voice directory not found: {healthy_path}")
    
    print(f"✓ Total healthy voice: {len(audio_files)} files")
    return audio_files, labels

def extract_opensmile_features(audio_path, feature_set='ComParE_2016'):
    """
    Extract OpenSMILE TEMPORAL FEATURES (Functionals).
    
    Functionals are statistical summaries computed over time:
    - Mean, Standard Deviation, Skewness, Kurtosis
    - Maximum, Minimum, Range
    - Quartiles, Percentiles
    - Linear regression coefficients
    - Zero-crossing rates
    
    These capture temporal dynamics and are more robust than raw LLDs.
    """
    try:
        smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet[feature_set],
            feature_level=opensmile.FeatureLevel.Functionals,  # TEMPORAL features
        )
        features = smile.process_file(audio_path)
        return features.values.flatten()
    except Exception as e:
        print(f"Error extracting features from {audio_path}: {e}")
        return None

def extract_all_features(audio_files):
    """Extract features from all audio files."""
    print("\n" + "=" * 70)
    print("🎵 Extracting OpenSMILE TEMPORAL FEATURES (Functionals)...")
    print("=" * 70)
    print("These features capture statistical properties over time:")
    print("  - Mean, Std, Max, Min, Range")
    print("  - Skewness, Kurtosis")
    print("  - Quartiles, Percentiles")
    print("  - Linear regression coefficients")
    print("=" * 70)
    
    features_list = []
    valid_files = []
    
    for audio_file in tqdm(audio_files, desc="Processing"):
        features = extract_opensmile_features(audio_file, FEATURE_SET)
        if features is not None:
            features_list.append(features)
            valid_files.append(audio_file)
    
    X = np.array(features_list)
    print(f"\n✓ Extracted temporal features: {X.shape}")
    print(f"  {X.shape[0]} samples × {X.shape[1]} temporal features")
    return X, valid_files

def visualize_features(X, y, timestamp):
    """
    Visualize extracted temporal features before training.
    
    Creates multiple visualizations:
    1. PCA 2D projection
    2. t-SNE 2D projection
    3. Feature importance (top features by variance)
    4. Feature distribution comparison
    """
    print("\n" + "=" * 70)
    print("📊 VISUALIZING TEMPORAL FEATURES")
    print("=" * 70)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    
    # ===== 1. PCA Visualization =====
    print("Creating PCA visualization...")
    ax1 = plt.subplot(2, 3, 1)
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X)
    
    scatter1 = ax1.scatter(X_pca[y==0, 0], X_pca[y==0, 1], 
                          c='green', alpha=0.6, s=50, label='Healthy', edgecolors='k')
    scatter2 = ax1.scatter(X_pca[y==1, 0], X_pca[y==1, 1], 
                          c='red', alpha=0.6, s=50, label="Parkinson's", edgecolors='k')
    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
    ax1.set_title('PCA: Temporal Features Projection', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ===== 2. t-SNE Visualization =====
    print("Creating t-SNE visualization...")
    ax2 = plt.subplot(2, 3, 2)
    tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=30)
    X_tsne = tsne.fit_transform(X)
    
    ax2.scatter(X_tsne[y==0, 0], X_tsne[y==0, 1], 
               c='green', alpha=0.6, s=50, label='Healthy', edgecolors='k')
    ax2.scatter(X_tsne[y==1, 0], X_tsne[y==1, 1], 
               c='red', alpha=0.6, s=50, label="Parkinson's", edgecolors='k')
    ax2.set_xlabel('t-SNE Dimension 1')
    ax2.set_ylabel('t-SNE Dimension 2')
    ax2.set_title('t-SNE: Temporal Features Clustering', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # ===== 3. Feature Variance (Top Features) =====
    print("Analyzing feature importance...")
    ax3 = plt.subplot(2, 3, 3)
    feature_variance = np.var(X, axis=0)
    top_indices = np.argsort(feature_variance)[-20:]  # Top 20
    
    ax3.barh(range(20), feature_variance[top_indices], color='steelblue', edgecolor='k')
    ax3.set_xlabel('Variance')
    ax3.set_ylabel('Feature Index')
    ax3.set_title('Top 20 Temporal Features by Variance', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='x')
    
    # ===== 4. Class Distribution =====
    ax4 = plt.subplot(2, 3, 4)
    class_counts = [np.sum(y==0), np.sum(y==1)]
    colors = ['green', 'red']
    bars = ax4.bar(['Healthy', "Parkinson's"], class_counts, color=colors, 
                   alpha=0.7, edgecolor='k', linewidth=2)
    ax4.set_ylabel('Number of Samples')
    ax4.set_title('Class Distribution', fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add counts on bars
    for bar, count in zip(bars, class_counts):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count)}',
                ha='center', va='bottom', fontweight='bold')
    
    # ===== 5. Feature Statistics Comparison =====
    print("Computing feature statistics...")
    ax5 = plt.subplot(2, 3, 5)
    
    # Compare mean feature values between classes
    mean_healthy = np.mean(X[y==0], axis=0)
    mean_parkinsons = np.mean(X[y==1], axis=0)
    feature_diff = np.abs(mean_healthy - mean_parkinsons)
    top_diff_indices = np.argsort(feature_diff)[-20:]
    
    ax5.barh(range(20), feature_diff[top_diff_indices], color='coral', edgecolor='k')
    ax5.set_xlabel('Absolute Difference in Mean')
    ax5.set_ylabel('Feature Index')
    ax5.set_title('Top 20 Discriminative Temporal Features', fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='x')
    
    # ===== 6. Explained Variance (PCA) =====
    ax6 = plt.subplot(2, 3, 6)
    pca_full = PCA(random_state=RANDOM_STATE)
    pca_full.fit(X)
    cumsum_variance = np.cumsum(pca_full.explained_variance_ratio_)
    
    ax6.plot(range(1, min(51, len(cumsum_variance)+1)), 
            cumsum_variance[:50], 
            marker='o', linewidth=2, markersize=4, color='steelblue')
    ax6.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
    ax6.set_xlabel('Number of Components')
    ax6.set_ylabel('Cumulative Explained Variance')
    ax6.set_title('PCA: Cumulative Explained Variance', fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save visualization
    viz_path = f"{RESULTS_DIR}/feature_visualization_{timestamp}.png"
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Feature visualization saved: {viz_path}")
    plt.close()
    
    # Print summary statistics
    print("\n" + "=" * 70)
    print("📊 TEMPORAL FEATURE STATISTICS")
    print("=" * 70)
    print(f"Total temporal features: {X.shape[1]}")
    print(f"PCA variance (first 2 components): {pca.explained_variance_ratio_[0]*100:.2f}% + {pca.explained_variance_ratio_[1]*100:.2f}% = {sum(pca.explained_variance_ratio_)*100:.2f}%")
    
    # Find optimal number of components for 95% variance
    n_components_95 = np.argmax(cumsum_variance >= 0.95) + 1
    print(f"Components needed for 95% variance: {n_components_95}")
    
    print(f"\nClass separability (t-SNE visualization shows clustering)")
    print("=" * 70)

def extract_all_features(audio_files):
    """Extract features from all audio files."""
    print("\n" + "=" * 70)
    print("🎵 Extracting OpenSMILE features...")
    print("=" * 70)
    
    features_list = []
    valid_files = []
    
    for audio_file in tqdm(audio_files, desc="Processing"):
        features = extract_opensmile_features(audio_file, FEATURE_SET)
        if features is not None:
            features_list.append(features)
            valid_files.append(audio_file)
    
    X = np.array(features_list)
    print(f"\n✓ Extracted features: {X.shape}")
    return X, valid_files

def main():
    print("=" * 70)
    print("🎯 Training Parkinson's Detection Model")
    print("🎯 FOCUS: TEMPORAL FEATURES (Functionals)")
    print("=" * 70)
    print(f"Feature Set: {FEATURE_SET}")
    print(f"Feature Level: Functionals (Temporal)")
    print(f"Test Size: {TEST_SIZE * 100}%")
    print("=" * 70)
    
    # Load figdata
    fig_files, fig_labels = load_figdata()
    
    # Load gitdata
    git_files, git_labels = load_gitdata()
    
    # Load healthy voice
    healthy_files, healthy_labels = load_healthy_voice()
    
    # Combine datasets
    all_files = fig_files + git_files + healthy_files
    all_labels = fig_labels + git_labels + healthy_labels
    
    print("\n" + "=" * 70)
    print("📊 Combined Dataset")
    print("=" * 70)
    print(f"Total files: {len(all_files)}")
    print(f"Healthy: {all_labels.count(0)}")
    print(f"  - figdata/HC_AH: {fig_labels.count(0)}")
    print(f"  - healthy voice: {healthy_labels.count(0)}")
    print(f"Parkinson's: {all_labels.count(1)}")
    print(f"  - figdata/PD_AH: {fig_labels.count(1)}")
    print(f"  - gitdata: {git_labels.count(1)}")
    
    # Extract TEMPORAL features
    X, valid_files = extract_all_features(all_files)
    y = np.array([all_labels[all_files.index(f)] for f in valid_files])
    
    print("\n" + "=" * 70)
    print("📊 Temporal Feature Extraction Complete")
    print("=" * 70)
    print(f"Valid samples: {len(valid_files)}")
    print(f"Healthy: {np.sum(y == 0)}")
    print(f"Parkinson's: {np.sum(y == 1)}")
    
    # ===== VISUALIZE TEMPORAL FEATURES BEFORE TRAINING =====
    visualize_features(X, y, timestamp)
    
    # Scale features
    print("\n" + "=" * 70)
    print("📊 Scaling temporal features...")
    print("=" * 70)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("✓ Temporal features scaled")
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    print("\n" + "=" * 70)
    print("🔀 Train-Test Split")
    print("=" * 70)
    print(f"Training: {len(X_train)} samples")
    print(f"  Healthy: {np.sum(y_train == 0)}")
    print(f"  Parkinson's: {np.sum(y_train == 1)}")
    print(f"Test: {len(X_test)} samples")
    print(f"  Healthy: {np.sum(y_test == 0)}")
    print(f"  Parkinson's: {np.sum(y_test == 1)}")
    
    # Train XGBoost
    print("\n" + "=" * 70)
    print("🚀 Training XGBoost model...")
    print("=" * 70)
    
    model = xgb.XGBClassifier(**XGB_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=True
    )
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    print("\n" + "=" * 70)
    print("📈 Test Results")
    print("=" * 70)
    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUC-ROC:   {auc:.4f}")
    
    print("\n" + classification_report(y_test, y_pred, 
                                       target_names=['Healthy', "Parkinson's"]))
    
    # Save model
    model_path = f"{MODELS_DIR}/xgboost_model_{timestamp}.json"
    model.save_model(model_path)
    print(f"\n✓ Model saved: {model_path}")
    
    # Save scaler
    scaler_path = f"{MODELS_DIR}/scaler_{timestamp}.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ Scaler saved: {scaler_path}")
    
    # Save metrics
    metrics = {
        'timestamp': timestamp,
        'feature_set': FEATURE_SET,
        'feature_level': 'Functionals (Temporal)',
        'test_size': TEST_SIZE,
        'total_samples': len(valid_files),
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'auc_roc': float(auc),
        'dataset': {
            'figdata_healthy': fig_labels.count(0),
            'figdata_parkinsons': fig_labels.count(1),
            'gitdata_parkinsons': len(git_labels),
            'healthy_voice': len(healthy_labels),
            'total_healthy': all_labels.count(0),
            'total_parkinsons': all_labels.count(1)
        }
    }
    
    metrics_path = f"{RESULTS_DIR}/metrics_{timestamp}.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"✓ Metrics saved: {metrics_path}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Healthy', "Parkinson's"],
                yticklabels=['Healthy', "Parkinson's"])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    cm_path = f"{RESULTS_DIR}/confusion_matrix_{timestamp}.png"
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confusion matrix saved: {cm_path}")
    plt.close()
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    roc_path = f"{RESULTS_DIR}/roc_curve_{timestamp}.png"
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    print(f"✓ ROC curve saved: {roc_path}")
    plt.close()
    
    print("\n" + "=" * 70)
    print("✅ Training Complete!")
    print("=" * 70)
    print(f"\n🎯 Temporal Features Approach:")
    print(f"  - Used Functionals (statistical summaries over time)")
    print(f"  - NOT raw LLDs (Low-Level Descriptors)")
    print(f"  - Total temporal features: {X.shape[1]}")
    print(f"\n📁 Saved Files:")
    print(f"  Model: {model_path}")
    print(f"  Scaler: {scaler_path}")
    print(f"  Feature Visualization: {RESULTS_DIR}/feature_visualization_{timestamp}.png")
    print(f"\n📊 Performance:")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  AUC-ROC: {auc:.4f}")
    print("=" * 70)
    print("=" * 70)

if __name__ == "__main__":
    main()
