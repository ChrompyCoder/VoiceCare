"""
Train Random Forest model on UCI Parkinson's tabular data
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
import json
from pathlib import Path

# Paths
DATA_DIR = Path("data/data_tabular/uci-healthy-diseased")
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

print("="*70)
print("🎯 TRAINING RANDOM FOREST ON UCI PARKINSON'S TABULAR DATA")
print("="*70)

# Load data
print("\n[1/6] Loading data...")
train_df = pd.read_csv(DATA_DIR / "train_data.csv")
test_df = pd.read_csv(DATA_DIR / "test_data.csv")

print(f"   Train samples: {len(train_df)}")
print(f"   Test samples: {len(test_df)}")
print(f"   Features: {train_df.shape[1] - 1}")

# Check class distribution
train_dist = train_df['class information'].value_counts()
print(f"\n   Class distribution (train):")
print(f"      Healthy (0): {train_dist.get(0, 0)}")
print(f"      Parkinson's (1): {train_dist.get(1, 0)}")

# Prepare features and target
print("\n[2/6] Preparing features...")

# Drop subject id and target - use only common features
train_feature_cols = [col for col in train_df.columns if col not in ['Subject id', 'class information', 'UPDRS']]
test_feature_cols = [col for col in test_df.columns if col not in ['Subject id', 'class information', 'UPDRS']]

# Use intersection of features
feature_cols = list(set(train_feature_cols) & set(test_feature_cols))
feature_cols.sort()  # Consistent order

print(f"   Using {len(feature_cols)} common features")

X_train = train_df[feature_cols].values
y_train = train_df['class information'].values

X_test = test_df[feature_cols].values
y_test = test_df['class information'].values if 'class information' in test_df.columns else None

print(f"   X_train shape: {X_train.shape}")
print(f"   X_test shape: {X_test.shape}")

# Handle missing values (replace 'AC' etc with NaN and fill)
print("\n[3/6] Cleaning data...")

# Convert each column, replacing non-numeric with NaN
X_train_clean = []
for i in range(X_train.shape[1]):
    col_train = pd.to_numeric(X_train[:, i], errors='coerce')
    X_train_clean.append(col_train)

X_test_clean = []
for i in range(X_test.shape[1]):
    col_test = pd.to_numeric(X_test[:, i], errors='coerce')
    X_test_clean.append(col_test)

X_train = np.column_stack(X_train_clean)
X_test = np.column_stack(X_test_clean)

# Fill NaN with column mean
col_means_train = np.nanmean(X_train, axis=0)
for i in range(X_train.shape[1]):
    mask_train = np.isnan(X_train[:, i])
    mask_test = np.isnan(X_test[:, i])
    if np.any(mask_train):
        X_train[mask_train, i] = col_means_train[i]
    if np.any(mask_test):
        X_test[mask_test, i] = col_means_train[i]

print(f"   ✅ Data cleaned, NaN filled with column means")

# Standardize features
print("\n[4/6] Standardizing features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Random Forest
print("\n[5/6] Training Random Forest model...")
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42,
    class_weight='balanced',  # Handle class imbalance
    n_jobs=-1,
    verbose=1
)

model.fit(X_train_scaled, y_train)

# Evaluate
print("\n[6/6] Evaluating model...")
y_train_pred = model.predict(X_train_scaled)
y_train_proba = model.predict_proba(X_train_scaled)[:, 1]

train_metrics = {
    'accuracy': accuracy_score(y_train, y_train_pred),
    'precision': precision_score(y_train, y_train_pred),
    'recall': recall_score(y_train, y_train_pred),
    'f1_score': f1_score(y_train, y_train_pred),
    'auc': roc_auc_score(y_train, y_train_proba)
}

print("\n📊 TRAINING METRICS:")
print(f"   Accuracy:  {train_metrics['accuracy']:.4f}")
print(f"   Precision: {train_metrics['precision']:.4f}")
print(f"   Recall:    {train_metrics['recall']:.4f}")
print(f"   F1 Score:  {train_metrics['f1_score']:.4f}")
print(f"   AUC:       {train_metrics['auc']:.4f}")

if y_test is not None:
    y_test_pred = model.predict(X_test_scaled)
    y_test_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    test_metrics = {
        'accuracy': accuracy_score(y_test, y_test_pred),
        'precision': precision_score(y_test, y_test_pred),
        'recall': recall_score(y_test, y_test_pred),
        'f1_score': f1_score(y_test, y_test_pred),
        'auc': roc_auc_score(y_test, y_test_proba)
    }
    
    print("\n📊 TEST METRICS:")
    print(f"   Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"   Precision: {test_metrics['precision']:.4f}")
    print(f"   Recall:    {test_metrics['recall']:.4f}")
    print(f"   F1 Score:  {test_metrics['f1_score']:.4f}")
    print(f"   AUC:       {test_metrics['auc']:.4f}")

# Save model and scaler
print("\n💾 Saving model...")
model_path = MODELS_DIR / "uci_tabular_rf.pkl"
scaler_path = MODELS_DIR / "uci_tabular_scaler.pkl"
features_path = MODELS_DIR / "uci_tabular_features.json"

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)

# Save feature names
with open(features_path, 'w') as f:
    json.dump({'features': feature_cols}, f, indent=2)

print(f"   ✅ Model: {model_path}")
print(f"   ✅ Scaler: {scaler_path}")
print(f"   ✅ Features: {features_path}")

print("\n" + "="*70)
print("✅ TRAINING COMPLETE!")
print("="*70)
