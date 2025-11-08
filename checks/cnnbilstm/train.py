"""
Project: Voice-Based Parkinson's Disease Detection using CNN + BiLSTM
Author: Tushar (Hackathon Build)
Description:
This script builds a high-accuracy deep learning model for detecting Parkinson's disease 
from voice recordings. The dataset consists of audio samples collected from 
patients and healthy controls.

Datasets:
- figdata/HC_AH: Healthy Controls (labeled as 0)
- figdata/PD_AH: Parkinson's Disease patients (labeled as 1)

Goal:
Build a CNN + BiLSTM hybrid model that analyzes Mel-spectrograms of audio samples 
to classify between Parkinson's and healthy controls with maximum accuracy.
"""

# =============================
# 📦 IMPORTS
# =============================
import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# =============================
# 📂 DATASET CONFIGURATION
# =============================
DATA_PATH = "data/figdata"
SAMPLE_RATE = 22050
DURATION = 5  # seconds
SAMPLES_PER_FILE = SAMPLE_RATE * DURATION

# =============================
# 🧹 DATA LOADER & FEATURE EXTRACTION
# =============================
def load_audio_files(data_path):
    """
    Loads audio files and corresponding labels from subdirectories.
    Directory structure:
    data/figdata/
        HC_AH/  -> Healthy Controls (label 0)
        PD_AH/  -> Parkinson's Disease (label 1)
    """
    X, y = [], []
    
    # Load Healthy Controls
    hc_path = os.path.join(data_path, "HC_AH")
    if os.path.exists(hc_path):
        print(f"Loading healthy controls from {hc_path}")
        for file in os.listdir(hc_path):
            if file.endswith(".wav"):
                file_path = os.path.join(hc_path, file)
                X.append(file_path)
                y.append(0)  # Healthy
        print(f"Loaded {len([f for f in os.listdir(hc_path) if f.endswith('.wav')])} healthy control files")
    
    # Load Parkinson's Disease patients
    pd_path = os.path.join(data_path, "PD_AH")
    if os.path.exists(pd_path):
        print(f"Loading Parkinson's patients from {pd_path}")
        for file in os.listdir(pd_path):
            if file.endswith(".wav"):
                file_path = os.path.join(pd_path, file)
                X.append(file_path)
                y.append(1)  # Parkinson's
        print(f"Loaded {len([f for f in os.listdir(pd_path) if f.endswith('.wav')])} Parkinson's patient files")
    
    print(f"\nTotal files loaded: {len(X)}")
    print(f"Healthy: {sum(1 for label in y if label == 0)}, Parkinson's: {sum(1 for label in y if label == 1)}")
    
    return np.array(X), np.array(y)

def extract_melspectrogram(file_path, n_mels=128, n_fft=2048, hop_length=512):
    """Extracts a Mel-spectrogram from an audio file."""
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        if len(y) > SAMPLES_PER_FILE:
            y = y[:SAMPLES_PER_FILE]
        else:
            y = np.pad(y, (0, max(0, SAMPLES_PER_FILE - len(y))), "constant")
        
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        return mel_db
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def prepare_dataset(X, y):
    """Extracts features and converts to NumPy arrays suitable for CNN input."""
    features = []
    labels = []
    for i, path in enumerate(X):
        mel = extract_melspectrogram(path)
        if mel is not None:
            features.append(mel)
            labels.append(y[i])
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(X)} files...")
    
    X = np.array(features)
    X = X[..., np.newaxis]  # add channel dimension
    print(f"Feature shape: {X.shape}")
    return X, np.array(labels)

# =============================
# 🧠 MODEL DEFINITION
# =============================
def build_cnn_bilstm(input_shape):
    """
    CNN + BiLSTM hybrid architecture:
    CNN extracts spectral patterns, BiLSTM captures temporal dependencies.
    """
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape),
        layers.MaxPooling2D((2,2)),
        layers.BatchNormalization(),

        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.BatchNormalization(),

        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.BatchNormalization(),

        layers.Reshape((-1, 128)),
        layers.Bidirectional(layers.LSTM(128, return_sequences=False)),
        layers.Dropout(0.4),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    return model

# =============================
# ⚙️ TRAINING PIPELINE
# =============================
def main():
    print("=" * 60)
    print("Voice-Based Parkinson's Disease Detection")
    print("CNN + BiLSTM Model Training")
    print("=" * 60)
    print()
    
    # Load dataset
    print("Loading audio files...")
    X_paths, y = load_audio_files(DATA_PATH)
    
    if len(X_paths) == 0:
        print("No audio files found! Please check the data directory.")
        return
    
    # Split dataset: 70% train, 15% validation (from train), 15% test
    X_train_paths, X_test_paths, y_train, y_test = train_test_split(
        X_paths, y, test_size=0.15, stratify=y, random_state=42
    )
    
    print(f"\nDataset split:")
    print(f"Training set: {len(X_train_paths)} samples")
    print(f"Test set: {len(X_test_paths)} samples")
    
    # Extract features
    print("\n" + "=" * 60)
    print("Extracting features for training set...")
    X_train, y_train = prepare_dataset(X_train_paths, y_train)
    
    print("\nExtracting features for test set...")
    X_test, y_test = prepare_dataset(X_test_paths, y_test)
    
    # Build model
    print("\n" + "=" * 60)
    print("Building CNN + BiLSTM model...")
    input_shape = X_train.shape[1:]
    print(f"Input shape: {input_shape}")
    model = build_cnn_bilstm(input_shape)
    model.summary()
    
    # Setup callbacks
    os.makedirs("models", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/best_model_{timestamp}.h5"
    
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        model_path, 
        monitor="val_accuracy", 
        save_best_only=True, 
        mode="max",
        verbose=1
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", 
        patience=15, 
        restore_best_weights=True,
        verbose=1
    )
    lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", 
        factor=0.5, 
        patience=7, 
        min_lr=1e-7,
        verbose=1
    )
    
    # Train model
    print("\n" + "=" * 60)
    print("Training model...")
    print("=" * 60)
    
    history = model.fit(
        X_train, y_train,
        validation_split=0.18,  # ~15% of total data for validation
        epochs=100,
        batch_size=16,
        callbacks=[checkpoint, early_stopping, lr_schedule],
        verbose=1
    )
    
    # Save training history
    print("\n" + "=" * 60)
    print("Saving training history...")
    os.makedirs("results", exist_ok=True)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'results/training_history_{timestamp}.png')
    print(f"Training history plot saved to results/training_history_{timestamp}.png")
    
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
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Healthy', 'Parkinson'], 
                yticklabels=['Healthy', 'Parkinson'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'results/confusion_matrix_{timestamp}.png')
    print(f"\nConfusion matrix saved to results/confusion_matrix_{timestamp}.png")
    
    # Test accuracy
    test_loss, test_acc, test_precision, test_recall = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest Results:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Accuracy: {test_acc:.4f}")
    print(f"  Precision: {test_precision:.4f}")
    print(f"  Recall: {test_recall:.4f}")
    print(f"  F1-Score: {2 * (test_precision * test_recall) / (test_precision + test_recall):.4f}")
    
    print("\n" + "=" * 60)
    print(f"Training complete! Best model saved to: {model_path}")
    print("=" * 60)

# =============================
# 🧪 TEST SCRIPT (RUN INFERENCE)
# =============================
def predict_parkinson(file_path, model):
    """Predicts whether the input audio file indicates Parkinson's disease."""
    mel = extract_melspectrogram(file_path)
    if mel is None:
        print(f"Error: Could not process file {file_path}")
        return None, None
    
    mel = mel[np.newaxis, ..., np.newaxis]
    prediction = model.predict(mel, verbose=0)[0][0]
    label = "Parkinson" if prediction > 0.5 else "Healthy"
    print(f"File: {file_path}")
    print(f"Prediction: {label} (confidence: {prediction*100:.2f}%)")
    return label, prediction

if __name__ == "__main__":
    main()
