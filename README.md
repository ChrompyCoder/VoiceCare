# Voice-Based Parkinson's Disease Detection

This project implements a deep learning model using CNN + BiLSTM architecture to detect Parkinson's disease from voice recordings.

## Dataset Structure

```
data/figdata/
├── HC_AH/     # Healthy Controls (41 audio samples)
└── PD_AH/     # Parkinson's Disease patients (40 audio samples)
```

**Total**: 81 audio samples (5-second WAV files at 22050 Hz)

## Model Architecture

- **CNN Layers**: Extract spectral patterns from Mel-spectrograms (32→64→128 filters)
- **BiLSTM Layer**: Capture temporal dependencies in audio features (128 units)
- **Regularization**: Batch normalization, dropout (0.4), early stopping
- **Output**: Binary classification (Healthy vs Parkinson's)

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Validate Setup (Optional but Recommended)

```bash
python validate_setup.py
```

This checks if all dependencies are installed and data is accessible.

### 3. Train the Model

**Option A**: Using Python directly
```bash
python train.py
```

**Option B**: Using the bash script
```bash
chmod +x run_training.sh
./run_training.sh
```

The script will:
- Load audio files from `data/figdata/`
- Split data into train (70%), validation (15%), and test (15%) sets
- Extract Mel-spectrogram features (128 mel bands)
- Train the CNN+BiLSTM model
- Save the best model to `models/` directory
- Generate training history plots and confusion matrix in `results/` directory

### 4. Make Predictions

Use the inference script to predict on new audio files:

```bash
python inference.py models/best_model_<timestamp>.h5 path/to/audio.wav
```

Example:
```bash
python inference.py models/best_model_20241108_143000.h5 data/figdata/HC_AH/AH_064F_7AB034C9-72E4-438B-A9B3-AD7FDA1596C5.wav
```

## Training Configuration

The training uses the following split strategy:
- **Training**: ~70% (~57 samples) - Used to train model weights
- **Validation**: ~15% (~12 samples) - Used during training for hyperparameter tuning
- **Testing**: ~15% (~12 samples) - Held out for final evaluation

### Key Parameters (see `config.py` to modify)

- **Batch Size**: 16
- **Max Epochs**: 100 (with early stopping)
- **Learning Rate**: 1e-4 (with adaptive reduction)
- **Feature**: Mel-spectrogram (128 bands)
- **Audio Length**: 5 seconds

## Model Performance

The model is evaluated using:
- **Accuracy**: Overall correctness
- **Precision**: True Positives / (True Positives + False Positives)
- **Recall**: True Positives / (True Positives + False Negatives)  
- **F1-Score**: Harmonic mean of Precision and Recall
- **Confusion Matrix**: Visual breakdown of predictions

Results are saved in the `results/` directory after training.

## Project Structure

```
HK-11/
├── train.py              # Main training script
├── inference.py          # Prediction script for new audio files
├── validate_setup.py     # Environment validation script
├── config.py            # Configuration parameters
├── requirements.txt     # Python dependencies
├── README.md            # This file
├── PROJECT_SUMMARY.md   # Detailed project documentation
├── run_training.sh      # Bash script to run training
├── prompt.txt           # Original project instructions
├── utils/
│   ├── __init__.py
│   └── data_loader.py   # Data loading and feature extraction
├── data/
│   └── figdata/
│       ├── HC_AH/       # Healthy control audio files
│       └── PD_AH/       # Parkinson's patient audio files
├── models/              # Saved models (created during training)
└── results/             # Plots and metrics (created during training)
```

## Expected Output

After training completes, you'll find:

1. **Trained Model**: `models/best_model_<timestamp>.h5`
2. **Training History**: `results/training_history_<timestamp>.png`
3. **Confusion Matrix**: `results/confusion_matrix_<timestamp>.png`
4. **Console Output**: Classification report with precision/recall/F1-score

## Requirements

- Python 3.10+
- TensorFlow 2.15+
- Librosa 0.10+
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn

See `requirements.txt` for complete list.

## Features

- ✅ **Mel-spectrogram Analysis**: Extracts rich spectral features from audio
- ✅ **Hybrid Architecture**: Combines spatial (CNN) and temporal (BiLSTM) learning
- ✅ **Stratified Split**: Maintains class balance across train/val/test sets
- ✅ **Early Stopping**: Prevents overfitting during training
- ✅ **Learning Rate Scheduling**: Adaptive learning rate for better convergence
- ✅ **Comprehensive Evaluation**: Detailed metrics and visualizations
- ✅ **Easy Inference**: Simple script to test new audio samples

## Troubleshooting

**Issue**: Import errors
```bash
# Solution: Install all dependencies
pip install -r requirements.txt
```

**Issue**: No audio files found
```bash
# Solution: Verify data structure
python validate_setup.py
```

**Issue**: Out of memory during training
```bash
# Solution: Reduce batch size in config.py
# Change BATCH_SIZE from 16 to 8
```

## Author

Tushar (Hackathon Build)

## Documentation

For detailed technical information, see `PROJECT_SUMMARY.md`
