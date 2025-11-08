# Parkinson's Detection: Surfboard + XGBoost

This directory contains an alternative approach to Parkinson's Disease detection using:
- **Surfboard** for comprehensive audio feature extraction
- **XGBoost** for gradient boosted tree classification

## 🎯 Approach

Unlike the deep learning CNN+BiLSTM approach, this method uses traditional machine learning with sophisticated feature engineering:

1. **Feature Extraction**: Surfboard extracts multiple audio features:
   - MFCC (Mel-frequency cepstral coefficients)
   - Spectral features (centroid, rolloff, flux, etc.)
   - Intensity and energy features
   - Temporal features (zero-crossing rate, etc.)
   - Formant frequencies
   - Loudness features
   - F0 (fundamental frequency/pitch)
   - Harmonicity (harmonics-to-noise ratio)
   - Jitter (pitch variation)
   - Shimmer (amplitude variation)

2. **Classification**: XGBoost gradient boosted trees for robust classification

## 📋 Requirements

- Python 3.8+
- Audio files with duration > 4 seconds
- See `requirements.txt` for full dependencies

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run Training

```bash
python train_xgboost.py
```

## 📊 Dataset

The script automatically loads:
- **gitdata/denoised-speech-dataset**: Parkinson's patients (label=1)
  - DL/ directory
  - LW/ directory
  - Faces/ directory
- **figdata**: Mixed dataset
  - HC_AH/: Healthy controls (label=0)
  - PD_AH/: Parkinson's patients (label=1)

Only audio files with **duration > 4 seconds** are included.

## 🔧 Configuration

Edit the following parameters in `train_xgboost.py`:

```python
MIN_DURATION = 4.0  # Minimum audio duration (seconds)
TEST_SIZE = 0.2     # Test set size (20%)
N_FOLDS = 5         # K-fold cross-validation folds

# XGBoost parameters
XGB_PARAMS = {
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 200,
    # ... more parameters
}
```

## 📈 Output

After training, you'll find:

### Models Directory (`models/`)
- `xgboost_model_<timestamp>.json` - Trained XGBoost model
- `scaler_<timestamp>.pkl` - Feature scaler

### Results Directory (`results/`)
- `metrics_<timestamp>.json` - Detailed metrics
- `confusion_matrix_<timestamp>.png` - Confusion matrix visualization
- `roc_curve_<timestamp>.png` - ROC curve
- `feature_importance_<timestamp>.png` - Top feature importances

## 📊 Evaluation Metrics

The model is evaluated using:
- **Accuracy**: Overall correctness
- **Precision**: Positive predictive value
- **Recall**: Sensitivity/True positive rate
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under ROC curve
- **K-Fold CV**: Cross-validation scores

## 🆚 Comparison with CNN+BiLSTM

| Aspect | Surfboard+XGBoost | CNN+BiLSTM |
|--------|------------------|------------|
| Features | Hand-crafted (Surfboard) | Learned (spectrograms) |
| Model | Gradient boosting | Deep neural network |
| Training Time | Faster | Slower |
| Interpretability | High (feature importance) | Low (black box) |
| Data Requirements | Works with smaller datasets | Needs more data |
| Overfitting Risk | Lower | Higher |

## 🎓 Feature Engineering

Surfboard extracts comprehensive acoustic features including:
- **Prosodic**: Pitch, intensity, duration
- **Voice Quality**: Jitter, shimmer, HNR
- **Spectral**: MFCC, spectral centroid, rolloff
- **Temporal**: Zero-crossing rate, energy

These features capture voice characteristics that may differ between healthy individuals and Parkinson's patients.

## 📝 Notes

- Requires `surfboard` package which has dependencies on audio processing libraries
- Feature extraction is computationally intensive but only done once
- Model training is fast compared to deep learning
- Results are highly interpretable through feature importance analysis

## 🔍 Troubleshooting

**If Surfboard installation fails:**
```bash
# Install dependencies first
pip install numpy scipy pandas librosa soundfile
# Then install surfboard
pip install surfboard
```

**If audio files are not found:**
- Check that `../data/` directory exists relative to this folder
- Verify audio file paths in the dataset directories

## 📚 References

- Surfboard: https://github.com/novoic/surfboard
- XGBoost: https://xgboost.readthedocs.io/
