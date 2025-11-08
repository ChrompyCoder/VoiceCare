# Parkinson's Detection: OpenSMILE + XGBoost

This directory contains a Parkinson's Disease detection system using **OpenSMILE** for feature extraction and **XGBoost** for classification.

## 🎯 Approach

This implementation uses:
- **OpenSMILE**: Industry-standard audio feature extraction toolkit
- **Original Speech Dataset**: Uses original-speech-dataset (NOT denoised)
- **XGBoost**: Gradient boosted decision trees for classification

## 📊 Dataset

### GitData (original-speech-dataset)
- **Location**: `../data/gitdata/original-speech-dataset/`
- **Label**: All Parkinson's (label=1)
- **Subdirectories**:
  - `DL/`: Speaker audio files
  - `LW/`: Speaker audio files  
  - `Faces/*_ori/`: Multiple speaker directories

### FigData
- **Location**: `../data/figdata/`
- **HC_AH/**: Healthy controls (label=0)
- **PD_AH/**: Parkinson's patients (label=1)

**Note**: This script uses ONLY the original-speech-dataset, NOT the denoised version.

## 🎵 OpenSMILE Feature Sets

The script supports multiple feature sets:

| Feature Set | Features | Description |
|-------------|----------|-------------|
| **ComParE_2016** | 6,373 | Computational Paralinguistics Challenge (default) |
| **GeMAPSv01b** | 62 | Geneva Minimalistic Acoustic Parameter Set |
| **eGeMAPSv02** | 88 | Extended GeMAPS |
| **emobase** | 988 | Emotion recognition base set |

Default: **ComParE_2016** (most comprehensive)

## 🚀 Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- OpenSMILE 2.4+
- XGBoost 2.0+
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn

## 📋 Usage

### 1. Check Dataset

The script automatically checks all files before training:

```bash
python train_opensmile_xgboost.py
```

This will display:
- GitData file counts (DL, LW, Faces)
- FigData file counts (HC_AH, PD_AH)
- Total dataset statistics

### 2. Training

Run the full training pipeline:

```bash
python train_opensmile_xgboost.py
```

The script will:
1. ✅ Check all audio files in the dataset
2. ✅ Load original-speech-dataset (Parkinson's)
3. ✅ Load figdata (HC_AH=Healthy, PD_AH=Parkinson's)
4. ✅ Extract OpenSMILE features
5. ✅ Scale features
6. ✅ Train XGBoost classifier
7. ✅ Perform 5-fold cross-validation
8. ✅ Evaluate on test set
9. ✅ Save model, scaler, and results

### 3. Inference

```bash
# Single file
python inference.py \
  --audio /path/to/audio.wav \
  --model models/xgboost_opensmile_<timestamp>.json \
  --scaler models/scaler_<timestamp>.pkl

# Directory of files
python inference.py \
  --audio ../data/figdata/HC_AH/ \
  --model models/xgboost_opensmile_<timestamp>.json \
  --scaler models/scaler_<timestamp>.pkl \
  --output results/predictions.json

# With custom feature set
python inference.py \
  --audio /path/to/audio.wav \
  --model models/xgboost_opensmile_<timestamp>.json \
  --scaler models/scaler_<timestamp>.pkl \
  --feature-set GeMAPSv01b
```

## 📈 Output

### Training Output

After training, you'll find:

**models/**
- `xgboost_opensmile_<timestamp>.json` - Trained XGBoost model
- `scaler_<timestamp>.pkl` - Feature scaler

**results/**
- `metrics_<timestamp>.json` - Detailed performance metrics
- `confusion_matrix_<timestamp>.png` - Confusion matrix visualization
- `roc_curve_<timestamp>.png` - ROC curve
- `feature_importance_<timestamp>.png` - Top 30 important features

### Metrics

The system reports:
- **Accuracy**: Overall classification accuracy
- **Precision**: Positive predictive value
- **Recall**: Sensitivity
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under ROC curve
- **Cross-validation**: Mean ± std across 5 folds

## 🔧 Configuration

Edit `train_opensmile_xgboost.py` to customize:

```python
# Feature set
FEATURE_SET = 'ComParE_2016'  # or 'GeMAPSv01b', 'eGeMAPSv02', 'emobase'

# Train-test split
TEST_SIZE = 0.2  # 20% for testing

# Cross-validation
N_FOLDS = 5

# XGBoost parameters
XGB_PARAMS = {
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 200,
    # ... more parameters
}
```

## 🆚 Comparison with Other Approaches

| Approach | Features | Model | Dataset Used |
|----------|----------|-------|--------------|
| **opxgboost** (this) | OpenSMILE (6373) | XGBoost | Original speech only |
| sf+xgboost | Custom acoustic | XGBoost | Denoised speech |
| cnnbilstm | Mel-spectrograms | CNN+BiLSTM | Denoised speech |

### Advantages

✅ **Industry-standard features**: OpenSMILE is used in research and production  
✅ **Comprehensive feature set**: 6,373 acoustic features  
✅ **Fast training**: Minutes instead of hours  
✅ **Interpretable**: Feature importance analysis  
✅ **Original dataset**: Uses unprocessed audio  
✅ **Robust**: Well-tested feature extraction  

### Use Cases

- Research requiring standard feature sets
- Comparison with published papers
- Feature importance analysis
- Baseline for other models
- Production systems requiring interpretability

## 📊 Feature Categories (ComParE_2016)

OpenSMILE ComParE_2016 extracts:

1. **Energy & Amplitude**: RMS energy, loudness
2. **Spectral**: MFCCs, spectral flux, rolloff, centroid
3. **Voicing**: F0, jitter, shimmer, HNR
4. **Temporal**: Zero-crossing rate, duration
5. **Formants**: F1-F3 frequencies and bandwidths
6. **Cepstral**: MFCC deltas and accelerations
7. **Statistical**: Mean, std, min, max, kurtosis, skewness for all features

## 🐛 Troubleshooting

### OpenSMILE Installation Issues

```bash
# If pip install fails, try:
pip install --upgrade pip
pip install opensmile --no-cache-dir

# Or install from conda
conda install -c conda-forge opensmile
```

### Audio File Not Found

- Ensure `../data/` directory exists
- Check file paths in `.txt` files
- Verify `.wav` files are present

### Feature Extraction Errors

- Check audio file format (must be .wav)
- Ensure audio files are not corrupted
- Verify OpenSMILE installation

## 📚 References

- OpenSMILE: https://github.com/audeering/opensmile
- ComParE Feature Set: Computational Paralinguistics Challenge
- XGBoost: https://xgboost.readthedocs.io/

## 📝 Notes

- Uses **original-speech-dataset** (not denoised)
- Feature extraction may take several minutes
- Model is saved automatically after training
- Results include detailed metrics and visualizations
