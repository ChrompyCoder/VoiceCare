# Parkinson's Detection: OpenSMILE + XGBoost

This directory contains a Parkinson's Disease detection system using **OpenSMILE** for feature extraction and **XGBoost** for classification.

## Approach

This implementation uses:
- **OpenSMILE**: Industry-standard audio feature extraction toolkit
- **XGBoost**: Gradient boosted decision trees for classification

## 🎵 OpenSMILE Feature Sets

The script supports multiple feature sets:

| Feature Set | Features | Description |
|-------------|----------|-------------|
| **ComParE_2016** | 6,373 | Computational Paralinguistics Challenge (default) |
| **GeMAPSv01b** | 62 | Geneva Minimalistic Acoustic Parameter Set |
| **eGeMAPSv02** | 88 | Extended GeMAPS |
| **emobase** | 988 | Emotion recognition base set |

Usage: **ComParE_2016** (most comprehensive)

## Installation

### 1. Training 
Install the dependencies(check out the projects main README.md)
Linux: `python3 ./train.py`
### 2. Inference
Add your test files to the test directory of backend.
Run tests using:
Linux: `python3 ./test.py`

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


### Advantages

**Industry-standard features**: OpenSMILE is used in research and production  
**Comprehensive feature set**: 6,373 acoustic features  
**Fast training**: Minutes instead of hours  
**Interpretable**: Feature importance analysis  
**Original dataset**: Uses unprocessed audio  
**Robust**: Well-tested feature extraction  

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

## References

- OpenSMILE: https://github.com/audeering/opensmile
- ComParE Feature Set: Computational Paralinguistics Challenge
- XGBoost: https://xgboost.readthedocs.io/
