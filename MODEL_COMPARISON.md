# Model Comparison: CNN+BiLSTM vs Surfboard+XGBoost

This document compares the two approaches for Parkinson's Disease detection.

## 📊 Overview

| Feature | CNN+BiLSTM | Surfboard+XGBoost |
|---------|------------|-------------------|
| **Location** | `cnnbilstm/` | `sf+xgboost/` |
| **Approach** | Deep Learning | Traditional ML |
| **Features** | Learned (Mel-spectrograms) | Hand-crafted (Surfboard) |
| **Min Duration** | 3 seconds | 4 seconds |
| **Training Time** | Slower (hours) | Faster (minutes) |
| **Interpretability** | Low | High |
| **Memory Usage** | High | Low |

## 🎯 CNN+BiLSTM Approach

### Architecture
- **Convolutional layers** for spatial feature extraction
- **Bidirectional LSTM** for temporal pattern learning
- **Deep neural network** with dropout and batch normalization

### Advantages
- Learns features automatically from raw spectrograms
- Can capture complex non-linear patterns
- State-of-the-art for audio classification tasks
- No feature engineering required

### Disadvantages
- Requires more data for training
- Longer training time
- More prone to overfitting
- Black box (hard to interpret)
- Higher computational requirements

### Best For
- Large datasets (>1000 samples)
- When you have GPU resources
- When interpretability is not critical
- Complex audio patterns

## 🎯 Surfboard+XGBoost Approach

### Architecture
- **Surfboard** extracts 10 types of audio features
- **StandardScaler** normalizes features
- **XGBoost** gradient boosted trees for classification

### Feature Categories
1. **MFCC**: Spectral envelope representation
2. **Spectral**: Frequency domain characteristics
3. **Intensity**: Energy and loudness
4. **Temporal**: Time-domain features
5. **Formants**: Vocal tract resonances
6. **F0**: Pitch information
7. **Harmonicity**: Voice quality
8. **Jitter/Shimmer**: Voice stability

### Advantages
- Fast training (minutes vs hours)
- Works well with smaller datasets
- Highly interpretable (feature importance)
- Less prone to overfitting
- Lower computational requirements
- Traditional ML robustness

### Disadvantages
- Requires feature engineering knowledge
- May miss complex patterns
- Fixed feature set (not learned)
- Longer inference time per sample

### Best For
- Smaller datasets (<500 samples)
- When interpretability is important
- Limited computational resources
- Clinical applications requiring explainability

## 📈 Expected Performance

### CNN+BiLSTM
```
Typical metrics (with sufficient data):
- Accuracy: 85-95%
- Precision: 85-95%
- Recall: 80-90%
- F1-Score: 82-92%
- AUC: 0.90-0.98
```

### Surfboard+XGBoost
```
Typical metrics:
- Accuracy: 80-92%
- Precision: 78-90%
- Recall: 75-88%
- F1-Score: 76-89%
- AUC: 0.85-0.95
```

## 🔧 When to Use Each

### Use CNN+BiLSTM when:
- ✅ You have >500 audio samples
- ✅ GPU resources available
- ✅ Training time is not critical
- ✅ Want state-of-the-art performance
- ✅ Can afford to retrain periodically

### Use Surfboard+XGBoost when:
- ✅ Limited dataset (<500 samples)
- ✅ Need fast training/iteration
- ✅ CPU-only environment
- ✅ Require interpretability
- ✅ Clinical/medical application
- ✅ Need to explain predictions

## 🧪 Ensemble Approach (Recommended)

For best results, you can combine both:

```python
# Pseudo-code for ensemble
prediction_cnn = cnn_bilstm_model.predict(audio)
prediction_xgb = xgboost_model.predict(audio)

# Weighted average
final_prediction = 0.6 * prediction_cnn + 0.4 * prediction_xgb

# Or voting
final_prediction = mode([prediction_cnn, prediction_xgb])
```

### Ensemble Benefits
- Higher accuracy
- More robust predictions
- Combines strengths of both approaches
- Better generalization

## 📝 Training Commands

### CNN+BiLSTM
```bash
cd cnnbilstm
python train_combined.py
```

### Surfboard+XGBoost
```bash
cd sf+xgboost
python train_xgboost.py
```

## 🔍 Inference Commands

### CNN+BiLSTM
```bash
cd cnnbilstm
python inference.py --audio path/to/audio.wav --model models/best_model.h5
```

### Surfboard+XGBoost
```bash
cd sf+xgboost
python inference.py --audio path/to/audio.wav \
    --model models/xgboost_model_<timestamp>.json \
    --scaler models/scaler_<timestamp>.pkl
```

## 📊 Feature Comparison

### CNN+BiLSTM Features
- Automatically learned from Mel-spectrograms
- 128 Mel bands × time steps
- Captures temporal and spectral patterns
- ~40,000+ implicit features

### Surfboard Features
- Explicitly extracted acoustic features
- ~200-500 handcrafted features
- Statistics: mean, std, min, max, median, skew, kurtosis
- Based on audio processing research

## 🎓 Clinical Relevance

### CNN+BiLSTM
- Good for screening/detection
- Requires validation before clinical use
- Hard to explain to clinicians

### Surfboard+XGBoost
- Features align with clinical knowledge:
  - Jitter/Shimmer: Known PD markers
  - Pitch variability: Voice dysfunction
  - Formants: Articulatory changes
- Easier to validate with domain experts
- Feature importance guides clinical interpretation

## 🚀 Recommendation

**Start with Surfboard+XGBoost** for:
- Initial exploration
- Fast prototyping
- Understanding which features matter

**Scale to CNN+BiLSTM** when:
- You have more data
- Initial results are promising
- Need maximum accuracy
- Have computational resources

**Use both** for:
- Production systems
- Critical applications
- Maximum confidence in predictions
