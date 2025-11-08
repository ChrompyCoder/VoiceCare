# Combined Dataset Training Guide

## Overview

This enhanced training script (`train_combined.py`) uses **ALL data from BOTH gitdata and figdata** with advanced features:

1. **Three-Phase Training**: Load gitdata → Load figdata → Combine ALL and train together
2. **Overfitting Detection**: Automatic monitoring and alerts
3. **Feature Verification**: Validates extracted features for quality
4. **Advanced Regularization**: L2 regularization, dropout, batch normalization
5. **Data Augmentation**: White noise, time stretching, pitch shifting

## Dataset Sources

### GITDATA (`data/gitdata/`)
- **Denoised Speech Dataset**: 113 Parkinson's samples
  - `denoised-speech-dataset/Faces/` - 6 patient folders (BG_au, JC_au, MJ_au, SK_au, TP_au, TS_au)
- **Original Speech Dataset**: 113 Parkinson's samples
  - `original-speech-dataset/Faces/` - 6 patient folders (BG_ori, JC_ori, MJ_ori, SK_ori, TP_ori, TS_ori)
- **Total**: 226 Parkinson's patient samples
- **Use**: Training data (combined with figdata)

### FIGDATA (`data/figdata/`)
- **Content**: Both healthy controls and Parkinson's patients
- **Structure**:
  - `HC_AH/` - 41 healthy control samples
  - `PD_AH/` - 40 Parkinson's patient samples
- **Total**: 81 samples (41 healthy + 40 Parkinson's)
- **Use**: Training data (combined with gitdata)

### COMBINED DATASET
- **Total Raw Samples**: 307 (41 healthy + 266 Parkinson's)
- **After Augmentation**: ~921 samples
- **Class Distribution**: 13.4% healthy, 86.6% Parkinson's

## Training Strategy

### Phase 1: Load GITDATA
```
Load denoised-speech-dataset (113 Parkinson's files)
     +
Load original-speech-dataset (113 Parkinson's files)
     ↓
Extract features with augmentation
     ↓
Verify feature quality
     ↓
Total: 226 Parkinson's samples prepared
```

### Phase 2: Load FIGDATA
```
Load HC_AH (41 Healthy files)
     +
Load PD_AH (40 Parkinson's files)
     ↓
Extract features with augmentation
     ↓
Verify feature quality
     ↓
Total: 81 mixed samples prepared
```

### Phase 3: Combine & Train on ALL Data
```
Combine gitdata + figdata
     ↓
307 total samples (41 healthy + 266 Parkinson's)
     ↓
Apply data augmentation (~3x multiplication)
     ↓
~921 training samples
     ↓
Split: 70% train / 15% validation / 15% test
     ↓
Train CNN+BiLSTM model on COMBINED dataset
     ↓
Monitor for overfitting
     ↓
Evaluate on test set
```

## Key Features

### 1. **Data Augmentation**
Automatically applies:
- White noise addition (0.5% noise)
- Time stretching (0.9x speed)
- Pitch shifting (+2 semitones)

This increases training samples ~3-4x for better generalization.

### 2. **Overfitting Detection**
Monitors:
- Training vs Validation accuracy gap
- Training vs Validation loss gap

**Alerts if:**
- Accuracy gap > 15%
- Loss gap > 0.3

### 3. **Feature Verification**
Checks for each dataset:
- Feature shape and dimensions
- Value ranges and statistics
- NaN/Inf detection
- Label distribution
- Sample visualizations

### 4. **Enhanced Regularization**

**L2 Regularization:**
- Applied to Conv2D, LSTM, and Dense layers
- Weight: 0.001

**Dropout:**
- After each CNN block: 0.3-0.4
- After LSTM: 0.5
- After Dense layer: 0.3

**Batch Normalization:**
- After each pooling layer
- Stabilizes training

## Configuration

Edit the script to adjust parameters:

```python
# Training configuration
USE_REGULARIZATION = True    # Enable/disable regularization
L2_REG = 0.001              # L2 regularization strength
DROPOUT_RATE = 0.5          # Dropout after LSTM
DATA_AUGMENTATION = True     # Enable/disable augmentation
```

## Usage

### Run Training

```bash
python train_combined.py
```

### Expected Output

1. **Console Output:**
   - PHASE 1: Gitdata loading progress (226 Parkinson's samples)
   - PHASE 2: Figdata loading progress (81 mixed samples)
   - PHASE 3: Dataset combination summary (307 total samples)
   - Feature extraction progress for each dataset
   - Feature verification statistics with visualizations
   - Training progress (epoch by epoch) on COMBINED dataset
   - Overfitting warnings (if detected)
   - Final test metrics

2. **Saved Files:**
   - `models/best_model_combined_<timestamp>.h5` - Best model trained on ALL data
   - `results/feature_verification_GITDATA_<timestamp>.png` - Gitdata features
   - `results/feature_verification_FIGDATA_<timestamp>.png` - Figdata features
   - `results/training_history_combined_<timestamp>.png` - 4-panel training plot
   - `results/confusion_matrix_combined_<timestamp>.png` - Confusion matrix
   - `results/roc_curve_combined_<timestamp>.png` - ROC curve
   - `results/metrics_combined_<timestamp>.json` - All metrics in JSON

## Training Metrics

The model tracks:
- **Accuracy**: Overall correctness
- **Precision**: How many predicted Parkinson's are actually Parkinson's
- **Recall**: How many actual Parkinson's cases were caught
- **F1-Score**: Balance between Precision and Recall
- **AUC**: Area Under ROC Curve (overall classifier quality)

## Training Time

Approximate training time:
- **Feature Extraction**: 3-5 minutes (depends on augmentation)
- **Model Training**: 15-45 minutes (depends on epochs and hardware)
- **Total**: ~20-50 minutes

## Interpreting Results

### Good Training Signs ✓
- Val accuracy within 10% of train accuracy
- Smooth training curves
- AUC > 0.85
- F1-score > 0.80

### Overfitting Signs ⚠️
- Train accuracy >> Val accuracy (gap > 15%)
- Val loss increasing while train loss decreasing
- Perfect train accuracy but poor val accuracy

### Solutions to Overfitting
If overfitting is detected:

1. **Increase Regularization**:
   ```python
   L2_REG = 0.005  # Increase from 0.001
   DROPOUT_RATE = 0.6  # Increase from 0.5
   ```

2. **More Data Augmentation**:
   ```python
   DATA_AUGMENTATION = True
   # Add more augmentation techniques in augment_audio()
   ```

3. **Reduce Model Complexity**:
   - Reduce CNN filters: [16, 32, 64] instead of [32, 64, 128]
   - Reduce LSTM units: 64 instead of 128

4. **Early Stopping**:
   - Already enabled with patience=20

## Advanced: Custom Augmentation

To add more augmentation techniques, edit the `augment_audio()` function:

```python
def augment_audio(y, sr):
    augmentations = [y]  # Original
    
    # Add your custom augmentations:
    # 1. Different noise levels
    # 2. Different time stretch rates
    # 3. Volume changes
    # 4. Frequency masking
    
    return augmentations
```

## Troubleshooting

**Issue**: High training time
- **Solution**: Disable augmentation or reduce epochs

**Issue**: Out of memory
- **Solution**: Reduce batch size to 4 or disable augmentation

**Issue**: Poor performance on both train and test
- **Solution**: Check feature verification plots, may need feature engineering

**Issue**: Overfitting detected
- **Solution**: Follow "Solutions to Overfitting" above

## Comparison: train.py vs train_combined.py

| Feature | train.py | train_combined.py |
|---------|----------|-------------------|
| Datasets | figdata only (81 samples) | **gitdata + figdata (307 samples)** |
| Training Samples | ~81 raw | **~307 raw, ~921 after augmentation** |
| Augmentation | No | Yes (3-4x data) |
| Overfitting Detection | No | Yes, automatic |
| Feature Verification | No | Yes, with plots |
| Regularization | Basic | Advanced (L2 + dropout) |
| Metrics Tracking | Basic | Comprehensive + JSON |
| Visualizations | 2 plots | 5+ plots |
| Training Data | Figdata only | **Gitdata (226) + Figdata (81) COMBINED** |

## Key Difference: Combined Training

**train_combined.py** trains on ALL 307 samples together:
- Uses all 226 Parkinson's samples from gitdata
- Uses all 41 healthy + 40 Parkinson's samples from figdata
- Combines them into one large training set
- Better generalization from more diverse data
- More robust model performance

## Next Steps

1. Run `python train_combined.py` to train on ALL data
2. Monitor console for:
   - Phase 1: Gitdata loading (226 samples)
   - Phase 2: Figdata loading (81 samples)
   - Phase 3: Combined dataset (307 samples)
   - Overfitting warnings during training
3. Check `results/` folder for visualizations
4. Review `metrics_combined_<timestamp>.json` for detailed stats
5. If overfitting detected, adjust regularization and retrain
6. Use best model for inference with `inference.py`

## Example Training Session

```bash
$ python train_combined.py

============================================================
Voice-Based Parkinson's Disease Detection
CNN + BiLSTM Model Training (Combined Dataset)
============================================================

============================================================
PHASE 1: Loading GITDATA (Parkinson's patients only)
============================================================
Loading denoised-speech-dataset from data/gitdata/denoised-speech-dataset/Faces
  BG_au: 38 files
  JC_au: 17 files
  MJ_au: 20 files
  SK_au: 17 files
  TP_au: 6 files
  TS_au: 15 files
  Total denoised: 113 samples

Loading original-speech-dataset from data/gitdata/original-speech-dataset/Faces
  BG_ori: 38 files
  JC_ori: 17 files
  MJ_ori: 20 files
  SK_ori: 17 files
  TP_ori: 6 files
  TS_ori: 15 files
  Total original: 113 samples

  TOTAL GITDATA: 226 Parkinson's samples

Extracting features from gitdata...
[Feature extraction progress...]

============================================================
PHASE 2: Loading FIGDATA (Healthy + Parkinson's)
============================================================
Loading healthy controls from data/figdata/HC_AH
  Loaded 41 healthy control files
Loading Parkinson's patients from data/figdata/PD_AH
  Loaded 40 Parkinson's patient files

Extracting features from figdata...
[Feature extraction progress...]

============================================================
PHASE 3: Combining datasets for training
============================================================
Combining 226 gitdata samples with 81 figdata samples
Total combined samples: 307

Combined dataset distribution:
  Healthy: 41 samples (13.4%)
  Parkinson's: 266 samples (86.6%)

Final dataset split:
Training set: 260 samples
Test set: 47 samples
Validation: ~47 samples (from training set)

[Training begins on COMBINED dataset...]
Epoch 1/150
[...]
```

## Performance Tips

1. **Use GPU if available** - 10-20x faster training
2. **Start with small epochs** (e.g., 50) to test quickly
3. **Monitor first 10 epochs** - If val_loss doesn't decrease, adjust learning rate
4. **Save intermediate models** - Already done via ModelCheckpoint
5. **Use cross-validation** - For more robust evaluation (future enhancement)

---

## Important: Combined Training Approach

**This script trains on ALL 307 samples from BOTH datasets together:**

- **Gitdata (226 Parkinson's)**: Provides diverse Parkinson's voice patterns from 6 different patients
- **Figdata (81 mixed)**: Provides both healthy and Parkinson's samples for balanced learning
- **Combined (307 total)**: The model learns from ALL available data simultaneously
  - More training samples = Better generalization
  - Diverse data sources = More robust model
  - Both classes represented = Proper binary classification

The model sees all data during training, not sequentially but combined, which allows it to:
1. Learn Parkinson's patterns from many sources (gitdata + figdata Parkinson's)
2. Learn healthy patterns (figdata healthy controls)
3. Distinguish between both classes effectively

---