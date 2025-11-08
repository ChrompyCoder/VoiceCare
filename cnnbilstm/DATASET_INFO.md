# Complete Dataset Information

## Overview

This project uses **ALL available data** from two main sources:

### Total: 443 WAV files → ~1300+ samples after augmentation

---

## Dataset Breakdown

### 1. GITDATA (226 Parkinson's samples)

**Location**: `data/gitdata/`

#### A. Denoised Speech Dataset (113 files)
Path: `data/gitdata/denoised-speech-dataset/Faces/`

| Patient ID | Files | Description |
|------------|-------|-------------|
| BG_au | 38 | Parkinson's patient BG (denoised) |
| JC_au | 17 | Parkinson's patient JC (denoised) |
| MJ_au | 20 | Parkinson's patient MJ (denoised) |
| SK_au | 17 | Parkinson's patient SK (denoised) |
| TP_au | 6  | Parkinson's patient TP (denoised) |
| TS_au | 15 | Parkinson's patient TS (denoised) |
| **Total** | **113** | All Parkinson's patients |

#### B. Original Speech Dataset (113 files)
Path: `data/gitdata/original-speech-dataset/Faces/`

| Patient ID | Files | Description |
|------------|-------|-------------|
| BG_ori | 38 | Parkinson's patient BG (original) |
| JC_ori | 17 | Parkinson's patient JC (original) |
| MJ_ori | 20 | Parkinson's patient MJ (original) |
| SK_ori | 17 | Parkinson's patient SK (original) |
| TP_ori | 6  | Parkinson's patient TP (original) |
| TS_ori | 15 | Parkinson's patient TS (original) |
| **Total** | **113** | All Parkinson's patients |

**Note**: These are the same 6 patients, with both denoised and original versions of their recordings.

---

### 2. FIGDATA (81 samples)

**Location**: `data/figdata/`

| Category | Path | Files | Label | Description |
|----------|------|-------|-------|-------------|
| Healthy Controls | `HC_AH/` | 41 | 0 | Healthy individuals |
| Parkinson's Patients | `PD_AH/` | 40 | 1 | Parkinson's disease patients |
| **Total** | | **81** | | Mixed dataset |

---

## Training Data Summary

### Raw Data
- **Healthy Controls**: 41 samples (from figdata only)
- **Parkinson's Patients**: 266 samples (226 from gitdata + 40 from figdata)
- **Total Raw**: 307 samples
- **Class Imbalance**: ~13% healthy, ~87% Parkinson's

### After Augmentation (3-4x multiplier)
- **Estimated Total**: ~1000-1300 samples
- **Augmentation Techniques**:
  - White noise addition
  - Time stretching (0.9x)
  - Pitch shifting (+2 semitones)

### Dataset Split
- **Training**: 70% (~215 raw samples, ~750 after augmentation)
- **Validation**: 15% (~46 raw samples, ~160 after augmentation)
- **Test**: 15% (~46 raw samples, NO augmentation)

---

## Training Strategy

### Phase 1: Load GITDATA
```
Load denoised-speech-dataset (113 Parkinson's files)
     +
Load original-speech-dataset (113 Parkinson's files)
     ↓
Total: 226 Parkinson's samples
```

### Phase 2: Load FIGDATA
```
Load HC_AH (41 Healthy files)
     +
Load PD_AH (40 Parkinson's files)
     ↓
Total: 81 mixed samples
```

### Phase 3: Combine & Train
```
Combine gitdata + figdata
     ↓
307 total samples (41 healthy + 266 Parkinson's)
     ↓
Apply data augmentation
     ↓
~1000-1300 training samples
     ↓
Split: 70% train / 15% val / 15% test
     ↓
Train CNN+BiLSTM model
```

---

## Key Features of train_combined.py

1. **Uses ALL 443 audio files**
   - Nothing is left out
   - Maximizes available data

2. **Smart Augmentation**
   - Only on training set
   - 3-4x data multiplication
   - Maintains original test set

3. **Overfitting Detection**
   - Monitors train vs validation gap
   - Automatic warnings
   - Suggests solutions

4. **Feature Verification**
   - Validates each dataset
   - Visualizes sample spectrograms
   - Checks for data quality issues

5. **Advanced Regularization**
   - L2 regularization (0.001)
   - Dropout (0.3-0.5)
   - Batch normalization
   - Early stopping

---

## File Naming Conventions

### Gitdata
- **Denoised**: `{PERSON}{NUMBER}.wav` (e.g., `BG1.wav`, `TS15.wav`)
- **Original**: `{PERSON}{NUMBER}.wav` (e.g., `BG1.wav`, `TS15.wav`)
- Same naming, different directories

### Figdata
- **Healthy**: `AH_{ID}_{UUID}.wav`
- **Parkinson's**: `AH_{ID}_{UUID}.wav`
- UUID format: `XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX`

---

## Audio Specifications

All audio files:
- **Format**: WAV
- **Sample Rate**: 22050 Hz (standardized during loading)
- **Duration**: ~5 seconds (padded/truncated as needed)
- **Channels**: Mono (converted if stereo)

---

## Usage

### Train on ALL data (recommended):
```bash
python train_combined.py
```

### Train on figdata only (faster, for testing):
```bash
python train.py
```

### Validate setup:
```bash
python validate_setup.py
```

---

## Expected Performance

With ~1300 training samples after augmentation:

**Target Metrics:**
- Accuracy: 85-95%
- Precision: 85-95%
- Recall: 85-95%
- F1-Score: 85-95%
- AUC: 0.90-0.98

**Note**: Performance depends on:
- Quality of audio recordings
- Effectiveness of feature extraction
- Model architecture tuning
- Regularization settings
- Train/test split randomness

---

## Class Imbalance Handling

The dataset has significant class imbalance (13% healthy vs 87% Parkinson's).

**Strategies implemented:**
1. **Stratified splitting** - Maintains class ratio in train/val/test
2. **Data augmentation** - Increases minority class samples
3. **Class weights** - Can be added to model.fit() if needed
4. **Evaluation metrics** - Focus on precision, recall, F1 (not just accuracy)

**To add class weights** (if needed):
```python
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight('balanced', 
                                      classes=np.unique(y_train), 
                                      y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

# In model.fit():
model.fit(..., class_weight=class_weight_dict)
```

---

## Data Quality Notes

1. **Gitdata**: Professional recordings, consistent quality, denoised versions available
2. **Figdata**: May have varying recording conditions
3. **Recommendation**: Feature verification plots help identify quality issues

---

## Next Steps

1. Run `python train_combined.py`
2. Monitor console for overfitting warnings
3. Check `results/` folder for visualizations
4. Review feature verification plots
5. Adjust regularization if overfitting detected
6. Test final model with `inference.py`

---

**Last Updated**: November 8, 2025
