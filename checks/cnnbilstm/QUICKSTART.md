# Quick Start Guide

## Train on ALL Data (Recommended)

```bash
# Verify your dataset first
python verify_dataset.py

# Train on combined dataset (gitdata + figdata = 307 samples)
python train_combined.py
```

## What Happens During Training

### ✅ Data Loading
- **Phase 1**: Loads 226 Parkinson's samples from gitdata (denoised + original)
- **Phase 2**: Loads 81 samples from figdata (41 healthy + 40 Parkinson's)
- **Phase 3**: Combines ALL 307 samples together

### ✅ Feature Processing
- Extracts Mel-spectrograms from each audio file
- Applies data augmentation (~3x multiplication)
- Verifies feature quality with visualizations
- Final training set: ~921 samples

### ✅ Model Training
- Trains CNN+BiLSTM on the COMBINED dataset
- Monitors for overfitting (automatic warnings)
- Saves best model based on validation accuracy
- Tracks: accuracy, precision, recall, F1-score, AUC

### ✅ Results
All outputs saved to:
- `models/best_model_combined_<timestamp>.h5` - Trained model
- `results/` - Plots, confusion matrix, ROC curve, metrics

## Expected Training Time

- Feature extraction: 5-10 minutes
- Model training: 20-60 minutes (depends on hardware)
- Total: ~30-70 minutes

## After Training

Test your model:
```bash
python inference.py models/best_model_combined_<timestamp>.h5 path/to/audio.wav
```

## Key Points

1. ✅ **Uses ALL 307 audio files** from both datasets
2. ✅ **Combines** gitdata and figdata for training
3. ✅ **Augments** data for better generalization
4. ✅ **Detects** overfitting automatically
5. ✅ **Saves** comprehensive metrics and visualizations

---

For detailed information, see:
- `TRAINING_GUIDE.md` - Complete training documentation
- `DATASET_INFO.md` - Dataset details
- `README.md` - Project overview
