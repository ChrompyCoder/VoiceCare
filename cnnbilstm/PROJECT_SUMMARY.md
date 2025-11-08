# Project Summary: Voice-Based Parkinson's Disease Detection

## Overview
This project implements a deep learning solution for detecting Parkinson's disease from voice recordings using a hybrid CNN+BiLSTM architecture.

## Dataset
- **Total Samples**: 81 audio files (5-second recordings at 22050 Hz)
  - Healthy Controls (HC_AH): 41 samples
  - Parkinson's Patients (PD_AH): 40 samples
- **Format**: WAV audio files
- **Location**: `data/figdata/`

## Data Split Strategy
- **Training Set**: 70% (~57 samples)
- **Validation Set**: 15% (~12 samples) 
- **Test Set**: 15% (~12 samples)

The split is stratified to maintain class balance across all sets.

## Feature Extraction
- **Method**: Mel-spectrogram analysis
- **Parameters**:
  - Sample Rate: 22050 Hz
  - Duration: 5 seconds
  - Mel Bands (n_mels): 128
  - FFT Window (n_fft): 2048
  - Hop Length: 512 samples

## Model Architecture

### Input
- Mel-spectrogram: (128 x time_steps x 1)

### Layers
1. **CNN Block 1**
   - Conv2D: 32 filters, 3x3 kernel, ReLU
   - MaxPooling2D: 2x2
   - BatchNormalization

2. **CNN Block 2**
   - Conv2D: 64 filters, 3x3 kernel, ReLU
   - MaxPooling2D: 2x2
   - BatchNormalization

3. **CNN Block 3**
   - Conv2D: 128 filters, 3x3 kernel, ReLU
   - MaxPooling2D: 2x2
   - BatchNormalization

4. **Sequence Processing**
   - Reshape: Convert to sequence
   - Bidirectional LSTM: 128 units
   - Dropout: 0.4

5. **Classification Head**
   - Dense: 64 units, ReLU
   - Dense: 1 unit, Sigmoid (binary output)

### Total Parameters
The model will be automatically summarized during training.

## Training Configuration

### Optimizer
- **Type**: Adam
- **Learning Rate**: 1e-4
- **Loss Function**: Binary Cross-entropy

### Callbacks
1. **ModelCheckpoint**: Saves best model based on validation accuracy
2. **EarlyStopping**: Stops training if validation loss doesn't improve for 15 epochs
3. **ReduceLROnPlateau**: Reduces learning rate by 0.5 if validation loss plateaus for 7 epochs

### Batch Size
- 16 samples per batch

### Maximum Epochs
- 100 (will stop early if performance plateaus)

## Evaluation Metrics
- **Accuracy**: Overall correctness
- **Precision**: True Positives / (True Positives + False Positives)
- **Recall**: True Positives / (True Positives + False Negatives)
- **F1-Score**: Harmonic mean of Precision and Recall
- **Confusion Matrix**: Detailed breakdown of predictions

## File Structure
```
HK-11/
├── train.py              # Main training script
├── inference.py          # Prediction script for new audio files
├── config.py            # Configuration parameters
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
├── run_training.sh      # Bash script to run training
├── utils/
│   ├── __init__.py
│   └── data_loader.py   # Data loading and feature extraction utilities
├── data/
│   └── figdata/
│       ├── HC_AH/       # Healthy control audio files
│       └── PD_AH/       # Parkinson's patient audio files
├── models/              # Saved trained models (created during training)
└── results/             # Training plots and metrics (created during training)
```

## Usage Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python train.py
# OR
./run_training.sh
```

### 3. Make Predictions
```bash
python inference.py models/best_model_<timestamp>.h5 path/to/audio.wav
```

## Expected Output

### During Training
- Progress updates for feature extraction
- Training progress with epoch-by-epoch metrics
- Validation accuracy and loss
- Learning rate adjustments
- Best model checkpoints

### After Training
1. **Saved Model**: `models/best_model_<timestamp>.h5`
2. **Training History Plot**: `results/training_history_<timestamp>.png`
3. **Confusion Matrix**: `results/confusion_matrix_<timestamp>.png`
4. **Console Output**:
   - Classification report (precision, recall, F1-score)
   - Test set performance metrics
   - Model summary

## Performance Expectations

Given the small dataset size (81 samples), the model will:
- Likely achieve good training accuracy
- May show some variance in test performance due to limited data
- Benefit from the train/validation/test split to prevent overfitting
- Use early stopping and dropout for regularization

### Tips for Better Performance
1. **Data Augmentation**: Consider adding noise, time-stretching, or pitch-shifting
2. **Feature Engineering**: Experiment with different audio features (MFCCs, chroma, etc.)
3. **Ensemble Methods**: Train multiple models and average predictions
4. **Cross-Validation**: Use k-fold cross-validation for more robust evaluation
5. **Transfer Learning**: Use pre-trained audio models if available

## Technical Details

### Why CNN + BiLSTM?
- **CNN**: Extracts local spectral patterns in the Mel-spectrogram (spatial features)
- **BiLSTM**: Captures temporal dependencies and context in both directions
- **Combined**: Leverages both spatial and temporal patterns in voice data

### Why Mel-spectrogram?
- Represents audio in a way that mimics human auditory perception
- Captures frequency content over time
- Commonly used in audio classification tasks
- Suitable for CNN processing

## Troubleshooting

### Issue: Out of Memory
- Reduce batch size in `config.py`
- Reduce number of training samples

### Issue: Model Not Training
- Check that audio files are in correct directories
- Verify audio files are valid WAV format
- Check console output for error messages

### Issue: Poor Performance
- Increase training epochs (if not hitting early stopping)
- Adjust learning rate
- Try different model architectures
- Add data augmentation

## Future Improvements
1. Collect more data for better generalization
2. Implement data augmentation techniques
3. Try different model architectures (ResNet, VGG, etc.)
4. Experiment with ensemble methods
5. Add explainability features (Grad-CAM visualization)
6. Deploy as web application or mobile app

## References
- Librosa Documentation: https://librosa.org/
- TensorFlow/Keras Documentation: https://www.tensorflow.org/
- Mel-spectrogram Theory: https://en.wikipedia.org/wiki/Mel_scale

## Author
Tushar (Hackathon Build)

## License
This is a hackathon project for educational and research purposes.
