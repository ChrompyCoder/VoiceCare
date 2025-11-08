"""
Configuration file for Parkinson's Disease Detection Model
Modify these parameters to customize training
"""

# Data Configuration
DATA_PATH = "data/figdata"
SAMPLE_RATE = 22050
DURATION = 5  # seconds

# Feature Extraction
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

# Model Architecture
CNN_FILTERS = [32, 64, 128]  # Number of filters in each CNN layer
LSTM_UNITS = 128
DROPOUT_RATE = 0.4
DENSE_UNITS = 64

# Training Configuration
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.18  # ~15% of total data

# Data Split
TEST_SIZE = 0.15  # 15% for testing, rest for training+validation
RANDOM_STATE = 42

# Callbacks
EARLY_STOPPING_PATIENCE = 15
LR_REDUCE_PATIENCE = 7
LR_REDUCE_FACTOR = 0.5
MIN_LR = 1e-7

# Paths
MODEL_DIR = "models"
RESULTS_DIR = "results"

# Metrics
METRICS = ['accuracy', 'precision', 'recall']
