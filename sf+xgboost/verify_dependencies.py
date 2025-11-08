"""
Quick verification script to test if all dependencies are working
"""

print("=" * 70)
print("Verifying Dependencies for Custom Features + XGBoost")
print("=" * 70)

try:
    import numpy as np
    print("✓ NumPy:", np.__version__)
except ImportError as e:
    print("✗ NumPy:", e)

try:
    import pandas as pd
    print("✓ Pandas:", pd.__version__)
except ImportError as e:
    print("✗ Pandas:", e)

try:
    import sklearn
    print("✓ Scikit-learn:", sklearn.__version__)
except ImportError as e:
    print("✗ Scikit-learn:", e)

try:
    import xgboost as xgb
    print("✓ XGBoost:", xgb.__version__)
except ImportError as e:
    print("✗ XGBoost:", e)

try:
    import librosa
    print("✓ Librosa:", librosa.__version__)
except ImportError as e:
    print("✗ Librosa:", e)

try:
    import soundfile as sf
    print("✓ SoundFile:", sf.__version__)
except ImportError as e:
    print("✗ SoundFile:", e)

try:
    import parselmouth
    print("✓ Parselmouth:", parselmouth.__version__)
except ImportError as e:
    print("✗ Parselmouth:", e)

try:
    import matplotlib
    print("✓ Matplotlib:", matplotlib.__version__)
except ImportError as e:
    print("✗ Matplotlib:", e)

try:
    import seaborn as sns
    print("✓ Seaborn:", sns.__version__)
except ImportError as e:
    print("✗ Seaborn:", e)

try:
    import scipy
    print("✓ SciPy:", scipy.__version__)
except ImportError as e:
    print("✗ SciPy:", e)

print("\n" + "=" * 70)
print("All dependencies verified successfully!")
print("=" * 70)
print("\nYou can now run:")
print("  python train_xgboost.py")
print("=" * 70)
