"""
Validation script to check if the project is set up correctly
Run this before training to verify everything works
"""

import os
import sys

def check_directories():
    """Check if required directories exist"""
    print("Checking directories...")
    
    dirs_to_check = [
        "data/figdata/HC_AH",
        "data/figdata/PD_AH",
        "utils"
    ]
    
    all_good = True
    for dir_path in dirs_to_check:
        if os.path.exists(dir_path):
            print(f"  ✓ {dir_path} exists")
        else:
            print(f"  ✗ {dir_path} NOT FOUND")
            all_good = False
    
    return all_good

def check_audio_files():
    """Check if audio files exist in the dataset"""
    print("\nChecking audio files...")
    
    hc_path = "data/figdata/HC_AH"
    pd_path = "data/figdata/PD_AH"
    
    hc_files = [f for f in os.listdir(hc_path) if f.endswith('.wav')] if os.path.exists(hc_path) else []
    pd_files = [f for f in os.listdir(pd_path) if f.endswith('.wav')] if os.path.exists(pd_path) else []
    
    print(f"  Healthy Controls: {len(hc_files)} files")
    print(f"  Parkinson's Patients: {len(pd_files)} files")
    print(f"  Total: {len(hc_files) + len(pd_files)} files")
    
    if len(hc_files) > 0 and len(pd_files) > 0:
        print("  ✓ Audio files found")
        return True
    else:
        print("  ✗ No audio files found")
        return False

def check_dependencies():
    """Check if required Python packages are installed"""
    print("\nChecking Python dependencies...")
    
    required_packages = [
        'tensorflow',
        'librosa',
        'numpy',
        'sklearn',
        'matplotlib',
        'seaborn',
        'pandas'
    ]
    
    all_good = True
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package} installed")
        except ImportError:
            print(f"  ✗ {package} NOT INSTALLED")
            all_good = False
    
    return all_good

def check_sample_audio():
    """Try to load a sample audio file"""
    print("\nChecking audio file processing...")
    
    try:
        import librosa
        
        # Try to find and load one audio file
        hc_path = "data/figdata/HC_AH"
        if os.path.exists(hc_path):
            wav_files = [f for f in os.listdir(hc_path) if f.endswith('.wav')]
            if wav_files:
                sample_file = os.path.join(hc_path, wav_files[0])
                y, sr = librosa.load(sample_file, sr=22050, duration=1)
                print(f"  ✓ Successfully loaded sample file: {wav_files[0]}")
                print(f"    Sample rate: {sr} Hz")
                print(f"    Duration: {len(y)/sr:.2f} seconds")
                return True
    except Exception as e:
        print(f"  ✗ Error loading audio: {e}")
        return False
    
    return False

def main():
    print("=" * 60)
    print("Parkinson's Disease Detection - Environment Validation")
    print("=" * 60)
    print()
    
    checks = []
    
    # Run all checks
    checks.append(("Directories", check_directories()))
    checks.append(("Audio Files", check_audio_files()))
    checks.append(("Dependencies", check_dependencies()))
    checks.append(("Audio Processing", check_sample_audio()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Validation Summary")
    print("=" * 60)
    
    all_passed = True
    for check_name, result in checks:
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"  {symbol} {check_name}: {status}")
        if not result:
            all_passed = False
    
    print()
    if all_passed:
        print("✓ All checks passed! You're ready to train the model.")
        print("  Run: python train.py")
    else:
        print("✗ Some checks failed. Please fix the issues above before training.")
        print("  - Install missing dependencies: pip install -r requirements.txt")
        print("  - Verify dataset is in the correct location")
    
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
