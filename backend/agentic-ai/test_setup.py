"""
Quick Test Script - Verify Gemini API Integration
Run this to test if your Gemini API key is working
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

import config


def test_gemini_connection():
    """Test if Gemini API key is working"""
    print("="*70)
    print("🧪 TESTING GEMINI API CONNECTION")
    print("="*70)
    
    # Check API key is set
    print("\n[1/3] Checking API key configuration...")
    if config.GEMINI_API_KEY == 'PASTE_YOUR_GEMINI_API_KEY_HERE':
        print("❌ FAILED: API key not set!")
        print("\n📝 TO FIX:")
        print("   1. Go to: https://makersuite.google.com/app/apikey")
        print("   2. Create an API key")
        print("   3. Open: agentic-ai/config.py")
        print("   4. Update line 20 with your key")
        return False
    
    print(f"✅ API key found: {config.GEMINI_API_KEY[:20]}...")
    
    # Try importing Gemini
    print("\n[2/3] Testing Gemini package...")
    try:
        import google.generativeai as genai
        print("✅ google-generativeai package installed")
    except ImportError:
        print("❌ FAILED: google-generativeai not installed")
        print("\n📝 TO FIX:")
        print("   Run: pip install google-generativeai")
        return False
    
    # Try connecting to Gemini
    print("\n[3/3] Testing API connection...")
    try:
        genai.configure(api_key=config.GEMINI_API_KEY)
        model = genai.GenerativeModel(config.GEMINI_MODEL)
        
        # Send test prompt
        response = model.generate_content(
            "Say 'Hello! Gemini is working!' in one sentence."
        )
        
        print("✅ Gemini API working!")
        print(f"\n💬 Test Response:\n   {response.text}")
        
        print("\n" + "="*70)
        print("✅ ALL TESTS PASSED - GEMINI READY!")
        print("="*70)
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        print("\n📝 POSSIBLE ISSUES:")
        print("   - API key is invalid")
        print("   - No internet connection")
        print("   - Gemini API quota exceeded")
        print("   - API key doesn't have correct permissions")
        print("\n💡 TRY:")
        print("   1. Verify key at: https://makersuite.google.com/app/apikey")
        print("   2. Generate a new key if needed")
        print("   3. Check internet connection")
        return False


def test_model_path():
    """Test if model file exists"""
    print("\n" + "="*70)
    print("🔍 CHECKING MODEL FILE")
    print("="*70)
    
    model_path = config.PRIMARY_MODEL_PATH
    print(f"\nLooking for: {model_path}")
    
    if model_path.exists():
        print("✅ Model file found!")
        print(f"   Size: {model_path.stat().st_size / 1024 / 1024:.2f} MB")
        return True
    else:
        print("❌ Model file not found!")
        print("\n📝 EXPECTED LOCATION:")
        print(f"   {model_path}")
        print("\n💡 VERIFY:")
        print("   - Model was trained and saved")
        print("   - File is in correct location")
        print("   - Filename matches config.py")
        return False


def test_directories():
    """Test if required directories exist"""
    print("\n" + "="*70)
    print("📁 CHECKING DIRECTORIES")
    print("="*70)
    
    dirs = {
        'Models': config.MODELS_DIR,
        'Reports': config.REPORTS_DIR,
        'SHAP': config.SHAP_DIR,
    }
    
    all_exist = True
    for name, path in dirs.items():
        if path.exists():
            print(f"✅ {name}: {path}")
        else:
            print(f"⚠️  {name}: {path} (will be created)")
            path.mkdir(parents=True, exist_ok=True)
            all_exist = False
    
    return True


def test_all_packages():
    """Test if all required packages are installed"""
    print("\n" + "="*70)
    print("📦 CHECKING PACKAGES")
    print("="*70)
    
    required = {
        'tensorflow': 'TensorFlow',
        'keras': 'Keras',
        'librosa': 'librosa',
        'numpy': 'NumPy',
        'google.generativeai': 'google-generativeai',
        'reportlab': 'reportlab'
    }
    
    all_installed = True
    for module, name in required.items():
        try:
            __import__(module)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name} - NOT INSTALLED")
            all_installed = False
    
    if not all_installed:
        print("\n📝 TO INSTALL MISSING PACKAGES:")
        print("   pip install -r agentic-ai/requirements.txt")
    
    return all_installed


def main():
    """Run all tests"""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "VOICECARE AI - SYSTEM CHECK" + " "*26 + "║")
    print("╚" + "="*68 + "╝")
    
    results = {
        'Gemini API': test_gemini_connection(),
        'Model File': test_model_path(),
        'Directories': test_directories(),
        'Packages': test_all_packages()
    }
    
    print("\n" + "="*70)
    print("📊 FINAL RESULTS")
    print("="*70)
    
    for test, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test:20s} {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n" + "="*70)
        print("🎉 ALL SYSTEMS GO - READY FOR PRODUCTION! 🎉")
        print("="*70)
        print("\n💡 NEXT STEP:")
        print("   python agentic-ai/production_inference.py <audio_file.wav>")
    else:
        print("\n" + "="*70)
        print("⚠️  SOME TESTS FAILED - FIX ISSUES ABOVE")
        print("="*70)
    
    return all_passed


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
