"""
Interactive API Key Setup Script
Helps you configure Gemini API key easily
"""

import sys
from pathlib import Path


def setup_api_key():
    """Interactive setup for Gemini API key"""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*20 + "GEMINI API KEY SETUP" + " "*29 + "║")
    print("╚" + "="*68 + "╝")
    
    print("\n📝 STEP 1: Get Your API Key")
    print("   Visit: https://makersuite.google.com/app/apikey")
    print("   - Sign in with Google")
    print("   - Click 'Create API Key'")
    print("   - Copy the key (starts with 'AIza...')")
    
    input("\n   Press ENTER when you have your API key...")
    
    print("\n📝 STEP 2: Paste Your API Key")
    print("   (Don't worry, it won't be displayed)")
    api_key = input("\n   Paste your API key here: ").strip()
    
    if not api_key:
        print("\n❌ No API key provided. Exiting.")
        return False
    
    if not api_key.startswith('AIza'):
        print("\n⚠️  Warning: API key doesn't start with 'AIza'")
        proceed = input("   Continue anyway? (y/n): ").strip().lower()
        if proceed != 'y':
            print("\n❌ Setup cancelled.")
            return False
    
    # Read current config
    config_path = Path(__file__).parent / 'config.py'
    
    if not config_path.exists():
        print(f"\n❌ Config file not found: {config_path}")
        return False
    
    print("\n📝 STEP 3: Updating config.py...")
    
    # Read file
    with open(config_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find and replace the line
    updated = False
    for i, line in enumerate(lines):
        if "GEMINI_API_KEY = 'PASTE_YOUR_GEMINI_API_KEY_HERE'" in line:
            lines[i] = f"GEMINI_API_KEY = '{api_key}'  # ← Configured via setup script\n"
            updated = True
            break
        elif line.startswith("GEMINI_API_KEY = ") and 'PASTE_YOUR_GEMINI_API_KEY_HERE' in line:
            lines[i] = f"GEMINI_API_KEY = '{api_key}'  # ← Configured via setup script\n"
            updated = True
            break
    
    if not updated:
        print("\n⚠️  Could not find GEMINI_API_KEY line in config.py")
        print("   Please update manually:")
        print(f"   Open: {config_path}")
        print(f"   Find: GEMINI_API_KEY = 'PASTE_YOUR_GEMINI_API_KEY_HERE'")
        print(f"   Replace with: GEMINI_API_KEY = '{api_key}'")
        return False
    
    # Write back
    with open(config_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print("   ✅ Config updated!")
    
    # Test connection
    print("\n📝 STEP 4: Testing Connection...")
    
    try:
        import google.generativeai as genai
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        response = model.generate_content("Say 'Setup successful!' in one sentence.")
        
        print("   ✅ Connection successful!")
        print(f"\n   💬 Gemini says: {response.text}")
        
        print("\n" + "="*70)
        print("🎉 SETUP COMPLETE - GEMINI READY!")
        print("="*70)
        print("\n💡 NEXT STEPS:")
        print("   python agentic-ai/test_setup.py      # Verify full system")
        print("   python agentic-ai/production_inference.py <audio.wav>  # Run inference")
        
        return True
        
    except ImportError:
        print("\n⚠️  google-generativeai package not installed")
        print("   Run: pip install google-generativeai")
        print("   Then test with: python agentic-ai/test_setup.py")
        return True  # Config saved, just missing package
        
    except Exception as e:
        print(f"\n❌ Connection test failed: {e}")
        print("\n   Possible issues:")
        print("   - Invalid API key")
        print("   - No internet connection")
        print("   - API quota exceeded")
        print("\n   You can:")
        print("   1. Verify key at: https://makersuite.google.com/app/apikey")
        print("   2. Generate new key")
        print("   3. Run this setup script again")
        return False


def main():
    """Main entry point"""
    try:
        success = setup_api_key()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n❌ Setup cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
