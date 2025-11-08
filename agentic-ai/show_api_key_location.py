"""
Visual guide showing exactly where to put the Gemini API key
Run this script to see highlighted instructions
"""

import os
from pathlib import Path


def show_api_key_location():
    """Display visual guide for API key placement"""
    
    config_path = Path(__file__).parent / 'config.py'
    
    print("\n" + "="*80)
    print(" "*25 + "🔑 WHERE TO PUT YOUR API KEY")
    print("="*80 + "\n")
    
    print("📍 FILE LOCATION:")
    print(f"   {config_path.absolute()}\n")
    
    print("📝 WHAT TO DO:\n")
    
    # Read the file
    if not config_path.exists():
        print("❌ ERROR: config.py not found!")
        return
    
    with open(config_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find the API key line
    target_line = None
    for i, line in enumerate(lines, 1):
        if 'GEMINI_API_KEY' in line and 'PASTE_YOUR_GEMINI_API_KEY_HERE' in line:
            target_line = i
            break
    
    if target_line:
        # Show context around the line
        start = max(0, target_line - 5)
        end = min(len(lines), target_line + 5)
        
        print("📄 FILE PREVIEW (with line numbers):\n")
        print("   " + "-"*70)
        
        for i in range(start, end):
            line_num = i + 1
            line_content = lines[i].rstrip()
            
            if line_num == target_line:
                # Highlight the target line
                print(f"   {line_num:3d} │ " + "🔴 " + line_content + " ← CHANGE THIS LINE")
            else:
                print(f"   {line_num:3d} │ {line_content}")
        
        print("   " + "-"*70)
        
        print("\n✏️  CHANGE THIS:")
        print("   " + "="*70)
        current_line = lines[target_line - 1].strip()
        print(f"   {current_line}")
        print("   " + "="*70)
        
        print("\n✅ TO THIS (with your actual key):")
        print("   " + "="*70)
        print("   GEMINI_API_KEY = 'AIzaSyC..._your_actual_api_key_here'")
        print("   " + "="*70)
        
        print("\n🎯 STEPS:")
        print("   1. Get API key from: https://makersuite.google.com/app/apikey")
        print("   2. Open config.py in any text editor")
        print(f"   3. Go to line {target_line}")
        print("   4. Replace 'PASTE_YOUR_GEMINI_API_KEY_HERE' with your actual key")
        print("   5. Save the file")
        print("   6. Run: python agentic-ai/test_setup.py")
        
    else:
        print("⚠️  Could not find GEMINI_API_KEY line with placeholder.")
        print("   It may already be configured!")
        print("\n   Current GEMINI_API_KEY line:")
        for i, line in enumerate(lines, 1):
            if 'GEMINI_API_KEY' in line and '=' in line:
                print(f"   Line {i}: {line.strip()}")
                break
    
    print("\n" + "="*80)
    print(" "*20 + "💡 TIP: Use the interactive script!")
    print(" "*15 + "python agentic-ai/setup_api_key.py")
    print("="*80 + "\n")


if __name__ == '__main__':
    show_api_key_location()
