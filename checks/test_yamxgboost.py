"""
Test YAMNet + OpenSMILE + XGBoost ensemble model on test files
"""
import sys
from pathlib import Path
sys.path.append('yamxgboost')

from yamxgboost.inference import load_yamnet_model, load_model_and_scaler, predict

# Paths
MODEL_PATH = "yamxgboost/models/ensemble_yamnet_opensmile_20251108_161015.json"
SCALER_PATH = "yamxgboost/models/scaler_20251108_161015.pkl"
TEST_FOLDER = Path("test")

# Get all wav files
audio_files = sorted(list(TEST_FOLDER.glob("*.wav")))

print("="*80)
print(f"🎵 TESTING YAMNET + OPENSMILE + XGBOOST ENSEMBLE ON {len(audio_files)} FILES")
print("="*80)

# Load models
print("\n📦 Loading models...")
print("  Loading YAMNet...")
yamnet_model = load_yamnet_model()
print("  ✅ YAMNet loaded")

print("  Loading XGBoost and scaler...")
model, scaler = load_model_and_scaler(MODEL_PATH, SCALER_PATH)
print("  ✅ XGBoost and scaler loaded\n")

# Test each file
results = []
for audio_file in audio_files:
    print("="*80)
    print(f"📂 File: {audio_file.name}")
    print("="*80)
    
    result = predict(str(audio_file), model, scaler, yamnet_model, feature_set='eGeMAPSv02')
    
    if result['error']:
        print(f"❌ Error: {result['error']}")
    else:
        print(f"✅ Prediction: {result['prediction']}")
        print(f"📊 Confidence: {result['confidence']:.2%}")
        print(f"📈 P(Healthy): {result['probability_healthy']*100:.2f}%")
        print(f"📈 P(Parkinson's): {result['probability_parkinsons']*100:.2f}%")
        
        # Risk level
        pd_prob = result['probability_parkinsons']
        if pd_prob < 0.30:
            risk = "🟢 LOW"
        elif pd_prob < 0.65:
            risk = "🟡 MODERATE"
        else:
            risk = "🔴 HIGH"
        print(f"⚠️  Risk Level: {risk}")
    
    results.append({
        'file': audio_file.name,
        'result': result
    })
    print()

# Summary
print("\n" + "="*80)
print("📊 SUMMARY")
print("="*80)

healthy_count = sum(1 for r in results if not r['result']['error'] and r['result']['prediction'] == 'Healthy')
pd_count = sum(1 for r in results if not r['result']['error'] and r['result']['prediction'] == "Parkinson's Disease")
error_count = sum(1 for r in results if r['result']['error'])

print(f"Total files: {len(results)}")
print(f"Healthy: {healthy_count}")
print(f"Parkinson's: {pd_count}")
print(f"Errors: {error_count}")
print("="*80)
