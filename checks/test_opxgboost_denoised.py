"""
Test OpenSMILE + XGBoost model with denoising on all test files
"""
import sys
from pathlib import Path
sys.path.append('opxgboost')

from opxgboost.inference import load_model_and_scaler, extract_opensmile_features

# Paths
MODEL_PATH = "opxgboost\\modelsoldcorrectONE\\xgboost_opensmile_20251108_154914.json"
SCALER_PATH = "opxgboost\\modelsoldcorrectONE\\scaler_20251108_154914.pkl"
TEST_FOLDER = Path("test")

# Get all wav files
audio_files = sorted(list(TEST_FOLDER.glob("*.wav")))

print("="*80)
print(f"🎵 TESTING OPENSMILE + XGBOOST WITH DENOISING ON {len(audio_files)} FILES")
print("="*80)

# Load model
print("\n📦 Loading model and scaler...")
model, scaler = load_model_and_scaler(MODEL_PATH, SCALER_PATH)
print("✅ Model loaded successfully\n")

# Test each file
results = []
for audio_file in audio_files:
    print("="*80)
    print(f"📂 File: {audio_file.name}")
    print("="*80)
    
    # Extract features with denoising
    features = extract_opensmile_features(str(audio_file), feature_set='ComParE_2016', denoise=True)
    if features is None:
        print(f"❌ Error extracting features")
        continue
    
    # Scale and predict
    features_scaled = scaler.transform(features.reshape(1, -1))
    probability = model.predict_proba(features_scaled)[0]
    prediction = model.predict(features_scaled)[0]
    
    label = "Parkinson's Disease" if prediction == 1 else 'Healthy'
    confidence = probability[1] if prediction == 1 else probability[0]
    
    print(f"✅ Prediction: {label}")
    print(f"📊 Confidence: {confidence:.2%}")
    print(f"📈 P(Healthy): {probability[0]*100:.2f}%")
    print(f"📈 P(Parkinson's): {probability[1]*100:.2f}%")
    
    # Risk level
    pd_prob = probability[1]
    if pd_prob < 0.30:
        risk = "🟢 LOW"
    elif pd_prob < 0.65:
        risk = "🟡 MODERATE"
    else:
        risk = "🔴 HIGH"
    print(f"⚠️  Risk Level: {risk}")
    
    results.append({
        'file': audio_file.name,
        'prediction': label,
        'pd_probability': pd_prob
    })
    print()

# Summary
print("\n" + "="*80)
print("📊 SUMMARY")
print("="*80)

healthy_count = sum(1 for r in results if r['prediction'] == 'Healthy')
pd_count = sum(1 for r in results if r['prediction'] == "Parkinson's Disease")

print(f"Total files: {len(results)}")
print(f"Healthy: {healthy_count}")
print(f"Parkinson's: {pd_count}")
print("="*80)
