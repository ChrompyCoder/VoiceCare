# Quick Start Guide - Testing the Enhanced System

## Prerequisites
- Python virtual environment activated
- All dependencies installed (librosa, opensmile, xgboost, etc.)
- Node.js and npm installed for frontend
- Gemini API key configured in `backend/agentic_ai/config.py`

---

## Step 1: Start Backend Server

```powershell
# Navigate to backend directory
cd backend

# Activate virtual environment (if not already active)
..\venv\Scripts\Activate.ps1

# Start Flask server
python app.py
```

**Expected Output:**
```
======================================================================
🎤 VOICECARE AI - PRODUCTION INFERENCE SYSTEM
======================================================================
Model: OpenSMILE + XGBoost (Accuracy ~90%)
User ID: default_user
======================================================================

[1/6] Loading trained model and scaler...
✅ Model loaded: xgboost_model_20251108_185046.json
✅ Scaler loaded: scaler_20251108_185046.pkl

[2/6] Initializing AI components...
✅ All components initialized

[3/6] Initializing voice stability analyzer...
✅ Voice stability analyzer ready

[4/6] Initializing OpenSMILE feature extractor...
✅ OpenSMILE initialized with ComParE_2016 feature set

✅ Backend server is ready!
📡 Listening on http://localhost:5000
🔗 Frontend should connect to: http://localhost:5000/predict
```

---

## Step 2: Start Frontend Development Server

```powershell
# Open a new terminal
# Navigate to frontend directory
cd frontend

# Install dependencies (if not already done)
npm install

# Start Vite dev server
npm run dev
```

**Expected Output:**
```
  VITE v5.x.x  ready in xxx ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
  ➜  press h + enter to show help
```

---

## Step 3: Test the Application

### Open Browser
Navigate to: `http://localhost:5173/`

### Test Flow:

1. **Splash Screen** → Click "Get Started"

2. **Home Page** → Click "Start Voice Test"

3. **Record Page** → Choose testing mode:
   
   **Option A: Live Recording**
   - Click "Live Recording" button
   - Click "Start Recording"
   - Speak clearly for 3-5 seconds (read a sentence or count)
   - Click "Stop Recording"
   - Wait for analysis (progress bar)

   **Option B: Upload File**
   - Click "Upload File" button
   - Select an audio file (.wav, .mp3, .m4a)
   - Click "Analyze Audio"
   - Wait for analysis

4. **Results Page** → Verify displays:
   - ✅ Risk Assessment (score, level, confidence)
   - ✅ AI Summary (Gemini-generated, history-aware)
   - ✅ **Voice Quality Metrics Card** ⭐ NEW
     - Jitter with status (Excellent/Good/Fair/Needs Attention)
     - Shimmer with status
     - HNR with status
     - Pitch Variation with status
     - Energy Variation with status
     - Color-coded indicators
     - Progress bars
   - ✅ Progress Chart (if multiple tests)
   - ✅ Next Steps buttons

5. **Test PDF Download**
   - Click "Download PDF Report"
   - Open PDF and verify:
     - ✅ Header and test info
     - ✅ Risk assessment
     - ✅ AI summary
     - ✅ **Voice Quality Metrics table** ⭐ NEW
     - ✅ Progress chart (if available)
     - ✅ Disclaimer

6. **Test Multiple Recordings**
   - Click "Take Another Test"
   - Record/upload another sample
   - Verify:
     - ✅ AI summary mentions trends ("improvement", "consistency", etc.)
     - ✅ Progress chart shows multiple data points
     - ✅ History tracking works

---

## What to Look For

### Backend Console Output (During Analysis):
```
======================================================================
🔬 STARTING FULL ANALYSIS PIPELINE
======================================================================

[1/4] Denoising audio...
✅ Audio denoised

[2/4] Extracting OpenSMILE features...
✅ 6373 features extracted

[3/4] Running prediction...
✅ Prediction complete:
   Risk Score: 0.234
   Risk Level: Low
   Confidence: 92.00%

[5/7] Calculating voice stability metrics...
✅ Stability Index: 0.847
   Jitter: 0.0234
   Shimmer: 0.0567
   HNR: 18.34 dB

[6/7] Generating AI insights...
📊 Found 0 previous tests
🤖 Generating empathetic AI summary...
```

### Frontend API Response (Check browser DevTools Network tab):
```json
{
  "risk_score": 0.234,
  "confidence": 0.92,
  "risk_level": "Low",
  "voice_stability_index": 0.847,
  "gemini_summary": "Your voice shows good stability...",
  "ai_findings": [
    "Voice characteristics analyzed using OpenSMILE features",
    "XGBoost model prediction with 92% confidence",
    "Voice frequency is stable (Jitter: 0.0234)",
    "Voice amplitude is consistent (Shimmer: 0.0567)",
    "Good voice clarity detected (HNR: 18.34 dB)"
  ],
  "acoustic_features": {
    "jitter": 0.0234,
    "shimmer": 0.0567,
    "hnr": 18.34,
    "pitch_variation": 12.3,
    "energy_variation": 19.8
  }
}
```

---

## Verification Checklist

### Backend:
- [ ] Server starts without errors
- [ ] Voice stability analyzer initializes
- [ ] OpenSMILE loads ComParE_2016 feature set
- [ ] XGBoost model and scaler load successfully
- [ ] Gemini API connection works
- [ ] Analysis pipeline completes all 7 steps
- [ ] Acoustic features calculated (jitter, shimmer, HNR, etc.)
- [ ] API returns acoustic_features in response
- [ ] History saved to JSON file

### Frontend:
- [ ] Application loads without console errors
- [ ] Both recording modes work (live + upload)
- [ ] File upload validates audio types
- [ ] Progress bar animates during analysis
- [ ] Results page displays all sections
- [ ] **AcousticFeaturesCard renders with all 5 metrics**
- [ ] Status colors match metric values (green/blue/yellow/orange)
- [ ] Progress bars display correctly
- [ ] Educational info panel shows
- [ ] PDF download works
- [ ] PDF includes Voice Quality Metrics section
- [ ] Chart renders in PDF (if multiple tests)
- [ ] localStorage persists tests
- [ ] Clear cache works

### AI Features:
- [ ] Gemini summary is contextual and empathetic
- [ ] AI findings include acoustic interpretations
- [ ] History-aware feedback after 2nd+ test
- [ ] Trend analysis mentions "improving" or "stable"
- [ ] Summaries avoid medical jargon
- [ ] Tone is supportive and encouraging

---

## Common Issues & Solutions

### Issue: Voice Stability Analyzer Fails
**Error:** `Could not calculate jitter/shimmer`
**Solution:** Check audio quality - may be too short or silent

### Issue: Gemini API Error
**Error:** `API key not configured`
**Solution:** Add API key in `backend/agentic_ai/config.py`

### Issue: OpenSMILE Error
**Error:** `Feature extraction failed`
**Solution:** Ensure audio file is valid format (16kHz+ sample rate recommended)

### Issue: Frontend Blank Screen
**Error:** API response format mismatch
**Solution:** Check browser console for errors, verify API returns acoustic_features

### Issue: PDF Missing Acoustic Features
**Error:** Acoustic section not in PDF
**Solution:** Verify latestTest.acoustic_features exists in state

---

## Test Data Interpretation

### Good Voice Quality Example:
```
Jitter: 0.0234 (< 0.05) ✅ Excellent
Shimmer: 0.0567 (< 0.10) ✅ Good
HNR: 18.34 dB (> 15) ✅ Good
Pitch Variation: 12.3% (< 15) ✅ Stable
Energy Variation: 19.8% (< 25) ✅ Consistent
→ Overall Stability Index: 0.847
```

### Voice Quality Needing Attention:
```
Jitter: 0.0789 (> 0.05) ⚠️ Needs Attention
Shimmer: 0.1234 (> 0.10) ⚠️ Needs Attention
HNR: 12.45 dB (< 15) ⚠️ Could be improved
Pitch Variation: 23.4% (> 15) ⚠️ Variable
Energy Variation: 31.2% (> 25) ⚠️ Variable
→ Overall Stability Index: 0.612
```

---

## Success Criteria

✅ **System is working correctly if:**
1. Backend calculates all 5 acoustic features without errors
2. Frontend displays AcousticFeaturesCard with correct values
3. Status indicators match metric thresholds
4. AI summary is empathetic and contextual
5. PDF includes Voice Quality Metrics table
6. History tracking persists across sessions
7. Trend analysis works after 2+ tests

---

## Files to Monitor

### Backend Logs:
- Console output during analysis
- `backend/data/history/default_user_history.json`
- `backend/reports/` directory (PDFs and JSONs)

### Frontend:
- Browser DevTools Console
- Network tab for API responses
- localStorage: `voicecare_tests` key

---

## Ready to Test!

The system is now fully integrated with:
- ✅ Voice Stability Calculator
- ✅ Enhanced Gemini AI with History Awareness
- ✅ Detailed Acoustic Features Display
- ✅ Comprehensive PDF Reports
- ✅ Complete History Tracking

**Start both servers and test the complete flow!**
