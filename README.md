# VoiceCare: Detection of Parkinson's disease using Voice(.wav or live recording)

@sanjayj @sakshamyadav

## Overview

Successfully implemented a comprehensive voice analysis system with acoustic feature analysis, enhanced AI feedback, and detailed reporting capabilities. The system now provides deep insights into SHAP (SHapley Additive exPlanations) alongside Parkinson's disease risk assessment.

---
## Dependencies
### Build a python venv
Linux:
`python3 -m venv ./.venv`
`source ./.venv/bin/activate`
`pip3 install -r requirements.txt`
---
## Train yourself
The features are extracted using OpenSmile and trained on XGBoost.
Activate the dependencies.
Train using train.py, for custom dataset, change the address in the code.
Linux: `python3 ./opxgboost/train.py`
Test the model using test.py
Linux: `python3 ./opxgboost/test.py`
---

## Running the app
Make sure the dependencies are correct.
Run the backend server using: 
Linux: `python3 ./backend/app.py`
Run the frontend:
Linux: `npm install` only for the first time
`npm run dev`

## Completed Features

### 1. Enhanced Gemini Empathetic Layer
**Status:** Fully Integrated

**Implementation:**
- **File:** `backend/agentic_ai/gemini_interface.py`
- Enhanced with telemedicine assistant persona

**Key Enhancements:**

**System Prompt Update:**
- Warm, supportive language (avoid clinical jargon)
- Never diagnose - focus on observations
- Encourage professional consultation for elevated risk
- Be tactful about concerns while celebrating improvements
- Keep summaries concise (3-4 sentences)
- Focus on trends and patterns

**Method Updates:**

1. **`generate_summary()`**
   - Now accepts `stability_metrics` parameter
   - Passes acoustic features to context builder

2. **`_build_context()`**
   - Extracts key acoustic features (jitter, shimmer, HNR)
   - Adds descriptive strings for each metric with thresholds
   - Calculates trend information from history
   - Returns direction (improving/worsening/stable), change percentage, number of tests

3. **`_create_prompt()`**
   - Structured format with risk score, risk level, acoustic features
   - Includes trend information if available
   - History-aware instructions for longitudinal perspective
   - Emphasizes comparing with previous tests

**Integration:**
- Receives stability_metrics from `production_inference.py`
- Generates context-aware, empathetic summaries
- Returns summaries stored in test results

---

### 2. SHAP Explainability Integration
**Status:** Fully Integrated

**Implementation:**
- **File:** `backend/agentic_ai/shap_xgboost.py` (XGBoost-specific SHAP explainer)
- **File:** `backend/agentic_ai/shap_explainer.py` (Deep learning SHAP support)
- **Component:** `XGBoostShapExplainer`

**Features:**
- SHAP (SHapley Additive exPlanations) values for model interpretability
- Feature importance analysis showing which voice characteristics influenced the prediction
- Waterfall plots visualizing individual prediction explanations
- Base64-encoded visualization images for frontend display
- Automatic feature name labeling using OpenSMILE feature names

**SHAP Analysis Provides:**
- Top contributing features (ranked by importance)
- Feature categories (prosodic, spectral, voice quality, temporal)
- Clinical interpretation of feature impacts
- Visual waterfall charts showing cumulative feature effects

**Integration:**
- Initialized in `production_inference.py` during system startup
- Uses historical feature vectors as SHAP background data
- Generates explanations for each prediction
- Results included in API response as `shap_analysis` field
- Graceful fallback if SHAP initialization fails

**Technical Details:**
- Uses TreeExplainer for XGBoost model compatibility
- Handles base_score normalization for accurate SHAP values

---

### 3. Improved PDF Report Generation
**Status:** Fully Integrated

**Backend PDF (`report_generator.py`):**

**Enhanced JSON Report:**
- Separate `acoustic_features` section with all 5 metrics
- Includes interpretation field
- Structured for programmatic access

**Enhanced PDF Report:**
- **Voice Quality Metrics Table:**
  - 3-column table (Metric | Value | Interpretation)
  - Color-coded header (green)
  - All 5 acoustic features with thresholds
  - Professional medical-grade formatting

**Frontend PDF (`ResultsPage.tsx`):**

**Added Acoustic Features Section:**
- Renders between AI Summary and Progress Chart
- Lists all 5 metrics with values and status
- Compact format (9pt font)
- Automatic page break handling

**Complete PDF Structure:**
1. Header (VoiceCare AI branding)
2. Test Information (ID, Date)
3. Risk Assessment (Score, Level, Confidence)
4. AI Summary
5. **Voice Quality Metrics** NEW
6. Progress Trend Chart 
7. Disclaimer

---

## Technical Architecture

### Backend Flow
```
Audio Input
    ↓
[1] Audio Denoising (noisereduce, prop_decrease=0.8)
    ↓
[2] OpenSMILE Feature Extraction (ComParE_2016 - 6,373 features)
    ↓
[3] StandardScaler Normalization
    ↓
[4] XGBoost Prediction (risk_score, confidence, risk_level)
    ↓
[5] Voice Stability Analysis (VoiceStabilityAnalyzer)
    ↓  - Jitter, Shimmer, HNR, Pitch Var, Energy Var
    ↓
[6] Gemini AI Summary Generation (with stability metrics + history)
    ↓
[7] History Storage (JSON persistence)
    ↓
[8] Report Generation (PDF + JSON)
    ↓
API Response (flat structure with acoustic_features)
```

### Frontend Flow
```
User Input (Live/Upload)
    ↓
API Call (/predict)
    ↓
Response Parsing
    ↓
Test Object Creation (with acoustic_features)
    ↓
Context Storage (localStorage + state)
    ↓
ResultsPage Display:
    - Risk Assessment
    - AI Summary
    - AcousticFeaturesCard
    - Progress Chart
    - PDF Download
```

---

## 📊 API Response Format

**Endpoint:** `POST /predict`

**Response Structure:**
```json
{
  "risk_score": float,
  "confidence": float,
  "risk_level": string,
  "voice_stability_index": float,
  "gemini_summary": string,
  "progress_analysis": string,  // MISSING in README
  "ai_findings": array,
  "acoustic_features": {
    "jitter": float,
    "shimmer": float,
    "hnr": float,
    "pitch_variation": float,
    "energy_variation": float
  },
  "shap_analysis": object  // MISSING in README
}
```

---

## User Experience Improvements

### For Patients:
1. **Detailed Voice Insights** - Understand specific aspects of voice quality
2. **Visual Feedback** - Color-coded status indicators for easy comprehension
3. **Trend Awareness** - AI summaries now reference historical patterns
4. **Educational Context** - Learn what jitter, shimmer, HNR mean
5. **Comprehensive Reports** - PDFs include all acoustic details

### For Healthcare Professionals:
1. **Clinical Metrics** - Standard acoustic biomarkers (jitter, shimmer, HNR)
2. **Quantitative Data** - Precise numerical values with thresholds
3. **Longitudinal Tracking** - History stores complete metric sets
4. **Structured Reports** - Professional PDF format with acoustic tables
5. **API Access** - JSON reports for programmatic integration

---
