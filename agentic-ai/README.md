# Agentic AI Features

This directory contains advanced AI features for the VoiceCare AI system. These features are ready for integration once the core model training is complete.

## Features Implemented

### 1. 🩺 SHAP Explainability Layer (`shap_explainer.py`)
- Generates transparent explanations for model predictions
- Creates heatmaps highlighting problematic voice regions
- Identifies top contributing acoustic features
- Saves visualizations for medical reports

### 2. 💬 Gemini Empathetic Layer (`gemini_interface.py`)
- Converts clinical results into empathetic, user-friendly language
- Provides context-aware feedback based on test history
- Acts as a telemedicine assistant persona
- Avoids medical diagnosis while staying informative

### 3. 🔁 Dual AI Model System (`dual_model_system.py`)
- Model A: Core CNN-BiLSTM Parkinson's predictor
- Model B: Audio quality guardian (detects noise, interference)
- Fusion logic for reliable results
- Flags unreliable inputs for re-recording

### 4. 📊 User History Management (`history_manager.py`)
- Stores test results with full context
- Tracks trends over time
- Generates comparison reports
- Feeds historical context to Gemini

### 5. 🧩 PDF/Text Export (`report_generator.py`)
- Creates professional medical reports
- Includes SHAP visualizations
- Exports as PDF and JSON
- QR code linking to cloud history

### 6. 🧠 Gemini Context Engine (`gemini_context.py`)
- Provides longitudinal awareness
- Analyzes trends across multiple tests
- Generates personalized progress summaries
- Smart recommendations based on patterns

## Directory Structure

```
agentic-ai/
├── README.md
├── shap_explainer.py       # SHAP visualization & analysis
├── gemini_interface.py     # Empathetic AI responses
├── gemini_context.py       # History-aware Gemini
├── dual_model_system.py    # Dual model architecture
├── history_manager.py      # Test history tracking
├── report_generator.py     # PDF/JSON export
├── audio_quality_model.py  # Audio quality classifier
└── config.py               # Configuration settings
```

## Integration Notes

These modules are designed to be plug-and-play. Once your core model is trained:

1. Update `config.py` with your model paths
2. Import the required modules into your main pipeline
3. Call the functions in sequence after getting model predictions

## Example Integration Flow

```python
# After getting prediction from your main model
result = main_model.predict(audio)

# 1. Generate SHAP explanation
shap_viz = shap_explainer.explain(audio, result)

# 2. Check audio quality
quality_score = audio_quality_model.assess(audio)

# 3. Get empathetic summary
gemini_summary = gemini_interface.generate_summary(result, quality_score)

# 4. Save to history
history_manager.save_test(result, shap_viz, gemini_summary)

# 5. Generate report
report_generator.create_pdf(result, shap_viz, gemini_summary)
```

## Requirements

See individual module files for specific dependencies.
