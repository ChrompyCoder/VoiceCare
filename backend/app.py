from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from pathlib import Path
import werkzeug

from agentic_ai.production_inference import ProductionInference
from agentic_ai.gemini_interface import GeminiInterface

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Create an uploads folder if it doesn't exist
UPLOADS_DIR = Path(__file__).parent / 'uploads'
UPLOADS_DIR.mkdir(exist_ok=True)

# Initialize the inference engine once
try:
    inference_engine = ProductionInference(user_id='frontend_user')
except Exception as e:
    print(f"FATAL: Could not initialize ProductionInference: {e}")
    inference_engine = None

# Separate lightweight Gemini interface for chat (avoid re-running heavy pipeline)
try:
    gemini_chat = GeminiInterface()
except Exception as e:
    print(f"⚠️ Gemini chat interface init failed: {e}")
    gemini_chat = None

@app.route('/predict', methods=['POST'])
def predict():
    if inference_engine is None:
        return jsonify({"error": "Inference engine is not available. Check server logs."}), 500

    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']
    
    if audio_file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Sanitize filename
        filename = werkzeug.utils.secure_filename(audio_file.filename)
        save_path = UPLOADS_DIR / filename
        
        # Save the audio file
        audio_file.save(str(save_path))
        print(f"File saved to {save_path}")

        # Run the full analysis
        analysis_result = inference_engine.full_analysis(str(save_path))

        # Clean up the uploaded file
        os.remove(save_path)

        # Extract the test result and format for frontend
        test_result = analysis_result['test_result']
        stability_metrics = test_result.get('stability_metrics', {})
        
        # Generate acoustic findings from stability metrics
        ai_findings = [
            f"Voice characteristics analyzed using OpenSMILE features",
            f"XGBoost model prediction with {int(test_result['confidence'] * 100)}% confidence"
        ]
        
        # Add stability-based findings
        if stability_metrics:
            jitter = stability_metrics.get('jitter', 0)
            shimmer = stability_metrics.get('shimmer', 0)
            hnr = stability_metrics.get('hnr', 0)
            
            if jitter > 0.05:
                ai_findings.append(f"Voice frequency shows slight variation (Jitter: {jitter:.4f})")
            else:
                ai_findings.append(f"Voice frequency is stable (Jitter: {jitter:.4f})")
                
            if shimmer > 0.10:
                ai_findings.append(f"Voice amplitude shows some variation (Shimmer: {shimmer:.4f})")
            else:
                ai_findings.append(f"Voice amplitude is consistent (Shimmer: {shimmer:.4f})")
                
            if hnr < 15:
                ai_findings.append(f"Harmonic-to-Noise Ratio could be improved (HNR: {hnr:.2f} dB)")
            else:
                ai_findings.append(f"Good voice clarity detected (HNR: {hnr:.2f} dB)")
        
        # Format response for frontend
        response = {
            'risk_score': test_result['risk_score'],
            'confidence': test_result['confidence'],
            'risk_level': test_result['risk_level'],
            'voice_stability_index': stability_metrics.get('stability_index', 0.85),
            'gemini_summary': test_result.get('ai_summary', ''),
            'progress_analysis': test_result.get('progress_analysis', ''),
            'ai_findings': ai_findings,
            'acoustic_features': {
                'jitter': stability_metrics.get('jitter', 0),
                'shimmer': stability_metrics.get('shimmer', 0),
                'hnr': stability_metrics.get('hnr', 0),
                'pitch_variation': stability_metrics.get('pitch_variation', 0),
                'energy_variation': stability_metrics.get('energy_variation', 0)
            },
            'shap_analysis': test_result.get('shap_analysis')  # Include SHAP data
        }

        return jsonify(response)

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        # Consider more specific error handling here
        return jsonify({"error": "An internal error occurred during analysis."}), 500

@app.route('/chat', methods=['POST'])
def chat():
    """Gemini-powered chatbot endpoint.
    Expects JSON: { message: str, conversation?: [{role, content}], feature_name?: str }
    Returns: { reply: str }
    """
    if gemini_chat is None:
        return jsonify({'error': 'Chat interface unavailable'}), 500
    data = request.get_json(silent=True) or {}
    message = (data.get('message') or '').strip()
    feature_name = (data.get('feature_name') or '').strip()
    conversation = data.get('conversation') or []

    if feature_name and not message:
        # Direct feature explanation
        result = gemini_chat.explain_feature(feature_name)
        return jsonify({'reply': result['text']})

    # If user explicitly asks "what is" followed by feature-like token, attempt feature extraction
    if message.lower().startswith('what is') and len(message.split()) <= 12 and not feature_name:
        # crude heuristic: last word(s)
        candidate = message[7:].strip().strip('?')
        if candidate:
            feature_name = candidate
            result = gemini_chat.explain_feature(feature_name)
            return jsonify({'reply': result['text'], 'interpreted_feature': feature_name})

    # Generic multi-turn chat
    if message:
        # Append current user message to conversation
        conversation.append({'role': 'user', 'content': message})
        result = gemini_chat.chat(conversation)
        return jsonify({'reply': result['text']})
    else:
        return jsonify({'reply': 'Hi! Ask me about any voice feature or your results.'})

if __name__ == '__main__':
    # Running on 0.0.0.0 makes it accessible from your local network
    print("\n✅ Backend server is ready!")
    print("📡 Listening on http://localhost:5000")
    print("🔗 Frontend should connect to: http://localhost:5000/predict")
    print("\nPress CTRL+C to stop the server\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
