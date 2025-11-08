from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from pathlib import Path
import werkzeug

from agentic_ai.production_inference import ProductionInference

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

        return jsonify(analysis_result)

    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        # Consider more specific error handling here
        return jsonify({"error": "An internal error occurred during analysis."}), 500

if __name__ == '__main__':
    # Running on 0.0.0.0 makes it accessible from your local network
    print("\n✅ Backend server is ready!")
    print("📡 Listening on http://localhost:5000")
    print("🔗 Frontend should connect to: http://localhost:5000/predict")
    print("\nPress CTRL+C to stop the server\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
