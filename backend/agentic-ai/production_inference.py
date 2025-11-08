"""
Production Inference Pipeline with Full Agentic AI Integration
Uses Fold 3 Model (Best Performance: AUC=0.9977, Precision=1.0)
"""

import sys
import numpy as np
import librosa
import noisereduce as nr
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

import tensorflow as tf
from tensorflow import keras

# Import agentic AI modules
import config
from gemini_interface import GeminiInterface
from gemini_context import GeminiContextEngine
from history_manager import HistoryManager
from report_generator import ReportGenerator


class ProductionInference:
    """
    Complete production inference pipeline with all agentic AI features
    """
    
    def __init__(self, user_id='default_user'):
        """
        Initialize production inference system
        
        Args:
            user_id: User identifier for history tracking
        """
        print("="*70)
        print("🎤 VOICECARE AI - PRODUCTION INFERENCE SYSTEM")
        print("="*70)
        print(f"Model: Fold 3 (AUC=0.9977, Precision=1.0, Recall=0.9813)")
        print(f"User ID: {user_id}")
        print("="*70)
        
        # Load model
        print("\n[1/5] Loading trained model...")
        model_path = config.PRIMARY_MODEL_PATH
        if not model_path.exists():
            raise FileNotFoundError(
                f"❌ Model not found: {model_path}\n"
                f"Expected: models/fold3_model_20251108_131148.h5"
            )
        
        self.model = keras.models.load_model(str(model_path))
        print(f"✅ Model loaded: {model_path.name}")
        
        # Initialize components
        print("\n[2/5] Initializing AI components...")
        self.gemini = GeminiInterface()
        self.context_engine = GeminiContextEngine()
        self.history_mgr = HistoryManager(user_id=user_id)
        self.report_gen = ReportGenerator()
        print("✅ All components initialized")
        
        # Audio parameters (must match training)
        self.sample_rate = 22050
        self.duration = 5
        self.n_mels = 128
        self.hop_length = 512
        
    def preprocess_audio(self, audio_path):
        """
        Preprocess audio file to model input format
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            np.array: Preprocessed features ready for model
        """
        print(f"\n[3/5] Preprocessing audio: {Path(audio_path).name}")
        
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, duration=self.duration)
        
        # 🔇 DENOISE AUDIO (Critical for matching training data)
        print("   🔇 Applying noise reduction...")
        # Use gentler denoising to avoid over-processing
        audio = nr.reduce_noise(y=audio, sr=sr, stationary=False, prop_decrease=0.8)
        print("   ✅ Audio denoised")
        
        # Pad or trim to exact duration
        target_length = self.sample_rate * self.duration
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        else:
            audio = audio[:target_length]
        
        # Extract mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            hop_length=self.hop_length
        )
        
        # Convert to log scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize
        mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-6)
        
        # Reshape for model input: (batch, height, width, channels)
        mel_spec_db = mel_spec_db[..., np.newaxis]
        mel_spec_db = np.expand_dims(mel_spec_db, axis=0)
        
        print(f"✅ Audio preprocessed: {mel_spec_db.shape}")
        return mel_spec_db
    
    def predict(self, audio_features):
        """
        Make prediction with model
        
        Args:
            audio_features: Preprocessed audio features
            
        Returns:
            dict: Prediction results
        """
        print("\n[4/5] Running model prediction...")
        
        # Get prediction
        prediction = self.model.predict(audio_features, verbose=0)
        risk_score = float(prediction[0][0])
        
        # Calculate confidence (distance from decision boundary at 0.5)
        confidence = abs(risk_score - 0.5) * 2
        
        # Determine risk level
        if risk_score < config.LOW_RISK_THRESHOLD:
            risk_level = 'Low'
        elif risk_score < config.MODERATE_RISK_THRESHOLD:
            risk_level = 'Moderate'
        else:
            risk_level = 'High'
        
        result = {
            'risk_score': risk_score,
            'confidence': confidence,
            'risk_level': risk_level,
            'model_name': 'Fold3_CNN-BiLSTM'
        }
        
        print(f"✅ Prediction complete:")
        print(f"   Risk Score: {risk_score:.3f}")
        print(f"   Risk Level: {risk_level}")
        print(f"   Confidence: {confidence:.2%}")
        
        return result
    
    def full_analysis(self, audio_path, generate_report=True):
        """
        Complete analysis pipeline with all agentic features
        
        Args:
            audio_path: Path to audio file
            generate_report: Whether to generate PDF/JSON reports
            
        Returns:
            dict: Complete analysis results
        """
        print("\n" + "="*70)
        print("🔬 STARTING FULL ANALYSIS PIPELINE")
        print("="*70)
        
        # Preprocess
        audio_features = self.preprocess_audio(audio_path)
        
        # Predict
        prediction = self.predict(audio_features)
        
        # Get test history
        print("\n[5/5] Generating AI insights...")
        test_history = self.history_mgr.get_all_tests(limit=10)
        print(f"📊 Found {len(test_history)} previous tests")
        
        # Create test result object
        test_result = {
            'id': f"VPX-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'date': datetime.now().isoformat(),
            'risk_score': prediction['risk_score'],
            'confidence': prediction['confidence'],
            'risk_level': prediction['risk_level'],
            'model': prediction['model_name'],
            'audio_file': str(Path(audio_path).name)
        }
        
        # Generate Gemini summary
        print("🤖 Generating empathetic AI summary...")
        gemini_result = self.gemini.generate_summary(
            prediction,
            include_history=len(test_history) > 0,
            test_history=test_history
        )
        test_result['ai_summary'] = gemini_result['summary']
        
        # Context-aware progress analysis (if history available)
        if len(test_history) >= 1:
            print("📈 Analyzing longitudinal progress...")
            progress = self.context_engine.analyze_progress(test_history, test_result)
            test_result['progress_analysis'] = progress['summary']
            test_result['trend'] = progress.get('trend', 'N/A')
        else:
            test_result['progress_analysis'] = "First test - baseline established"
            test_result['trend'] = 'baseline'
        
        # Save to history
        print("💾 Saving to history...")
        self.history_mgr.save_test(test_result)
        
        # Generate reports
        reports = {}
        if generate_report:
            print("📄 Generating reports...")
            reports = self.report_gen.generate_full_report(
                test_result,
                shap_summary=None,  # Can add SHAP later
                gemini_summary=test_result['ai_summary'],
                test_history=test_history,
                format='both'
            )
            print(f"✅ PDF Report: {reports.get('pdf', 'N/A')}")
            print(f"✅ JSON Report: {reports.get('json', 'N/A')}")
        
        # Complete result
        result = {
            'test_result': test_result,
            'reports': reports,
            'history_stats': self.history_mgr.calculate_statistics()
        }
        
        print("\n" + "="*70)
        print("✅ ANALYSIS COMPLETE")
        print("="*70)
        print(f"\n📋 SUMMARY:")
        print(f"   Test ID: {test_result['id']}")
        print(f"   Risk Level: {test_result['risk_level']}")
        print(f"   Risk Score: {test_result['risk_score']:.3f}")
        print(f"   Trend: {test_result['trend']}")
        print(f"\n💬 AI Summary:")
        print(f"   {test_result['ai_summary']}")
        
        if test_result.get('progress_analysis'):
            print(f"\n📊 Progress Analysis:")
            print(f"   {test_result['progress_analysis']}")
        
        return result
    
    def quick_analysis(self, audio_path):
        """
        Quick analysis without reports (for testing)
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            dict: Prediction results
        """
        audio_features = self.preprocess_audio(audio_path)
        prediction = self.predict(audio_features)
        
        # Quick AI summary
        gemini_result = self.gemini.generate_summary(prediction)
        prediction['ai_summary'] = gemini_result['summary']
        
        print(f"\n💬 AI Summary: {prediction['ai_summary']}")
        
        return prediction


def main():
    """
    Example usage
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='VoiceCare AI Production Inference')
    parser.add_argument('audio_file', help='Path to audio file')
    parser.add_argument('--user-id', default='demo_user', help='User ID')
    parser.add_argument('--quick', action='store_true', help='Quick analysis without reports')
    
    args = parser.parse_args()
    
    # Initialize system
    inference = ProductionInference(user_id=args.user_id)
    
    # Run analysis
    if args.quick:
        result = inference.quick_analysis(args.audio_file)
    else:
        result = inference.full_analysis(args.audio_file)
    
    return result


if __name__ == '__main__':
    main()
