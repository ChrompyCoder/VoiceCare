"""
Production Inference Pipeline with Full Agentic AI Integration
Uses OpenSMILE + XGBoost Model (Best Accuracy: ~90%)
"""

import sys
import numpy as np
import librosa
import opensmile
import xgboost as xgb
import joblib
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import agentic AI modules
from . import config
from .gemini_interface import GeminiInterface
from .gemini_context import GeminiContextEngine
from .history_manager import HistoryManager
from .report_generator import ReportGenerator


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
        print(f"Model: OpenSMILE + XGBoost (Accuracy ~90%)")
        print(f"User ID: {user_id}")
        print("="*70)
        
        # Load model and scaler
        print("\n[1/5] Loading trained model and scaler...")
        model_path = config.PRIMARY_MODEL_PATH
        scaler_path = config.SCALER_PATH
        if not model_path.exists() or not scaler_path.exists():
            raise FileNotFoundError(
                f"❌ Model or scaler not found.\n"
                f"Expected model: {model_path}\n"
                f"Expected scaler: {scaler_path}"
            )
        
        self.model = xgb.Booster()
        self.model.load_model(str(model_path))
        self.scaler = joblib.load(scaler_path)
        print(f"✅ Model loaded: {model_path.name}")
        print(f"✅ Scaler loaded: {scaler_path.name}")
        
        # Initialize components
        print("\n[2/5] Initializing AI components...")
        self.gemini = GeminiInterface()
        self.context_engine = GeminiContextEngine()
        self.history_mgr = HistoryManager(user_id=user_id)
        self.report_gen = ReportGenerator()
        print("✅ All components initialized")

        # Initialize OpenSMILE
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.ComParE_2016,
            feature_level=opensmile.FeatureLevel.Functionals,
            num_workers=None,
            verbose=False
        )
        print("✅ OpenSMILE initialized with ComParE_2016 feature set")
        
    def preprocess_audio(self, audio_path):
        """
        Preprocess audio file to model input format using OpenSMILE
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            np.array: Preprocessed features ready for model
        """
        print(f"\n[3/5] Preprocessing audio: {Path(audio_path).name}")
        
        try:
            # Extract features with OpenSMILE
            print("   Extracting OpenSMILE features (ComParE_2016)...")
            features = self.smile.process_file(audio_path)
            
            # Drop non-numeric columns if any
            features = features.select_dtypes(include=np.number)

            # Scale features
            print("   Scaling features...")
            scaled_features = self.scaler.transform(features)
            
            print(f"✅ Audio preprocessed: {scaled_features.shape}")
            return scaled_features
        except Exception as e:
            print(f"❌ Error during audio preprocessing: {e}")
            raise

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
        dmatrix = xgb.DMatrix(audio_features)
        prediction = self.model.predict(dmatrix, output_margin=True)
        risk_score = float(1 / (1 + np.exp(-prediction[0]))) # Sigmoid function to convert raw margin to probability
        
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
            'model_name': 'OpenSMILE_XGBoost'
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
                shap_summary=None,  # SHAP for XGBoost requires different setup
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
