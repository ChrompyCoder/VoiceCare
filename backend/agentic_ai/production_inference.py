"""
Production Inference Pipeline with Full Agentic AI Integration
Uses OpenSMILE + XGBoost Model (Best Accuracy: ~90%)
Includes Voice Stability Analysis with Acoustic Features
"""

import sys
import numpy as np
import librosa
import noisereduce as nr
import soundfile as sf
import tempfile
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
from .voice_stability_analyzer import VoiceStabilityAnalyzer
from .shap_xgboost import XGBoostShapExplainer


class ProductionInference:
    """Complete production inference pipeline with all agentic AI features"""

    def __init__(self, user_id='default_user'):
        """Initialize production inference system

        Args:
            user_id: User identifier for history tracking
        """
        print("=" * 70)
        print("🎤 VOICECARE AI - PRODUCTION INFERENCE SYSTEM")
        print("=" * 70)
        print("Model: OpenSMILE + XGBoost (Accuracy ~90%)")
        print(f"User ID: {user_id}")
        print("=" * 70)

        # Load model and scaler
        print("\n[1/6] Loading trained model and scaler...")
        model_path = config.PRIMARY_MODEL_PATH
        scaler_path = config.SCALER_PATH
        if not model_path.exists() or not scaler_path.exists():
            raise FileNotFoundError(
                f"❌ Model or scaler not found.\n"
                f"Expected model: {model_path}\n"
                f"Expected scaler: {scaler_path}"
            )
        
        # Fix malformed base_score in raw JSON file before load (if present)
        try:
            import re
            if model_path.exists():
                raw = model_path.read_text(encoding='utf-8')
                m = re.search(r'"base_score":"([^"]+)"', raw)
                if m:
                    bs_val = m.group(1)
                    if '[' in bs_val or ']' in bs_val or 'E--' in bs_val:
                        cleaned = bs_val.strip('[]').replace('E--', 'E-')
                        try:
                            numeric = float(cleaned)
                            raw_fixed = raw.replace(f'"base_score":"{bs_val}"', f'"base_score":"{numeric}"')
                            model_path.write_text(raw_fixed, encoding='utf-8')
                            print(f"   🔧 Patched model file base_score '{bs_val}' -> '{numeric}'")
                        except Exception:
                            pass
        except Exception as e:
            print(f"   ⚠️ Could not patch model file base_score: {e}")

        self.model = xgb.Booster()
        self.model.load_model(str(model_path))
        
        # Fix base_score issue for SHAP compatibility
        try:
            import json
            import tempfile
            model_config = json.loads(self.model.save_config())
            
            # Check and fix base_score if needed
            if 'learner' in model_config and 'learner_model_param' in model_config['learner']:
                base_score = model_config['learner']['learner_model_param'].get('base_score', '0.5')
                
                if isinstance(base_score, str) and ('[' in base_score or ']' in base_score):
                    # Extract float from array format like '[5.40404E-1]'
                    base_score_clean = base_score.strip('[]').strip()
                    base_score_float = float(base_score_clean)
                    model_config['learner']['learner_model_param']['base_score'] = str(base_score_float)
                    
                    # Save model with fixed config
                    with tempfile.NamedTemporaryFile(suffix='.ubj', delete=False) as f:
                        temp_model_path = f.name
                    
                    # Save current model structure
                    self.model.save_model(temp_model_path)
                    
                    # Reload and apply fixed config
                    self.model = xgb.Booster()
                    self.model.load_model(temp_model_path)
                    self.model.load_config(json.dumps(model_config))
                    
                    # Save with fixed config permanently to temp location
                    self.model.save_model(temp_model_path)
                    
                    # Reload the fully fixed model
                    self.model = xgb.Booster()
                    self.model.load_model(temp_model_path)
                    
                    # Clean up temp file
                    import os
                    os.unlink(temp_model_path)
                    
                    print(f"   ✅ Fixed model base_score: {base_score_float} (for SHAP compatibility)")
        except Exception as e:
            print(f"   ⚠️ Could not fix base_score: {e}")
            # Continue with original model
        
        self.scaler = joblib.load(scaler_path)
        print(f"✅ Model loaded: {model_path.name}")
        print(f"✅ Scaler loaded: {scaler_path.name}")
        
        # Initialize components
        print("\n[2/6] Initializing AI components...")
        self.gemini = GeminiInterface()
        self.context_engine = GeminiContextEngine()
        self.history_mgr = HistoryManager(user_id=user_id)
        self.report_gen = ReportGenerator()
        print("✅ All components initialized")

        # Initialize voice stability analyzer
        print("\n[3/6] Initializing voice stability analyzer...")
        self.stability_analyzer = VoiceStabilityAnalyzer()
        print("✅ Voice stability analyzer ready")

        # Initialize OpenSMILE
        print("\n[4/6] Initializing OpenSMILE feature extractor...")
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.ComParE_2016,
            feature_level=opensmile.FeatureLevel.Functionals,
            num_workers=None,
            verbose=False
        )
        print("✅ OpenSMILE initialized with ComParE_2016 feature set")
        
        # Initialize SHAP explainer
        print("\n[5/6] Initializing SHAP explainer...")
        try:
            # Get feature names from scaler if available
            feature_names = None
            if hasattr(self.scaler, 'feature_names_in_'):
                feature_names = self.scaler.feature_names_in_.tolist()
            
            self.shap_explainer = XGBoostShapExplainer(self.model, feature_names)
            print("✅ SHAP explainer ready")
            self.shap_enabled = True
            # Provide historical feature vectors (if stored) for mean background
            hist_vectors = []
            for h in self.history_mgr.get_all_tests(limit=50):
                vec = h.get('scaled_features')
                if isinstance(vec, list) and len(vec) == (len(feature_names) if feature_names else len(vec)):
                    hist_vectors.append(vec)
            if hist_vectors:
                self.shap_explainer.set_background_vectors(hist_vectors)
        except Exception as e:
            print(f"⚠️ SHAP initialization failed: {e}")
            print("   Continuing without SHAP analysis...")
            self.shap_explainer = None
            self.shap_enabled = False
        
    def denoise_audio(self, audio_path):
        """
        Denoise audio file and return path to denoised version
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            str: Path to denoised audio file
        """
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=None)
            
            # Apply noise reduction
            y_denoised = nr.reduce_noise(y=y, sr=sr, stationary=False, prop_decrease=0.8)
            
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            sf.write(temp_file.name, y_denoised, sr)
            
            return temp_file.name
        except Exception as e:
            print(f"   ⚠️ Denoising failed, using original audio: {e}")
            return audio_path
    
    def preprocess_audio(self, audio_path, denoise=True):
        """
        Preprocess audio file to model input format using OpenSMILE with optional denoising
        
        Args:
            audio_path: Path to audio file
            denoise: Whether to apply noise reduction (default: True)
            
        Returns:
            np.array: Preprocessed features ready for model
        """
        print(f"\n[3/5] Preprocessing audio: {Path(audio_path).name}")
        
        audio_path_processed = audio_path
        try:
            # Denoise audio if enabled
            if denoise:
                print("   Denoising audio...")
                audio_path_processed = self.denoise_audio(audio_path)
            
            # Extract features with OpenSMILE
            print("   Extracting OpenSMILE features (ComParE_2016)...")
            features = self.smile.process_file(audio_path_processed)
            
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
        finally:
            # Clean up temp file if denoising was used
            if denoise and audio_path_processed != audio_path:
                try:
                    import os
                    os.unlink(audio_path_processed)
                except:
                    pass

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
        
        # Calculate voice stability metrics
        print("\n[5/8] Calculating voice stability metrics...")
        stability_metrics = self.stability_analyzer.calculate_stability(audio_path)
        print(f"✅ Stability Index: {stability_metrics['stability_index']:.3f}")
        print(f"   Jitter: {stability_metrics['jitter']:.4f}")
        print(f"   Shimmer: {stability_metrics['shimmer']:.4f}")
        print(f"   HNR: {stability_metrics['hnr']:.2f} dB")
        
        # Generate test ID early (needed for caching paths)
        test_id = f"VPX-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # Generate SHAP explanation (with caching)
        print("\n[6/8] Generating SHAP feature importance analysis...")
        shap_explanation = None
        shap_cache_path = config.SHAP_DIR / f"{test_id}.json"
        if shap_cache_path.exists():
            try:
                import json, base64
                shap_explanation = json.loads(shap_cache_path.read_text(encoding='utf-8'))
                print("   ♻️ Loaded SHAP explanation from cache")

                # --- Schema / field upgrade logic (labels, heatmaps) ---
                upgraded = False
                # Backfill heatmap_base64 if path exists but base64 missing
                if shap_explanation and shap_explanation.get('heatmap_path') and not shap_explanation.get('heatmap_base64'):
                    hp = Path(shap_explanation['heatmap_path'])
                    if hp.exists():
                        try:
                            b64 = base64.b64encode(hp.read_bytes()).decode('utf-8')
                            shap_explanation['heatmap_base64'] = f"data:image/png;base64,{b64}"
                            upgraded = True
                            print("   🔁 Added missing heatmap_base64 to cached SHAP explanation")
                        except Exception as e:
                            print(f"   ⚠️ Failed to backfill heatmap_base64: {e}")
                # Backfill raw_name & friendly name if only numeric Feature_x present
                if shap_explanation and shap_explanation.get('top_features'):
                    for f in shap_explanation['top_features']:
                        if 'raw_name' not in f:
                            f['raw_name'] = f.get('name')
                        # If name still looks like 'Feature_123', attempt friendly conversion using explainer helper
                        if f.get('name','').startswith('Feature_') and hasattr(self, 'shap_explainer') and self.shap_explainer:
                            try:
                                f['name'] = self.shap_explainer._friendly_name(f['raw_name'])
                                upgraded = True
                            except Exception:
                                pass
                # Inject / bump schema_version
                target_schema = 3
                if shap_explanation and shap_explanation.get('schema_version', 0) < target_schema:
                    shap_explanation['schema_version'] = target_schema
                    upgraded = True
                    print(f"   🔁 Updated schema_version={target_schema} in cached SHAP explanation")
                # Persist upgrade if any modifications
                if upgraded:
                    try:
                        shap_cache_path.write_text(json.dumps(shap_explanation, indent=2), encoding='utf-8')
                        print("   💾 Upgraded SHAP cache saved")
                    except Exception as e:
                        print(f"   ⚠️ Could not save upgraded SHAP cache: {e}")
            except Exception:
                shap_explanation = None
        if shap_explanation is None and self.shap_enabled and self.shap_explainer:
            try:
                shap_image_path = config.SHAP_DIR / f"shap_{test_id}.png"
                shap_explanation = self.shap_explainer.explain_prediction(
                    audio_features,
                    save_path=shap_image_path,
                    return_base64=True
                )
                print(f"✅ SHAP analysis complete")
                if shap_explanation.get('top_features'):
                    print(f"   Top feature: {shap_explanation['top_features'][0]['name']}")
                # Persist cache
                try:
                    import json
                    # Ensure schema_version present in newly generated explanation
                    if shap_explanation.get('schema_version', 0) < 3:
                        shap_explanation['schema_version'] = 3
                    shap_cache_path.write_text(json.dumps(shap_explanation, indent=2), encoding='utf-8')
                    print("   💾 SHAP explanation cached")
                except Exception as e:
                    print(f"   ⚠️ Could not cache SHAP explanation: {e}")
            except Exception as e:
                print(f"⚠️ SHAP analysis failed: {e}")
                shap_explanation = None
        elif shap_explanation is None:
            print("⏭️  SHAP analysis skipped (not enabled)")
        
        # Get test history
        print("\n[7/8] Generating AI insights...")
        test_history = self.history_mgr.get_all_tests(limit=10)
        print(f"📊 Found {len(test_history)} previous tests")
        
        # Create test result object with stability metrics
        test_result = {
            'id': test_id,
            'date': datetime.now().isoformat(),
            'risk_score': prediction['risk_score'],
            'confidence': prediction['confidence'],
            'risk_level': prediction['risk_level'],
            'model': prediction['model_name'],
            'audio_file': str(Path(audio_path).name),
            'stability_metrics': stability_metrics,  # Include full stability data
            'shap_analysis': shap_explanation,  # Include SHAP analysis
            'scaled_features': audio_features[0].tolist()  # Store vector for future background
        }
        
        # Generate Gemini summary with stability metrics
        print("🤖 Generating empathetic AI summary...")
        try:
            gemini_result = self.gemini.generate_summary(
                prediction,
                include_history=len(test_history) > 0,
                test_history=test_history,
                stability_metrics=stability_metrics  # Pass stability metrics to Gemini
            )
            
            # Safely extract summary
            if isinstance(gemini_result, dict):
                test_result['ai_summary'] = gemini_result.get('summary', 'Analysis complete.')
            elif isinstance(gemini_result, str):
                test_result['ai_summary'] = gemini_result
            else:
                test_result['ai_summary'] = 'Your voice analysis is complete. Please consult with a healthcare professional for detailed interpretation.'
                
        except Exception as e:
            print(f"⚠️ Gemini summary generation failed: {e}")
            test_result['ai_summary'] = 'Your voice analysis is complete. Please consult with a healthcare professional for detailed interpretation.'
        
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
            print("\n[8/8] Generating reports...")
            # Prepare SHAP summary for report
            shap_summary = None
            if shap_explanation:
                # Prepare formatted feature strings for report
                top_feats = shap_explanation.get('top_features', [])[:3]
                top_3_formatted = [
                    f"{f['name']} (imp: {f['importance']:.3f}, val: {f['value']:.3f})" for f in top_feats
                ]
                shap_summary = {
                    'clinical_interpretation': shap_explanation.get('explanation', ''),
                    'top_3_features': top_3_formatted,
                    'visualization': shap_explanation.get('visualization_path'),
                    'visualization_base64': shap_explanation.get('visualization_base64'),
                    'heatmap': shap_explanation.get('heatmap_path'),
                    'heatmap_base64': shap_explanation.get('heatmap_base64')
                }
            
            reports = self.report_gen.generate_full_report(
                test_result,
                shap_summary=shap_summary,
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
        
        if shap_explanation and shap_explanation.get('top_features'):
            print(f"\n🔍 Top Contributing Feature:")
            top_feat = shap_explanation['top_features'][0]
            print(f"   {top_feat['name']} (importance: {top_feat['importance']:.4f})")
        
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
