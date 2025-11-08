"""
Example Integration Script
Demonstrates how to use all agentic AI features together
"""

import numpy as np
from pathlib import Path

# Import agentic AI modules
from shap_explainer import ShapExplainer
from gemini_interface import GeminiInterface
from gemini_context import GeminiContextEngine
from dual_model_system import DualModelSystem
from history_manager import HistoryManager
from report_generator import ReportGenerator
import config


def full_pipeline_example(audio_features, user_id='demo_user'):
    """
    Complete pipeline demonstrating all features
    
    Args:
        audio_features: Preprocessed audio features (mel-spectrogram)
        user_id: User identifier
        
    Returns:
        dict: Complete analysis results
    """
    print("="*60)
    print("VOICECARE AI - AGENTIC PIPELINE")
    print("="*60)
    
    # Step 1: Initialize components
    print("\n[1/7] Initializing AI components...")
    
    # Load models (update paths after training)
    dual_system = DualModelSystem(
        primary_model_path=config.PRIMARY_MODEL_PATH,
        quality_model_path=config.AUDIO_QUALITY_MODEL_PATH
    )
    
    # Initialize history manager
    history_mgr = HistoryManager(user_id=user_id)
    
    # Step 2: Make prediction with dual model system
    print("[2/7] Running dual model prediction...")
    prediction = dual_system.predict(audio_features, return_quality=True)
    
    print(f"   Risk Score: {prediction['adjusted_risk_score']:.3f}")
    print(f"   Risk Level: {prediction['risk_level']}")
    print(f"   Audio Quality: {prediction['quality_flag']}")
    
    # Check if re-recording needed
    rerecord_check = dual_system.should_rerecord(prediction['quality_score'])
    if rerecord_check['should_rerecord']:
        print(f"   ⚠ Warning: {rerecord_check['reason']}")
        return {'status': 'rerecord_needed', 'details': rerecord_check}
    
    # Step 3: Generate SHAP explanation
    print("[3/7] Generating SHAP explanations...")
    
    # Note: Need background data for SHAP - load from your training set
    # For demo, we'll skip if not available
    try:
        background_data = load_background_data()  # Implement this
        explainer = ShapExplainer(dual_system.primary_model, background_data)
        shap_explanation = explainer.explain_prediction(audio_features)
        shap_summary = explainer.generate_summary_report(shap_explanation)
        print(f"   ✓ SHAP visualization saved: {shap_explanation['visualization_path']}")
    except Exception as e:
        print(f"   ⚠ SHAP explanation skipped: {e}")
        shap_summary = None
    
    # Step 4: Get test history for context
    print("[4/7] Loading test history...")
    test_history = history_mgr.get_all_tests(limit=10)
    print(f"   Found {len(test_history)} previous tests")
    
    # Step 5: Generate empathetic Gemini summary
    print("[5/7] Generating AI-powered empathetic summary...")
    
    result_context = {
        'risk_level': prediction['risk_level'],
        'risk_score': prediction['adjusted_risk_score'],
        'confidence': prediction['adjusted_confidence'],
        'voice_stability_index': prediction.get('voice_stability_index', 0.7),
        'shap_summary': shap_summary
    }
    
    try:
        gemini_interface = GeminiInterface()
        gemini_result = gemini_interface.generate_summary(
            result_context,
            include_history=len(test_history) > 0,
            test_history=test_history
        )
        gemini_summary = gemini_result['summary']
        print(f"   ✓ Gemini summary generated")
    except Exception as e:
        print(f"   ⚠ Gemini unavailable: {e}")
        gemini_summary = "AI summary unavailable"
    
    # Step 6: Save to history
    print("[6/7] Saving test result to history...")
    
    test_result = {
        'id': f"VPX-{int(np.random.rand() * 1000000)}",
        'date': np.datetime64('now').astype(str),
        'risk_score': prediction['adjusted_risk_score'],
        'confidence': prediction['adjusted_confidence'],
        'risk_level': prediction['risk_level'],
        'voice_stability_index': prediction.get('voice_stability_index', 0.7),
        'shap_summary': shap_summary,
        'gemini_summary': gemini_summary,
        'quality_score': prediction['quality_score']
    }
    
    history_mgr.save_test(test_result)
    print(f"   ✓ Test saved with ID: {test_result['id']}")
    
    # Step 7: Generate reports
    print("[7/7] Generating reports...")
    
    report_gen = ReportGenerator()
    reports = report_gen.generate_full_report(
        test_result,
        shap_summary,
        gemini_summary,
        test_history,
        format='both'
    )
    
    print(f"   ✓ PDF Report: {reports.get('pdf', 'N/A')}")
    print(f"   ✓ JSON Report: {reports.get('json', 'N/A')}")
    
    # Bonus: Context-aware progress analysis
    if len(test_history) >= 1:
        print("\n[BONUS] Context-aware progress analysis...")
        try:
            context_engine = GeminiContextEngine()
            progress_analysis = context_engine.analyze_progress(test_history, test_result)
            print(f"   Trend: {progress_analysis['trend']}")
            print(f"   Summary: {progress_analysis['summary'][:200]}...")
        except Exception as e:
            print(f"   ⚠ Progress analysis skipped: {e}")
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    
    return {
        'status': 'success',
        'prediction': prediction,
        'test_result': test_result,
        'reports': reports,
        'gemini_summary': gemini_summary
    }


def quick_analysis_example(audio_features):
    """
    Simplified example for quick analysis without full pipeline
    """
    print("Running quick analysis...")
    
    # Just prediction and basic summary
    dual_system = DualModelSystem(primary_model_path=config.PRIMARY_MODEL_PATH)
    prediction = dual_system.predict(audio_features, return_quality=False)
    
    result = {
        'risk_level': prediction['risk_level'],
        'risk_score': prediction['risk_score'],
        'confidence': prediction['model_confidence']
    }
    
    # Generate basic report
    report_gen = ReportGenerator()
    summary = report_gen.generate_quick_summary(result)
    
    print(summary)
    return result


def load_background_data():
    """
    Load background data for SHAP
    This should load a subset of your training data
    """
    # TODO: Implement actual loading from your training dataset
    # For now, return None to indicate unavailable
    raise NotImplementedError("Load your training data here for SHAP background")


if __name__ == '__main__':
    print(__doc__)
    print("\nThis is an example integration script.")
    print("Update model paths in config.py before running.")
    print("\nUsage:")
    print("  from example_integration import full_pipeline_example")
    print("  result = full_pipeline_example(audio_features, user_id='user123')")
