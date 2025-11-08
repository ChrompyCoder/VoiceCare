"""
SHAP Explainability for XGBoost Model
Provides transparent explanations for OpenSMILE + XGBoost predictions
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import shap
from pathlib import Path
from datetime import datetime
from . import config
import base64
from io import BytesIO


class XGBoostShapExplainer:
    """
    Generates SHAP-based explanations for XGBoost voice analysis predictions
    """
    
    def __init__(self, model, feature_names=None):
        """
        Initialize SHAP explainer for XGBoost
        
        Args:
            model: Trained XGBoost model
            feature_names: List of feature names (optional)
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = shap.TreeExplainer(self.model)
        
    def explain_prediction(self, audio_features, save_path=None, return_base64=True):
        """
        Generate SHAP explanation for a single prediction
        
        Args:
            audio_features: Preprocessed OpenSMILE feature array (1, 6373)
            save_path: Optional path to save visualization
            return_base64: Whether to return base64 encoded image
            
        Returns:
            dict: Explanation summary with SHAP values and insights
        """
        try:
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(audio_features)
            
            # Get feature importance
            feature_importance = self._analyze_feature_importance(shap_values, audio_features)
            
            # Generate visualization
            viz_result = self._create_visualization(
                shap_values, 
                audio_features, 
                save_path,
                return_base64
            )
            
            # Create summary
            summary = {
                'top_features': feature_importance['top_features'],
                'feature_categories': feature_importance['categories'],
                'explanation': feature_importance['explanation'],
                'visualization_path': viz_result.get('path'),
                'visualization_base64': viz_result.get('base64'),
                'timestamp': datetime.now().isoformat()
            }
            
            return summary
            
        except Exception as e:
            print(f"⚠️ SHAP analysis failed: {e}")
            return {
                'top_features': [],
                'feature_categories': {},
                'explanation': 'SHAP analysis unavailable for this prediction.',
                'visualization_path': None,
                'visualization_base64': None,
                'error': str(e)
            }
    
    def _analyze_feature_importance(self, shap_values, features):
        """
        Analyze SHAP values to identify key contributing factors
        
        Returns:
            dict: Feature importance analysis
        """
        # Get absolute SHAP values
        shap_abs = np.abs(shap_values[0])
        
        # Get indices of top contributing features
        top_indices = np.argsort(shap_abs)[-20:][::-1]
        
        # Categorize features (OpenSMILE ComParE_2016 feature groups)
        feature_categories = {
            'prosodic': [],  # F0, energy, duration
            'spectral': [],  # MFCC, spectral features
            'voice_quality': [],  # Jitter, shimmer, HNR
            'temporal': []  # Rate of change features
        }
        
        top_features = []
        
        for idx in top_indices[:10]:  # Top 10 features
            feature_name = f"Feature_{idx}" if self.feature_names is None else self.feature_names[idx]
            importance = float(shap_abs[idx])
            value = float(features[0, idx])
            
            feature_info = {
                'index': int(idx),
                'name': feature_name,
                'importance': importance,
                'value': value
            }
            
            top_features.append(feature_info)
            
            # Categorize feature
            if 'F0' in feature_name or 'pitch' in feature_name.lower():
                feature_categories['prosodic'].append(feature_info)
            elif 'mfcc' in feature_name.lower() or 'spectral' in feature_name.lower():
                feature_categories['spectral'].append(feature_info)
            elif 'jitter' in feature_name.lower() or 'shimmer' in feature_name.lower() or 'hnr' in feature_name.lower():
                feature_categories['voice_quality'].append(feature_info)
            else:
                feature_categories['temporal'].append(feature_info)
        
        # Generate explanation text
        explanation = self._generate_explanation(top_features, feature_categories)
        
        return {
            'top_features': top_features,
            'categories': feature_categories,
            'explanation': explanation
        }
    
    def _generate_explanation(self, top_features, categories):
        """Generate human-readable explanation"""
        if not top_features:
            return "Unable to generate detailed explanation."
        
        explanation_parts = []
        
        # Analyze by category
        if categories['voice_quality']:
            explanation_parts.append(
                "Voice quality features (jitter, shimmer, harmonic-to-noise ratio) "
                "show the most significant impact on the prediction."
            )
        
        if categories['prosodic']:
            explanation_parts.append(
                "Prosodic features (pitch variation, energy dynamics) "
                "contributed to the risk assessment."
            )
        
        if categories['spectral']:
            explanation_parts.append(
                "Spectral characteristics of the voice signal "
                "indicate patterns consistent with the predicted risk level."
            )
        
        # Add top feature
        if top_features:
            top = top_features[0]
            explanation_parts.append(
                f"The most influential factor was {top['name']} "
                f"(importance: {top['importance']:.4f})."
            )
        
        return " ".join(explanation_parts) if explanation_parts else "Analysis complete."
    
    def _create_visualization(self, shap_values, features, save_path=None, return_base64=True):
        """
        Create SHAP waterfall plot visualization
        
        Returns:
            dict: Visualization path and/or base64 encoded image
        """
        try:
            # Create figure
            fig = plt.figure(figsize=(10, 8))
            
            # Create waterfall plot
            shap.plots.waterfall(
                shap.Explanation(
                    values=shap_values[0],
                    base_values=self.explainer.expected_value,
                    data=features[0],
                    feature_names=self.feature_names
                ),
                max_display=15,
                show=False
            )
            
            plt.title('SHAP Feature Impact Analysis', fontsize=14, fontweight='bold')
            plt.tight_layout()
            
            result = {}
            
            # Save to file if requested
            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
                result['path'] = str(save_path)
            
            # Convert to base64 if requested
            if return_base64:
                buffer = BytesIO()
                plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
                buffer.seek(0)
                image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
                result['base64'] = f"data:image/png;base64,{image_base64}"
                buffer.close()
            
            plt.close(fig)
            
            return result
            
        except Exception as e:
            print(f"⚠️ Visualization creation failed: {e}")
            plt.close('all')
            return {'path': None, 'base64': None, 'error': str(e)}
    
    def generate_summary_report(self, explanation):
        """
        Generate a structured summary for inclusion in reports
        
        Args:
            explanation: Output from explain_prediction()
            
        Returns:
            dict: Formatted summary for reports
        """
        return {
            'top_5_features': [
                {
                    'name': f['name'],
                    'importance': f['importance'],
                    'value': f['value']
                }
                for f in explanation['top_features'][:5]
            ],
            'feature_categories': {
                cat: len(features) 
                for cat, features in explanation['feature_categories'].items()
            },
            'clinical_interpretation': explanation['explanation'],
            'visualization': explanation.get('visualization_path')
        }


def create_explainer(model, feature_names=None):
    """
    Utility function to create XGBoost SHAP explainer
    
    Args:
        model: Trained XGBoost model
        feature_names: Optional list of feature names
        
    Returns:
        XGBoostShapExplainer instance
    """
    return XGBoostShapExplainer(model, feature_names)
