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
            model: Trained XGBoost model (xgb.Booster)
            feature_names: List of feature names (optional)
        """
        self.model = model
        self.feature_names = feature_names

        # Paths: Prefer XGBoost's built-in contributions; fallback to SHAP Tree/Kernel
        self.explainer = None
        self._use_kernel = False
        self._kernel_background = None
        self._precomputed_background_mean = None  # Mean feature vector from history (scaled)
        self._mode = 'auto'  # 'xgb_contrib' | 'tree' | 'kernel' | 'auto'

    def set_background_vectors(self, vectors):
        """Set historical feature vectors to build a mean-based KernelExplainer background.

        Args:
            vectors (list[list[float]] or np.ndarray): Collection of past scaled feature vectors.
        """
        try:
            if vectors is None:
                return
            arr = np.array(vectors, dtype=float)
            if arr.ndim == 1:
                # Single vector
                self._precomputed_background_mean = arr
            elif arr.ndim == 2 and arr.shape[0] >= 1:
                self._precomputed_background_mean = arr.mean(axis=0)
            else:
                return
            print(f"   📦 Loaded historical background mean (shape: {self._precomputed_background_mean.shape})")
        except Exception as e:
            print(f"   ⚠️ Could not set background vectors: {e}")
        
    def explain_prediction(self, audio_features, save_path=None, return_base64=True):
        """
        Generate SHAP explanation for a single prediction
        
        Args:
            audio_features: Preprocessed OpenSMILE feature array (numpy array or DMatrix)
            save_path: Optional path to save visualization
            return_base64: Whether to return base64 encoded image
            
        Returns:
            dict: Explanation summary with SHAP values and insights
        """
        try:
            # Convert to numpy array if it's a different type
            if hasattr(audio_features, 'values'):
                audio_features = audio_features.values
            
            # Ensure it's a 2D numpy array
            if len(audio_features.shape) == 1:
                audio_features = audio_features.reshape(1, -1)
            
            # Calculate contributions/SHAP values
            shap_values, base_value = self._compute_contrib_or_shap(audio_features)
            
            # Get feature importance
            feature_importance = self._analyze_feature_importance(shap_values, audio_features)
            
            # Generate visualization
            viz_result = self._create_visualization(
                shap_values=shap_values,
                features=audio_features,
                base_value=base_value,
                save_path=save_path,
                return_base64=return_base64
            )

            # Also create a heatmap visualization
            heatmap_path = None
            heatmap_b64 = None
            try:
                heatmap_out = self._create_heatmap(
                    shap_values=shap_values,
                    features=audio_features,
                    save_path=(Path(save_path).with_name(f"heatmap_{Path(save_path).name}") if save_path else None),
                    return_base64=return_base64
                )
                heatmap_path = heatmap_out.get('path')
                heatmap_b64 = heatmap_out.get('base64')
            except Exception as _:
                pass
            
            # Create summary
            # Build final summary (schema_version introduced for future upgrades)
            summary = {
                'schema_version': 2,
                'top_features': feature_importance['top_features'],
                'feature_categories': feature_importance['categories'],
                'explanation': feature_importance['explanation'],
                'visualization_path': viz_result.get('path'),
                'visualization_base64': viz_result.get('base64'),
                'heatmap_path': heatmap_path,
                'heatmap_base64': heatmap_b64,
                'timestamp': datetime.now().isoformat()
            }
            
            return summary
            
        except Exception as e:
            print(f"⚠️ SHAP analysis failed: {e}")
            return {
                'top_features': [],
                'feature_categories': {},
                'explanation': 'Feature contribution analysis unavailable for this prediction.',
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
        # Ensure we have a 1D array of SHAP values for the single sample
        if isinstance(shap_values, list):
            # Some explainers return [values] for single output
            shap_vals_1d = np.array(shap_values[0])
        else:
            shap_vals_2d = np.array(shap_values)
            shap_vals_1d = shap_vals_2d[0] if shap_vals_2d.ndim > 1 else shap_vals_2d

        # Get absolute SHAP values
        shap_abs = np.abs(shap_vals_1d)
        
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
            raw_name = f"Feature_{idx}" if self.feature_names is None else self.feature_names[idx]
            feature_name = self._friendly_name(raw_name)
            importance = float(shap_abs[idx])
            value = float(features[0, idx])
            
            feature_info = {
                'index': int(idx),
                'raw_name': raw_name,
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

    def _compute_contrib_or_shap(self, features):
        """Compute local feature contributions.

        Tries XGBoost's built-in pred_contribs first (fast, reliable). If that fails,
        falls back to SHAP TreeExplainer, and finally KernelExplainer.

        Returns:
            (values, base_value): tuple of (np.ndarray, float)
        """
        # 1) Try XGBoost's built-in contributions
        try:
            import xgboost as xgb
            dm = xgb.DMatrix(features)
            contribs = self.model.predict(dm, pred_contribs=True)
            # contribs shape: (n, n_features + 1), last column is bias term
            contrib_vector = contribs[0]
            base_value = float(contrib_vector[-1])
            values = np.array(contrib_vector[:-1])
            self._mode = 'xgb_contrib'
            return values, base_value
        except Exception as e:
            print(f"   ⚠️ XGBoost pred_contribs failed: {e}")

        # 2) Try TreeExplainer
        try:
            if self.explainer is None or self._mode not in ('tree', 'kernel'):
                print("   Creating SHAP TreeExplainer...")
                self.explainer = shap.TreeExplainer(self.model)
            vals = self.explainer.shap_values(features)
            expected = self.explainer.expected_value
            if isinstance(vals, list):
                vals = np.array(vals[0])
            values = vals[0] if vals.ndim == 2 else vals
            base_value = float(np.array(expected).reshape(-1)[0])
            self._mode = 'tree'
            print("   ✅ SHAP TreeExplainer created successfully")
            return values, base_value
        except Exception as e:
            print(f"   ⚠️ TreeExplainer failed: {e}")

        # 3) KernelExplainer fallback
        n_features = features.shape[1]

        def predict_fn(X):
            import xgboost as xgb
            dm = xgb.DMatrix(X)
            margins = self.model.predict(dm, output_margin=True)
            probs = 1.0 / (1.0 + np.exp(-margins))
            return probs

        if self._kernel_background is None:
            if self._precomputed_background_mean is not None and len(self._precomputed_background_mean) == n_features:
                noise_scale = 0.01
                self._kernel_background = np.repeat(self._precomputed_background_mean.reshape(1, -1), 50, axis=0)
                self._kernel_background += np.random.normal(0, noise_scale, self._kernel_background.shape)
                print("   🟣 KernelExplainer background initialized (mean-based)")
            else:
                self._kernel_background = np.zeros((20, n_features), dtype=float)
                print("   🟣 KernelExplainer background initialized (zeros)")

        if self.explainer is None or self._mode != 'kernel':
            self.explainer = shap.KernelExplainer(predict_fn, self._kernel_background)
            print("   ✅ KernelExplainer created")
        shap_values = self.explainer.shap_values(features, nsamples=100)
        if isinstance(shap_values, list):
            shap_values = np.array(shap_values[0])
        values = shap_values[0] if shap_values.ndim == 2 else shap_values
        # Approximate base value as model average probability on background
        base_value = float(np.mean(predict_fn(self._kernel_background)))
        self._mode = 'kernel'
        return values, base_value
    
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

    def _friendly_name(self, raw):
        """Convert a raw ComParE/OpenSMILE feature string into a concise human label.

        Heuristics based on common patterns. Keeps original if no mapping found.
        """
        if not raw:
            return raw
        lower = raw.lower()
        # Direct mappings / contains checks
        mappings = [
            (['jitter'], 'Jitter (Pitch Stability)'),
            (['shimmer'], 'Shimmer (Amplitude Stability)'),
            (['hnr'], 'Harmonic-to-Noise Ratio (HNR)'),
            (['f0', 'pitch'], 'Pitch (F0)'),
            (['loudness'], 'Loudness'),
            (['energy'], 'Energy'),
            (['zcr'], 'Zero-Crossing Rate'),
            (['spectralflux', 'specflux'], 'Spectral Flux'),
            (['spectralrolloff', 'rolloff'], 'Spectral Rolloff'),
            (['spectralcentroid', 'centroid'], 'Spectral Centroid'),
            (['mfcc'], 'MFCC Coefficient'),
            (['formant'], 'Formant Frequency'),
            (['voicing'], 'Voicing Probability'),
            (['delta'], 'Delta (Change Rate)'),
            (['stddev', 'std'], 'Standard Deviation'),
            (['kurtosis'], 'Kurtosis'),
            (['skewness'], 'Skewness'),
            (['min'], 'Minimum'),
            (['max'], 'Maximum'),
            (['range'], 'Range'),
            (['amean', 'mean'], 'Mean'),
            (['quartile1'], '1st Quartile'),
            (['quartile2', 'median'], 'Median'),
            (['quartile3'], '3rd Quartile'),
            (['iqr'], 'Inter-Quartile Range'),
        ]
        for keys, label in mappings:
            if any(k in lower for k in keys):
                # Add more context for mfcc with index inside []
                if 'mfcc' in lower:
                    import re
                    m = re.search(r'mfcc.*\[(\d+)\]', lower)
                    if m:
                        return f"MFCC {m.group(1)}"
                return label
        # Clean generic artifacts
        cleaned = raw
        cleaned = cleaned.replace('F0final_sma', 'Pitch').replace('F0final', 'Pitch')
        cleaned = cleaned.replace('_sma', '')
        cleaned = cleaned.replace('_', ' ').replace('[', ' ').replace(']', '')
        # Collapse multiple spaces
        cleaned = " ".join(cleaned.split())
        # Title case but keep MFCC capitalized
        if 'mfcc' in lower:
            cleaned = cleaned.replace('mfcc', 'MFCC')
        cleaned = cleaned.title()
        return cleaned[:60]
    
    def _create_visualization(self, shap_values, features, base_value, save_path=None, return_base64=True):
        """Create a simple contribution bar chart visualization.

        Returns:
            dict: Visualization path and/or base64 encoded image
        """
        try:
            vec = np.array(shap_values)
            # Select top 15 by absolute contribution
            k = min(15, len(vec))
            idx = np.argsort(np.abs(vec))[-k:][::-1]
            vals = vec[idx]
            names = [self.feature_names[i] if self.feature_names is not None else f"Feature_{i}" for i in idx]
            colors = ['#2E7D32' if v < 0 else '#C62828' for v in vals]  # green for lowering risk, red for increasing

            fig, ax = plt.subplots(figsize=(10, 7))
            y = np.arange(k)
            ax.barh(y, vals, color=colors)
            ax.set_yticks(y)
            ax.set_yticklabels(names)
            ax.invert_yaxis()
            ax.axvline(0, color='#444', linewidth=0.8)
            ax.set_xlabel('Contribution to risk score')
            ax.set_title('Feature Contribution Analysis')
            plt.tight_layout()

            result = {}
            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
                result['path'] = str(save_path)
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

    def _create_heatmap(self, shap_values, features, save_path=None, return_base64=True):
        """Create a heatmap of top-K feature values and contributions.

        For a single test, we present two columns per feature: value and contribution.
        """
        try:
            vec = np.array(shap_values)
            k = min(20, len(vec))
            idx = np.argsort(np.abs(vec))[-k:][::-1]

            contribs = vec[idx]
            vals = features[0, idx]

            data = np.vstack([vals, contribs]).T  # shape (k, 2)
            row_labels = [self.feature_names[i] if self.feature_names is not None else f"Feature_{i}" for i in idx]
            col_labels = ["Value", "Contribution"]

            fig, ax = plt.subplots(figsize=(8, 0.4 * k + 2))
            im = ax.imshow(data, aspect='auto', cmap='coolwarm')
            ax.set_yticks(np.arange(k))
            ax.set_yticklabels(row_labels)
            ax.set_xticks(np.arange(2))
            ax.set_xticklabels(col_labels)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title('Top Feature Values and Contributions (Heatmap)')
            plt.tight_layout()

            result = {}
            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
                result['path'] = str(save_path)
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
            print(f"⚠️ Heatmap creation failed: {e}")
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
