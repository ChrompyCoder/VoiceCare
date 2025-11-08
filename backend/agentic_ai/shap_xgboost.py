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
            # Prefer provided feature_names; fall back to generic placeholder
            raw_name = None
            if self.feature_names is not None and idx < len(self.feature_names):
                raw_name = str(self.feature_names[idx])
            if not raw_name or raw_name.strip() == '' or raw_name.lower().startswith('feature_'):
                raw_name = f"Feature_{idx}"
            # Build friendly label
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
        # Shorten statistical suffix clutter
        cleaned = cleaned.replace(' F0', ' Pitch')
        cleaned = cleaned.replace(' AudSpec', ' Audio Spectrum')
        cleaned = cleaned.replace(' Pcm', ' PCM')
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
        """Create a "liquidation-style" heatmap for top-K features.

        Visual style:
        - Dark background with neon gradient (viridis) like trading liquidation maps
        - Horizontal streaks via tiling and slight noise for a fluid look
        - Overlay a polyline representing contribution magnitude per feature
        - Colorbar on the right
        """
        try:
            vec = np.array(shap_values)
            k = min(20, len(vec))
            idx = np.argsort(np.abs(vec))[-k:][::-1]

            # Values and contributions for selected features
            contribs = vec[idx]
            vals = features[0, idx]

            # Normalize to [0,1] for color mapping
            def norm(x):
                x = np.array(x, dtype=float)
                if np.allclose(np.max(x), np.min(x)):
                    return np.zeros_like(x)
                return (x - np.min(x)) / (np.max(x) - np.min(x))

            v_norm = norm(vals)
            c_abs = np.abs(contribs)
            c_norm = norm(c_abs)

            # Build a wide matrix with horizontal bands to mimic liquidation heatmap
            # Left half encodes feature values, right half encodes contribution magnitude
            width_per_panel = 180  # columns per panel
            H = k
            W = width_per_panel * 2
            heat = np.zeros((H, W), dtype=float)
            # Fill left and right panels with row-wise constants then add slight noise bands
            for i in range(H):
                heat[i, :width_per_panel] = v_norm[i]
                heat[i, width_per_panel:] = c_norm[i]
            # Add subtle horizontal banding and blur-like effect
            rng = np.random.default_rng(42)
            noise = rng.normal(0, 0.03, size=heat.shape)
            heat = np.clip(heat + noise, 0, 1)

            # Figure setup (dark theme)
            fig_height = max(3.0, 0.28 * k + 2.0)
            fig, ax = plt.subplots(figsize=(10, fig_height))
            fig.patch.set_facecolor('#0b0f19')
            ax.set_facecolor('#0b0f19')

            im = ax.imshow(
                heat,
                aspect='auto',
                cmap='viridis',
                interpolation='bilinear',
                vmin=0,
                vmax=1
            )

            # Y labels: feature-friendly names
            row_labels = [self._friendly_name(self.feature_names[i] if self.feature_names is not None else f"Feature_{i}") for i in idx]
            ax.set_yticks(np.arange(H))
            ax.set_yticklabels(row_labels, color='#e5e7eb', fontsize=9)

            # X labels: panels
            ax.set_xticks([width_per_panel/2, width_per_panel + width_per_panel/2])
            ax.set_xticklabels(['Value', 'Contribution'], color='#e5e7eb', fontsize=10)

            # Gridlines and spines minimal
            for spine in ax.spines.values():
                spine.set_visible(False)
            ax.tick_params(axis='both', colors='#9ca3af', length=0)

            # Overlay polyline mapping contribution sign/magnitude to X in Contribution panel
            # Map contribution value to x in right panel
            if np.max(np.abs(contribs)) > 0:
                # scale contributions to [0, width_per_panel]
                s = (contribs - np.min(contribs)) / (np.max(contribs) - np.min(contribs) + 1e-9)
                x_coords = width_per_panel + s * (width_per_panel - 1)
                y_coords = np.arange(H)
                ax.plot(x_coords, y_coords, color='#ff9800', linewidth=1.6, alpha=0.9)

            # Colorbar styled
            cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
            cbar.outline.set_edgecolor('#374151')
            cbar.ax.tick_params(colors='#e5e7eb', labelsize=8)
            cbar.set_label('Intensity', color='#e5e7eb')

            ax.set_title('Top Features: Value (L) vs Contribution (R)', color='#e5e7eb')
            plt.tight_layout()

            result = {}
            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=160, bbox_inches='tight', facecolor=fig.get_facecolor())
                result['path'] = str(save_path)
            if return_base64:
                buffer = BytesIO()
                plt.savefig(buffer, format='png', dpi=160, bbox_inches='tight', facecolor=fig.get_facecolor())
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
