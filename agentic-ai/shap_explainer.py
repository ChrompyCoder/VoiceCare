"""
SHAP Explainability Layer
Provides transparent explanations for model predictions using SHAP values
"""

import numpy as np
import matplotlib.pyplot as plt
import shap
from pathlib import Path
import librosa
import librosa.display
from datetime import datetime
import config

class ShapExplainer:
    """
    Generates SHAP-based explanations for voice analysis predictions
    """
    
    def __init__(self, model, background_data=None):
        """
        Initialize SHAP explainer
        
        Args:
            model: Trained Keras model
            background_data: Background dataset for SHAP (optional)
        """
        self.model = model
        self.explainer = None
        self.background_data = background_data
        
        if background_data is not None:
            self._initialize_explainer()
    
    def _initialize_explainer(self):
        """Initialize SHAP DeepExplainer"""
        # Use a subset of background data for efficiency
        background_subset = self.background_data[:config.SHAP_BACKGROUND_SIZE]
        self.explainer = shap.DeepExplainer(self.model, background_subset)
    
    def explain_prediction(self, audio_features, save_path=None):
        """
        Generate SHAP explanation for a single prediction
        
        Args:
            audio_features: Mel-spectrogram or feature array (numpy array)
            save_path: Optional path to save visualization
            
        Returns:
            dict: Explanation summary with SHAP values and insights
        """
        if self.explainer is None:
            raise ValueError("Explainer not initialized. Provide background_data first.")
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(audio_features)
        
        # Extract top contributing features
        feature_importance = self._analyze_feature_importance(shap_values, audio_features)
        
        # Generate visualization
        viz_path = self._create_visualization(
            shap_values, 
            audio_features, 
            save_path
        )
        
        # Create summary
        summary = {
            'shap_values': shap_values,
            'top_features': feature_importance['top_features'],
            'spectral_regions': feature_importance['spectral_regions'],
            'confidence_explanation': feature_importance['explanation'],
            'visualization_path': viz_path,
            'timestamp': datetime.now().isoformat()
        }
        
        return summary
    
    def _analyze_feature_importance(self, shap_values, features):
        """
        Analyze SHAP values to identify key contributing factors
        
        Returns:
            dict: Feature importance analysis
        """
        # Flatten SHAP values for analysis
        shap_flat = np.abs(shap_values).flatten()
        feature_flat = features.flatten()
        
        # Get indices of top contributing features
        top_indices = np.argsort(shap_flat)[-10:][::-1]
        
        # Map to mel-spectrogram coordinates (time, frequency)
        if len(features.shape) == 4:  # (batch, height, width, channels)
            height, width = features.shape[1:3]
        else:
            height, width = features.shape[:2]
        
        top_features = []
        spectral_regions = []
        
        for idx in top_indices[:5]:  # Top 5 features
            # Convert flat index to 2D coordinates
            time_idx = (idx % (height * width)) // width
            freq_idx = (idx % (height * width)) % width
            
            # Approximate frequency range (assuming mel scale)
            freq_hz = self._mel_to_hz(freq_idx, width)
            
            top_features.append({
                'time_frame': int(time_idx),
                'frequency_bin': int(freq_idx),
                'frequency_hz': freq_hz,
                'importance': float(shap_flat[idx]),
                'value': float(feature_flat[idx])
            })
            
            # Group into spectral regions
            if 200 <= freq_hz <= 300:
                spectral_regions.append('Tremor-related jitter region (200-300Hz)')
            elif 300 <= freq_hz <= 500:
                spectral_regions.append('Mid-frequency instability region (300-500Hz)')
            elif freq_hz < 200:
                spectral_regions.append('Low-frequency baseline region (<200Hz)')
        
        # Remove duplicate regions
        spectral_regions = list(set(spectral_regions))
        
        # Generate explanation text
        explanation = self._generate_explanation(top_features, spectral_regions)
        
        return {
            'top_features': top_features,
            'spectral_regions': spectral_regions,
            'explanation': explanation
        }
    
    def _mel_to_hz(self, mel_bin, total_bins):
        """Convert mel bin to approximate Hz"""
        # Rough approximation
        mel_max = config.MEL_SPEC_PARAMS.get('fmax', 8000)
        hz = (mel_bin / total_bins) * mel_max
        return round(hz, 1)
    
    def _generate_explanation(self, top_features, spectral_regions):
        """Generate human-readable explanation"""
        if not top_features:
            return "Unable to generate detailed explanation."
        
        explanation_parts = []
        
        # Identify primary region
        primary_freq = top_features[0]['frequency_hz']
        if 200 <= primary_freq <= 300:
            explanation_parts.append(
                f"Primary instability detected around {primary_freq:.0f}Hz, "
                "consistent with tremor-related jitter patterns."
            )
        elif primary_freq > 300:
            explanation_parts.append(
                f"Spectral irregularities in the {primary_freq:.0f}Hz range "
                "suggest potential voice instability."
            )
        else:
            explanation_parts.append(
                f"Low-frequency variations around {primary_freq:.0f}Hz "
                "may indicate baseline voice quality issues."
            )
        
        # Add affected regions
        if len(spectral_regions) > 1:
            explanation_parts.append(
                f"Multiple spectral regions affected: {', '.join(spectral_regions[:2])}."
            )
        
        return " ".join(explanation_parts)
    
    def _create_visualization(self, shap_values, features, save_path=None):
        """
        Create SHAP heatmap visualization
        
        Returns:
            str: Path to saved visualization
        """
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Original mel-spectrogram
        if len(features.shape) == 4:
            feature_2d = features[0, :, :, 0]
        else:
            feature_2d = features
        
        # Plot 1: Original mel-spectrogram
        librosa.display.specshow(
            feature_2d,
            y_axis='mel',
            x_axis='time',
            ax=axes[0],
            cmap='viridis'
        )
        axes[0].set_title('Original Mel-Spectrogram')
        axes[0].set_ylabel('Frequency (Hz)')
        
        # Plot 2: SHAP heatmap
        if len(shap_values.shape) == 4:
            shap_2d = shap_values[0, :, :, 0]
        else:
            shap_2d = shap_values
        
        im = axes[1].imshow(
            np.abs(shap_2d),
            aspect='auto',
            cmap='hot',
            interpolation='bilinear'
        )
        axes[1].set_title('SHAP Importance Heatmap (Brighter = More Important)')
        axes[1].set_xlabel('Time Frame')
        axes[1].set_ylabel('Frequency Bin')
        
        plt.colorbar(im, ax=axes[1], label='Absolute SHAP Value')
        plt.tight_layout()
        
        # Save visualization
        if save_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            save_path = config.SHAP_DIR / f'shap_explanation_{timestamp}.png'
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        return str(save_path)
    
    def generate_summary_report(self, explanation):
        """
        Generate a structured summary for inclusion in reports
        
        Args:
            explanation: Output from explain_prediction()
            
        Returns:
            dict: Formatted summary for reports
        """
        return {
            'top_3_features': [
                f"{f['frequency_hz']:.0f}Hz at time {f['time_frame']}"
                for f in explanation['top_features'][:3]
            ],
            'affected_regions': explanation['spectral_regions'],
            'clinical_interpretation': explanation['confidence_explanation'],
            'visualization': explanation['visualization_path']
        }


# Utility function for batch processing
def explain_batch(model, audio_batch, background_data):
    """
    Generate SHAP explanations for a batch of audio samples
    
    Args:
        model: Trained model
        audio_batch: Batch of audio features
        background_data: Background dataset
        
    Returns:
        list: List of explanation dictionaries
    """
    explainer = ShapExplainer(model, background_data)
    explanations = []
    
    for i, audio in enumerate(audio_batch):
        try:
            explanation = explainer.explain_prediction(
                np.expand_dims(audio, 0),
                save_path=config.SHAP_DIR / f'batch_sample_{i}.png'
            )
            explanations.append(explanation)
        except Exception as e:
            print(f"Error explaining sample {i}: {e}")
            explanations.append(None)
    
    return explanations
