"""
Dual Model System
Implements a two-model architecture for robust predictions
Model A: Core Parkinson's classifier
Model B: Audio quality guardian
"""

import numpy as np
from tensorflow import keras
import config


class DualModelSystem:
    """
    Manages two-model system for robust voice analysis
    """
    
    def __init__(self, primary_model_path=None, quality_model_path=None):
        """
        Initialize dual model system
        
        Args:
            primary_model_path: Path to primary Parkinson's model
            quality_model_path: Path to audio quality model
        """
        self.primary_model = None
        self.quality_model = None
        
        if primary_model_path:
            self.load_primary_model(primary_model_path)
        if quality_model_path:
            self.load_quality_model(quality_model_path)
    
    def load_primary_model(self, model_path):
        """Load primary Parkinson's classification model"""
        try:
            self.primary_model = keras.models.load_model(model_path)
            print(f"✓ Primary model loaded from {model_path}")
        except Exception as e:
            print(f"✗ Error loading primary model: {e}")
    
    def load_quality_model(self, model_path):
        """Load audio quality assessment model"""
        try:
            self.quality_model = keras.models.load_model(model_path)
            print(f"✓ Quality model loaded from {model_path}")
        except Exception as e:
            print(f"✗ Error loading quality model: {e}")
    
    def predict(self, audio_features, return_quality=True):
        """
        Make prediction using both models
        
        Args:
            audio_features: Preprocessed audio features
            return_quality: Whether to return quality assessment
            
        Returns:
            dict: Combined prediction results
        """
        if self.primary_model is None:
            raise ValueError("Primary model not loaded")
        
        # Model A: Primary prediction
        primary_prediction = self.primary_model.predict(audio_features, verbose=0)
        risk_score = float(primary_prediction[0][0])
        
        results = {
            'risk_score': risk_score,
            'risk_level': self._get_risk_level(risk_score),
            'model_confidence': self._calculate_confidence(primary_prediction)
        }
        
        # Model B: Quality assessment
        if return_quality and self.quality_model is not None:
            quality_score = self._assess_quality(audio_features)
            results['quality_score'] = quality_score
            results['quality_flag'] = self._get_quality_flag(quality_score)
            
            # Adjust final score based on quality
            results['adjusted_risk_score'] = self._adjust_score(
                risk_score,
                quality_score
            )
            results['adjusted_confidence'] = self._adjust_confidence(
                results['model_confidence'],
                quality_score
            )
        else:
            results['quality_score'] = 1.0
            results['quality_flag'] = 'unknown'
            results['adjusted_risk_score'] = risk_score
            results['adjusted_confidence'] = results['model_confidence']
        
        return results
    
    def _assess_quality(self, audio_features):
        """
        Assess audio quality using Model B
        
        Returns:
            float: Quality score (0-1, higher is better)
        """
        if self.quality_model is None:
            return 1.0  # Assume perfect quality if no model
        
        try:
            quality_prediction = self.quality_model.predict(audio_features, verbose=0)
            return float(quality_prediction[0][0])
        except Exception as e:
            print(f"Error assessing quality: {e}")
            return 1.0
    
    def _get_quality_flag(self, quality_score):
        """
        Determine quality flag based on score
        
        Returns:
            str: Quality flag (excellent/good/fair/poor)
        """
        if quality_score >= 0.9:
            return 'excellent'
        elif quality_score >= config.QUALITY_WARNING_THRESHOLD:
            return 'good'
        elif quality_score >= config.MIN_AUDIO_QUALITY:
            return 'fair'
        else:
            return 'poor'
    
    def _adjust_score(self, risk_score, quality_score):
        """
        Adjust risk score based on audio quality
        
        Args:
            risk_score: Raw risk score from primary model
            quality_score: Audio quality score
            
        Returns:
            float: Adjusted risk score
        """
        if quality_score >= config.QUALITY_WARNING_THRESHOLD:
            # High quality - trust the score
            return risk_score
        elif quality_score >= config.MIN_AUDIO_QUALITY:
            # Medium quality - slight adjustment
            adjustment_factor = 0.9 + (quality_score - config.MIN_AUDIO_QUALITY) * 0.5
            return risk_score * adjustment_factor
        else:
            # Low quality - significant uncertainty
            return risk_score * quality_score
    
    def _adjust_confidence(self, model_confidence, quality_score):
        """
        Adjust confidence based on audio quality
        
        Returns:
            float: Adjusted confidence
        """
        # Confidence is reduced proportionally to quality issues
        return model_confidence * quality_score
    
    def _calculate_confidence(self, prediction):
        """
        Calculate model confidence from prediction
        
        Args:
            prediction: Raw model output
            
        Returns:
            float: Confidence score (0-1)
        """
        # For binary classification, confidence is how far from 0.5
        score = float(prediction[0][0])
        confidence = abs(score - 0.5) * 2
        return min(confidence, 1.0)
    
    def _get_risk_level(self, risk_score):
        """
        Convert risk score to categorical level
        
        Returns:
            str: Risk level (Low/Moderate/High)
        """
        if risk_score < config.LOW_RISK_THRESHOLD:
            return 'Low'
        elif risk_score < config.MODERATE_RISK_THRESHOLD:
            return 'Moderate'
        else:
            return 'High'
    
    def should_rerecord(self, quality_score):
        """
        Determine if audio should be re-recorded
        
        Args:
            quality_score: Audio quality score
            
        Returns:
            dict: Recommendation for re-recording
        """
        if quality_score < config.MIN_AUDIO_QUALITY:
            return {
                'should_rerecord': True,
                'reason': 'Audio quality is too low for reliable analysis',
                'suggestions': [
                    'Find a quieter environment',
                    'Hold microphone closer to mouth',
                    'Ensure no background noise',
                    'Check microphone settings'
                ]
            }
        elif quality_score < config.QUALITY_WARNING_THRESHOLD:
            return {
                'should_rerecord': False,
                'warning': 'Audio quality is fair but could be improved',
                'suggestions': [
                    'Consider re-recording in a quieter space for better accuracy'
                ]
            }
        else:
            return {
                'should_rerecord': False,
                'status': 'Audio quality is good'
            }


class AudioQualityDetector:
    """
    Standalone audio quality detector
    Can be used independently or as part of DualModelSystem
    """
    
    @staticmethod
    def detect_confounds(audio_features):
        """
        Detect confounding factors in audio
        
        Args:
            audio_features: Audio feature array
            
        Returns:
            dict: Detected confounds
        """
        confounds = {
            'noise_detected': False,
            'clipping_detected': False,
            'silence_detected': False,
            'overall_quality': 'unknown'
        }
        
        # Simple heuristics (improve with actual audio quality model)
        # These are placeholder implementations
        
        # Check for clipping (values near 1.0 or -1.0)
        if len(audio_features.shape) > 2:
            max_val = np.max(np.abs(audio_features))
            if max_val > 0.95:
                confounds['clipping_detected'] = True
        
        # Check for excessive silence (low energy)
        mean_energy = np.mean(np.abs(audio_features))
        if mean_energy < 0.01:
            confounds['silence_detected'] = True
        
        # Check for noise (high variance in low frequencies)
        if len(audio_features.shape) >= 2:
            low_freq_var = np.var(audio_features[:, :10] if audio_features.shape[1] > 10 else audio_features)
            if low_freq_var > 0.5:
                confounds['noise_detected'] = True
        
        # Overall quality assessment
        if confounds['clipping_detected'] or confounds['silence_detected']:
            confounds['overall_quality'] = 'poor'
        elif confounds['noise_detected']:
            confounds['overall_quality'] = 'fair'
        else:
            confounds['overall_quality'] = 'good'
        
        return confounds


# Utility functions
def create_dual_system(primary_model_path=None, quality_model_path=None):
    """
    Factory function to create dual model system
    
    Returns:
        DualModelSystem: Initialized system
    """
    return DualModelSystem(primary_model_path, quality_model_path)


def quick_predict(audio_features, primary_model_path):
    """
    Quick prediction without quality model
    
    Args:
        audio_features: Preprocessed audio
        primary_model_path: Path to primary model
        
    Returns:
        dict: Prediction results
    """
    system = DualModelSystem(primary_model_path)
    return system.predict(audio_features, return_quality=False)
