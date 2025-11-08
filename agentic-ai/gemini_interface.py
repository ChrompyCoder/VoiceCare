"""
Gemini Empathetic Interface
Converts clinical results into user-friendly, empathetic language
"""

import os
import json
from datetime import datetime
import config

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: google-generativeai not installed. Install with: pip install google-generativeai")


class GeminiInterface:
    """
    Interface for generating empathetic, user-friendly explanations using Gemini
    """
    
    def __init__(self, api_key=None):
        """
        Initialize Gemini interface
        
        Args:
            api_key: Gemini API key (defaults to config or env variable)
        """
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai package not installed")
        
        self.api_key = api_key or config.GEMINI_API_KEY
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(config.GEMINI_MODEL)
        self.system_prompt = config.GEMINI_SYSTEM_PROMPT
    
    def generate_summary(self, result_context, include_history=False, test_history=None):
        """
        Generate empathetic summary from analysis results
        
        Args:
            result_context: Dict containing analysis results
            include_history: Whether to include historical context
            test_history: List of previous test results (if include_history=True)
            
        Returns:
            dict: Empathetic summary and recommendations
        """
        # Build structured context
        context = self._build_context(result_context, include_history, test_history)
        
        # Create prompt
        prompt = self._create_prompt(context, include_history)
        
        # Get Gemini response
        try:
            response = self.model.generate_content(
                f"{self.system_prompt}\n\n{prompt}"
            )
            summary_text = response.text
        except Exception as e:
            print(f"Error generating Gemini response: {e}")
            summary_text = self._fallback_summary(context)
        
        return {
            'summary': summary_text,
            'context': context,
            'timestamp': datetime.now().isoformat()
        }
    
    def _build_context(self, result_context, include_history, test_history):
        """
        Build structured context for Gemini
        
        Returns:
            dict: Structured context
        """
        # Extract key information
        risk_level = result_context.get('risk_level', 'Unknown')
        risk_score = result_context.get('risk_score', 0)
        confidence = result_context.get('confidence', 0)
        voice_stability = result_context.get('voice_stability_index', 0)
        
        # Determine key features
        key_features = []
        if risk_score > 0.5:
            if voice_stability < 0.6:
                key_features.append("Voice stability irregularities")
            if 'shap_summary' in result_context:
                shap_info = result_context['shap_summary']
                if 'spectral_regions' in shap_info:
                    key_features.extend(shap_info['spectral_regions'][:2])
        
        context = {
            'risk_level': risk_level,
            'risk_score': round(risk_score, 2),
            'confidence': round(confidence, 2),
            'voice_stability_score': round(voice_stability, 2),
            'key_features': key_features if key_features else ["General voice patterns analyzed"],
            'emotion_tone': 'neutral'
        }
        
        # Add historical context if available
        if include_history and test_history:
            context['previous_tests'] = self._format_history(test_history)
            context['trend'] = self._analyze_trend(test_history)
        
        return context
    
    def _format_history(self, test_history):
        """Format test history for context"""
        formatted = []
        for test in test_history[-3:]:  # Last 3 tests
            formatted.append({
                'date': test.get('date', 'Unknown'),
                'risk_score': round(test.get('risk_score', 0), 2)
            })
        return formatted
    
    def _analyze_trend(self, test_history):
        """Analyze trend in test results"""
        if len(test_history) < 2:
            return 'insufficient_data'
        
        recent_scores = [t.get('risk_score', 0) for t in test_history[-3:]]
        
        if len(recent_scores) < 2:
            return 'insufficient_data'
        
        # Simple trend analysis
        if recent_scores[-1] < recent_scores[0]:
            return 'improving'
        elif recent_scores[-1] > recent_scores[0]:
            return 'worsening'
        else:
            return 'stable'
    
    def _create_prompt(self, context, include_history):
        """
        Create Gemini prompt from context
        
        Returns:
            str: Formatted prompt
        """
        context_json = json.dumps(context, indent=2)
        
        base_prompt = f"""
Based on this voice analysis context:

{context_json}

Task: Explain these results to the user in a gentle, empathetic tone. 

Guidelines:
- Keep it brief (2-3 sentences)
- Use simple language, avoid medical jargon
- Be reassuring but honest
- Focus on what was observed, not diagnosis
- Suggest next steps if risk is elevated
"""
        
        if include_history:
            base_prompt += """
- Reference the trend from previous tests
- Acknowledge progress if improvement is seen
- Encourage consistency in testing
"""
        
        return base_prompt
    
    def _fallback_summary(self, context):
        """
        Generate fallback summary if Gemini is unavailable
        
        Returns:
            str: Basic summary
        """
        risk_level = context['risk_level']
        risk_score = context['risk_score']
        
        if risk_level == 'Low':
            return (
                f"Your voice analysis shows a low risk score ({risk_score:.0%}). "
                "Your voice patterns appear stable. Continue monitoring regularly for peace of mind."
            )
        elif risk_level == 'Moderate':
            return (
                f"Your voice analysis indicates a moderate risk score ({risk_score:.0%}). "
                "Some minor irregularities were detected. Consider consulting a healthcare professional "
                "for a thorough evaluation."
            )
        else:  # High
            return (
                f"Your voice analysis shows an elevated risk score ({risk_score:.0%}). "
                "We recommend scheduling an appointment with a neurologist for a comprehensive assessment. "
                "Early consultation can be very helpful."
            )
    
    def generate_progress_summary(self, test_history):
        """
        Generate a progress summary based on test history
        
        Args:
            test_history: List of test results
            
        Returns:
            str: Progress summary
        """
        if len(test_history) < 2:
            return "Not enough test history to analyze progress. Continue testing regularly."
        
        context = {
            'previous_tests': self._format_history(test_history),
            'trend': self._analyze_trend(test_history)
        }
        
        prompt = f"""
Based on this test history:

{json.dumps(context, indent=2)}

Task: Provide a brief progress summary for the user.

Guidelines:
- Highlight the overall trend (improving, stable, or concerning)
- Be encouraging if progress is positive
- Suggest consistent monitoring
- Keep it 2-3 sentences
- Be empathetic and supportive
"""
        
        try:
            response = self.model.generate_content(
                f"{self.system_prompt}\n\n{prompt}"
            )
            return response.text
        except Exception as e:
            print(f"Error generating progress summary: {e}")
            
            # Fallback
            trend = context['trend']
            if trend == 'improving':
                return "Your recent tests show positive progress. Keep up the good work!"
            elif trend == 'stable':
                return "Your voice patterns have remained consistent. Continue regular monitoring."
            else:
                return "Your recent tests show some variation. Consider scheduling a consultation."


# Utility function
def generate_empathetic_response(result_dict, history=None, api_key=None):
    """
    Quick utility to generate empathetic response
    
    Args:
        result_dict: Analysis results
        history: Optional test history
        api_key: Optional API key
        
    Returns:
        str: Empathetic summary text
    """
    if not GEMINI_AVAILABLE:
        print("Warning: Gemini not available, using fallback")
        interface = type('obj', (object,), {
            '_fallback_summary': lambda self, ctx: GeminiInterface._fallback_summary(None, ctx)
        })()
        context = {
            'risk_level': result_dict.get('risk_level', 'Unknown'),
            'risk_score': result_dict.get('risk_score', 0)
        }
        return interface._fallback_summary(context)
    
    interface = GeminiInterface(api_key)
    result = interface.generate_summary(
        result_dict,
        include_history=history is not None,
        test_history=history
    )
    return result['summary']
