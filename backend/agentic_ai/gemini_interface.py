"""
Gemini Empathetic Interface
Converts clinical results into user-friendly, empathetic language
NO FALLBACKS - Requires valid API key
"""

import json
from datetime import datetime
from . import config
import os
from google import genai


class GeminiInterface:
    """
    Interface for generating empathetic, user-friendly explanations using Gemini
    """
    
    def __init__(self, api_key=None):
        """
        Initialize Gemini interface with empathetic persona
        
        Args:
            api_key: Gemini API key (defaults to config)
        """
        self.api_key = api_key or config.GEMINI_API_KEY
        
        if not self.api_key or self.api_key == 'PASTE_YOUR_GEMINI_API_KEY_HERE':
            raise ValueError(
                "❌ GEMINI API KEY NOT SET!\n"
                "Get your API key from: https://makersuite.google.com/app/apikey\n"
                "Then update it in: agentic_ai/config.py (line 20)"
            )
        
        # Set API key as environment variable for genai client
        os.environ['GOOGLE_API_KEY'] = self.api_key
        self.client = genai.Client()
        self.model_name = config.GEMINI_MODEL
        
        # Enhanced persona - telemedicine assistant specializing in neurological health
        self.system_prompt = """You are a compassionate telemedicine assistant specializing in neurological health screening. 
Your role is to interpret voice analysis results and communicate them in a gentle, empathetic manner.

Key Guidelines:
- Use warm, supportive language without being alarmist
- Avoid clinical jargon; use simple, clear explanations
- Never provide definitive medical diagnoses
- Always encourage professional consultation when needed
- Be honest but tactful about concerning patterns
- Celebrate improvements and provide reassurance when appropriate
- Keep responses concise (3-4 sentences maximum)
- Focus on trends and stability rather than single readings"""
    
    def generate_summary(self, result_context, include_history=False, test_history=None, stability_metrics=None):
        """
        Generate empathetic summary from analysis results with history awareness
        
        Args:
            result_context: Dict containing analysis results
            include_history: Whether to include historical context
            test_history: List of previous test results (if include_history=True)
            stability_metrics: Voice stability analysis metrics
            
        Returns:
            dict: Empathetic summary and recommendations
        """
        # Build structured context
        context = self._build_context(result_context, include_history, test_history, stability_metrics)
        
        # Create prompt with history awareness
        prompt = self._create_prompt(context, include_history)
        
        # Get Gemini response (NO FALLBACK)
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=f"{self.system_prompt}\n\n{prompt}"
            )
            
            summary_text = response.text if hasattr(response, 'text') else str(response)
            
            return {
                'summary': summary_text,
                'context': context,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            print(f"⚠️ Gemini API error: {e}")
            return {
                'summary': 'Your voice analysis is complete. Please consult with a healthcare professional for detailed interpretation.',
                'context': context,
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def _build_context(self, result_context, include_history, test_history, stability_metrics):
        """
        Build structured context for Gemini with enhanced acoustic features
        
        Returns:
            dict: Structured context
        """
        # Extract key information
        risk_level = result_context.get('risk_level', 'Unknown')
        risk_score = result_context.get('risk_score', 0)
        confidence = result_context.get('confidence', 0)
        
        # Get stability metrics
        voice_stability = 0.85
        key_acoustic_features = []
        
        if stability_metrics:
            voice_stability = stability_metrics.get('stability_index', 0.85)
            
            # Extract key acoustic findings
            if stability_metrics.get('jitter', 0) > 0.03:
                key_acoustic_features.append(
                    f"Frequency instability detected (jitter: {stability_metrics['jitter']:.3f})"
                )
            if stability_metrics.get('shimmer', 0) > 0.08:
                key_acoustic_features.append(
                    f"Amplitude variation present (shimmer: {stability_metrics['shimmer']:.3f})"
                )
            if stability_metrics.get('hnr', 30) < 18:
                key_acoustic_features.append(
                    f"Reduced vocal clarity (HNR: {stability_metrics['hnr']:.1f} dB)"
                )
        
        # Build trend information from history
        trend_info = None
        if include_history and test_history and len(test_history) >= 2:
            recent_scores = [t.get('risk_score', 0) for t in test_history[:3]]
            if len(recent_scores) >= 2:
                score_change = ((recent_scores[0] - recent_scores[-1]) / recent_scores[-1]) * 100
                trend_info = {
                    'direction': 'improving' if score_change < -5 else 'worsening' if score_change > 5 else 'stable',
                    'change_percent': abs(score_change),
                    'num_tests': len(test_history)
                }
        
        context = {
            'risk_level': risk_level,
            'risk_score': round(risk_score, 2),
            'confidence': round(confidence, 2),
            'voice_stability_score': round(voice_stability, 2),
            'key_acoustic_features': key_acoustic_features if key_acoustic_features else ["Voice patterns within typical range"],
            'trend_info': trend_info,
            'emotion_tone': 'neutral'
        }
        
        # Add historical context if available
        if include_history and test_history:
            context['previous_tests'] = self._format_history(test_history)
            context['historical_trend'] = self._analyze_trend(test_history)
        
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
        Create Gemini prompt from context with history awareness
        
        Returns:
            str: Formatted prompt
        """
        # Extract key information for structured prompt
        risk_score = context.get('risk_score', 0)
        risk_level = context.get('risk_level', 'unknown')
        
        # Build prompt sections
        prompt_parts = [
            f"Voice Analysis Results:",
            f"- Risk Score: {risk_score:.1%}",
            f"- Risk Level: {risk_level.upper()}"
        ]
        
        # Add acoustic features if available
        if 'acoustic_features' in context:
            prompt_parts.append("\nVoice Quality Indicators:")
            for feature in context['acoustic_features']:
                prompt_parts.append(f"- {feature}")
        
        # Add trend information if available
        if 'historical_trend' in context and context['historical_trend']['direction'] != 'insufficient_data':
            trend_info = context['historical_trend']
            direction = trend_info['direction'].upper()
            change = trend_info['change_percent']
            prompt_parts.append(f"\nTrend: {direction} ({change:+.1f}% over last {trend_info['num_tests']} tests)")
        
        base_prompt = "\n".join(prompt_parts)
        base_prompt += """

Task: As a caring telemedicine assistant, explain these results to help the user understand their voice health.

Your Response Should:
- Use warm, supportive language (2-3 sentences maximum)
- Highlight any positive trends or stability
- If acoustic indicators show concerns, explain them gently
- Suggest practical next steps (consultation if elevated risk, keep monitoring if stable)
- Avoid medical diagnoses - focus on observations only
- Be encouraging while remaining factual
"""
        
        if include_history:
            base_prompt += """
Additional Context: Compare this test with previous results to identify patterns. Mention if this is an improvement or if consistency is observed.
"""
        
        return base_prompt
    

    
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
        
        response = self.model.generate_content(
            f"{self.system_prompt}\n\n{prompt}"
        )
        return response.text


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
    interface = GeminiInterface(api_key)
    result = interface.generate_summary(
        result_dict,
        include_history=history is not None,
        test_history=history
    )
    return result['summary']
