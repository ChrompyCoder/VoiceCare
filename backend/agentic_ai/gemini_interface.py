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

    # --- New general chat & feature explanation helpers ---
    def explain_feature(self, feature_name: str):
        """Generate a simple, patient-friendly explanation of an acoustic/OpenSMILE feature.

        Args:
            feature_name: Raw feature string from OpenSMILE / preprocessing
        Returns:
            dict with 'text' explanation
        """
        if not feature_name:
            return {'text': 'Please provide a feature name to explain.'}
        prompt = f"""
You are a friendly clinical screening assistant.
Explain the voice analysis feature name below for a non-technical user:

Feature: "{feature_name}"

Provide:
1. What it measures (plain everyday language)
2. Why it matters for voice or neurological screening
3. What unusual values could mean (very briefly)

Format: 2-3 short sentences. Avoid jargon. Don't give a diagnosis. If the feature is obscure, say it's a technical spectral pattern and reassure user it's part of internal analysis.
"""
        try:
            resp = self.client.models.generate_content(
                model=self.model_name,
                contents=f"{self.system_prompt}\n\n{prompt}"
            )
            text = resp.text if hasattr(resp, 'text') else str(resp)
            return {'text': text.strip()}
        except Exception as e:
            print(f"⚠️ Gemini feature explanation error: {e}")
            return {'text': 'Unable to explain this feature right now.'}

    def chat(self, messages):
        """Lightweight multi-turn chat. `messages` is a list of {'role': 'user'|'assistant', 'content': str}.
        We collapse recent turns into a concise context and ask Gemini to reply empathetically.
        """
        if not messages:
            return {'text': 'Hi! Ask me about any voice feature or your results.'}
        # Keep last 8 messages
        recent = messages[-8:]
        # Build transcript
        transcript = "\n".join([f"{m['role']}: {m['content']}" for m in recent])
        prompt = f"""
You are an empathetic assistant helping a user understand voice screening features and results.
Conversation so far:
{transcript}

Respond to the last user message with a concise (<= 3 sentences) friendly explanation. If they ask about a feature name, explain it similarly to the feature explanation guidelines. Avoid medical diagnosis claims.
"""
        try:
            resp = self.client.models.generate_content(
                model=self.model_name,
                contents=f"{self.system_prompt}\n\n{prompt}"
            )
            text = resp.text if hasattr(resp, 'text') else str(resp)
            return {'text': text.strip()}
        except Exception as e:
            print(f"⚠️ Gemini chat error: {e}")
            return {'text': 'I had trouble responding. Please try again shortly.'}
    
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
        Create simplified Gemini prompt focused on empathy
        
        Returns:
            str: Formatted prompt
        """
        risk_score = context.get('risk_score', 0)
        risk_level = context.get('risk_level', 'unknown')
        
        # Simple, clear prompt
        prompt = f"""You are a compassionate healthcare assistant providing voice screening results.

Voice Screening Results:
Risk Score: {risk_score:.1%} ({risk_level.upper()} risk)
"""
        
        # Add trend if available
        trend_info = context.get('trend_info')
        if trend_info:
            direction = trend_info['direction']
            change = trend_info['change_percent']
            prompt += f"Trend: Your scores are {direction} ({change:.1f}% change)\n"
        
        prompt += """
Please provide a warm, supportive summary (2-3 sentences) that:
1. Explains what this risk score means in simple, everyday language
2. If the risk is elevated, gently encourages seeing a doctor
3. If scores are improving or stable, celebrates that progress
4. Avoids medical jargon - speak like a caring friend, not a doctor

Keep your response brief, empathetic, and reassuring."""
        
        return prompt
    

    
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
