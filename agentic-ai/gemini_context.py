"""
Gemini Context Engine
Provides history-aware, longitudinal analysis for personalized insights
"""

import json
from datetime import datetime, timedelta
import config

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class GeminiContextEngine:
    """
    Manages context-aware Gemini interactions with test history
    """
    
    def __init__(self, api_key=None):
        """Initialize context engine"""
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai package not installed")
        
        self.api_key = api_key or config.GEMINI_API_KEY
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(config.GEMINI_MODEL)
    
    def analyze_progress(self, test_history, current_result):
        """
        Analyze progress over time with full context awareness
        
        Args:
            test_history: List of previous test results
            current_result: Latest test result
            
        Returns:
            dict: Comprehensive progress analysis
        """
        if len(test_history) < 1:
            return {
                'summary': 'This is your first test. Continue testing regularly to track trends.',
                'trend': 'baseline',
                'recommendations': ['Test again in 1-2 weeks to establish a baseline']
            }
        
        # Build comprehensive context
        context = self._build_longitudinal_context(test_history, current_result)
        
        # Generate analysis
        analysis = self._generate_context_aware_analysis(context)
        
        return analysis
    
    def _build_longitudinal_context(self, test_history, current_result):
        """
        Build comprehensive longitudinal context
        
        Returns:
            dict: Full context with trends and patterns
        """
        # Sort by date
        sorted_history = sorted(
            test_history,
            key=lambda x: x.get('date', ''),
            reverse=True
        )
        
        # Extract recent tests (last 3)
        recent_tests = sorted_history[:3]
        
        # Calculate statistics
        risk_scores = [t.get('risk_score', 0) for t in sorted_history]
        avg_risk = sum(risk_scores) / len(risk_scores) if risk_scores else 0
        
        # Identify trend
        trend = self._calculate_detailed_trend(sorted_history)
        
        # Time analysis
        time_analysis = self._analyze_time_intervals(sorted_history)
        
        # Feature evolution
        feature_evolution = self._track_feature_evolution(sorted_history)
        
        context = {
            'current_result': {
                'date': current_result.get('date', datetime.now().isoformat()),
                'risk_score': round(current_result.get('risk_score', 0), 3),
                'risk_level': current_result.get('risk_level', 'Unknown'),
                'confidence': round(current_result.get('confidence', 0), 3)
            },
            'recent_history': [
                {
                    'date': t.get('date', ''),
                    'risk_score': round(t.get('risk_score', 0), 3),
                    'risk_level': t.get('risk_level', '')
                }
                for t in recent_tests
            ],
            'statistics': {
                'total_tests': len(sorted_history) + 1,
                'average_risk': round(avg_risk, 3),
                'min_risk': round(min(risk_scores), 3) if risk_scores else 0,
                'max_risk': round(max(risk_scores), 3) if risk_scores else 0
            },
            'trend_analysis': trend,
            'time_analysis': time_analysis,
            'feature_evolution': feature_evolution
        }
        
        return context
    
    def _calculate_detailed_trend(self, test_history):
        """Calculate detailed trend analysis"""
        if len(test_history) < 2:
            return {'direction': 'insufficient_data', 'magnitude': 0}
        
        scores = [t.get('risk_score', 0) for t in test_history[:5]]  # Last 5 tests
        
        # Calculate trend
        first_avg = sum(scores[-2:]) / 2 if len(scores) >= 2 else scores[-1]
        last_avg = sum(scores[:2]) / 2 if len(scores) >= 2 else scores[0]
        
        change = last_avg - first_avg
        change_pct = (change / first_avg * 100) if first_avg > 0 else 0
        
        # Determine direction
        if abs(change_pct) < 5:
            direction = 'stable'
        elif change_pct < -5:
            direction = 'improving'
        else:
            direction = 'worsening'
        
        return {
            'direction': direction,
            'magnitude': round(abs(change_pct), 1),
            'change': round(change, 3)
        }
    
    def _analyze_time_intervals(self, test_history):
        """Analyze testing frequency and consistency"""
        if len(test_history) < 2:
            return {'consistency': 'insufficient_data'}
        
        try:
            dates = [
                datetime.fromisoformat(t.get('date', '').replace('Z', '+00:00'))
                for t in test_history[:5]
                if t.get('date')
            ]
            
            if len(dates) < 2:
                return {'consistency': 'insufficient_data'}
            
            # Calculate intervals
            intervals = []
            for i in range(len(dates) - 1):
                delta = abs((dates[i] - dates[i+1]).days)
                intervals.append(delta)
            
            avg_interval = sum(intervals) / len(intervals)
            
            # Determine consistency
            if avg_interval <= 7:
                consistency = 'excellent'
            elif avg_interval <= 14:
                consistency = 'good'
            elif avg_interval <= 30:
                consistency = 'moderate'
            else:
                consistency = 'irregular'
            
            return {
                'consistency': consistency,
                'average_interval_days': round(avg_interval, 1),
                'last_test_days_ago': (datetime.now() - dates[0]).days
            }
        except Exception as e:
            print(f"Error analyzing time intervals: {e}")
            return {'consistency': 'error'}
    
    def _track_feature_evolution(self, test_history):
        """Track how specific features have evolved"""
        if not test_history:
            return {}
        
        # Extract voice stability scores if available
        stability_scores = [
            t.get('voice_stability_index', 0)
            for t in test_history[:5]
            if 'voice_stability_index' in t
        ]
        
        if not stability_scores:
            return {}
        
        avg_stability = sum(stability_scores) / len(stability_scores)
        stability_trend = 'improving' if stability_scores[0] > stability_scores[-1] else 'declining'
        
        return {
            'voice_stability': {
                'average': round(avg_stability, 3),
                'trend': stability_trend,
                'recent': round(stability_scores[0], 3) if stability_scores else 0
            }
        }
    
    def _generate_context_aware_analysis(self, context):
        """
        Generate context-aware analysis using Gemini
        
        Returns:
            dict: Analysis with summary and recommendations
        """
        context_json = json.dumps(context, indent=2)
        
        prompt = f"""
You are analyzing voice health screening results with full historical context.

Context (JSON):
{context_json}

Task: Provide a comprehensive yet empathetic progress analysis.

Include:
1. Overall trend interpretation (improving/stable/concerning)
2. Key observations from the data
3. Personalized recommendations based on the trend
4. Encouragement and next steps

Guidelines:
- Reference specific data points when relevant
- Acknowledge progress if improvement is seen
- Be empathetic and supportive
- Focus on patterns over time, not single readings
- Suggest actionable next steps
- Keep total response to 4-5 sentences
- DO NOT diagnose or provide medical advice
"""
        
        try:
            response = self.model.generate_content(prompt)
            summary_text = response.text
            
            # Extract recommendations (simple parsing)
            recommendations = self._extract_recommendations(summary_text, context)
            
            return {
                'summary': summary_text,
                'trend': context['trend_analysis']['direction'],
                'recommendations': recommendations,
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            print(f"Error generating context-aware analysis: {e}")
            return self._fallback_analysis(context)
    
    def _extract_recommendations(self, summary_text, context):
        """Extract actionable recommendations"""
        recommendations = []
        
        trend_dir = context['trend_analysis']['direction']
        consistency = context['time_analysis'].get('consistency', 'unknown')
        
        # Trend-based recommendations
        if trend_dir == 'improving':
            recommendations.append('Continue current health practices')
            recommendations.append('Maintain regular testing schedule')
        elif trend_dir == 'worsening':
            recommendations.append('Consider consulting a healthcare professional')
            recommendations.append('Monitor symptoms closely')
        else:  # stable
            recommendations.append('Keep up consistent testing')
        
        # Consistency-based recommendations
        if consistency in ['irregular', 'moderate']:
            recommendations.append('Test more regularly for better trend analysis')
        
        return recommendations[:3]  # Top 3 recommendations
    
    def _fallback_analysis(self, context):
        """Fallback analysis if Gemini fails"""
        trend_dir = context['trend_analysis']['direction']
        magnitude = context['trend_analysis']['magnitude']
        
        if trend_dir == 'improving':
            summary = (
                f"Your voice health shows positive progress with a {magnitude}% improvement. "
                f"You've completed {context['statistics']['total_tests']} tests. "
                "Keep up the consistent monitoring."
            )
        elif trend_dir == 'stable':
            summary = (
                f"Your voice patterns have remained stable across {context['statistics']['total_tests']} tests. "
                "Consistency in results is good. Continue regular monitoring."
            )
        else:
            summary = (
                f"Your recent tests show some variation ({magnitude}% change). "
                "Consider scheduling a consultation with a healthcare professional for evaluation."
            )
        
        return {
            'summary': summary,
            'trend': trend_dir,
            'recommendations': ['Continue regular testing', 'Consult healthcare provider if concerned'],
            'timestamp': datetime.now().isoformat()
        }


# Utility function
def analyze_with_context(test_history, current_result, api_key=None):
    """
    Quick utility for context-aware analysis
    
    Args:
        test_history: List of previous tests
        current_result: Current test result
        api_key: Optional Gemini API key
        
    Returns:
        dict: Analysis results
    """
    if not GEMINI_AVAILABLE:
        print("Warning: Gemini not available")
        return {
            'summary': 'Context analysis unavailable. Install google-generativeai package.',
            'trend': 'unknown',
            'recommendations': []
        }
    
    engine = GeminiContextEngine(api_key)
    return engine.analyze_progress(test_history, current_result)
