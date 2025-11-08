import { AnalysisResult } from '../types';

export const analyzeVoice = async (audioBlob: Blob): Promise<AnalysisResult> => {
  await new Promise(resolve => setTimeout(resolve, 2000));

  const mockResult: AnalysisResult = {
    risk_score: 0.74,
    confidence: 0.82,
    voice_stability_index: 67,
    shap_image: 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==',
    gemini_summary: 'Your voice shows minor irregularities that could be related to fatigue or mild tremor. It\'s recommended to monitor over time and consult a specialist if needed.',
    ai_findings: [
      'Mid-frequency jitter detected',
      'Amplitude shimmer variation moderate',
      'Spectral flatness increased'
    ]
  };

  return mockResult;
};

export const getRiskLevel = (score: number): 'Low' | 'Moderate' | 'High' => {
  if (score < 0.4) return 'Low';
  if (score < 0.7) return 'Moderate';
  return 'High';
};

export const getRiskColor = (level: 'Low' | 'Moderate' | 'High'): string => {
  switch (level) {
    case 'Low': return '#43A047';
    case 'Moderate': return '#FFA000';
    case 'High': return '#E53935';
  }
};
