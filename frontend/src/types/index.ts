export interface AcousticFeatures {
  jitter: number;
  shimmer: number;
  hnr: number;
  pitch_variation: number;
  energy_variation: number;
}

export interface VoiceTest {
  id: string;
  date: string;
  risk_score: number;
  confidence: number;
  risk_level: 'Low' | 'Moderate' | 'High';
  voice_stability_index: number;
  gemini_summary: string;
  ai_findings: string[];
  acoustic_features?: AcousticFeatures;
}

export interface AnalysisResult {
  risk_score: number;
  confidence: number;
  gemini_summary: string;
  voice_stability_index: number;
  ai_findings: string[];
  acoustic_features?: AcousticFeatures;
}
