export interface AcousticFeatures {
  jitter: number;
  shimmer: number;
  hnr: number;
  pitch_variation: number;
  energy_variation: number;
}

export interface SHAPFeature {
  name: string;
  raw_name?: string;
  index: number;
  importance: number;
  value: number;
}

export interface SHAPAnalysis {
  top_features: SHAPFeature[];
  feature_categories: Record<string, SHAPFeature[]>;
  explanation: string;
  visualization_base64?: string;
  heatmap_base64?: string;
}

export interface VoiceTest {
  id: string;
  date: string;
  risk_score: number;
  confidence: number;
  risk_level: 'Low' | 'Moderate' | 'High';
  voice_stability_index: number;
  gemini_summary: string;
  progress_analysis?: string;
  ai_findings: string[];
  acoustic_features?: AcousticFeatures;
  shap_analysis?: SHAPAnalysis;
}

export interface AnalysisResult {
  risk_score: number;
  confidence: number;
  gemini_summary: string;
  progress_analysis?: string;
  voice_stability_index: number;
  ai_findings: string[];
  acoustic_features?: AcousticFeatures;
  shap_analysis?: SHAPAnalysis;
}
