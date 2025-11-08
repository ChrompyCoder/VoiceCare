export interface VoiceTest {
  id: string;
  date: string;
  risk_score: number;
  confidence: number;
  risk_level: 'Low' | 'Moderate' | 'High';
  voice_stability_index: number;
  shap_image?: string;
  gemini_summary: string;
  ai_findings: string[];
}

export interface AnalysisResult {
  risk_score: number;
  confidence: number;
  shap_image: string;
  gemini_summary: string;
  voice_stability_index: number;
  ai_findings: string[];
}
