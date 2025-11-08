import { AnalysisResult } from '../types';

const API_URL = 'http://localhost:5000/predict';

export const analyzeVoice = async (audioFile: File): Promise<any> => {
  const formData = new FormData();
  formData.append('audio', audioFile);

  try {
    const response = await fetch(API_URL, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || 'Analysis failed');
    }

    const result = await response.json();
    return result;
  } catch (error) {
    console.error('Error calling analysis API:', error);
    throw error;
  }
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
