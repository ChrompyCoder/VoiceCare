import React, { useState, useRef } from 'react';
import { ArrowLeft, Loader, Upload } from 'lucide-react';
import { Button } from '../components/Button';
import { useApp } from '../context/AppContext';
import { analyzeVoice } from '../utils/api';

export const UploadPage: React.FC = () => {
  const { setCurrentPage, addTest } = useApp();
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const [fileName, setFileName] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      setFileName(file.name);
      handleAnalysis(file);
    }
  };

  const handleUploadClick = () => {
    fileInputRef.current?.click();
  };

  const handleAnalysis = async (audioFile: File) => {
    setIsAnalyzing(true);
    setProgress(0);

    const progressInterval = setInterval(() => {
      setProgress(prev => {
        if (prev >= 90) {
          clearInterval(progressInterval);
          return 90;
        }
        return prev + 10;
      });
    }, 200);

    try {
      const result = await analyzeVoice(audioFile);
      clearInterval(progressInterval);
      setProgress(100);

      const test = {
        id: result.test_result.id,
        date: result.test_result.date,
        risk_score: result.test_result.risk_score,
        confidence: result.test_result.confidence,
        risk_level: result.test_result.risk_level,
        ai_summary: result.test_result.ai_summary,
        voice_stability_index: 0, // Placeholder
        gemini_summary: result.test_result.ai_summary, // Use AI summary for both
        ai_findings: [] // Placeholder
      };

      addTest(test);

      setTimeout(() => {
        setCurrentPage('results');
      }, 500);
    } catch (error) {
      console.error('Analysis error:', error);
      setIsAnalyzing(false);
      alert('Analysis failed. Please try again.');
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#FAFAFA] via-[#E8F5E9] to-[#FAFAFA] flex flex-col">
      <header className="p-6">
        <Button
          onClick={() => setCurrentPage('home')}
          variant="outline"
          icon={<ArrowLeft className="w-5 h-5" />}
        >
          Back
        </Button>
      </header>

      <div className="flex-1 flex flex-col items-center justify-center p-6">
        <div className="max-w-md w-full text-center">
          {!isAnalyzing ? (
            <>
              <div className="bg-white rounded-2xl shadow-lg p-8 border-2 border-dashed border-gray-300">
                <div className="mb-6">
                  <Upload className="w-16 h-16 text-[#2E7D32] mx-auto" />
                </div>
                <h3 className="text-xl font-semibold text-[#263238] mb-2">
                  Upload Your Voice Recording
                </h3>
                <p className="text-[#546E7A] text-sm mb-6">
                  Please upload a WAV, MP3, or M4A file.
                </p>
                <input
                  type="file"
                  ref={fileInputRef}
                  onChange={handleFileChange}
                  className="hidden"
                  accept="audio/wav, audio/mpeg, audio/mp4, audio/x-m4a"
                />
                <Button onClick={handleUploadClick} size="lg">
                  Choose File
                </Button>
                {fileName && (
                  <p className="text-sm text-gray-500 mt-4">
                    Selected: {fileName}
                  </p>
                )}
              </div>
            </>
          ) : (
            <div className="bg-white rounded-2xl shadow-2xl p-8">
              <div className="mb-6">
                <Loader className="w-16 h-16 text-[#2E7D32] mx-auto animate-spin" />
              </div>
              <h3 className="text-xl font-semibold text-[#263238] mb-4">
                Analyzing your voice with AI...
              </h3>
              <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
                <div
                  className="h-full bg-gradient-to-r from-[#2E7D32] to-[#4CAF50] transition-all duration-300 rounded-full"
                  style={{ width: `${progress}%` }}
                />
              </div>
              <p className="text-sm text-[#546E7A] mt-4">
                Extracting acoustic biomarkers...
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
