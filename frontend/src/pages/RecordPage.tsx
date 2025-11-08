import React, { useState, useRef, useEffect } from 'react';
import { Mic, ArrowLeft, Loader } from 'lucide-react';
import { Button } from '../components/Button';
import { useApp } from '../context/AppContext';
import { analyzeVoice, getRiskLevel } from '../utils/api';

export const RecordPage: React.FC = () => {
  const { setCurrentPage, addTest } = useApp();
  const [isRecording, setIsRecording] = useState(false);
  const [timer, setTimer] = useState(0);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [progress, setProgress] = useState(0);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  useEffect(() => {
    let interval: NodeJS.Timeout;
    if (isRecording && timer < 5) {
      interval = setInterval(() => {
        setTimer(prev => {
          if (prev >= 4) {
            stopRecording();
            return 5;
          }
          return prev + 1;
        });
      }, 1000);
    }
    return () => clearInterval(interval);
  }, [isRecording, timer]);

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      chunksRef.current = [];

      mediaRecorder.ondataavailable = (e) => {
        chunksRef.current.push(e.data);
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(chunksRef.current, { type: 'audio/wav' });
        stream.getTracks().forEach(track => track.stop());
        await handleAnalysis(audioBlob);
      };

      mediaRecorder.start();
      setIsRecording(true);
      setTimer(0);
    } catch (error) {
      console.error('Error accessing microphone:', error);
      alert('Could not access microphone. Please grant permission.');
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const handleAnalysis = async (audioBlob: Blob) => {
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
      const result = await analyzeVoice(audioBlob);
      clearInterval(progressInterval);
      setProgress(100);

      const test = {
        id: `VPX-${Date.now()}`,
        date: new Date().toISOString(),
        risk_score: result.risk_score,
        confidence: result.confidence,
        risk_level: getRiskLevel(result.risk_score),
        voice_stability_index: result.voice_stability_index,
        shap_image: result.shap_image,
        gemini_summary: result.gemini_summary,
        ai_findings: result.ai_findings
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

      <div className="flex-1 flex flex-col items-center justify-center p-6 relative">
        <div className="absolute inset-0 flex items-center justify-center">
          {isRecording && (
            <>
              <div className="absolute w-64 h-64 bg-[#2E7D32] rounded-full opacity-20 animate-ping"></div>
              <div className="absolute w-96 h-96 bg-[#4CAF50] rounded-full opacity-10 animate-ping" style={{ animationDelay: '0.5s' }}></div>
              <div className="absolute w-[32rem] h-[32rem] bg-[#81C784] rounded-full opacity-5 animate-ping" style={{ animationDelay: '1s' }}></div>
            </>
          )}
        </div>

        <div className="relative z-10 max-w-md w-full text-center">
          {!isAnalyzing ? (
            <>
              <div className="mb-8">
                <div className="text-6xl font-bold text-[#263238] mb-2">
                  {timer < 5 ? `0${timer}` : '05'} / 05
                </div>
                <p className="text-[#546E7A] font-medium">seconds</p>
              </div>

              <div className="mb-12">
                <button
                  onClick={isRecording ? stopRecording : startRecording}
                  disabled={timer >= 5}
                  className={`w-40 h-40 rounded-full shadow-2xl flex items-center justify-center transition-all duration-300 ${
                    isRecording
                      ? 'bg-gradient-to-r from-[#E53935] to-[#C62828] scale-110'
                      : 'bg-gradient-to-r from-[#2E7D32] to-[#4CAF50] hover:scale-110'
                  } ${timer >= 5 ? 'opacity-50' : ''}`}
                >
                  <Mic className="w-16 h-16 text-white" />
                </button>
              </div>

              <div className="bg-white rounded-2xl shadow-lg p-6 mb-6">
                <p className="text-lg text-[#263238] font-medium mb-2">
                  {isRecording ? 'Recording...' : 'Tap the microphone to start'}
                </p>
                <p className="text-[#546E7A] text-sm">
                  Speak this sentence clearly:
                </p>
                <p className="text-[#2E7D32] font-semibold mt-2 text-lg">
                  "Today is a bright sunny day."
                </p>
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
