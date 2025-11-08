import React, { useState } from 'react';
import { Mic, TrendingUp } from 'lucide-react';
import { Button } from '../components/Button';
import { useApp } from '../context/AppContext';

export const SplashPage: React.FC = () => {
  const { setCurrentPage } = useApp();
  const [currentSlide, setCurrentSlide] = useState(0);

  const slides = [
    {
      icon: <Mic className="w-16 h-16" />,
      title: 'Record your voice for 5 seconds',
      description: 'Simply speak a sentence and let AI analyze your voice patterns'
    },
    {
      icon: <TrendingUp className="w-16 h-16" />,
      title: 'Get instant AI insights',
      description: 'Receive detailed analysis and a shareable report powered by Gemini AI'
    }
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#FAFAFA] via-[#E8F5E9] to-[#FAFAFA] flex flex-col items-center justify-center p-6 relative overflow-hidden">
      <div className="absolute inset-0 opacity-30">
        <div className="absolute top-20 left-10 w-64 h-64 bg-[#81C784] rounded-full blur-3xl animate-pulse"></div>
        <div className="absolute bottom-20 right-10 w-96 h-96 bg-[#4CAF50] rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }}></div>
      </div>

      <div className="relative z-10 max-w-md w-full">
        <div className="flex items-center justify-center mb-8 animate-bounce">
          <div className="bg-gradient-to-r from-[#2E7D32] to-[#4CAF50] p-6 rounded-full shadow-2xl">
            <Mic className="w-12 h-12 text-white" />
          </div>
        </div>

        <h1 className="text-5xl font-bold text-center mb-4 text-[#263238]">
          VoiceCare AI
        </h1>

        <p className="text-center text-lg text-[#546E7A] mb-12 font-medium">
          AI-powered voice analysis for early Parkinson's detection
        </p>

        <div className="bg-white rounded-2xl shadow-2xl p-8 mb-8 min-h-[280px] flex flex-col justify-between">
          <div className="flex flex-col items-center text-center mb-6">
            <div className="text-[#2E7D32] mb-4 animate-pulse">
              {slides[currentSlide].icon}
            </div>
            <h3 className="text-xl font-semibold text-[#263238] mb-3">
              {slides[currentSlide].title}
            </h3>
            <p className="text-[#546E7A]">
              {slides[currentSlide].description}
            </p>
          </div>

          <div className="flex justify-center gap-2">
            {slides.map((_, idx) => (
              <button
                key={idx}
                onClick={() => setCurrentSlide(idx)}
                className={`w-2 h-2 rounded-full transition-all duration-300 ${
                  idx === currentSlide ? 'bg-[#2E7D32] w-8' : 'bg-[#81C784]'
                }`}
              />
            ))}
          </div>
        </div>

        <Button onClick={() => setCurrentPage('home')} className="w-full text-lg">
          Start Test →
        </Button>

        <p className="text-center text-xs text-[#546E7A] mt-6">
          Powered by Gemini AI + SHAP Explainability
        </p>
      </div>
    </div>
  );
};
