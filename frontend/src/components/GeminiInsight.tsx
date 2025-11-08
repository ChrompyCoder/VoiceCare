import React, { useState, useEffect } from 'react';
import { Brain, TrendingUp } from 'lucide-react';

interface GeminiInsightProps {
  summary: string;
  progressAnalysis?: string;
}

export const GeminiInsight: React.FC<GeminiInsightProps> = ({ summary, progressAnalysis }) => {
  const [displayedSummary, setDisplayedSummary] = useState('');
  const [displayedProgress, setDisplayedProgress] = useState('');
  const [isTypingSummary, setIsTypingSummary] = useState(true);
  const [isTypingProgress, setIsTypingProgress] = useState(false);

  useEffect(() => {
    // Type out AI Summary first
    let index = 0;
    const interval = setInterval(() => {
      if (index < summary.length) {
        setDisplayedSummary(summary.slice(0, index + 1));
        index++;
      } else {
        setIsTypingSummary(false);
        clearInterval(interval);
        
        // Start typing progress analysis after summary is done
        if (progressAnalysis) {
          setIsTypingProgress(true);
          let progressIndex = 0;
          const progressInterval = setInterval(() => {
            if (progressIndex < progressAnalysis.length) {
              setDisplayedProgress(progressAnalysis.slice(0, progressIndex + 1));
              progressIndex++;
            } else {
              setIsTypingProgress(false);
              clearInterval(progressInterval);
            }
          }, 15);
        }
      }
    }, 20);

    return () => clearInterval(interval);
  }, [summary, progressAnalysis]);

  return (
    <div className="space-y-4">
      {/* AI Summary */}
      <div className="bg-gradient-to-br from-[#E8F5E9] to-[#F1F8E9] rounded-2xl p-6 shadow-md">
        <div className="flex items-start gap-4">
          <div className="bg-gradient-to-r from-[#2E7D32] to-[#4CAF50] p-3 rounded-full flex-shrink-0">
            <Brain className="w-6 h-6 text-white" />
          </div>
          <div className="flex-1">
            <h3 className="text-lg font-semibold text-[#263238] mb-3">
              AI Summary
            </h3>
            <p className="text-[#546E7A] leading-relaxed">
              {displayedSummary}
              {isTypingSummary && <span className="inline-block w-1 h-4 bg-[#2E7D32] ml-1 animate-pulse"></span>}
            </p>
          </div>
        </div>
      </div>

      {/* Progress Analysis */}
      {progressAnalysis && (
        <div className="bg-gradient-to-br from-[#E3F2FD] to-[#F3E5F5] rounded-2xl p-6 shadow-md">
          <div className="flex items-start gap-4">
            <div className="bg-gradient-to-r from-[#1565C0] to-[#7B1FA2] p-3 rounded-full flex-shrink-0">
              <TrendingUp className="w-6 h-6 text-white" />
            </div>
            <div className="flex-1">
              <h3 className="text-lg font-semibold text-[#263238] mb-3">
                Progress Analysis
              </h3>
              <p className="text-[#546E7A] leading-relaxed">
                {displayedProgress}
                {isTypingProgress && <span className="inline-block w-1 h-4 bg-[#1565C0] ml-1 animate-pulse"></span>}
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
