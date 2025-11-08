import React, { useState, useEffect } from 'react';
import { Brain } from 'lucide-react';

interface GeminiInsightProps {
  summary: string;
}

export const GeminiInsight: React.FC<GeminiInsightProps> = ({ summary }) => {
  const [displayedText, setDisplayedText] = useState('');
  const [isTyping, setIsTyping] = useState(true);

  useEffect(() => {
    let index = 0;
    const interval = setInterval(() => {
      if (index < summary.length) {
        setDisplayedText(summary.slice(0, index + 1));
        index++;
      } else {
        setIsTyping(false);
        clearInterval(interval);
      }
    }, 20);

    return () => clearInterval(interval);
  }, [summary]);

  return (
    <div className="bg-gradient-to-br from-[#E8F5E9] to-[#F1F8E9] rounded-2xl p-6 shadow-md">
      <div className="flex items-start gap-4">
        <div className="bg-gradient-to-r from-[#2E7D32] to-[#4CAF50] p-3 rounded-full flex-shrink-0">
          <Brain className="w-6 h-6 text-white" />
        </div>
        <div className="flex-1">
          <h3 className="text-lg font-semibold text-[#263238] mb-3">
            Gemini AI Insight
          </h3>
          <p className="text-[#546E7A] leading-relaxed">
            {displayedText}
            {isTyping && <span className="inline-block w-1 h-4 bg-[#2E7D32] ml-1 animate-pulse"></span>}
          </p>
        </div>
      </div>
    </div>
  );
};
