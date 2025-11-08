import React from 'react';

interface ConfidenceMeterProps {
  confidence: number;
}

export const ConfidenceMeter: React.FC<ConfidenceMeterProps> = ({ confidence }) => {
  const percentage = Math.round(confidence * 100);
  const circumference = 2 * Math.PI * 45;
  const offset = circumference - (confidence * circumference);

  return (
    <div className="flex flex-col items-center gap-2">
      <div className="relative w-32 h-32">
        <svg className="transform -rotate-90 w-32 h-32">
          <circle
            cx="64"
            cy="64"
            r="45"
            stroke="#E0E0E0"
            strokeWidth="8"
            fill="none"
          />
          <circle
            cx="64"
            cy="64"
            r="45"
            stroke="url(#gradient)"
            strokeWidth="8"
            fill="none"
            strokeDasharray={circumference}
            strokeDashoffset={offset}
            strokeLinecap="round"
            className="transition-all duration-1000 ease-out"
          />
          <defs>
            <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#2E7D32" />
              <stop offset="100%" stopColor="#4CAF50" />
            </linearGradient>
          </defs>
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <span className="text-2xl font-bold text-[#263238]">{percentage}%</span>
        </div>
      </div>
      <p className="text-sm text-[#546E7A] font-medium">Model Confidence</p>
    </div>
  );
};
