import React from 'react';
import { getRiskColor } from '../utils/api';

interface RiskChipProps {
  riskLevel: 'Low' | 'Moderate' | 'High';
  score: number;
  size?: 'small' | 'large';
}

export const RiskChip: React.FC<RiskChipProps> = ({ riskLevel, score, size = 'large' }) => {
  const color = getRiskColor(riskLevel);
  const sizeClasses = size === 'large' ? 'text-2xl px-8 py-4' : 'text-sm px-4 py-2';

  return (
    <div
      className={`${sizeClasses} rounded-full font-bold text-white shadow-lg inline-flex items-center gap-2`}
      style={{ backgroundColor: color }}
    >
      <span className="text-3xl">●</span>
      {riskLevel} Risk ({(score * 100).toFixed(0)}%)
    </div>
  );
};
