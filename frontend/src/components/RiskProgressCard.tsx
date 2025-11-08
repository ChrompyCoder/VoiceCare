import React, { useState } from 'react';
import { TrendingUp, ChevronDown, ChevronUp, Info } from 'lucide-react';
import { Card } from './Card';
import { ProgressChart } from './ProgressChart';
import { VoiceTest } from '../types';

interface RiskProgressCardProps {
  tests: VoiceTest[];
  progressAnalysis?: string;
  chartRef: React.RefObject<HTMLDivElement>;
}

export const RiskProgressCard: React.FC<RiskProgressCardProps> = ({ tests, progressAnalysis, chartRef }) => {
  const [open, setOpen] = useState(false);

  return (
    <Card>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="bg-gradient-to-r from-[#6366F1] to-[#8B5CF6] p-3 rounded-full">
            <TrendingUp className="w-6 h-6 text-white" />
          </div>
          <div>
            <h3 className="text-xl font-bold text-[#263238]">Risk Score & Progress</h3>
            <p className="text-sm text-[#546E7A]">Risk trend and progression</p>
          </div>
        </div>
        <button
          onClick={() => setOpen(!open)}
          className="p-2 hover:bg-gray-100 rounded-full transition-colors"
          aria-label={open ? 'Collapse' : 'Expand'}
        >
          {open ? (
            <ChevronUp className="w-5 h-5 text-[#546E7A]" />
          ) : (
            <ChevronDown className="w-5 h-5 text-[#546E7A]" />
          )}
        </button>
      </div>

      {/* Info Banner to match SHAP look */}
      <div className="bg-blue-50 border-l-4 border-blue-400 p-3 mb-4 rounded">
        <div className="flex items-start gap-2">
          <Info className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
          <p className="text-sm text-blue-900">
            This shows how your risk score changes over time. Lower scores and downward trends usually indicate improvement; 
            consistently high scores suggest discussing results with a healthcare professional.
          </p>
        </div>
      </div>

      {open && (
        <div className="mt-2 space-y-4 animate-fadeIn">
          {tests.length > 1 && (
            <div ref={chartRef}>
              <ProgressChart tests={tests} />
            </div>
          )}
          {progressAnalysis && (
            <div className="bg-gradient-to-r from-purple-50 to-blue-50 p-4 rounded-lg">
              <p className="text-sm text-[#263238] leading-relaxed whitespace-pre-wrap">{progressAnalysis}</p>
            </div>
          )}
        </div>
      )}
    </Card>
  );
};
