import React from 'react';
import { VoiceTest } from '../types';

interface ProgressChartProps {
  tests: VoiceTest[];
}

export const ProgressChart: React.FC<ProgressChartProps> = ({ tests }) => {
  if (tests.length === 0) return null;

  // Reverse to show oldest to newest (left to right)
  const chartData = [...tests].reverse().slice(-10); // Show last 10 tests max
  const maxScore = 1.0;
  const chartHeight = 200;

  // Calculate positions
  const points = chartData.map((test, index) => {
    const x = (index / (chartData.length - 1 || 1)) * 100;
    const y = (1 - (test.risk_score / maxScore)) * chartHeight;
    return { x, y, test };
  });

  // Create SVG path for the line
  const linePath = points
    .map((point, index) => `${index === 0 ? 'M' : 'L'} ${point.x}% ${point.y}`)
    .join(' ');

  // Create area fill path
  const areaPath = `${linePath} L ${points[points.length - 1].x}% ${chartHeight} L 0% ${chartHeight} Z`;

  // Get risk level color
  const getRiskColor = (score: number) => {
    if (score < 0.4) return '#43A047'; // Low - Green
    if (score < 0.7) return '#FFA000'; // Moderate - Orange
    return '#E53935'; // High - Red
  };

  return (
    <div className="w-full">
      <div className="relative bg-gradient-to-br from-gray-50 to-gray-100 rounded-xl p-6 border border-gray-200">
        <h3 className="text-lg font-semibold text-[#263238] mb-4">
          Risk Score Trend (Last {chartData.length} Tests)
        </h3>
        
        <div className="relative" style={{ height: `${chartHeight}px` }}>
          {/* Y-axis labels */}
          <div className="absolute left-0 top-0 bottom-0 w-12 flex flex-col justify-between text-xs text-gray-500">
            <span>High</span>
            <span>Moderate</span>
            <span>Low</span>
          </div>

          {/* Chart area */}
          <div className="absolute left-14 right-0 top-0 bottom-0">
            <svg
              className="w-full h-full"
              viewBox={`0 0 100 ${chartHeight}`}
              preserveAspectRatio="none"
            >
              {/* Background grid lines */}
              <line x1="0" y1={chartHeight * 0.3} x2="100%" y2={chartHeight * 0.3} stroke="#E0E0E0" strokeWidth="0.5" strokeDasharray="2,2" />
              <line x1="0" y1={chartHeight * 0.7} x2="100%" y2={chartHeight * 0.7} stroke="#E0E0E0" strokeWidth="0.5" strokeDasharray="2,2" />

              {/* Area fill with gradient */}
              <defs>
                <linearGradient id="areaGradient" x1="0%" y1="0%" x2="0%" y2="100%">
                  <stop offset="0%" stopColor="#E53935" stopOpacity="0.3" />
                  <stop offset="50%" stopColor="#FFA000" stopOpacity="0.2" />
                  <stop offset="100%" stopColor="#43A047" stopOpacity="0.1" />
                </linearGradient>
              </defs>
              <path
                d={areaPath}
                fill="url(#areaGradient)"
              />

              {/* Line */}
              <path
                d={linePath}
                fill="none"
                stroke="#2E7D32"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              />

              {/* Data points */}
              {points.map((point, index) => (
                <g key={index}>
                  <circle
                    cx={`${point.x}%`}
                    cy={point.y}
                    r="4"
                    fill={getRiskColor(point.test.risk_score)}
                    stroke="white"
                    strokeWidth="2"
                    className="hover:r-6 transition-all cursor-pointer"
                  />
                </g>
              ))}
            </svg>
          </div>
        </div>

        {/* X-axis labels (dates) */}
        <div className="flex justify-between mt-4 text-xs text-gray-500 ml-14">
          <span>{new Date(chartData[0].date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}</span>
          {chartData.length > 1 && (
            <span>{new Date(chartData[chartData.length - 1].date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}</span>
          )}
        </div>

        {/* Legend */}
        <div className="flex flex-wrap gap-4 mt-6 justify-center">
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-[#43A047]"></div>
            <span className="text-xs text-gray-600">Low Risk (&lt;40%)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-[#FFA000]"></div>
            <span className="text-xs text-gray-600">Moderate (40-70%)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-3 h-3 rounded-full bg-[#E53935]"></div>
            <span className="text-xs text-gray-600">High Risk (&gt;70%)</span>
          </div>
        </div>

        {/* Stats summary */}
        <div className="grid grid-cols-3 gap-4 mt-6 pt-4 border-t border-gray-200">
          <div className="text-center">
            <div className="text-2xl font-bold text-[#263238]">
              {(chartData[chartData.length - 1].risk_score * 100).toFixed(0)}%
            </div>
            <div className="text-xs text-gray-500">Latest Score</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-[#263238]">
              {((chartData.reduce((sum, t) => sum + t.risk_score, 0) / chartData.length) * 100).toFixed(0)}%
            </div>
            <div className="text-xs text-gray-500">Average</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-[#263238]">
              {chartData.length > 1 
                ? `${((chartData[0].risk_score - chartData[chartData.length - 1].risk_score) * 100).toFixed(0)}%`
                : 'N/A'
              }
            </div>
            <div className="text-xs text-gray-500">Change</div>
          </div>
        </div>
      </div>
    </div>
  );
};
