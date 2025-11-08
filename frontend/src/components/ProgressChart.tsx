import React from 'react';
import { VoiceTest } from '../types';

interface ProgressChartProps {
  tests: VoiceTest[];
}

export const ProgressChart: React.FC<ProgressChartProps> = ({ tests }) => {
  if (tests.length === 0) return null;

  // Reverse to show oldest to newest (left to right)
  const chartData = [...tests].reverse().slice(-10); // Show last 10 tests max
  
  const chartWidth = 600;
  const chartHeight = 300;
  const padding = { top: 20, right: 30, bottom: 50, left: 60 };
  const innerWidth = chartWidth - padding.left - padding.right;
  const innerHeight = chartHeight - padding.top - padding.bottom;

  // Calculate positions for line graph
  const points = chartData.map((test, index) => {
    const x = padding.left + (index / Math.max(chartData.length - 1, 1)) * innerWidth;
    const y = padding.top + (1 - test.risk_score) * innerHeight;
    return { x, y, test, index };
  });

  // Create SVG path for the line
  const linePath = points
    .map((point, index) => `${index === 0 ? 'M' : 'L'} ${point.x} ${point.y}`)
    .join(' ');

  // Get risk level color
  const getRiskColor = (score: number) => {
    if (score < 0.4) return '#43A047'; // Low - Green
    if (score < 0.7) return '#FFA000'; // Moderate - Orange
    return '#E53935'; // High - Red
  };

  // Generate Y-axis labels (0% to 100%)
  const yAxisLabels = [0, 0.25, 0.5, 0.75, 1.0];

  return (
    <div className="w-full">
      <div className="relative bg-white rounded-xl p-6 border border-gray-200 shadow-sm">
        <h3 className="text-lg font-semibold text-[#263238] mb-6">
          Risk Score Trend (Last {chartData.length} Test{chartData.length !== 1 ? 's' : ''})
        </h3>
        
        <div className="w-full overflow-x-auto">
          <svg
            width={chartWidth}
            height={chartHeight}
            className="mx-auto"
            style={{ minWidth: '400px' }}
          >
            {/* Y-axis */}
            <line
              x1={padding.left}
              y1={padding.top}
              x2={padding.left}
              y2={chartHeight - padding.bottom}
              stroke="#333"
              strokeWidth="2"
            />

            {/* X-axis */}
            <line
              x1={padding.left}
              y1={chartHeight - padding.bottom}
              x2={chartWidth - padding.right}
              y2={chartHeight - padding.bottom}
              stroke="#333"
              strokeWidth="2"
            />

            {/* Y-axis labels and grid lines */}
            {yAxisLabels.map((value) => {
              const y = padding.top + (1 - value) * innerHeight;
              return (
                <g key={value}>
                  {/* Grid line */}
                  <line
                    x1={padding.left}
                    y1={y}
                    x2={chartWidth - padding.right}
                    y2={y}
                    stroke="#E0E0E0"
                    strokeWidth="1"
                    strokeDasharray="4,4"
                  />
                  {/* Label */}
                  <text
                    x={padding.left - 10}
                    y={y}
                    textAnchor="end"
                    dominantBaseline="middle"
                    className="text-xs fill-gray-600"
                  >
                    {(value * 100).toFixed(0)}%
                  </text>
                </g>
              );
            })}

            {/* Risk zones (background) */}
            <defs>
              <linearGradient id="riskGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#E53935" stopOpacity="0.05" />
                <stop offset="30%" stopColor="#FFA000" stopOpacity="0.05" />
                <stop offset="70%" stopColor="#43A047" stopOpacity="0.05" />
              </linearGradient>
            </defs>
            <rect
              x={padding.left}
              y={padding.top}
              width={innerWidth}
              height={innerHeight}
              fill="url(#riskGradient)"
            />

            {/* Line */}
            <path
              d={linePath}
              fill="none"
              stroke="#2E7D32"
              strokeWidth="3"
              strokeLinecap="round"
              strokeLinejoin="round"
            />

            {/* Data points */}
            {points.map((point) => (
              <g key={point.index}>
                <circle
                  cx={point.x}
                  cy={point.y}
                  r="6"
                  fill={getRiskColor(point.test.risk_score)}
                  stroke="white"
                  strokeWidth="2"
                  className="transition-all hover:r-8"
                  style={{ cursor: 'pointer' }}
                />
                {/* Enhanced tooltip on hover */}
                <title>
                  {`${new Date(point.test.date).toLocaleDateString('en-US', { 
                    month: 'short', 
                    day: 'numeric',
                    year: 'numeric'
                  })}\nRisk Score: ${(point.test.risk_score * 100).toFixed(1)}%\nConfidence: ${(point.test.confidence * 100).toFixed(1)}%\nRisk Level: ${point.test.risk_level}`}
                </title>
              </g>
            ))}

            {/* X-axis labels (dates) - improved spacing */}
            {points.map((point, index) => {
              // Smart label display based on number of points
              let shouldShowLabel = false;
              
              if (chartData.length <= 3) {
                // Show all labels for 3 or fewer points
                shouldShowLabel = true;
              } else if (chartData.length <= 6) {
                // Show first, last, and every other for 4-6 points
                shouldShowLabel = index === 0 || index === points.length - 1 || index % 2 === 1;
              } else {
                // Show first, last, and every third for more points
                shouldShowLabel = index === 0 || index === points.length - 1 || index % 3 === 0;
              }
              
              if (!shouldShowLabel) return null;

              return (
                <text
                  key={`label-${index}`}
                  x={point.x}
                  y={chartHeight - padding.bottom + 20}
                  textAnchor="middle"
                  className="text-xs fill-gray-600"
                >
                  {new Date(point.test.date).toLocaleDateString('en-US', { 
                    month: 'short', 
                    day: 'numeric' 
                  })}
                </text>
              );
            })}

            {/* Axis labels */}
            <text
              x={padding.left / 2}
              y={chartHeight / 2}
              textAnchor="middle"
              transform={`rotate(-90, ${padding.left / 2}, ${chartHeight / 2})`}
              className="text-sm fill-gray-700 font-semibold"
            >
              Risk Score (%)
            </text>

            <text
              x={chartWidth / 2}
              y={chartHeight - 10}
              textAnchor="middle"
              className="text-sm fill-gray-700 font-semibold"
            >
              Test Date
            </text>
          </svg>
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
