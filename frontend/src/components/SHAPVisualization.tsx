import React, { useState } from 'react';
import { BarChart3, ChevronDown, ChevronUp, Info } from 'lucide-react';
import { Card } from './Card';
import { SHAPAnalysis } from '../types';

interface SHAPVisualizationProps {
  shapAnalysis: SHAPAnalysis;
}

export const SHAPVisualization: React.FC<SHAPVisualizationProps> = ({ shapAnalysis }) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const [showWaterfall, setShowWaterfall] = useState(false);
  const [showHeatmap, setShowHeatmap] = useState(false);

  if (!shapAnalysis || !shapAnalysis.top_features || shapAnalysis.top_features.length === 0) {
    return null;
  }

  const topFeatures = shapAnalysis.top_features.slice(0, 5);
  const maxImportance = Math.max(...topFeatures.map(f => f.importance));

  return (
    <Card>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="bg-gradient-to-r from-[#6366F1] to-[#8B5CF6] p-3 rounded-full">
            <BarChart3 className="w-6 h-6 text-white" />
          </div>
          <div>
            <h3 className="text-xl font-bold text-[#263238]">SHAP Feature Analysis</h3>
            <p className="text-sm text-[#546E7A]">Model explainability & feature importance</p>
          </div>
        </div>
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="p-2 hover:bg-gray-100 rounded-full transition-colors"
          aria-label={isExpanded ? "Collapse" : "Expand"}
        >
          {isExpanded ? (
            <ChevronUp className="w-5 h-5 text-[#546E7A]" />
          ) : (
            <ChevronDown className="w-5 h-5 text-[#546E7A]" />
          )}
        </button>
      </div>

      {/* Info Banner */}
      <div className="bg-blue-50 border-l-4 border-blue-400 p-3 mb-4 rounded">
        <div className="flex items-start gap-2">
          <Info className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
          <p className="text-sm text-blue-900">
            <strong>What is SHAP?</strong> SHAP (SHapley Additive exPlanations) shows which voice features 
            contributed most to your risk prediction, making the AI's decision transparent and interpretable.
          </p>
        </div>
      </div>

      {/* Explanation Text */}
      {shapAnalysis.explanation && (
        <div className="bg-gradient-to-r from-purple-50 to-blue-50 p-4 rounded-lg mb-4">
          <p className="text-sm text-[#263238] leading-relaxed">
            {shapAnalysis.explanation}
          </p>
        </div>
      )}

      {/* Top 5 Features */}
      <div className="space-y-3">
        <h4 className="font-semibold text-[#263238] flex items-center gap-2">
          <span className="w-1 h-4 bg-gradient-to-b from-[#6366F1] to-[#8B5CF6] rounded"></span>
          Top 5 Contributing Features
        </h4>
        
        {topFeatures.map((feature, index) => {
          const percentage = (feature.importance / maxImportance) * 100;
          const isPositive = feature.importance > 0;
          
          return (
            <div key={index} className="space-y-1">
              <div className="flex justify-between items-center">
                <span className="text-sm font-medium text-[#263238] truncate max-w-[70%]" title={feature.raw_name || feature.name}>
                  {index + 1}. {feature.name}
                  {feature.raw_name && feature.raw_name !== feature.name && (
                    <span className="text-xs text-[#546E7A] ml-2">({feature.raw_name})</span>
                  )}
                </span>
                <span className="text-xs font-mono text-[#546E7A]">
                  {feature.importance.toFixed(4)}
                </span>
              </div>
              <div className="relative h-6 bg-gray-100 rounded-full overflow-hidden">
                <div
                  className={`absolute top-0 left-0 h-full rounded-full transition-all duration-500 ${
                    isPositive
                      ? 'bg-gradient-to-r from-purple-500 to-indigo-500'
                      : 'bg-gradient-to-r from-green-500 to-teal-500'
                  }`}
                  style={{ width: `${percentage}%` }}
                >
                  <div className="absolute inset-0 bg-white opacity-20"></div>
                </div>
                <div className="absolute inset-0 flex items-center px-3">
                  <span className="text-xs font-semibold text-[#263238]">
                    Value: {feature.value.toFixed(4)}
                  </span>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Expanded Content */}
      {isExpanded && (
        <div className="mt-6 space-y-4 animate-fadeIn">
          {/* Contribution Plot Toggle (bar chart) */}
          {shapAnalysis.visualization_base64 && (
            <div>
              <button
                onClick={() => setShowWaterfall(!showWaterfall)}
                className="w-full bg-gradient-to-r from-[#6366F1] to-[#8B5CF6] text-white py-3 rounded-lg 
                         hover:shadow-lg transition-all duration-300 font-medium flex items-center justify-center gap-2"
              >
                <BarChart3 className="w-5 h-5" />
                {showWaterfall ? 'Hide' : 'Show'} Feature Contribution Plot
              </button>

              {showWaterfall && (
                <div className="mt-4 bg-white p-4 rounded-lg border border-gray-200 shadow-inner">
                  <img
                    src={shapAnalysis.visualization_base64}
                    alt="Feature Contribution Plot"
                    className="w-full h-auto rounded"
                  />
                  <p className="text-xs text-[#546E7A] mt-2 text-center">
                    Bar chart showing how each feature pushes the prediction higher or lower
                  </p>
                </div>
              )}
            </div>
          )}

          {/* Feature Categories */}
          {shapAnalysis.feature_categories && Object.keys(shapAnalysis.feature_categories).length > 0 && (
            <div>
              <h4 className="font-semibold text-[#263238] mb-3">Feature Categories</h4>
              <div className="grid grid-cols-2 gap-3">
                {Object.entries(shapAnalysis.feature_categories).map(([category, features]) => (
                  <div key={category} className="bg-gradient-to-br from-gray-50 to-gray-100 p-3 rounded-lg">
                    <div className="text-xs font-semibold text-[#546E7A] uppercase tracking-wide mb-1">
                      {category}
                    </div>
                    <div className="text-2xl font-bold text-[#263238]">
                      {Array.isArray(features) ? features.length : 0}
                    </div>
                    <div className="text-xs text-[#546E7A]">features</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Heatmap Toggle (prefer base64; fallback to path) */}
          {(shapAnalysis.heatmap_base64 || (shapAnalysis as any).heatmap_path) && (
            <div>
              <button
                onClick={() => setShowHeatmap(!showHeatmap)}
                className="w-full mt-3 bg-gradient-to-r from-[#06b6d4] to-[#0ea5e9] text-white py-3 rounded-lg 
                         hover:shadow-lg transition-all duration-300 font-medium flex items-center justify-center gap-2"
              >
                <BarChart3 className="w-5 h-5" />
                {showHeatmap ? 'Hide' : 'Show'} Feature Heatmap
              </button>

              {showHeatmap && (
                <div className="mt-4 bg-white p-4 rounded-lg border border-gray-200 shadow-inner">
                  {shapAnalysis.heatmap_base64 ? (
                    <img
                      src={shapAnalysis.heatmap_base64}
                      alt="SHAP Feature Heatmap"
                      className="w-full h-auto rounded"
                    />
                  ) : (
                    <img
                      src={(shapAnalysis as any).heatmap_path}
                      alt="SHAP Feature Heatmap"
                      className="w-full h-auto rounded"
                    />
                  )}
                  <p className="text-xs text-[#546E7A] mt-2 text-center">
                    Heatmap of top features: value (left) and contribution (right)
                  </p>
                </div>
              )}
            </div>
          )}

          {/* Interpretation Guide */}
          <div className="bg-gradient-to-r from-amber-50 to-orange-50 p-4 rounded-lg border border-amber-200">
            <h4 className="font-semibold text-amber-900 mb-2 flex items-center gap-2">
              <Info className="w-4 h-4" />
              How to Interpret
            </h4>
            <ul className="text-sm text-amber-900 space-y-1">
              <li className="flex items-start gap-2">
                <span className="text-purple-600 font-bold">•</span>
                <span>Higher importance values = stronger influence on prediction</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-green-600 font-bold">•</span>
                <span>Each feature's value shows its measured characteristic</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-blue-600 font-bold">•</span>
                <span>The waterfall plot visualizes cumulative feature contributions</span>
              </li>
            </ul>
          </div>
        </div>
      )}
    </Card>
  );
};
