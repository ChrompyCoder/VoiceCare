import React from 'react';
import { ShapExplanation } from '../types';
import { TrendingUp, Info } from 'lucide-react';

interface ShapExplanationCardProps {
  explanation: ShapExplanation;
}

const ShapExplanationCard: React.FC<ShapExplanationCardProps> = ({ explanation }) => {
  if (!explanation.available || !explanation.top_features || explanation.top_features.length === 0) {
    return null;
  }

  // Find max importance for scaling bars
  const maxImportance = Math.max(...explanation.top_features.map(f => Math.abs(f.importance)));

  return (
    <div className="bg-white rounded-lg shadow-md p-6 space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <TrendingUp className="w-6 h-6 text-purple-600" />
          <h3 className="text-xl font-semibold text-gray-800">AI Explainability (SHAP)</h3>
        </div>
        <span className="text-sm text-gray-500">Feature Importance</span>
      </div>

      <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
        <div className="flex items-start gap-3">
          <Info className="w-5 h-5 text-purple-600 mt-0.5 flex-shrink-0" />
          <div className="flex-1">
            <h4 className="font-semibold text-purple-900 mb-1">Understanding SHAP Analysis</h4>
            <p className="text-sm text-purple-800">
              SHAP (SHapley Additive exPlanations) shows which voice features had the most impact on 
              your risk assessment. Higher bars indicate features that contributed more to the prediction.
            </p>
          </div>
        </div>
      </div>

      <div className="space-y-3">
        <h4 className="text-sm font-semibold text-gray-700 uppercase tracking-wide">
          Top Contributing Features
        </h4>
        
        {explanation.top_features.map((feature, index) => {
          const percentage = (Math.abs(feature.importance) / maxImportance) * 100;
          const isPositive = feature.importance > 0;
          
          return (
            <div key={feature.index} className="border border-gray-200 rounded-lg p-4 hover:border-purple-300 transition-colors">
              <div className="flex items-start justify-between mb-2">
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <span className="text-xs font-bold text-purple-600 bg-purple-100 px-2 py-1 rounded">
                      #{index + 1}
                    </span>
                    <h5 className="font-semibold text-gray-800 text-sm">{feature.name}</h5>
                  </div>
                  <p className="text-xs text-gray-600 mt-1">
                    Value: {feature.value.toFixed(4)}
                  </p>
                </div>
                <div className="text-right ml-4">
                  <div className={`text-lg font-bold ${isPositive ? 'text-red-600' : 'text-green-600'}`}>
                    {isPositive ? '+' : ''}{feature.importance.toFixed(4)}
                  </div>
                  <div className="text-xs text-gray-500">Impact</div>
                </div>
              </div>
              
              {/* Impact bar */}
              <div className="mt-3">
                <div className="h-3 bg-gray-100 rounded-full overflow-hidden">
                  <div 
                    className={`h-full ${isPositive ? 'bg-gradient-to-r from-red-400 to-red-600' : 'bg-gradient-to-r from-green-400 to-green-600'} transition-all duration-500`}
                    style={{ width: `${percentage}%` }}
                  />
                </div>
                <div className="flex justify-between mt-1">
                  <span className="text-xs text-gray-500">
                    {isPositive ? 'Increases risk' : 'Decreases risk'}
                  </span>
                  <span className="text-xs font-medium text-gray-700">
                    {percentage.toFixed(1)}%
                  </span>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* SHAP Visualization Image */}
      {explanation.visualization_base64 && (
        <div className="mt-6">
          <h4 className="text-sm font-semibold text-gray-700 uppercase tracking-wide mb-3">
            Feature Importance Visualization
          </h4>
          <div className="border border-gray-200 rounded-lg p-4 bg-gray-50">
            <img 
              src={`data:image/png;base64,${explanation.visualization_base64}`}
              alt="SHAP Feature Importance"
              className="w-full h-auto rounded"
            />
          </div>
        </div>
      )}

      <div className="bg-gray-50 border border-gray-200 rounded-lg p-4 mt-4">
        <h4 className="font-semibold text-gray-800 mb-2 text-sm">How to Interpret</h4>
        <ul className="space-y-1 text-sm text-gray-700">
          <li className="flex items-start gap-2">
            <span className="text-red-600 font-bold">•</span>
            <span><strong className="text-red-600">Red bars:</strong> Features that increased the risk score</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-green-600 font-bold">•</span>
            <span><strong className="text-green-600">Green bars:</strong> Features that decreased the risk score</span>
          </li>
          <li className="flex items-start gap-2">
            <span className="text-purple-600 font-bold">•</span>
            <span><strong>Longer bars:</strong> Greater impact on the final prediction</span>
          </li>
        </ul>
      </div>
    </div>
  );
};

export default ShapExplanationCard;
