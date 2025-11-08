import React from 'react';
import { AcousticFeatures } from '../types';

interface AcousticFeaturesCardProps {
  features: AcousticFeatures;
}

const AcousticFeaturesCard: React.FC<AcousticFeaturesCardProps> = ({ features }) => {
  // Helper to determine status color and label
  const getFeatureStatus = (feature: string, value: number) => {
    switch (feature) {
      case 'jitter':
        if (value <= 0.03) return { color: 'text-green-600', bg: 'bg-green-50', label: 'Excellent' };
        if (value <= 0.05) return { color: 'text-blue-600', bg: 'bg-blue-50', label: 'Good' };
        if (value <= 0.08) return { color: 'text-yellow-600', bg: 'bg-yellow-50', label: 'Fair' };
        return { color: 'text-orange-600', bg: 'bg-orange-50', label: 'Needs Attention' };
      
      case 'shimmer':
        if (value <= 0.05) return { color: 'text-green-600', bg: 'bg-green-50', label: 'Excellent' };
        if (value <= 0.10) return { color: 'text-blue-600', bg: 'bg-blue-50', label: 'Good' };
        if (value <= 0.15) return { color: 'text-yellow-600', bg: 'bg-yellow-50', label: 'Fair' };
        return { color: 'text-orange-600', bg: 'bg-orange-50', label: 'Needs Attention' };
      
      case 'hnr':
        if (value >= 20) return { color: 'text-green-600', bg: 'bg-green-50', label: 'Excellent' };
        if (value >= 15) return { color: 'text-blue-600', bg: 'bg-blue-50', label: 'Good' };
        if (value >= 10) return { color: 'text-yellow-600', bg: 'bg-yellow-50', label: 'Fair' };
        return { color: 'text-orange-600', bg: 'bg-orange-50', label: 'Needs Attention' };
      
      case 'pitch_variation':
        if (value <= 5) return { color: 'text-green-600', bg: 'bg-green-50', label: 'Very Stable' };
        if (value <= 15) return { color: 'text-blue-600', bg: 'bg-blue-50', label: 'Stable' };
        if (value <= 25) return { color: 'text-yellow-600', bg: 'bg-yellow-50', label: 'Moderate' };
        return { color: 'text-orange-600', bg: 'bg-orange-50', label: 'Variable' };
      
      case 'energy_variation':
        if (value <= 10) return { color: 'text-green-600', bg: 'bg-green-50', label: 'Very Consistent' };
        if (value <= 25) return { color: 'text-blue-600', bg: 'bg-blue-50', label: 'Consistent' };
        if (value <= 40) return { color: 'text-yellow-600', bg: 'bg-yellow-50', label: 'Moderate' };
        return { color: 'text-orange-600', bg: 'bg-orange-50', label: 'Variable' };
      
      default:
        return { color: 'text-gray-600', bg: 'bg-gray-50', label: 'Unknown' };
    }
  };

  const featureConfig = [
    {
      key: 'jitter' as keyof AcousticFeatures,
      name: 'Jitter',
      description: 'Voice frequency stability',
      unit: '',
      format: (v: number) => v.toFixed(4)
    },
    {
      key: 'shimmer' as keyof AcousticFeatures,
      name: 'Shimmer',
      description: 'Voice amplitude consistency',
      unit: '',
      format: (v: number) => v.toFixed(4)
    },
    {
      key: 'hnr' as keyof AcousticFeatures,
      name: 'HNR',
      description: 'Harmonic-to-noise ratio',
      unit: 'dB',
      format: (v: number) => v.toFixed(2)
    },
    {
      key: 'pitch_variation' as keyof AcousticFeatures,
      name: 'Pitch Variation',
      description: 'Fundamental frequency variability',
      unit: '%',
      format: (v: number) => v.toFixed(1)
    },
    {
      key: 'energy_variation' as keyof AcousticFeatures,
      name: 'Energy Variation',
      description: 'Voice intensity consistency',
      unit: '%',
      format: (v: number) => v.toFixed(1)
    }
  ];

  return (
    <div className="bg-white rounded-lg shadow-md p-6 space-y-6">
      <div className="flex items-center justify-between">
        <h3 className="text-xl font-semibold text-gray-800">Voice Quality Metrics</h3>
        <span className="text-sm text-gray-500">Acoustic Features Analysis</span>
      </div>
      
      <div className="space-y-4">
        {featureConfig.map(({ key, name, description, unit, format }) => {
          const value = features[key];
          const status = getFeatureStatus(key, value);
          
          return (
            <div key={key} className="border border-gray-200 rounded-lg p-4 hover:border-blue-300 transition-colors">
              <div className="flex items-center justify-between mb-2">
                <div className="flex-1">
                  <div className="flex items-center gap-2">
                    <h4 className="font-semibold text-gray-800">{name}</h4>
                    <span className={`text-xs px-2 py-1 rounded-full ${status.bg} ${status.color} font-medium`}>
                      {status.label}
                    </span>
                  </div>
                  <p className="text-sm text-gray-600 mt-1">{description}</p>
                </div>
                <div className="text-right ml-4">
                  <div className="text-2xl font-bold text-gray-900">
                    {format(value)}
                  </div>
                  {unit && <div className="text-xs text-gray-500">{unit}</div>}
                </div>
              </div>
              
              {/* Visual indicator bar */}
              <div className="mt-3 h-2 bg-gray-100 rounded-full overflow-hidden">
                <div 
                  className={`h-full ${status.bg.replace('50', '400')} transition-all duration-500`}
                  style={{ 
                    width: key === 'hnr' 
                      ? `${Math.min((value / 30) * 100, 100)}%`
                      : `${Math.min((value / (key.includes('variation') ? 50 : 0.2)) * 100, 100)}%`
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>
      
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mt-4">
        <div className="flex items-start gap-3">
          <div className="flex-shrink-0 w-6 h-6 bg-blue-500 rounded-full flex items-center justify-center text-white text-sm font-bold">
            i
          </div>
          <div className="flex-1">
            <h4 className="font-semibold text-blue-900 mb-1">Understanding Your Metrics</h4>
            <p className="text-sm text-blue-800">
              These acoustic features provide insights into your voice quality and stability. 
              Lower jitter and shimmer values indicate more stable voice production, while higher HNR 
              suggests better voice clarity. Variations in pitch and energy reflect natural speech dynamics.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AcousticFeaturesCard;
