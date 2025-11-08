import React from 'react';
import { ArrowLeft, FileText, RotateCcw, TrendingUp } from 'lucide-react';
import { Button } from '../components/Button';
import { Card } from '../components/Card';
import { RiskChip } from '../components/RiskChip';
import { ConfidenceMeter } from '../components/ConfidenceMeter';
import { GeminiInsight } from '../components/GeminiInsight';
import { useApp } from '../context/AppContext';

export const ResultsPage: React.FC = () => {
  const { setCurrentPage, latestTest, tests } = useApp();

  if (!latestTest) {
    return (
      <div className="min-h-screen bg-[#FAFAFA] p-6 flex items-center justify-center">
        <Card>
          <p className="text-[#546E7A]">No test results available</p>
          <Button onClick={() => setCurrentPage('home')} className="mt-4">
            Go Home
          </Button>
        </Card>
      </div>
    );
  }

  const getComparison = () => {
    if (tests.length < 2) return null;
    const previous = tests[1];
    const change = ((latestTest.risk_score - previous.risk_score) / previous.risk_score) * 100;
    return {
      improved: change < 0,
      percentage: Math.abs(change).toFixed(1)
    };
  };

  const comparison = getComparison();

  return (
    <div className="min-h-screen bg-[#FAFAFA] pb-6">
      <header className="bg-white shadow-sm p-6 mb-6">
        <div className="max-w-4xl mx-auto">
          <Button
            onClick={() => setCurrentPage('home')}
            variant="outline"
            icon={<ArrowLeft className="w-5 h-5" />}
          >
            Back
          </Button>
        </div>
      </header>

      <div className="max-w-4xl mx-auto px-6 space-y-6">
        <Card>
          <h1 className="text-3xl font-bold text-[#263238] mb-6 text-center">
            Your Voice Health Result
          </h1>

          <div className="flex flex-col items-center mb-6">
            <RiskChip
              riskLevel={latestTest.risk_level}
              score={latestTest.risk_score}
              size="large"
            />
          </div>

          <div className="flex justify-center mb-6">
            <ConfidenceMeter confidence={latestTest.confidence} />
          </div>

          <div className="bg-[#FFF9C4] border-l-4 border-[#FFA000] rounded-lg p-4">
            <p className="text-sm text-[#263238]">
              <strong>Disclaimer:</strong> This analysis is for screening purposes only and is not a medical diagnosis. Please consult a healthcare professional for proper evaluation.
            </p>
          </div>
        </Card>

        <Card>
          <h2 className="text-xl font-semibold text-[#263238] mb-4">
            AI Feature Focus
          </h2>
          <div className="bg-gradient-to-br from-gray-100 to-gray-200 rounded-xl p-6 mb-4">
            <div className="relative aspect-video bg-gradient-to-br from-blue-900 via-purple-900 to-red-900 rounded-lg overflow-hidden">
              <div className="absolute inset-0 opacity-60" style={{
                backgroundImage: 'repeating-linear-gradient(90deg, transparent, transparent 10px, rgba(255,255,255,0.1) 10px, rgba(255,255,255,0.1) 20px)',
              }}></div>
              <div className="absolute top-1/4 left-1/4 w-1/2 h-1/2 bg-yellow-400 opacity-40 blur-3xl"></div>
              <div className="absolute top-1/3 right-1/4 w-1/3 h-1/3 bg-red-500 opacity-30 blur-2xl"></div>
              <div className="absolute inset-0 flex items-center justify-center">
                <p className="text-white text-sm font-mono opacity-75">
                  SHAP Visualization: Mel-Spectrogram Analysis
                </p>
              </div>
            </div>
          </div>
          <p className="text-sm text-[#546E7A]">
            Highlighted zones indicate parts of your speech that influenced the prediction most.
          </p>
        </Card>

        <GeminiInsight summary={latestTest.gemini_summary} />

        <Card>
          <h2 className="text-xl font-semibold text-[#263238] mb-4">
            Voice Analysis Details
          </h2>

          <div className="space-y-4">
            <div className="flex items-center justify-between p-4 bg-[#F5F5F5] rounded-lg">
              <span className="text-[#546E7A]">Voice Stability Index</span>
              <div className="flex items-center gap-2">
                <div className="w-32 bg-gray-300 rounded-full h-2 overflow-hidden">
                  <div
                    className="h-full bg-gradient-to-r from-[#2E7D32] to-[#4CAF50] rounded-full"
                    style={{ width: `${latestTest.voice_stability_index}%` }}
                  />
                </div>
                <span className="font-bold text-[#263238]">{latestTest.voice_stability_index}/100</span>
              </div>
            </div>

            <div className="p-4 bg-[#F5F5F5] rounded-lg">
              <h3 className="font-semibold text-[#263238] mb-3">AI Findings</h3>
              <ul className="space-y-2">
                {latestTest.ai_findings.map((finding, idx) => (
                  <li key={idx} className="flex items-start gap-2">
                    <span className="text-[#2E7D32] mt-1">•</span>
                    <span className="text-[#546E7A]">{finding}</span>
                  </li>
                ))}
              </ul>
            </div>

            {comparison && (
              <div className={`p-4 rounded-lg ${comparison.improved ? 'bg-green-50 border border-green-200' : 'bg-orange-50 border border-orange-200'}`}>
                <div className="flex items-center gap-2">
                  <TrendingUp className={`w-5 h-5 ${comparison.improved ? 'text-green-600 rotate-180' : 'text-orange-600'}`} />
                  <p className="text-sm font-medium text-[#263238]">
                    Compared to your last test, your voice stability {comparison.improved ? 'improved' : 'changed'} by {comparison.percentage}%
                  </p>
                </div>
              </div>
            )}
          </div>
        </Card>

        <Card>
          <h2 className="text-xl font-semibold text-[#263238] mb-4">
            Next Steps
          </h2>
          <div className="space-y-3">
            <Button
              onClick={() => setCurrentPage('report')}
              icon={<FileText className="w-5 h-5" />}
              className="w-full"
            >
              Export as PDF Report
            </Button>
            <Button
              onClick={() => setCurrentPage('record')}
              icon={<RotateCcw className="w-5 h-5" />}
              variant="secondary"
              className="w-full"
            >
              Retest
            </Button>
            {tests.length > 1 && (
              <Button
                onClick={() => setCurrentPage('history')}
                icon={<TrendingUp className="w-5 h-5" />}
                variant="outline"
                className="w-full"
              >
                Compare with Previous Tests
              </Button>
            )}
          </div>
        </Card>
      </div>
    </div>
  );
};
