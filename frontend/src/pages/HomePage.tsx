import React from 'react';
import { Mic, History, Brain, TrendingUp } from 'lucide-react';
import { Card } from '../components/Card';
import { Button } from '../components/Button';
import { RiskChip } from '../components/RiskChip';
import { useApp } from '../context/AppContext';

export const HomePage: React.FC = () => {
  const { setCurrentPage, latestTest, tests } = useApp();

  const getTrend = () => {
    if (tests.length < 2) return null;
    const recent = tests.slice(0, 5);
    return recent.map(t => t.risk_score);
  };

  const trend = getTrend();

  return (
    <div className="min-h-screen bg-[#FAFAFA] p-6">
      <div className="max-w-4xl mx-auto">
        <header className="mb-8 text-center">
          <div className="flex items-center justify-center gap-3 mb-3">
            <div className="bg-gradient-to-r from-[#2E7D32] to-[#4CAF50] p-3 rounded-full">
              <Mic className="w-6 h-6 text-white" />
            </div>
            <h1 className="text-3xl font-bold text-[#263238]">VoiceCare AI</h1>
          </div>
          <p className="text-[#546E7A]">Your Voice Health Dashboard</p>
        </header>

        {latestTest ? (
          <Card className="mb-6">
            <div className="flex items-start justify-between mb-4">
              <h2 className="text-xl font-semibold text-[#263238]">Last Test Summary</h2>
              <TrendingUp className="w-5 h-5 text-[#2E7D32]" />
            </div>

            <div className="flex flex-col items-center mb-6">
              <RiskChip
                riskLevel={latestTest.risk_level}
                score={latestTest.risk_score}
                size="large"
              />
              <p className="text-sm text-[#546E7A] mt-3">
                Analyzed {new Date(latestTest.date).toLocaleDateString('en-US', {
                  month: 'short',
                  day: 'numeric',
                  year: 'numeric'
                })}
              </p>
            </div>

            {trend && trend.length > 1 && (
              <div className="bg-gradient-to-r from-[#E8F5E9] to-[#F1F8E9] rounded-lg p-4">
                <p className="text-xs text-[#546E7A] mb-2 font-medium">Last 5 Tests Trend</p>
                <div className="flex items-end gap-2 h-16">
                  {trend.map((score, idx) => (
                    <div
                      key={idx}
                      className="flex-1 bg-gradient-to-t from-[#2E7D32] to-[#4CAF50] rounded-t transition-all duration-500 hover:opacity-80"
                      style={{ height: `${score * 100}%` }}
                      title={`Test ${trend.length - idx}: ${(score * 100).toFixed(0)}%`}
                    />
                  ))}
                </div>
              </div>
            )}
          </Card>
        ) : (
          <Card className="mb-6 text-center py-12">
            <Mic className="w-16 h-16 text-[#81C784] mx-auto mb-4" />
            <h3 className="text-xl font-semibold text-[#263238] mb-2">No tests yet</h3>
            <p className="text-[#546E7A]">Start your first voice health screening</p>
          </Card>
        )}

        <Card>
          <h2 className="text-xl font-semibold text-[#263238] mb-4">Actions</h2>
          <div className="space-y-3">
            <Button
              onClick={() => setCurrentPage('record')}
              icon={<Mic className="w-5 h-5" />}
              className="w-full"
            >
              Start New Test
            </Button>
            <Button
              onClick={() => setCurrentPage('history')}
              icon={<History className="w-5 h-5" />}
              variant="secondary"
              className="w-full"
              disabled={tests.length === 0}
            >
              View History
            </Button>
            {latestTest && (
              <Button
                onClick={() => setCurrentPage('results')}
                icon={<Brain className="w-5 h-5" />}
                variant="outline"
                className="w-full"
              >
                AI Insights
              </Button>
            )}
          </div>
        </Card>

        <div className="mt-8 text-center">
          <p className="text-xs text-[#546E7A]">
            Powered by Gemini AI + Advanced ML
          </p>
        </div>
      </div>
    </div>
  );
};
