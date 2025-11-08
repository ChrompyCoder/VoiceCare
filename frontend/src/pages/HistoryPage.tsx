import React, { useState } from 'react';
import { ArrowLeft, Calendar, TrendingUp, ChevronDown, ChevronUp } from 'lucide-react';
import { Button } from '../components/Button';
import { Card } from '../components/Card';
import { RiskChip } from '../components/RiskChip';
import { useApp } from '../context/AppContext';

export const HistoryPage: React.FC = () => {
  const { setCurrentPage, tests } = useApp();
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [sortBy, setSortBy] = useState<'date' | 'risk'>('date');

  const sortedTests = [...tests].sort((a, b) => {
    if (sortBy === 'date') {
      return new Date(b.date).getTime() - new Date(a.date).getTime();
    }
    return b.risk_score - a.risk_score;
  });

  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return {
      day: date.getDate().toString().padStart(2, '0'),
      month: date.toLocaleDateString('en-US', { month: 'short' }),
      year: date.getFullYear()
    };
  };

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

      <div className="max-w-4xl mx-auto px-6">
        <div className="flex items-center justify-between mb-6">
          <h1 className="text-3xl font-bold text-[#263238]">Voice Test History</h1>
          <div className="flex gap-2">
            <button
              onClick={() => setSortBy('date')}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                sortBy === 'date'
                  ? 'bg-[#2E7D32] text-white'
                  : 'bg-white text-[#546E7A] border border-gray-300'
              }`}
            >
              By Date
            </button>
            <button
              onClick={() => setSortBy('risk')}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
                sortBy === 'risk'
                  ? 'bg-[#2E7D32] text-white'
                  : 'bg-white text-[#546E7A] border border-gray-300'
              }`}
            >
              By Risk
            </button>
          </div>
        </div>

        {tests.length === 0 ? (
          <Card className="text-center py-12">
            <Calendar className="w-16 h-16 text-[#81C784] mx-auto mb-4" />
            <h3 className="text-xl font-semibold text-[#263238] mb-2">No history yet</h3>
            <p className="text-[#546E7A] mb-6">Start your first test to see results here</p>
            <Button onClick={() => setCurrentPage('record')}>
              Start Test
            </Button>
          </Card>
        ) : (
          <div className="relative">
            <div className="absolute left-8 top-0 bottom-0 w-0.5 bg-gradient-to-b from-[#2E7D32] to-[#81C784]"></div>

            <div className="space-y-6">
              {sortedTests.map((test, idx) => {
                const date = formatDate(test.date);
                const isExpanded = expandedId === test.id;

                return (
                  <div key={test.id} className="relative pl-20">
                    <div className="absolute left-0 top-0 flex flex-col items-center">
                      <div className="w-16 h-16 rounded-full bg-gradient-to-r from-[#2E7D32] to-[#4CAF50] flex items-center justify-center text-white shadow-lg">
                        <div className="text-center">
                          <div className="text-xs font-medium">{date.month}</div>
                          <div className="text-lg font-bold">{date.day}</div>
                        </div>
                      </div>
                    </div>

                    <Card className="cursor-pointer hover:shadow-2xl transition-shadow">
                      <div
                        onClick={() => setExpandedId(isExpanded ? null : test.id)}
                        className="flex items-center justify-between"
                      >
                        <div className="flex-1">
                          <div className="flex items-center gap-3 mb-2">
                            <RiskChip
                              riskLevel={test.risk_level}
                              score={test.risk_score}
                              size="small"
                            />
                            <span className="text-xs text-[#546E7A]">
                              {new Date(test.date).toLocaleTimeString('en-US', {
                                hour: '2-digit',
                                minute: '2-digit'
                              })}
                            </span>
                          </div>
                          <div className="flex items-center gap-4 text-sm text-[#546E7A]">
                            <span>Confidence: {(test.confidence * 100).toFixed(0)}%</span>
                            <span>Stability: {test.voice_stability_index}/100</span>
                          </div>
                        </div>
                        <button className="text-[#2E7D32]">
                          {isExpanded ? (
                            <ChevronUp className="w-6 h-6" />
                          ) : (
                            <ChevronDown className="w-6 h-6" />
                          )}
                        </button>
                      </div>

                      {isExpanded && (
                        <div className="mt-6 pt-6 border-t border-gray-200 space-y-4 animate-fadeIn">
                          <div className="bg-[#F5F5F5] rounded-lg p-4">
                            <h4 className="font-semibold text-[#263238] mb-2">AI Findings</h4>
                            <ul className="space-y-1">
                              {test.ai_findings.map((finding, idx) => (
                                <li key={idx} className="text-sm text-[#546E7A] flex items-start gap-2">
                                  <span className="text-[#2E7D32]">•</span>
                                  {finding}
                                </li>
                              ))}
                            </ul>
                          </div>

                          <div className="bg-gradient-to-br from-[#E8F5E9] to-[#F1F8E9] rounded-lg p-4">
                            <p className="text-sm text-[#546E7A] leading-relaxed">
                              {test.gemini_summary}
                            </p>
                          </div>

                          {idx > 0 && (
                            <div className="flex items-center gap-2 text-sm">
                              <TrendingUp className={`w-4 h-4 ${
                                test.risk_score < sortedTests[idx - 1].risk_score
                                  ? 'text-green-600 rotate-180'
                                  : 'text-orange-600'
                              }`} />
                              <span className="text-[#546E7A]">
                                {((test.risk_score - sortedTests[idx - 1].risk_score) * 100).toFixed(1)}% change from previous
                              </span>
                            </div>
                          )}

                          <Button
                            onClick={(e) => {
                              e.stopPropagation();
                              setCurrentPage('report');
                            }}
                            variant="outline"
                            className="w-full"
                          >
                            View Full Report
                          </Button>
                        </div>
                      )}
                    </Card>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        <button
          onClick={() => setCurrentPage('record')}
          className="fixed bottom-8 right-8 w-16 h-16 bg-gradient-to-r from-[#2E7D32] to-[#4CAF50] rounded-full shadow-2xl flex items-center justify-center text-white hover:scale-110 transition-transform"
        >
          <TrendingUp className="w-8 h-8" />
        </button>
      </div>
    </div>
  );
};
