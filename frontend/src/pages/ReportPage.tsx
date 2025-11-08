import React from 'react';
import { ArrowLeft, Download, Share2, FileText } from 'lucide-react';
import { Button } from '../components/Button';
import { Card } from '../components/Card';
import { RiskChip } from '../components/RiskChip';
import { useApp } from '../context/AppContext';

export const ReportPage: React.FC = () => {
  const { setCurrentPage, latestTest } = useApp();

  if (!latestTest) {
    return (
      <div className="min-h-screen bg-[#FAFAFA] p-6 flex items-center justify-center">
        <Card>
          <p className="text-[#546E7A]">No report available</p>
          <Button onClick={() => setCurrentPage('home')} className="mt-4">
            Go Home
          </Button>
        </Card>
      </div>
    );
  }

  const handleDownload = () => {
    alert('PDF download functionality would be implemented here using jsPDF');
  };

  const handleShare = () => {
    if (navigator.share) {
      navigator.share({
        title: 'VoiceCare AI Report',
        text: `Voice Health Report - ${latestTest.risk_level} Risk (${(latestTest.risk_score * 100).toFixed(0)}%)`,
      }).catch(() => {});
    } else {
      alert('Share functionality would copy link to clipboard');
    }
  };

  const formatDate = (dateStr: string) => {
    return new Date(dateStr).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'long',
      day: 'numeric'
    });
  };

  return (
    <div className="min-h-screen bg-[#FAFAFA] pb-6">
      <header className="bg-white shadow-sm p-6 mb-6">
        <div className="max-w-4xl mx-auto flex items-center justify-between">
          <Button
            onClick={() => setCurrentPage('results')}
            variant="outline"
            icon={<ArrowLeft className="w-5 h-5" />}
          >
            Back
          </Button>
          <div className="flex gap-3">
            <Button
              onClick={handleShare}
              variant="secondary"
              icon={<Share2 className="w-5 h-5" />}
            >
              Share
            </Button>
            <Button
              onClick={handleDownload}
              icon={<Download className="w-5 h-5" />}
            >
              Download PDF
            </Button>
          </div>
        </div>
      </header>

      <div className="max-w-4xl mx-auto px-6">
        <Card className="bg-white shadow-2xl">
          <div className="border-b-4 border-[#2E7D32] pb-6 mb-6">
            <div className="flex items-start justify-between mb-4">
              <div>
                <h1 className="text-3xl font-bold text-[#263238] mb-2">
                  Voice Parkinson Screening Report
                </h1>
                <div className="space-y-1 text-sm text-[#546E7A]">
                  <p><strong>Date:</strong> {formatDate(latestTest.date)}</p>
                  <p><strong>Analysis ID:</strong> {latestTest.id}</p>
                </div>
              </div>
              <FileText className="w-12 h-12 text-[#2E7D32]" />
            </div>
          </div>

          <div className="space-y-8">
            <div>
              <h2 className="text-xl font-semibold text-[#263238] mb-4 flex items-center gap-2">
                <span className="w-1 h-6 bg-[#2E7D32] rounded"></span>
                Assessment Summary
              </h2>
              <div className="bg-[#F5F5F5] rounded-xl p-6 space-y-4">
                <div className="flex items-center justify-between">
                  <span className="text-[#546E7A] font-medium">Risk Level:</span>
                  <RiskChip
                    riskLevel={latestTest.risk_level}
                    score={latestTest.risk_score}
                    size="small"
                  />
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-[#546E7A] font-medium">Model Confidence:</span>
                  <span className="text-2xl font-bold text-[#263238]">
                    {(latestTest.confidence * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-[#546E7A] font-medium">Voice Stability Index:</span>
                  <div className="flex items-center gap-3">
                    <div className="w-32 bg-gray-300 rounded-full h-3 overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-[#2E7D32] to-[#4CAF50] rounded-full"
                        style={{ width: `${latestTest.voice_stability_index}%` }}
                      />
                    </div>
                    <span className="text-xl font-bold text-[#263238]">
                      {latestTest.voice_stability_index}/100
                    </span>
                  </div>
                </div>
              </div>
            </div>

            <div>
              <h2 className="text-xl font-semibold text-[#263238] mb-4 flex items-center gap-2">
                <span className="w-1 h-6 bg-[#2E7D32] rounded"></span>
                AI Findings
              </h2>
              <div className="bg-[#F5F5F5] rounded-xl p-6">
                <p className="text-sm text-[#546E7A] mb-4">
                  The following acoustic features were analyzed in your voice sample:
                </p>
                <ul className="space-y-3">
                  {latestTest.ai_findings.map((finding, idx) => (
                    <li key={idx} className="flex items-start gap-3">
                      <div className="w-6 h-6 rounded-full bg-gradient-to-r from-[#2E7D32] to-[#4CAF50] flex items-center justify-center text-white text-xs font-bold flex-shrink-0">
                        {idx + 1}
                      </div>
                      <span className="text-[#263238] font-medium">{finding}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </div>

            <div>
              <h2 className="text-xl font-semibold text-[#263238] mb-4 flex items-center gap-2">
                <span className="w-1 h-6 bg-[#2E7D32] rounded"></span>
                Gemini AI Summary
              </h2>
              <div className="bg-gradient-to-br from-[#E8F5E9] to-[#F1F8E9] rounded-xl p-6 border-2 border-[#81C784]">
                <p className="text-[#263238] leading-relaxed">
                  {latestTest.gemini_summary}
                </p>
              </div>
            </div>

            <div className="bg-[#FFF9C4] border-l-4 border-[#FFA000] rounded-lg p-6">
              <h3 className="font-semibold text-[#263238] mb-2">Important Notice</h3>
              <p className="text-sm text-[#263238] leading-relaxed">
                This analysis is for screening purposes only and is not a medical diagnosis. The results should be interpreted in consultation with a qualified healthcare professional. Regular monitoring and professional medical evaluation are recommended for accurate assessment.
              </p>
            </div>
          </div>

          <div className="mt-8 pt-6 border-t border-gray-200">
            <div className="flex items-center justify-between text-sm text-[#546E7A]">
              <div className="flex items-center gap-2">
                <div className="w-8 h-8 bg-gradient-to-r from-[#2E7D32] to-[#4CAF50] rounded-full flex items-center justify-center">
                  <FileText className="w-4 h-4 text-white" />
                </div>
                <span className="font-medium">Generated by VoiceCare AI</span>
              </div>
              <span>Powered by Gemini AI</span>
            </div>
          </div>
        </Card>

        <div className="mt-6 text-center">
          <Button
            onClick={() => setCurrentPage('record')}
            variant="secondary"
            className="inline-flex"
          >
            Take Another Test
          </Button>
        </div>
      </div>
    </div>
  );
};
