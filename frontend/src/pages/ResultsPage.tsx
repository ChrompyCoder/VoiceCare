import React, { useRef } from 'react';
import { ArrowLeft, RotateCcw, TrendingUp, Download, Trash2 } from 'lucide-react';
import { Button } from '../components/Button';
import { Card } from '../components/Card';
import { RiskChip } from '../components/RiskChip';
import { ConfidenceMeter } from '../components/ConfidenceMeter';
import { GeminiInsight } from '../components/GeminiInsight';
import { ProgressChart } from '../components/ProgressChart';
import AcousticFeaturesCard from '../components/AcousticFeaturesCard';
import { useApp } from '../context/AppContext';

export const ResultsPage: React.FC = () => {
  const { setCurrentPage, latestTest, tests, clearCache } = useApp();
  const chartRef = useRef<HTMLDivElement>(null);

  const handleDownloadPDF = async () => {
    if (!latestTest) return;

    try {
      // Import jsPDF dynamically
      const { jsPDF } = await import('jspdf');
      const html2canvas = (await import('html2canvas')).default;

      const pdf = new jsPDF('p', 'mm', 'a4');
      const pageWidth = pdf.internal.pageSize.getWidth();
      const pageHeight = pdf.internal.pageSize.getHeight();
      let yPosition = 20;

      // Header
      pdf.setFillColor(46, 125, 50);
      pdf.rect(0, 0, pageWidth, 30, 'F');
      pdf.setTextColor(255, 255, 255);
      pdf.setFontSize(24);
      pdf.text('VoiceCare AI', pageWidth / 2, 15, { align: 'center' });
      pdf.setFontSize(12);
      pdf.text('Voice Health Screening Report', pageWidth / 2, 22, { align: 'center' });

      yPosition = 40;
      pdf.setTextColor(38, 50, 56);

      // Test Information
      pdf.setFontSize(10);
      pdf.setTextColor(128, 128, 128);
      pdf.text(`Test ID: ${latestTest.id}`, 15, yPosition);
      pdf.text(`Date: ${new Date(latestTest.date).toLocaleString()}`, pageWidth - 15, yPosition, { align: 'right' });
      
      yPosition += 15;

      // Risk Score Section
      pdf.setFontSize(16);
      pdf.setTextColor(38, 50, 56);
      pdf.text('Risk Assessment', 15, yPosition);
      yPosition += 8;

      const riskScore = (latestTest.risk_score * 100).toFixed(1);
      pdf.setFontSize(32);
      
      // Color based on risk level
      if (latestTest.risk_level === 'Low') pdf.setTextColor(67, 160, 71);
      else if (latestTest.risk_level === 'Moderate') pdf.setTextColor(255, 160, 0);
      else pdf.setTextColor(229, 57, 53);
      
      pdf.text(`${riskScore}%`, pageWidth / 2, yPosition, { align: 'center' });
      yPosition += 8;
      
      pdf.setFontSize(14);
      pdf.text(`${latestTest.risk_level} Risk`, pageWidth / 2, yPosition, { align: 'center' });
      yPosition += 5;
      
      pdf.setFontSize(10);
      pdf.setTextColor(128, 128, 128);
      pdf.text(`Confidence: ${(latestTest.confidence * 100).toFixed(0)}%`, pageWidth / 2, yPosition, { align: 'center' });
      
      yPosition += 15;

      // AI Summary
      pdf.setFontSize(14);
      pdf.setTextColor(38, 50, 56);
      pdf.text('AI Analysis Summary', 15, yPosition);
      yPosition += 8;
      
      pdf.setFontSize(10);
      pdf.setTextColor(84, 110, 122);
      const summaryLines = pdf.splitTextToSize(latestTest.gemini_summary || 'No summary available', pageWidth - 30);
      pdf.text(summaryLines, 15, yPosition);
      yPosition += summaryLines.length * 5 + 10;

      // Acoustic Features Section
      if (latestTest.acoustic_features) {
        if (yPosition > pageHeight - 80) {
          pdf.addPage();
          yPosition = 20;
        }

        pdf.setFontSize(14);
        pdf.setTextColor(38, 50, 56);
        pdf.text('Voice Quality Metrics', 15, yPosition);
        yPosition += 8;

        const features = latestTest.acoustic_features;
        const metricsData = [
          ['Jitter (Frequency Stability)', features.jitter.toFixed(4), features.jitter <= 0.05 ? 'Good' : 'Needs Attention'],
          ['Shimmer (Amplitude Consistency)', features.shimmer.toFixed(4), features.shimmer <= 0.10 ? 'Good' : 'Needs Attention'],
          ['HNR (Voice Clarity)', `${features.hnr.toFixed(2)} dB`, features.hnr >= 15 ? 'Good' : 'Could be improved'],
          ['Pitch Variation', `${features.pitch_variation.toFixed(1)}%`, features.pitch_variation <= 15 ? 'Stable' : 'Variable'],
          ['Energy Variation', `${features.energy_variation.toFixed(1)}%`, features.energy_variation <= 25 ? 'Consistent' : 'Variable']
        ];

        pdf.setFontSize(9);
        pdf.setTextColor(84, 110, 122);
        metricsData.forEach(([metric, value, status]) => {
          pdf.text(`${metric}: ${value} - ${status}`, 15, yPosition);
          yPosition += 5;
        });
        yPosition += 5;
      }

      // Add chart if available
      if (chartRef.current && tests.length > 1) {
        try {
          const canvas = await html2canvas(chartRef.current, {
            scale: 2,
            backgroundColor: '#ffffff'
          });
          const imgData = canvas.toDataURL('image/png');
          const imgWidth = pageWidth - 30;
          const imgHeight = (canvas.height * imgWidth) / canvas.width;
          
          if (yPosition + imgHeight > pageHeight - 20) {
            pdf.addPage();
            yPosition = 20;
          }
          
          pdf.setFontSize(14);
          pdf.setTextColor(38, 50, 56);
          pdf.text('Progress Trend', 15, yPosition);
          yPosition += 8;
          
          pdf.addImage(imgData, 'PNG', 15, yPosition, imgWidth, imgHeight);
          yPosition += imgHeight + 10;
        } catch (error) {
          console.error('Error adding chart to PDF:', error);
        }
      }

      // Disclaimer
      if (yPosition > pageHeight - 40) {
        pdf.addPage();
        yPosition = 20;
      }
      
      pdf.setFillColor(255, 249, 196);
      pdf.rect(15, yPosition, pageWidth - 30, 25, 'F');
      pdf.setFontSize(8);
      pdf.setTextColor(38, 50, 56);
      const disclaimer = 'DISCLAIMER: This analysis is for screening purposes only and is not a medical diagnosis. Please consult a qualified healthcare professional for proper evaluation and diagnosis.';
      const disclaimerLines = pdf.splitTextToSize(disclaimer, pageWidth - 40);
      pdf.text(disclaimerLines, 20, yPosition + 5);

      // Footer
      pdf.setFontSize(8);
      pdf.setTextColor(128, 128, 128);
      pdf.text('Generated by VoiceCare AI | For Healthcare Professional Reference Only', pageWidth / 2, pageHeight - 10, { align: 'center' });

      // Save the PDF
      pdf.save(`VoiceCare_Report_${latestTest.id}.pdf`);
    } catch (error) {
      console.error('Error generating PDF:', error);
      alert('Failed to generate PDF. Please try again.');
    }
  };

  const handleClearCache = () => {
    if (confirm('Are you sure you want to clear all test history? This action cannot be undone.')) {
      clearCache();
    }
  };

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

        <GeminiInsight summary={latestTest.gemini_summary} />

        {tests.length > 1 && (
          <div ref={chartRef}>
            <ProgressChart tests={tests} />
          </div>
        )}

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

        {/* Acoustic Features Card */}
        {latestTest.acoustic_features && (
          <AcousticFeaturesCard features={latestTest.acoustic_features} />
        )}

        <Card>
          <h2 className="text-xl font-semibold text-[#263238] mb-4">
            Next Steps
          </h2>
          <div className="space-y-3">
            <Button
              onClick={handleDownloadPDF}
              icon={<Download className="w-5 h-5" />}
              className="w-full"
            >
              Download PDF Report
            </Button>
            <Button
              onClick={() => setCurrentPage('record')}
              icon={<RotateCcw className="w-5 h-5" />}
              variant="secondary"
              className="w-full"
            >
              Take Another Test
            </Button>
            {tests.length > 1 && (
              <Button
                onClick={() => setCurrentPage('history')}
                icon={<TrendingUp className="w-5 h-5" />}
                variant="outline"
                className="w-full"
              >
                View Test History
              </Button>
            )}
            {tests.length > 0 && (
              <Button
                onClick={handleClearCache}
                icon={<Trash2 className="w-5 h-5" />}
                variant="outline"
                className="w-full text-red-600 hover:bg-red-50 border-red-300"
              >
                Clear All Data
              </Button>
            )}
          </div>
        </Card>
      </div>
    </div>
  );
};
