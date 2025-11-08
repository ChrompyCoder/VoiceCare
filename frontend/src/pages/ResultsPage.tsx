import React, { useRef } from 'react';
import { ArrowLeft, RotateCcw, TrendingUp, Download, Trash2 } from 'lucide-react';
import { Button } from '../components/Button';
import { Card } from '../components/Card';
import { RiskChip } from '../components/RiskChip';
import { ConfidenceMeter } from '../components/ConfidenceMeter';
import { GeminiInsight } from '../components/GeminiInsight';
import { ProgressChart } from '../components/ProgressChart';
// Removed AcousticFeaturesCard (metrics UI suppressed per user request)
import { SHAPVisualization } from '../components/SHAPVisualization';
import { useApp } from '../context/AppContext';

export const ResultsPage: React.FC = () => {
  const { setCurrentPage, latestTest, tests, clearCache } = useApp();
  const chartRef = useRef<HTMLDivElement>(null);

  // Prefer backend-provided ai_summary; fallback to legacy gemini_summary
  const sanitizeSummary = (text: string) => {
    if (!text) return '';
    let t = text.trim();
    // Remove surrounding quotes
    if ((t.startsWith('"') && t.endsWith('"')) || (t.startsWith("'") && t.endsWith("'"))) {
      t = t.slice(1, -1);
    }
    // Remove informal preambles
    t = t.replace(/^[\s\S]*?(?=Your|This|Based on|The)/i, (m) => (m.length > 200 ? '' : m));
    // Collapse excessive whitespace
    t = t.replace(/\s+\n/g, '\n').replace(/\n{3,}/g, '\n\n').replace(/\s{2,}/g, ' ');
    return t.trim();
  };

  const aiSummary = sanitizeSummary(
    // @ts-expect-error allow backend field name
    (latestTest?.ai_summary as string) ?? latestTest?.gemini_summary ?? ''
  );

  const handleDownloadPDF = async () => {
    if (!latestTest) return;

    try {
      // Import jsPDF dynamically
      const { jsPDF } = await import('jspdf');
      const html2canvas = (await import('html2canvas')).default;

      // Helper: ensure we have a data URL for images
      const toDataUrlIfNeeded = async (imgSrc: string): Promise<string | null> => {
        try {
          if (!imgSrc) return null;
          if (imgSrc.startsWith('data:image')) return imgSrc;
          // Attempt to load and convert to data URL
          const img = new Image();
          // Allow CORS if server supports it
          img.crossOrigin = 'anonymous';
          const loaded: HTMLImageElement = await new Promise((resolve, reject) => {
            img.onload = () => resolve(img);
            img.onerror = reject;
            img.src = imgSrc;
          });
          const canvas = document.createElement('canvas');
          canvas.width = loaded.naturalWidth;
          canvas.height = loaded.naturalHeight;
          const ctx = canvas.getContext('2d');
          if (!ctx) return null;
          ctx.drawImage(loaded, 0, 0);
          return canvas.toDataURL('image/png');
        } catch {
          return null;
        }
      };

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
      pdf.text('AI Summary', 15, yPosition);
      yPosition += 8;
      
      pdf.setFontSize(10);
      pdf.setTextColor(84, 110, 122);
      const summaryLines = pdf.splitTextToSize(aiSummary || 'No summary available', pageWidth - 30);
      pdf.text(summaryLines, 15, yPosition);
      yPosition += summaryLines.length * 5 + 10;

      // SHAP Explainability (Contribution plot and Heatmap)
      if (latestTest.shap_analysis) {
        const sa = latestTest.shap_analysis as any;
        if (yPosition > pageHeight - 80) {
          pdf.addPage();
          yPosition = 20;
        }
        pdf.setFontSize(14);
        pdf.setTextColor(38, 50, 56);
        pdf.text('Explainability (SHAP)', 15, yPosition);
        yPosition += 8;

        // Contribution Plot
        const contribImgSrc = sa.visualization_base64 ?? sa.visualization_path ?? sa.visualization;
        if (contribImgSrc) {
          const dataUrl = await toDataUrlIfNeeded(contribImgSrc);
          if (dataUrl) {
            try {
              const imgWidth = pageWidth - 30;
              const imgHeight = 60; // mm; approximate height
              pdf.addImage(dataUrl, 'PNG', 15, yPosition, imgWidth, imgHeight);
              yPosition += imgHeight + 6;
            } catch {}
          }
        }

        // Heatmap
        const heatmapImgSrc = sa.heatmap_base64 ?? sa.heatmap_path ?? sa.heatmap;
        if (heatmapImgSrc) {
          const dataUrl = await toDataUrlIfNeeded(heatmapImgSrc);
          if (dataUrl) {
            try {
              if (yPosition > pageHeight - 80) {
                pdf.addPage();
                yPosition = 20;
              }
              const imgWidth = pageWidth - 30;
              const imgHeight = 60; // mm; approximate height
              pdf.addImage(dataUrl, 'PNG', 15, yPosition, imgWidth, imgHeight);
              yPosition += imgHeight + 10;
            } catch {}
          }
        }
      }

      // Progress Analysis (if available)
      if (latestTest.progress_analysis) {
        if (yPosition > pageHeight - 40) {
          pdf.addPage();
          yPosition = 20;
        }

        pdf.setFontSize(14);
        pdf.setTextColor(38, 50, 56);
        pdf.text('Progress Analysis', 15, yPosition);
        yPosition += 8;
        
        pdf.setFontSize(10);
        pdf.setTextColor(84, 110, 122);
        const progressLines = pdf.splitTextToSize(latestTest.progress_analysis, pageWidth - 30);
        pdf.text(progressLines, 15, yPosition);
        yPosition += progressLines.length * 5 + 10;
      }

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

  // Comparison helper removed (UI section suppressed per request)

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

        <GeminiInsight 
          summary={aiSummary}
          progressAnalysis={latestTest.progress_analysis}
        />

        {tests.length > 1 && (
          <div ref={chartRef}>
            <ProgressChart tests={tests} />
          </div>
        )}

        {/* Metrics & detailed acoustic findings removed per user request */}

        {/* Acoustic Features Card removed */}

        {/* SHAP Explainability */}
        {latestTest.shap_analysis && (
          <SHAPVisualization shapAnalysis={latestTest.shap_analysis} />
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
