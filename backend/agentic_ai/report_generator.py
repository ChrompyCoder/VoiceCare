"""
PDF/Text Report Generator
Creates professional medical reports for doctor consultation
"""

import json
from datetime import datetime
from pathlib import Path
from . import config
import base64
from io import BytesIO

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import (SimpleDocTemplate, Table, TableStyle, Paragraph, 
                                    Spacer, Image, PageBreak)
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("Warning: reportlab not installed. Install with: pip install reportlab")


class ReportGenerator:
    """
    Generates PDF and JSON reports for medical consultation
    """
    
    def __init__(self, output_dir=None):
        """
        Initialize report generator
        
        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = output_dir or config.REPORTS_DIR
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_full_report(self, test_result, shap_summary=None, gemini_summary=None, 
                            test_history=None, format='both'):
        """
        Generate complete report in PDF and/or JSON format
        
        Args:
            test_result: Main test result dictionary
            shap_summary: SHAP explanation summary
            gemini_summary: Gemini-generated summary
            test_history: Optional historical test data
            format: 'pdf', 'json', or 'both'
            
        Returns:
            dict: Paths to generated reports
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        test_id = test_result.get('id', f'VPX-{timestamp}')
        
        reports = {}
        
        # Generate JSON report
        if format in ['json', 'both']:
            json_path = self._generate_json_report(
                test_result, shap_summary, gemini_summary, test_history, test_id
            )
            reports['json'] = str(json_path)
        
        # Generate PDF report
        if format in ['pdf', 'both']:
            if not REPORTLAB_AVAILABLE:
                print("Warning: PDF generation unavailable. Install reportlab.")
                reports['pdf'] = None
            else:
                pdf_path = self._generate_pdf_report(
                    test_result, shap_summary, gemini_summary, test_history, test_id
                )
                reports['pdf'] = str(pdf_path)
        
        return reports
    
    def _generate_json_report(self, test_result, shap_summary, gemini_summary, 
                             test_history, test_id):
        """Generate JSON format report"""
        
        # Extract stability metrics if available
        stability_metrics = test_result.get('stability_metrics', {})
        
        report_data = {
            'report_id': test_id,
            'generated_date': datetime.now().isoformat(),
            'report_type': 'Voice-Based Parkinson Screening',
            'test_result': {
                'test_id': test_id,
                'date': test_result.get('date', datetime.now().isoformat()),
                'risk_score': test_result.get('risk_score', 0),
                'risk_level': test_result.get('risk_level', 'Unknown'),
                'confidence': test_result.get('confidence', 0),
                'voice_stability_index': test_result.get('voice_stability_index', 0)
            },
            'acoustic_features': {
                'jitter': stability_metrics.get('jitter', 0),
                'shimmer': stability_metrics.get('shimmer', 0),
                'hnr': stability_metrics.get('hnr', 0),
                'pitch_variation': stability_metrics.get('pitch_variation', 0),
                'energy_variation': stability_metrics.get('energy_variation', 0),
                'interpretation': stability_metrics.get('interpretation', 'Not available')
            },
            'ai_analysis': {
                'shap_findings': shap_summary if shap_summary else {},
                'gemini_summary': gemini_summary if gemini_summary else 'Not available',
                'key_features': test_result.get('ai_findings', [])
            },
            'test_history': test_history if test_history else [],
            'disclaimer': config.REPORT_DISCLAIMER
        }
        
        json_path = self.output_dir / f'report_{test_id}.json'
        with open(json_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        return json_path
    
    def _generate_pdf_report(self, test_result, shap_summary, gemini_summary, 
                            test_history, test_id):
        """Generate PDF format report"""
        pdf_path = self.output_dir / f'report_{test_id}.pdf'
        
        # Create PDF document
        doc = SimpleDocTemplate(
            str(pdf_path),
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18
        )
        
        # Container for PDF elements
        elements = []
        styles = getSampleStyleSheet()
        
        # Custom styles
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#2E7D32'),
            spaceAfter=30,
            alignment=TA_CENTER
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#263238'),
            spaceAfter=12,
            spaceBefore=12
        )
        
        # Title
        elements.append(Paragraph(config.REPORT_TITLE, title_style))
        elements.append(Spacer(1, 0.2*inch))
        
        # Header information
        header_data = [
            ['Report ID:', test_id],
            ['Generated:', datetime.now().strftime('%Y-%m-%d %H:%M:%S')],
            ['Facility:', config.CLINIC_NAME]
        ]
        header_table = Table(header_data, colWidths=[2*inch, 4*inch])
        header_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.grey),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ]))
        elements.append(header_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Test Results Section
        elements.append(Paragraph('Test Results', heading_style))
        
        risk_score = test_result.get('risk_score', 0)
        risk_level = test_result.get('risk_level', 'Unknown')
        confidence = test_result.get('confidence', 0)
        stability_metrics = test_result.get('stability_metrics', {})
        
        results_data = [
            ['Test Date:', test_result.get('date', 'N/A')[:10]],
            ['Risk Level:', risk_level],
            ['Risk Score:', f"{risk_score:.2%}"],
            ['Model Confidence:', f"{confidence:.2%}"],
            ['Voice Stability:', f"{stability_metrics.get('stability_index', 0):.3f}"]
        ]
        
        results_table = Table(results_data, colWidths=[2.5*inch, 3.5*inch])
        results_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#546E7A')),
            ('TEXTCOLOR', (1, 1), (1, 1), self._get_risk_color(risk_level)),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F5F5F5')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('PADDING', (0, 0), (-1, -1), 8),
        ]))
        elements.append(results_table)
        elements.append(Spacer(1, 0.3*inch))
        
        # Acoustic Features Section
        if stability_metrics:
            elements.append(Paragraph('Voice Quality Metrics', heading_style))
            
            acoustic_data = [
                ['Metric', 'Value', 'Interpretation'],
                ['Jitter (Frequency Stability)', f"{stability_metrics.get('jitter', 0):.4f}", 
                 'Good' if stability_metrics.get('jitter', 0) <= 0.05 else 'Needs attention'],
                ['Shimmer (Amplitude Consistency)', f"{stability_metrics.get('shimmer', 0):.4f}",
                 'Good' if stability_metrics.get('shimmer', 0) <= 0.10 else 'Needs attention'],
                ['HNR (Voice Clarity)', f"{stability_metrics.get('hnr', 0):.2f} dB",
                 'Good' if stability_metrics.get('hnr', 0) >= 15 else 'Could be improved'],
                ['Pitch Variation', f"{stability_metrics.get('pitch_variation', 0):.1f}%",
                 'Stable' if stability_metrics.get('pitch_variation', 0) <= 15 else 'Variable'],
                ['Energy Variation', f"{stability_metrics.get('energy_variation', 0):.1f}%",
                 'Consistent' if stability_metrics.get('energy_variation', 0) <= 25 else 'Variable']
            ]
            
            acoustic_table = Table(acoustic_data, colWidths=[2.5*inch, 1.5*inch, 2*inch])
            acoustic_table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E7D32')),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#F5F5F5')),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('PADDING', (0, 0), (-1, -1), 8),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ]))
            elements.append(acoustic_table)
            elements.append(Spacer(1, 0.3*inch))
        
        # Key Findings Section
        if shap_summary:
            elements.append(Paragraph('Key Findings (SHAP Analysis)', heading_style))
            
            findings_text = shap_summary.get('clinical_interpretation', 'No detailed findings available.')
            elements.append(Paragraph(findings_text, styles['BodyText']))
            elements.append(Spacer(1, 0.1*inch))
            
            # Top features
            if 'top_3_features' in shap_summary:
                features_text = "<b>Contributing Factors:</b><br/>"
                for i, feature in enumerate(shap_summary['top_3_features'], 1):
                    features_text += f"&bull; {feature}<br/>"
                elements.append(Paragraph(features_text, styles['BodyText']))
            
            # SHAP visualization
            added_image = False
            if 'visualization' in shap_summary and shap_summary['visualization'] and Path(shap_summary['visualization']).exists():
                elements.append(Spacer(1, 0.2*inch))
                img = Image(shap_summary['visualization'], width=5*inch, height=3*inch)
                elements.append(img)
                added_image = True
            elif 'visualization_base64' in shap_summary and shap_summary['visualization_base64']:
                try:
                    b64 = shap_summary['visualization_base64']
                    if b64.startswith('data:image'):
                        b64 = b64.split(',', 1)[1]
                    img_bytes = base64.b64decode(b64)
                    buf = BytesIO(img_bytes)
                    elements.append(Spacer(1, 0.2*inch))
                    img = Image(buf, width=5*inch, height=3*inch)
                    elements.append(img)
                    added_image = True
                except Exception:
                    added_image = False
            if added_image:
                elements.append(Spacer(1, 0.1*inch))

            # SHAP heatmap visualization (path or base64)
            heatmap_added = False
            if 'heatmap' in shap_summary and shap_summary['heatmap'] and Path(str(shap_summary['heatmap'])).exists():
                elements.append(Spacer(1, 0.2*inch))
                img2 = Image(str(shap_summary['heatmap']), width=5*inch, height=3*inch)
                elements.append(img2)
                heatmap_added = True
            elif 'heatmap_base64' in shap_summary and shap_summary['heatmap_base64']:
                try:
                    b64 = shap_summary['heatmap_base64']
                    if b64.startswith('data:image'):
                        b64 = b64.split(',', 1)[1]
                    img_bytes = base64.b64decode(b64)
                    buf = BytesIO(img_bytes)
                    elements.append(Spacer(1, 0.2*inch))
                    img2 = Image(buf, width=5*inch, height=3*inch)
                    elements.append(img2)
                    heatmap_added = True
                except Exception:
                    heatmap_added = False
            if heatmap_added:
                elements.append(Spacer(1, 0.1*inch))
            
            elements.append(Spacer(1, 0.3*inch))
        
        # AI Summary Section
        if gemini_summary:
            elements.append(Paragraph('AI-Generated Summary', heading_style))
            summary_text = gemini_summary if isinstance(gemini_summary, str) else gemini_summary.get('summary', 'N/A')
            elements.append(Paragraph(summary_text, styles['BodyText']))
            elements.append(Spacer(1, 0.3*inch))
        
        # Trend Analysis
        if test_history and len(test_history) > 1:
            elements.append(Paragraph('Historical Trend', heading_style))
            
            trend_data = [['Date', 'Risk Score', 'Risk Level']]
            for test in test_history[-5:]:  # Last 5 tests
                trend_data.append([
                    test.get('date', 'N/A')[:10],
                    f"{test.get('risk_score', 0):.2%}",
                    test.get('risk_level', 'N/A')
                ])
            
            trend_table = Table(trend_data, colWidths=[2*inch, 2*inch, 2*inch])
            trend_table.setStyle(TableStyle([
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#E8F5E9')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#2E7D32')),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('PADDING', (0, 0), (-1, -1), 6),
            ]))
            elements.append(trend_table)
            elements.append(Spacer(1, 0.3*inch))
        
        # Doctor Notes Section (blank)
        elements.append(Paragraph('Healthcare Provider Notes', heading_style))
        elements.append(Spacer(1, 0.5*inch))
        elements.append(Paragraph('_' * 100, styles['Normal']))
        elements.append(Spacer(1, 0.2*inch))
        elements.append(Paragraph('_' * 100, styles['Normal']))
        elements.append(Spacer(1, 0.3*inch))
        
        # Disclaimer
        elements.append(Spacer(1, 0.5*inch))
        disclaimer_style = ParagraphStyle(
            'Disclaimer',
            parent=styles['Normal'],
            fontSize=8,
            textColor=colors.grey,
            alignment=TA_LEFT
        )
        elements.append(Paragraph(f"<b>Disclaimer:</b> {config.REPORT_DISCLAIMER}", disclaimer_style))
        
        # Build PDF
        doc.build(elements)
        
        return pdf_path
    
    def _get_risk_color(self, risk_level):
        """Get color for risk level"""
        if risk_level == 'Low':
            return colors.HexColor('#4CAF50')
        elif risk_level == 'Moderate':
            return colors.HexColor('#FF9800')
        else:
            return colors.HexColor('#F44336')
    
    def generate_quick_summary(self, test_result):
        """
        Generate a quick text summary for sharing
        
        Returns:
            str: Formatted text summary
        """
        summary = f"""
{config.REPORT_TITLE}
{'=' * 50}
Date: {test_result.get('date', 'N/A')[:10]}
Test ID: {test_result.get('id', 'N/A')}

RESULTS:
Risk Level: {test_result.get('risk_level', 'Unknown')}
Risk Score: {test_result.get('risk_score', 0):.1%}
Confidence: {test_result.get('confidence', 0):.1%}

RECOMMENDATION:
{'Consult healthcare professional for evaluation.' if test_result.get('risk_score', 0) > 0.5 else 'Continue regular monitoring.'}

{'=' * 50}
{config.REPORT_DISCLAIMER}
"""
        return summary.strip()


# Utility functions
def generate_pdf_report(test_result, shap_summary=None, gemini_summary=None, output_dir=None):
    """
    Quick utility to generate PDF report
    
    Returns:
        str: Path to generated PDF
    """
    generator = ReportGenerator(output_dir)
    reports = generator.generate_full_report(
        test_result,
        shap_summary,
        gemini_summary,
        format='pdf'
    )
    return reports.get('pdf')


def generate_json_report(test_result, shap_summary=None, gemini_summary=None, output_dir=None):
    """
    Quick utility to generate JSON report
    
    Returns:
        str: Path to generated JSON
    """
    generator = ReportGenerator(output_dir)
    reports = generator.generate_full_report(
        test_result,
        shap_summary,
        gemini_summary,
        format='json'
    )
    return reports.get('json')
