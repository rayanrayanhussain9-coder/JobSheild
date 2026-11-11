"""
JobShield Report Generator
Generates exportable JSON and self-contained, sanitized HTML audit reports
for individual or comparative job advertisement fraud screenings.
"""
import html
import json
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from src.predictor import JobShieldResult, JobAdvertisementInput


def generate_json_report(ad: JobAdvertisementInput, result: JobShieldResult) -> str:
    """Produces a formatted, machine-readable JSON screening report."""
    report_data = {
        "report_id": f"JS-AUDIT-{int(datetime.now(timezone.utc).timestamp())}",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_advertisement": {
            "title": ad.title,
            "company_name": ad.company_name,
            "description": ad.description,
            "requirements": ad.requirements,
            "benefits": ad.benefits,
            "telecommuting": ad.telecommuting,
        },
        "screening_results": result.to_dict(),
    }
    return json.dumps(report_data, indent=2)


def generate_html_report(ad: JobAdvertisementInput, result: JobShieldResult) -> str:
    """
    Generates a secure, self-contained, styled HTML report.
    All user strings are strictly HTML-escaped.
    """
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    pred_label = result.model_prediction or "INCOMPLETE"
    score_str = f"{result.suspicion_score:.4f}" if result.suspicion_score is not None else "N/A"

    badge_color = "#28a745"
    if pred_label == "SUSPICIOUS_PATTERN":
        badge_color = "#dc3545"
    elif pred_label == "NEEDS_REVIEW":
        badge_color = "#ffc107"
    elif pred_label == "INCOMPLETE":
        badge_color = "#6c757d"

    # Escape all inputs
    title_safe = html.escape(ad.title or "Untitled Advertisement")
    company_safe = html.escape(ad.company_name or "Not Specified")
    desc_safe = html.escape(ad.description or "No description provided.")
    req_safe = html.escape(ad.requirements or "None provided.")
    ben_safe = html.escape(ad.benefits or "None provided.")
    synth_safe = html.escape(result.synthesis_summary)

    # Format warnings
    warnings_html = ""
    if result.rule_results and result.rule_results.get("warnings"):
        warnings_html += "<ul style='margin-top: 5px;'>"
        for w in result.rule_results["warnings"]:
            cat = html.escape(w.get("category", ""))
            match = html.escape(w.get("matched_text", ""))
            exp = html.escape(w.get("explanation", ""))
            guid = html.escape(w.get("guidance", ""))
            warnings_html += (
                f"<li style='margin-bottom: 8px;'>"
                f"<strong>[{cat}]</strong> Matched text: <em>\"{match}\"</em><br>"
                f"<span>{exp}</span><br>"
                f"<small style='color: #666;'>Advice: {guid}</small>"
                f"</li>"
            )
        warnings_html += "</ul>"
    else:
        warnings_html = "<p style='color: #28a745;'>No explicit rule-based warning signs triggered.</p>"

    # Format top features
    features_html = ""
    if result.explanation:
        top_susp = result.explanation.get("top_suspicious", [])
        top_legit = result.explanation.get("top_legitimate", [])

        features_html += "<table style='width: 100%; border-collapse: collapse; font-size: 13px; margin-top: 8px;'>"
        features_html += "<tr style='background: #f1f3f5;'><th style='padding: 6px; text-align: left; border-bottom: 1px solid #ddd;'>Word / Phrase</th><th style='padding: 6px; text-align: left; border-bottom: 1px solid #ddd;'>Direction</th><th style='padding: 6px; text-align: right; border-bottom: 1px solid #ddd;'>Local Weight</th></tr>"

        for f in top_susp[:6]:
            feat = html.escape(f["feature"])
            contrib = f["contribution"]
            features_html += f"<tr><td style='padding: 5px; border-bottom: 1px solid #eee;'>{feat}</td><td style='padding: 5px; border-bottom: 1px solid #eee; color: #dc3545;'>Pushing Suspicious</td><td style='padding: 5px; border-bottom: 1px solid #eee; text-align: right;'>+{contrib:.4f}</td></tr>"

        for f in top_legit[:6]:
            feat = html.escape(f["feature"])
            contrib = f["contribution"]
            features_html += f"<tr><td style='padding: 5px; border-bottom: 1px solid #eee;'>{feat}</td><td style='padding: 5px; border-bottom: 1px solid #eee; color: #28a745;'>Pushing Legitimate</td><td style='padding: 5px; border-bottom: 1px solid #eee; text-align: right;'>{contrib:.4f}</td></tr>"

        features_html += "</table>"
    else:
        features_html = "<p style='color: #666;'>No model feature explanations available.</p>"

    # Format missing info
    missing_html = ""
    if result.missing_info_observations:
        missing_html += "<ul style='margin-top: 5px;'>"
        for obs in result.missing_info_observations:
            fld = html.escape(obs["field"])
            msg = html.escape(obs["observation"])
            missing_html += f"<li><strong>{fld}:</strong> {msg}</li>"
        missing_html += "</ul>"
    else:
        missing_html = "<p style='color: #666;'>All standard informational fields were present.</p>"

    disagreement_html = ""
    if result.disagreement_notice:
        disagreement_html = (
            f"<div style='background: #fff3cd; border-left: 4px solid #ffeeba; padding: 10px; margin: 15px 0;'>"
            f"<strong>Attention:</strong> {html.escape(result.disagreement_notice)}"
            f"</div>"
        )

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>JobShield Fraud-Screening Audit Report</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      line-height: 1.5;
      color: #212529;
      background-color: #f8f9fa;
      margin: 0;
      padding: 20px;
    }}
    .container {{
      max-width: 800px;
      margin: 0 auto;
      background: #ffffff;
      border: 1px solid #dee2e6;
      border-radius: 6px;
      padding: 30px;
      box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }}
    .header {{
      border-bottom: 2px solid #0056b3;
      padding-bottom: 15px;
      margin-bottom: 20px;
    }}
    .badge {{
      display: inline-block;
      padding: 4px 10px;
      border-radius: 4px;
      color: #fff;
      font-weight: bold;
      font-size: 14px;
      background: {badge_color};
    }}
    .card {{
      background: #fdfdfe;
      border: 1px solid #e9ecef;
      border-radius: 4px;
      padding: 15px;
      margin-bottom: 18px;
    }}
    h1 {{ font-size: 22px; margin: 0 0 5px 0; color: #1e3a8a; }}
    h2 {{ font-size: 16px; margin: 0 0 10px 0; border-bottom: 1px solid #e9ecef; padding-bottom: 5px; }}
    .footer {{
      margin-top: 25px;
      border-top: 1px solid #dee2e6;
      padding-top: 10px;
      font-size: 11px;
      color: #6c757d;
    }}
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <h1>JobShield — Fraud Screening Audit Report</h1>
      <p style="margin: 0; font-size: 13px; color: #6c757d;">Generated on: {now_str} | Model Version: {html.escape(result.model_version)}</p>
    </div>

    <div class="card" style="background: #eef2ff; border-color: #c7d2fe;">
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
        <span class="badge">{pred_label}</span>
        <strong>{result.score_label}: {score_str}</strong>
      </div>
      <p style="margin: 0; font-size: 14px;"><strong>Assessment Summary:</strong> {synth_safe}</p>
    </div>

    {disagreement_html}

    <div class="card">
      <h2>Target Advertisement Details</h2>
      <p style="margin: 4px 0;"><strong>Job Title:</strong> {title_safe}</p>
      <p style="margin: 4px 0;"><strong>Company Name:</strong> {company_safe}</p>
      <details style="margin-top: 8px;">
        <summary style="cursor: pointer; color: #0056b3;">View Submitted Description & Text</summary>
        <div style="margin-top: 6px; padding: 8px; background: #f8f9fa; border-radius: 4px; font-size: 12px; white-space: pre-wrap;">{desc_safe}</div>
      </details>
    </div>

    <div class="card">
      <h2>Rule-Based Warning Engine Checks</h2>
      {warnings_html}
    </div>

    <div class="card">
      <h2>Local Feature Attributions (Faithful Linear Weights)</h2>
      {features_html}
    </div>

    <div class="card">
      <h2>Missing Information Observations</h2>
      {missing_html}
    </div>

    <div class="footer">
      <strong>Important Boundaries & Disclaimer:</strong> {html.escape(result.limitations_disclaimer)}<br>
      JobShield evaluates text patterns using machine learning trained on historical benchmark data. It does not certify employers or ensure safety.
    </div>
  </div>
</body>
</html>"""
    return html_content
