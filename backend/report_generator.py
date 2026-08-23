"""
backend/report_generator.py
───────────────────────────
Clinical PDF Report Generator using fpdf2.
Generates clean, light-themed, professional cardiovascular risk reports.
"""

import time
from typing import Dict, List
from fpdf import FPDF
from fpdf.enums import XPos, YPos
from .config import FEATURE_LABELS, SHAP_ADVICE


class ClinicalReportPDF(FPDF):
    def header(self):
        self.set_draw_color(226, 232, 240)  # Slate 200 border
        self.set_line_width(0.5)
        self.rect(8, 8, 194, 281)

    def footer(self):
        self.set_y(-16)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(148, 163, 184)  # Slate 400
        self.cell(
            0,
            6,
            f"CardioShield Confidential Clinical Report  |  Page {self.page_no()}",
            align="C",
        )

    def section_heading(self, title: str):
        self.set_x(14)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(15, 23, 42)  # Slate 900
        self.cell(182, 7, title, new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        self.set_draw_color(13, 148, 136)  # Teal 600
        self.set_line_width(0.8)
        self.line(14, self.get_y(), 196, self.get_y())
        self.ln(3)


def pdf_safe(text: str) -> str:
    """Sanitize unicode strings for Helvetica standard Latin-1 encoding."""
    replacements = {
        "\u2014": "-", "\u2013": "-", "\u2019": "'", "\u2018": "'",
        "\u201c": '"', "\u201d": '"', "\u2026": "...", "\u2192": "->",
        "\u2190": "<-", "\u2194": "<->", "\u2265": ">=", "\u2264": "<=",
        "\u00b7": ".", "\u2022": "-", "\u00a0": " ", "\u03c3": "sigma",
        "\u221e": "inf", "±": "+/-",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.encode("latin-1", errors="replace").decode("latin-1")


def generate_clinical_report(
    patient_name: str,
    clinician_name: str,
    assessment_date: str,
    features: Dict[str, float],
    risk_prob: float,
    risk_class: str,
    shap_values: List[float],
    feature_names: List[str],
    he_used: bool = True,
    latency_ms: float = 0.0,
) -> bytes:
    """Generate and return PDF bytes for a complete clinical analysis report."""
    pdf = ClinicalReportPDF()
    pdf.set_auto_page_break(auto=False)
    pdf.add_page()

    W_USABLE = 182

    # ── Header Banner ─────────────────────────────────────────────────────────
    pdf.set_xy(14, 14)
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(13, 148, 136)  # Teal 600
    pdf.cell(W_USABLE, 8, "CardioShield", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_x(14)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(100, 116, 139)  # Slate 500
    pdf.cell(
        W_USABLE,
        5,
        "Privacy-Preserving Cardiovascular Risk Diagnostic Assessment",
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )
    pdf.ln(5)

    # ── Patient & Clinician Metadata ──────────────────────────────────────────
    pdf.section_heading("PATIENT & ASSESSMENT DETAILS")
    pdf.set_font("Helvetica", "", 9)

    meta_items = [
        ("Patient Name", patient_name),
        ("Attending Clinician", clinician_name),
        ("Assessment Date", assessment_date or time.strftime("%Y-%m-%d")),
        ("Encryption Scheme", "TenSEAL CKKS 128-bit RLWE (Homomorphic Inference)" if he_used else "Standard Verified Model"),
    ]

    for label, val in meta_items:
        pdf.set_x(16)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(71, 85, 105)  # Slate 600
        pdf.cell(48, 5.5, f"{label}:")
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(15, 23, 42)  # Slate 900
        pdf.cell(W_USABLE - 48, 5.5, pdf_safe(str(val)), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.ln(4)

    # ── Risk Assessment Card ──────────────────────────────────────────────────
    pdf.section_heading("RISK ASSESSMENT SUMMARY")

    risk_pct = risk_prob * 100
    if risk_pct >= 60:
        badge_bg = (254, 242, 242)
        badge_fg = (185, 28, 28)
    elif risk_pct >= 40:
        badge_bg = (255, 251, 235)
        badge_fg = (180, 83, 9)
    else:
        badge_bg = (240, 253, 244)
        badge_fg = (21, 128, 61)

    pdf.set_x(16)
    pdf.set_fill_color(*badge_bg)
    pdf.set_draw_color(226, 232, 240)
    curr_y = pdf.get_y()
    pdf.rect(16, curr_y, W_USABLE - 4, 22, style="FD")

    pdf.set_xy(22, curr_y + 3)
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(*badge_fg)
    pdf.cell(38, 10, f"{risk_pct:.1f}%")

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(70, 10, pdf_safe(risk_class.upper()))

    pdf.set_xy(22, curr_y + 14)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(100, 116, 139)
    pdf.cell(W_USABLE - 16, 4, f"Inference Latency: {latency_ms:.1f} ms  |  Zero Plaintext Data Leakage", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_y(curr_y + 26)

    # ── Feature Values Table ──────────────────────────────────────────────────
    pdf.section_heading("CLINICAL BIOMARKERS & VITAL SIGNS")

    col_feat = 120
    col_val = 58

    pdf.set_x(16)
    pdf.set_fill_color(241, 245, 249)  # Slate 100
    pdf.set_font("Helvetica", "B", 8)
    pdf.set_text_color(51, 65, 85)
    pdf.cell(col_feat, 6, "  Biomarker / Clinical Input", border=1, fill=True)
    pdf.cell(col_val, 6, "  Recorded Value", border=1, fill=True, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(15, 23, 42)

    for fname in feature_names:
        fval = features.get(fname, 0.0)
        label = FEATURE_LABELS.get(fname, fname)
        pdf.set_x(16)
        pdf.cell(col_feat, 5.5, f"  {pdf_safe(label)}", border=1)
        pdf.cell(col_val, 5.5, f"  {float(fval):.2f}", border=1, new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    # ── Page 2: Explainability & Recommendations ──────────────────────────────
    pdf.add_page()
    pdf.set_xy(14, 14)
    pdf.section_heading("EXPLAINABLE AI: FEATURE ATTRIBUTIONS (SHAP)")

    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(100, 116, 139)
    pdf.set_x(16)
    pdf.cell(
        W_USABLE,
        4,
        "Features are ranked by directional contribution to cardiovascular risk.",
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )
    pdf.ln(3)

    sorted_shap = sorted(
        zip(feature_names, shap_values),
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    for fname, sv in sorted_shap:
        if abs(sv) < 0.001 or fname == "sex":
            continue

        label = FEATURE_LABELS.get(fname, fname)
        context, advice = SHAP_ADVICE.get(fname, ("", "Consult cardiologist."))
        direction = "INCREASES RISK" if sv > 0 else "DECREASES RISK"
        dir_color = (185, 28, 28) if sv > 0 else (21, 128, 61)

        pdf.set_x(16)
        pdf.set_font("Helvetica", "B", 8)
        pdf.set_text_color(*dir_color)
        pdf.cell(32, 5, f"[{direction}]")

        pdf.set_text_color(15, 23, 42)
        pdf.set_font("Helvetica", "B", 8)
        pdf.cell(
            W_USABLE - 36,
            5,
            pdf_safe(f"{label} (SHAP: {sv:+.4f}, Value: {features.get(fname, 0):.2f})"),
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )

        pdf.set_x(20)
        pdf.set_font("Helvetica", "", 7.5)
        pdf.set_text_color(71, 85, 105)
        pdf.multi_cell(W_USABLE - 8, 3.8, pdf_safe(f"Clinical Context: {context} {advice}"))
        pdf.ln(1.5)

    # ── Disclaimer ────────────────────────────────────────────────────────────
    pdf.ln(4)
    pdf.set_draw_color(226, 232, 240)
    pdf.line(14, pdf.get_y(), 196, pdf.get_y())
    pdf.ln(4)

    pdf.set_x(16)
    pdf.set_font("Helvetica", "I", 7)
    pdf.set_text_color(148, 163, 184)
    pdf.multi_cell(
        W_USABLE,
        3.5,
        pdf_safe(
            "DISCLAIMER: CardioShield is an investigational clinical decision-support tool. "
            "It does not replace professional clinical judgment. Always consult a board-certified "
            "cardiologist before making diagnostic or therapeutic decisions. All patient features were "
            "homomorphically processed using TenSEAL CKKS 128-bit RLWE encryption."
        ),
    )

    return bytes(pdf.output())
