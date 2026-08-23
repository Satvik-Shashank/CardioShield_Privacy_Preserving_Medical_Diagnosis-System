import React, { useState } from 'react';
import { Download, ShieldCheck, Clock, FileText, CheckCircle2, AlertTriangle, AlertCircle, RefreshCw } from 'lucide-react';
import ShapExplainer from './ShapExplainer';
import CiphertextProof from './CiphertextProof';
import { downloadPdfReport } from '../api/client';

export default function ResultsPanel({ result, onReset }) {
  const [isDownloading, setIsDownloading] = useState(false);

  if (!result) {
    return (
      <div className="clinical-card p-12 text-center">
        <div className="w-12 h-12 rounded-full bg-slate-100 flex items-center justify-center text-slate-400 mx-auto mb-4">
          <FileText className="w-6 h-6" />
        </div>
        <h3 className="text-base font-bold text-slate-800 mb-1">No Assessment Results Available</h3>
        <p className="text-xs text-slate-500 max-w-sm mx-auto mb-6">
          Please complete and submit the Patient Assessment form to run the TenSEAL CKKS inference engine.
        </p>
        <button
          onClick={onReset}
          className="clinical-btn-primary text-xs"
        >
          Go to Assessment Form
        </button>
      </div>
    );
  }

  const {
    risk_score_pct,
    risk_class,
    he_prob,
    plain_prob,
    prob_delta,
    shap_values,
    feature_values,
    hex_proof,
    ciphertext_bytes,
    t_enc_ms,
    t_he_ms,
    t_dec_ms,
    t_total_ms,
    patient_id,
  } = result;

  const handleDownloadPdf = async () => {
    try {
      setIsDownloading(true);
      const payload = {
        patient_name: result.patient_name || 'Patient',
        clinician_name: result.clinician_name || 'Attending Physician',
        assessment_date: result.assessment_date || new Date().toISOString().split('T')[0],
        features: feature_values,
        risk_prob: he_prob,
        risk_class: risk_class,
        shap_values: shap_values,
        latency_ms: t_total_ms,
        he_used: true,
      };

      const blob = await downloadPdfReport(payload);
      const url = window.URL.createObjectURL(new Blob([blob]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', `CardioShield_Report_${Date.now()}.pdf`);
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (err) {
      alert('Failed to generate PDF report: ' + err.message);
    } finally {
      setIsDownloading(false);
    }
  };

  // Color mapping based on risk triage
  let triageConfig = {
    bg: 'bg-emerald-50 border-emerald-200 text-emerald-800',
    badge: 'bg-emerald-600 text-white',
    icon: CheckCircle2,
    accent: '#059669',
  };

  if (risk_score_pct >= 60) {
    triageConfig = {
      bg: 'bg-rose-50 border-rose-200 text-rose-800',
      badge: 'bg-rose-600 text-white',
      icon: AlertCircle,
      accent: '#DC2626',
    };
  } else if (risk_score_pct >= 40) {
    triageConfig = {
      bg: 'bg-amber-50 border-amber-200 text-amber-800',
      badge: 'bg-amber-600 text-white',
      icon: AlertTriangle,
      accent: '#D97706',
    };
  }

  const TriageIcon = triageConfig.icon;

  return (
    <div className="space-y-6">
      {/* Top Banner with Risk Score & Actions */}
      <div className={`clinical-card p-6 border ${triageConfig.bg} relative overflow-hidden`}>
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
          <div className="flex items-start gap-4">
            <div className="p-3 rounded-xl bg-white shadow-xs">
              <TriageIcon className="w-8 h-8" style={{ color: triageConfig.accent }} />
            </div>
            <div>
              <div className="flex items-center gap-2 mb-1">
                <span className={`text-xs font-bold uppercase tracking-wider px-2.5 py-0.5 rounded-full ${triageConfig.badge}`}>
                  {risk_class}
                </span>
                {patient_id && (
                  <span className="text-[11px] font-mono text-slate-500 bg-white px-2 py-0.5 rounded-md border border-slate-200">
                    Encrypted DB Record #{patient_id}
                  </span>
                )}
              </div>
              <div className="flex items-baseline gap-3">
                <span className="text-4xl font-extrabold tracking-tight font-display text-slate-900">
                  {risk_score_pct.toFixed(1)}%
                </span>
                <span className="text-xs text-slate-600 font-medium">Estimated 10-Year Cardiovascular Event Risk</span>
              </div>
              <p className="text-xs text-slate-600 mt-2 max-w-xl">
                Diagnostic score computed homomorphically via TenSEAL CKKS 128-bit RLWE encryption. No raw clinical biomarkers were decrypted on the inference server.
              </p>
            </div>
          </div>

          {/* Action Button */}
          <div className="flex flex-wrap gap-2.5 shrink-0">
            <button
              onClick={handleDownloadPdf}
              disabled={isDownloading}
              className="clinical-btn-primary text-xs shadow-md"
            >
              {isDownloading ? (
                <>
                  <RefreshCw className="w-4 h-4 animate-spin" />
                  Generating PDF...
                </>
              ) : (
                <>
                  <Download className="w-4 h-4" />
                  Download Clinical Report (.pdf)
                </>
              )}
            </button>
            <button
              onClick={onReset}
              className="clinical-btn-secondary text-xs"
            >
              New Assessment
            </button>
          </div>
        </div>
      </div>

      {/* Latency & Encryption Performance Row */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
        <div className="clinical-card p-4">
          <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider block mb-1">
            Client Vector Encryption
          </span>
          <span className="text-xl font-bold font-display text-slate-800">{t_enc_ms.toFixed(1)} ms</span>
          <span className="text-[10px] text-slate-400 block mt-0.5">13 features to CKKS</span>
        </div>

        <div className="clinical-card p-4">
          <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider block mb-1">
            HE Ciphertext Inference
          </span>
          <span className="text-xl font-bold font-display text-teal-700">{t_he_ms.toFixed(1)} ms</span>
          <span className="text-[10px] text-slate-400 block mt-0.5">Dot product + Poly σ</span>
        </div>

        <div className="clinical-card p-4">
          <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider block mb-1">
            Scalar Decryption
          </span>
          <span className="text-xl font-bold font-display text-slate-800">{t_dec_ms.toFixed(1)} ms</span>
          <span className="text-[10px] text-slate-400 block mt-0.5">Final scalar probability</span>
        </div>

        <div className="clinical-card p-4">
          <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider block mb-1">
            HE vs Plain Delta
          </span>
          <span className="text-xl font-bold font-display text-emerald-700">{(prob_delta * 100).toFixed(2)} pp</span>
          <span className="text-[10px] text-slate-400 block mt-0.5">Preserved classification</span>
        </div>
      </div>

      {/* SHAP Feature Attributions Panel */}
      <ShapExplainer
        shapValues={shap_values}
        featureValues={feature_values}
      />

      {/* Live Ciphertext Proof Panel */}
      <CiphertextProof
        hexProof={hex_proof}
        ciphertextBytes={ciphertext_bytes}
      />
    </div>
  );
}
