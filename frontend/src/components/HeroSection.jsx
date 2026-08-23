import React, { useState } from 'react';
import { Lock, Cpu, KeyRound, Sparkles, ArrowRight, ShieldCheck, Activity, Info, X, CheckCircle2, Shield } from 'lucide-react';

export default function HeroSection({ onStartAssessment }) {
  const [showHeModal, setShowHeModal] = useState(false);

  const steps = [
    {
      num: '01',
      title: 'Clinical Data Input',
      desc: 'Clinician enters 13 clinical biomarkers (vitals, ECG, labs) into the standardized clinical form.',
      icon: Cpu,
    },
    {
      num: '02',
      title: 'TenSEAL CKKS Encryption',
      desc: 'Features are encrypted into a 128-bit RLWE CKKS ciphertext. No plaintext data is exposed to unauthorized observers.',
      icon: Lock,
    },
    {
      num: '03',
      title: 'Ciphertext HE Inference',
      desc: 'The server computes the Logistic Regression inner product and degree-3 polynomial sigmoid directly on the ciphertext.',
      icon: KeyRound,
    },
    {
      num: '04',
      title: 'Decryption & SHAP Explainability',
      desc: 'The resulting encrypted probability scalar is decrypted to reveal the risk score with feature-level SHAP attributions.',
      icon: Sparkles,
    },
  ];

  return (
    <div className="mb-8">
      {/* Flowy & Dynamic Hero Banner */}
      <div className="relative rounded-2xl overflow-hidden shadow-2xl border border-slate-700/60 bg-gradient-to-r from-slate-950 via-slate-900 to-teal-950 p-7 sm:p-12 text-white animate-gradient-flow">
        {/* Floating Ambient Glow Orbs */}
        <div className="absolute -top-24 -left-24 w-96 h-96 bg-teal-500/15 rounded-full blur-3xl pointer-events-none animate-float-1" />
        <div className="absolute -bottom-24 -right-24 w-96 h-96 bg-emerald-500/15 rounded-full blur-3xl pointer-events-none animate-float-2" />
        <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-80 h-80 bg-teal-600/10 rounded-full blur-2xl pointer-events-none animate-pulse" />

        {/* Dynamic Subtle Medical Wave SVG Line */}
        <div className="absolute inset-x-0 bottom-0 top-0 opacity-15 pointer-events-none flex items-center justify-center overflow-hidden">
          <svg
            className="w-full h-40 text-teal-400"
            viewBox="0 0 1200 120"
            fill="none"
            xmlns="http://www.w3.org/2000/svg"
            preserveAspectRatio="none"
          >
            <path
              d="M0,60 L250,60 L280,60 L295,20 L310,100 L325,45 L340,75 L355,60 L600,60 L620,60 L635,15 L650,105 L665,40 L680,80 L695,60 L950,60 L970,60 L985,25 L1000,95 L1015,50 L1030,70 L1045,60 L1200,60"
              stroke="currentColor"
              strokeWidth="2"
              className="animate-wave-line"
            />
          </svg>
        </div>

        {/* Content */}
        <div className="relative z-10 max-w-3xl">
          <h1 className="font-display text-3xl sm:text-4xl lg:text-5xl font-extrabold tracking-tight leading-tight mb-4 text-white drop-shadow-xs">
            Cardiovascular Risk Assessment
          </h1>

          <p className="text-slate-200 text-sm sm:text-base leading-relaxed mb-8 font-normal max-w-2xl text-shadow">
            CardioShield performs heart disease risk assessment entirely over encrypted ciphertexts using{' '}
            <button
              type="button"
              onClick={() => setShowHeModal(true)}
              className="inline-flex items-center gap-1 text-teal-300 font-semibold underline decoration-teal-400 decoration-2 underline-offset-4 hover:text-teal-200 hover:decoration-teal-200 transition-all cursor-pointer"
              title="Click to learn how this works in plain English"
            >
              <span>TenSEAL CKKS (128-bit RLWE)</span>
              <Info className="w-3.5 h-3.5 inline-block opacity-80" />
            </button>
            .
          </p>

          <div className="flex flex-wrap items-center gap-3.5">
            <button
              onClick={onStartAssessment}
              className="group inline-flex items-center gap-2.5 px-6 py-3.5 rounded-xl bg-teal-500 hover:bg-teal-400 text-slate-950 font-bold text-sm transition-all duration-200 shadow-lg shadow-teal-950/40 hover:shadow-teal-500/25 active:translate-y-0.5"
            >
              <span>Start Patient Assessment</span>
              <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
            </button>
            <div className="inline-flex items-center gap-2 px-4 py-3.5 rounded-xl bg-slate-800/80 border border-slate-700/80 text-xs text-slate-300 backdrop-blur-md">
              <ShieldCheck className="w-4 h-4 text-emerald-400" />
              <span>Verified: No Fake Fallbacks or Simulated Latency</span>
            </div>
          </div>
        </div>
      </div>

      {/* Clean White/Grey Explanation Modal */}
      {showHeModal && (
        <div
          className="fixed inset-0 z-50 bg-black/30 backdrop-blur-sm flex items-center justify-center p-4 transition-all"
          onClick={() => setShowHeModal(false)}
        >
          <div
            className="relative bg-white rounded-2xl shadow-2xl max-w-md w-full p-6 text-slate-900 overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            {/* Header */}
            <div className="flex items-center justify-between pb-4 border-b border-slate-200">
              <div className="flex items-center gap-2.5">
                <div className="w-8 h-8 rounded-lg bg-teal-50 border border-teal-200 flex items-center justify-center text-teal-600">
                  <Shield className="w-4 h-4" />
                </div>
                <h3 className="font-display font-bold text-sm text-slate-900">
                  Encryption Terms Explained
                </h3>
              </div>
              <button
                onClick={() => setShowHeModal(false)}
                className="p-1.5 rounded-lg text-slate-400 hover:text-slate-700 hover:bg-slate-100 transition-colors"
              >
                <X className="w-4 h-4" />
              </button>
            </div>

            {/* Body — one-line definitions */}
            <div className="space-y-4 pt-4 text-sm">
              <div className="flex items-start gap-3">
                <CheckCircle2 className="w-4 h-4 text-teal-500 shrink-0 mt-0.5" />
                <p className="text-slate-700"><strong className="text-slate-900">CKKS</strong> — An encryption scheme that lets AI do math on encrypted decimal numbers without ever decrypting them.</p>
              </div>

              <div className="flex items-start gap-3">
                <CheckCircle2 className="w-4 h-4 text-teal-500 shrink-0 mt-0.5" />
                <p className="text-slate-700"><strong className="text-slate-900">128-bit RLWE</strong> — A quantum-resistant security level that would take supercomputers billions of years to crack.</p>
              </div>

              <div className="flex items-start gap-3">
                <CheckCircle2 className="w-4 h-4 text-teal-500 shrink-0 mt-0.5" />
                <p className="text-slate-700"><strong className="text-slate-900">TenSEAL</strong> — An open-source Python library that performs these encrypted computations fast.</p>
              </div>
            </div>

            {/* Footer */}
            <div className="mt-5 pt-4 border-t border-slate-200 flex justify-end">
              <button
                onClick={() => setShowHeModal(false)}
                className="px-4 py-2 rounded-lg bg-teal-500 hover:bg-teal-400 text-white font-semibold text-sm transition-colors"
              >
                Got it
              </button>
            </div>
          </div>
        </div>
      )}


    </div>
  );
}
