import React, { useState } from 'react';
import { FEATURE_CONFIGS, SAMPLE_PRESETS } from '../utils/constants';
import { Lock, Sparkles, Database, RefreshCw, CheckCircle2, User, Stethoscope, Calendar } from 'lucide-react';

export default function PatientForm({ onSubmit, isLoading }) {
  const [patientName, setPatientName] = useState('');
  const [clinicianName, setClinicianName] = useState('');
  const [assessmentDate, setAssessmentDate] = useState(new Date().toISOString().split('T')[0]);
  const [saveToDb, setSaveToDb] = useState(true);

  // Initialize form state with defaults
  const [formData, setFormData] = useState(() => {
    const initial = {};
    FEATURE_CONFIGS.forEach((f) => {
      initial[f.name] = f.default;
    });
    return initial;
  });

  const handleInputChange = (name, value) => {
    setFormData((prev) => ({
      ...prev,
      [name]: parseFloat(value),
    }));
  };

  const applyPreset = (preset) => {
    setFormData(preset.data);
  };

  const handleSubmit = (e) => {
    e.preventDefault();
    onSubmit({
      patient_name: patientName.trim() || 'Patient Record',
      clinician_name: clinicianName.trim() || 'Attending Clinician',
      assessment_date: assessmentDate,
      save_to_db: saveToDb,
      features: formData,
    });
  };

  // Group features by section
  const demographicsFeatures = FEATURE_CONFIGS.filter((f) => ['demographics', 'symptoms'].includes(f.section));
  const vitalsFeatures = FEATURE_CONFIGS.filter((f) => f.section === 'vitals');
  const diagnosticsFeatures = FEATURE_CONFIGS.filter((f) => f.section === 'diagnostics');

  return (
    <div className="space-y-6">
      {/* Presets Bar */}
      <div className="clinical-card p-4 sm:p-5 bg-gradient-to-r from-slate-50 to-teal-50/40">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <span className="text-xs font-bold uppercase tracking-wider text-teal-800 flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5" /> Clinical Test Presets
            </span>
            <p className="text-xs text-slate-600 mt-0.5">Quickly populate inputs with validated medical cases</p>
          </div>
          <div className="flex flex-wrap gap-2">
            {SAMPLE_PRESETS.map((preset) => (
              <button
                key={preset.id}
                type="button"
                onClick={() => applyPreset(preset)}
                className="px-3 py-1.5 rounded-lg border border-teal-200 bg-white hover:bg-teal-50 text-xs font-semibold text-teal-900 transition-colors shadow-2xs hover:border-teal-300"
              >
                {preset.title}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Main Form */}
      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Patient & Clinician Details */}
        <div className="clinical-card p-5 sm:p-6">
          <h2 className="text-sm font-bold uppercase tracking-wider text-slate-800 mb-4 pb-2 border-b border-slate-100 flex items-center gap-2">
            <User className="w-4 h-4 text-teal-600" /> Patient & Assessment Metadata
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1.5">Patient Identifier / Name</label>
              <input
                type="text"
                value={patientName}
                onChange={(e) => setPatientName(e.target.value)}
                className="clinical-input"
                placeholder="Enter your name"
                required
              />
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1.5">Attending Clinician</label>
              <input
                type="text"
                value={clinicianName}
                onChange={(e) => setClinicianName(e.target.value)}
                className="clinical-input"
                placeholder="Enter clinician name"
                required
              />
            </div>
            <div>
              <label className="block text-xs font-semibold text-slate-700 mb-1.5">Assessment Date</label>
              <input
                type="date"
                value={assessmentDate}
                onChange={(e) => setAssessmentDate(e.target.value)}
                className="clinical-input"
                required
              />
            </div>
          </div>

          <div className="mt-4 pt-4 border-t border-slate-100 flex items-center justify-between">
            <label className="flex items-center gap-2.5 cursor-pointer">
              <input
                type="checkbox"
                checked={saveToDb}
                onChange={(e) => setSaveToDb(e.target.checked)}
                className="w-4 h-4 text-teal-600 rounded border-slate-300 focus:ring-teal-500"
              />
              <span className="text-xs font-medium text-slate-700">
                Save record to patient database
              </span>
            </label>
            <span className="text-[11px] text-slate-400">Zero plaintext storage</span>
          </div>
        </div>

        {/* 3-Column Clinical Biomarkers */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Section 1: Demographics & Symptoms */}
          <div className="clinical-card p-5">
            <h3 className="text-xs font-bold uppercase tracking-wider text-teal-800 mb-4 pb-2 border-b border-slate-100 flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-teal-500"></span> Demographics & Symptoms
            </h3>
            <div className="space-y-4">
              {demographicsFeatures.map((f) => (
                <div key={f.name}>
                  <div className="flex items-center justify-between mb-1">
                    <label className="text-xs font-semibold text-slate-700">{f.label}</label>
                    {f.unit && <span className="text-[11px] text-slate-400 font-mono">{f.unit}</span>}
                  </div>
                  {f.type === 'select' ? (
                    <select
                      value={formData[f.name]}
                      onChange={(e) => handleInputChange(f.name, e.target.value)}
                      className="clinical-input"
                    >
                      {f.options.map((opt) => (
                        <option key={opt.value} value={opt.value}>
                          {opt.label}
                        </option>
                      ))}
                    </select>
                  ) : (
                    <input
                      type="number"
                      min={f.min}
                      max={f.max}
                      step={f.step}
                      value={formData[f.name]}
                      onChange={(e) => handleInputChange(f.name, e.target.value)}
                      className="clinical-input"
                      placeholder="Enter value"
                      required
                    />
                  )}
                  <p className="text-[11px] text-slate-400 mt-1">{f.description}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Section 2: Vitals & Laboratory */}
          <div className="clinical-card p-5">
            <h3 className="text-xs font-bold uppercase tracking-wider text-teal-800 mb-4 pb-2 border-b border-slate-100 flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-teal-500"></span> Vitals & Laboratory
            </h3>
            <div className="space-y-4">
              {vitalsFeatures.map((f) => (
                <div key={f.name}>
                  <div className="flex items-center justify-between mb-1">
                    <label className="text-xs font-semibold text-slate-700">{f.label}</label>
                    {f.unit && <span className="text-[11px] text-slate-400 font-mono">{f.unit}</span>}
                  </div>
                  {f.type === 'select' ? (
                    <select
                      value={formData[f.name]}
                      onChange={(e) => handleInputChange(f.name, e.target.value)}
                      className="clinical-input"
                    >
                      {f.options.map((opt) => (
                        <option key={opt.value} value={opt.value}>
                          {opt.label}
                        </option>
                      ))}
                    </select>
                  ) : (
                    <input
                      type="number"
                      min={f.min}
                      max={f.max}
                      step={f.step}
                      value={formData[f.name]}
                      onChange={(e) => handleInputChange(f.name, e.target.value)}
                      className="clinical-input"
                      required
                    />
                  )}
                  <p className="text-[11px] text-slate-400 mt-1">{f.description}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Section 3: ECG & Diagnostics */}
          <div className="clinical-card p-5">
            <h3 className="text-xs font-bold uppercase tracking-wider text-teal-800 mb-4 pb-2 border-b border-slate-100 flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-teal-500"></span> ECG & Imaging Findings
            </h3>
            <div className="space-y-4">
              {diagnosticsFeatures.map((f) => (
                <div key={f.name}>
                  <div className="flex items-center justify-between mb-1">
                    <label className="text-xs font-semibold text-slate-700">{f.label}</label>
                    {f.unit && <span className="text-[11px] text-slate-400 font-mono">{f.unit}</span>}
                  </div>
                  {f.type === 'select' ? (
                    <select
                      value={formData[f.name]}
                      onChange={(e) => handleInputChange(f.name, e.target.value)}
                      className="clinical-input"
                    >
                      {f.options.map((opt) => (
                        <option key={opt.value} value={opt.value}>
                          {opt.label}
                        </option>
                      ))}
                    </select>
                  ) : (
                    <input
                      type="number"
                      min={f.min}
                      max={f.max}
                      step={f.step}
                      value={formData[f.name]}
                      onChange={(e) => handleInputChange(f.name, e.target.value)}
                      className="clinical-input"
                      required
                    />
                  )}
                  <p className="text-[11px] text-slate-400 mt-1">{f.description}</p>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Submit Actions */}
        <div className="clinical-card p-6 bg-slate-900 text-white flex flex-col sm:flex-row items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2 text-teal-400 text-xs font-bold uppercase tracking-wider mb-1">
              <Lock className="w-4 h-4" /> TenSEAL CKKS Vector Encryption Ready
            </div>
            <p className="text-xs text-slate-300 max-w-xl">
              13 clinical biomarkers will be scaled, homomorphically evaluated on ciphertext, and decrypted into actionable risk stratifications.
            </p>
          </div>

          <button
            type="submit"
            disabled={isLoading}
            className="w-full sm:w-auto px-8 py-3.5 rounded-lg bg-teal-500 hover:bg-teal-400 disabled:bg-slate-700 text-slate-950 font-bold text-sm transition-all shadow-lg flex items-center justify-center gap-2 shrink-0 cursor-pointer disabled:cursor-not-allowed"
          >
            {isLoading ? (
              <>
                <RefreshCw className="w-4 h-4 animate-spin text-slate-950" />
                <span>Computing CKKS Ciphertext...</span>
              </>
            ) : (
              <>
                <Lock className="w-4 h-4" />
                <span>Execute Encrypted Analysis</span>
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  );
}
