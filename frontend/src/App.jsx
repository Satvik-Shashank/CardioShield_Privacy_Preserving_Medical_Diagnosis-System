import React, { useState, useEffect } from 'react';
import Navbar from './components/Navbar';
import HeroSection from './components/HeroSection';
import PatientForm from './components/PatientForm';
import ResultsPanel from './components/ResultsPanel';
import PatientRecords from './components/PatientRecords';
import MetricsDashboard from './components/MetricsDashboard';
import { getHealth, predictRisk } from './api/client';
import { ShieldCheck, Heart, Lock, AlertCircle, FileCode, CheckCircle } from 'lucide-react';

export default function App() {
  const [activeTab, setActiveTab] = useState('form');
  const [healthStatus, setHealthStatus] = useState(null);
  const [predictionResult, setPredictionResult] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    let intervalId;
    async function checkBackend() {
      try {
        const res = await getHealth();
        setHealthStatus(res);
      } catch (err) {
        setHealthStatus({ status: 'CONNECTING' });
      }
    }

    checkBackend();
    intervalId = setInterval(checkBackend, 5000);

    return () => clearInterval(intervalId);
  }, []);

  const handleAssessmentSubmit = async (formData) => {
    try {
      setIsLoading(true);
      setError(null);
      const res = await predictRisk(formData);
      // Merge patient metadata into result for PDF generation
      setPredictionResult({
        ...res,
        patient_name: formData.patient_name,
        clinician_name: formData.clinician_name,
        assessment_date: formData.assessment_date,
      });
      setActiveTab('results');
    } catch (err) {
      setError(err.response?.data?.detail || err.message || 'Homomorphic prediction failed.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 flex flex-col font-sans">
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        healthStatus={healthStatus}
      />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Error notification */}
        {error && (
          <div className="mb-6 p-4 rounded-xl bg-rose-50 border border-rose-200 text-rose-900 text-sm flex items-start gap-3 shadow-xs">
            <AlertCircle className="w-5 h-5 text-rose-600 shrink-0 mt-0.5" />
            <div>
              <h4 className="font-bold text-xs uppercase tracking-wider text-rose-800">Inference Error</h4>
              <p className="text-xs text-rose-700 mt-0.5">{error}</p>
            </div>
          </div>
        )}

        {/* Tab 1: Form & Home */}
        {activeTab === 'form' && (
          <div>
            <HeroSection onStartAssessment={() => {}} />
            <PatientForm onSubmit={handleAssessmentSubmit} isLoading={isLoading} />
          </div>
        )}

        {/* Tab 2: Clinical Results */}
        {activeTab === 'results' && (
          <ResultsPanel
            result={predictionResult}
            onReset={() => setActiveTab('form')}
          />
        )}

        {/* Tab 3: Encrypted Patient Records */}
        {activeTab === 'records' && <PatientRecords />}

        {/* Tab 4: Metrics Dashboard */}
        {activeTab === 'metrics' && <MetricsDashboard />}


      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-slate-200 py-6 mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center text-xs text-slate-500">
          <div className="flex items-center justify-center gap-2 mb-2">
            <ShieldCheck className="w-4 h-4 text-teal-600" />
            <span className="font-semibold text-slate-700">CardioShield Privacy-Preserving Medical Diagnosis System</span>
          </div>
          <p className="text-[11px] text-slate-400">
            Research and clinical decision-support prototype. Compliant with 128-bit RLWE Homomorphic Cryptographic Standards.
          </p>
        </div>
      </footer>
    </div>
  );
}
