import React, { useState, useEffect } from 'react';
import { Database, RefreshCw, Trash2, Eye, Lock, User, Calendar, Stethoscope, AlertCircle, X } from 'lucide-react';
import { getPatients, getPatientDetails, deletePatient } from '../api/client';
import { FEATURE_LABELS } from '../utils/constants';

export default function PatientRecords() {
  const [patients, setPatients] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedPatient, setSelectedPatient] = useState(null);
  const [isDetailLoading, setIsDetailLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchPatients = async () => {
    try {
      setIsLoading(true);
      setError(null);
      const data = await getPatients();
      setPatients(data);
    } catch (err) {
      setError('Failed to fetch patient records: ' + err.message);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchPatients();
  }, []);

  const handleViewDetails = async (patientId) => {
    try {
      setIsDetailLoading(true);
      const details = await getPatientDetails(patientId);
      setSelectedPatient(details);
    } catch (err) {
      alert('Failed to load patient details: ' + err.message);
    } finally {
      setIsDetailLoading(false);
    }
  };

  const handleDelete = async (patientId) => {
    if (window.confirm(`Are you sure you want to delete patient #${patientId}? This will cascade and delete all associated encrypted predictions.`)) {
      try {
        await deletePatient(patientId);
        setPatients((prev) => prev.filter((p) => p.id !== patientId));
        if (selectedPatient?.id === patientId) {
          setSelectedPatient(null);
        }
      } catch (err) {
        alert('Failed to delete patient: ' + err.message);
      }
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="clinical-card p-5 sm:p-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <h2 className="text-base font-bold text-slate-900 flex items-center gap-2">
              <Database className="w-5 h-5 text-teal-600" /> Encrypted Patient Database
            </h2>
            <p className="text-xs text-slate-500 mt-1">
              All records are securely encrypted at rest. Zero plaintext clinical data touches persistent storage.
            </p>
          </div>

          <button
            onClick={fetchPatients}
            disabled={isLoading}
            className="clinical-btn-secondary text-xs self-start sm:self-auto"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin' : ''}`} />
            Refresh Records
          </button>
        </div>
      </div>

      {error && (
        <div className="p-4 rounded-lg bg-rose-50 border border-rose-200 text-rose-800 text-xs flex items-center gap-2">
          <AlertCircle className="w-4 h-4 text-rose-600 shrink-0" />
          <span>{error}</span>
        </div>
      )}

      {/* Table */}
      <div className="clinical-card overflow-hidden">
        {isLoading ? (
          <div className="p-12 text-center text-slate-500 text-xs flex items-center justify-center gap-2">
            <RefreshCw className="w-4 h-4 animate-spin text-teal-600" />
            Loading encrypted patient records...
          </div>
        ) : patients.length === 0 ? (
          <div className="p-12 text-center text-slate-500 text-xs">
            <Database className="w-8 h-8 text-slate-300 mx-auto mb-3" />
            <p className="font-semibold text-slate-700">No patient records found in database.</p>
            <p className="text-slate-400 mt-1">Perform a patient assessment with "Save to Database" checked.</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-slate-50 border-b border-slate-200 text-slate-600 font-semibold uppercase tracking-wider">
                <tr>
                  <th className="px-4 py-3">ID</th>
                  <th className="px-4 py-3">Patient Name</th>
                  <th className="px-4 py-3">Clinician</th>
                  <th className="px-4 py-3">Assessment Date</th>
                  <th className="px-4 py-3">Latest Risk Score</th>
                  <th className="px-4 py-3">At-Rest Security</th>
                  <th className="px-4 py-3 text-right">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {patients.map((p) => {
                  const pred = p.latest_prediction;
                  return (
                    <tr key={p.id} className="hover:bg-slate-50/70 transition-colors">
                      <td className="px-4 py-3 font-mono font-bold text-slate-700">#{p.id}</td>
                      <td className="px-4 py-3 font-semibold text-slate-900 flex items-center gap-2">
                        <User className="w-3.5 h-3.5 text-slate-400" />
                        {p.patient_name}
                      </td>
                      <td className="px-4 py-3 text-slate-600">{p.clinician_name}</td>
                      <td className="px-4 py-3 text-slate-500">{p.assessment_date}</td>
                      <td className="px-4 py-3">
                        {pred ? (
                          <span
                            className={`inline-flex items-center px-2 py-0.5 rounded-full font-bold text-[10px] ${
                              pred.risk_score >= 60
                                ? 'bg-rose-100 text-rose-800'
                                : pred.risk_score >= 40
                                ? 'bg-amber-100 text-amber-800'
                                : 'bg-emerald-100 text-emerald-800'
                            }`}
                          >
                            {pred.risk_score.toFixed(1)}% — {pred.risk_class}
                          </span>
                        ) : (
                          <span className="text-slate-400">N/A</span>
                        )}
                      </td>
                      <td className="px-4 py-3">
                        <span className="inline-flex items-center gap-1 text-[11px] text-teal-700 font-medium bg-teal-50 px-2 py-0.5 rounded border border-teal-200">
                          <Lock className="w-3 h-3" /> Encrypted
                        </span>
                      </td>
                      <td className="px-4 py-3 text-right">
                        <div className="flex items-center justify-end gap-2">
                          <button
                            onClick={() => handleViewDetails(p.id)}
                            className="p-1.5 rounded hover:bg-slate-200 text-slate-600 hover:text-slate-900 transition-colors"
                            title="View Decrypted Record"
                          >
                            <Eye className="w-4 h-4" />
                          </button>
                          <button
                            onClick={() => handleDelete(p.id)}
                            className="p-1.5 rounded hover:bg-rose-100 text-slate-400 hover:text-rose-700 transition-colors"
                            title="Delete Record"
                          >
                            <Trash2 className="w-4 h-4" />
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Details Modal */}
      {selectedPatient && (
        <div className="fixed inset-0 z-50 bg-slate-900/40 backdrop-blur-xs flex items-center justify-center p-4">
          <div className="bg-white rounded-xl shadow-2xl max-w-2xl w-full max-h-[90vh] overflow-y-auto border border-slate-200 animate-in fade-in zoom-in-95 duration-150">
            <div className="p-6 border-b border-slate-100 flex items-center justify-between sticky top-0 bg-white z-10">
              <div>
                <span className="text-[11px] font-mono text-teal-700 font-bold uppercase tracking-wider">
                  Decrypted Clinical View (Patient #{selectedPatient.id})
                </span>
                <h3 className="text-lg font-bold text-slate-900">{selectedPatient.patient_name}</h3>
              </div>
              <button
                onClick={() => setSelectedPatient(null)}
                className="p-1.5 rounded-lg hover:bg-slate-100 text-slate-400 hover:text-slate-700"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="p-6 space-y-6">
              {/* Metadata */}
              <div className="grid grid-cols-3 gap-3 p-3 rounded-lg bg-slate-50 text-xs">
                <div>
                  <span className="text-slate-400 block text-[10px] uppercase font-semibold">Clinician</span>
                  <span className="font-semibold text-slate-800">{selectedPatient.clinician_name}</span>
                </div>
                <div>
                  <span className="text-slate-400 block text-[10px] uppercase font-semibold">Date</span>
                  <span className="font-semibold text-slate-800">{selectedPatient.assessment_date}</span>
                </div>
                <div>
                  <span className="text-slate-400 block text-[10px] uppercase font-semibold">Database Security</span>
                  <span className="font-semibold text-teal-700">Encrypted at Rest</span>
                </div>
              </div>

              {/* Decrypted Clinical Features */}
              <div>
                <h4 className="text-xs font-bold uppercase tracking-wider text-slate-700 mb-3">
                  Decrypted Clinical Feature Set
                </h4>
                <div className="grid grid-cols-2 sm:grid-cols-3 gap-2.5">
                  {selectedPatient.features &&
                    Object.entries(selectedPatient.features).map(([key, val]) => (
                      <div key={key} className="p-2.5 rounded-md border border-slate-100 bg-slate-50/50">
                        <span className="text-[10px] text-slate-500 uppercase font-semibold block">
                          {key}
                        </span>
                        <span className="text-sm font-bold text-slate-900 font-mono">
                          {typeof val === 'number' ? val.toFixed(1) : val}
                        </span>
                      </div>
                    ))}
                </div>
              </div>

              {/* Prediction History */}
              {selectedPatient.predictions && selectedPatient.predictions.length > 0 && (
                <div>
                  <h4 className="text-xs font-bold uppercase tracking-wider text-slate-700 mb-3">
                    Homomorphic Prediction History
                  </h4>
                  <div className="space-y-2">
                    {selectedPatient.predictions.map((pred) => (
                      <div
                        key={pred.id}
                        className="p-3 rounded-lg border border-slate-200 bg-slate-50 flex items-center justify-between text-xs"
                      >
                        <div>
                          <div className="flex items-center gap-2">
                            <span className="font-bold text-slate-900 text-sm">
                              {pred.risk_score_pct.toFixed(1)}% Risk
                            </span>
                            <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-slate-200 text-slate-800">
                              {pred.risk_class}
                            </span>
                          </div>
                          <span className="text-[11px] text-slate-500">
                            Latency: {pred.total_time_ms.toFixed(1)} ms | HE Inference: {pred.inference_time_ms.toFixed(1)} ms
                          </span>
                        </div>
                        <span className="text-[10px] font-mono text-slate-400">
                          {new Date(pred.created_at * 1000).toLocaleString()}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            <div className="p-4 border-t border-slate-100 bg-slate-50 flex justify-end">
              <button
                onClick={() => setSelectedPatient(null)}
                className="clinical-btn-secondary text-xs"
              >
                Close View
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
