import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_URL
  ? `${import.meta.env.VITE_API_URL.replace(/\/$/, '')}/api`
  : '/api';

const api = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000,
});

export const getHealth = async () => {
  const res = await api.get('/health');
  return res.data;
};

export const getMetrics = async () => {
  const res = await api.get('/metrics');
  return res.data;
};

export const predictRisk = async (payload) => {
  const res = await api.post('/predict', payload);
  return res.data;
};

export const getPatients = async () => {
  const res = await api.get('/patients');
  return res.data;
};

export const getPatientDetails = async (patientId) => {
  const res = await api.get(`/patients/${patientId}`);
  return res.data;
};

export const deletePatient = async (patientId) => {
  const res = await api.delete(`/patients/${patientId}`);
  return res.data;
};

export const downloadPdfReport = async (payload) => {
  const res = await api.post('/report/pdf', payload, {
    responseType: 'blob',
  });
  return res.data;
};

export default api;
