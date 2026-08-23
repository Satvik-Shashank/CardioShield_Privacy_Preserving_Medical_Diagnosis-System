import React from 'react';
import { ShieldCheck, Activity, Users, BarChart3, Info, Lock } from 'lucide-react';

export default function Navbar({ activeTab, setActiveTab, healthStatus }) {
  const tabs = [
    { id: 'form', label: 'Patient Assessment', icon: Activity },
    { id: 'results', label: 'Clinical Analysis', icon: ShieldCheck },
    { id: 'records', label: 'Patient Records', icon: Users },
    { id: 'metrics', label: 'System Metrics', icon: BarChart3 },
  ];

  const isHealthy = healthStatus?.status === 'HEALTHY';

  return (
    <header className="bg-white border-b border-slate-200 sticky top-0 z-50 shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Brand */}
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-lg bg-teal-600 flex items-center justify-center text-white shadow-sm">
              <ShieldCheck className="w-6 h-6" />
            </div>
            <div>
              <span className="font-display text-xl font-bold tracking-tight text-slate-900">
                CardioShield
              </span>
              <p className="text-xs text-slate-500 hidden sm:block">
                Privacy-Preserving Cardiovascular Risk Diagnostic System
              </p>
            </div>
          </div>

          {/* Navigation Tabs */}
          <nav className="flex space-x-1 sm:space-x-2">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`inline-flex items-center gap-2 px-3 sm:px-3.5 py-2 rounded-lg text-xs sm:text-sm font-medium transition-colors ${
                    isActive
                      ? 'bg-teal-50 text-teal-700 font-semibold shadow-xs'
                      : 'text-slate-600 hover:text-slate-900 hover:bg-slate-100'
                  }`}
                >
                  <Icon className={`w-4 h-4 ${isActive ? 'text-teal-600' : 'text-slate-400'}`} />
                  <span className="hidden md:inline">{tab.label}</span>
                </button>
              );
            })}
          </nav>

          {/* Status Badge */}
          <div className="hidden lg:flex items-center gap-2 px-3 py-1.5 rounded-full bg-slate-100 border border-slate-200 text-xs">
            <span className={`w-2 h-2 rounded-full ${isHealthy ? 'bg-emerald-500 animate-pulse' : 'bg-amber-500'}`}></span>
            <span className="font-medium text-slate-700">
              {isHealthy ? 'TenSEAL CKKS 128-bit Active' : 'HE Engine Initializing...'}
            </span>
          </div>
        </div>
      </div>
    </header>
  );
}
