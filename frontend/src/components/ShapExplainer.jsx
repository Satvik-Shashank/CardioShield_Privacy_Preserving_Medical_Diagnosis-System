import React from 'react';
import { Sparkles, TrendingUp, TrendingDown, Info } from 'lucide-react';
import { SHAP_ADVICE } from '../utils/constants';

export default function ShapExplainer({ shapValues, featureValues }) {
  if (!shapValues || Object.keys(shapValues).length === 0) {
    return null;
  }

  // Sort features by absolute impact magnitude
  const sortedFeatures = Object.entries(shapValues)
    .filter(([key]) => key !== 'sex') // filter sex from prominent listing if desired, or include all
    .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));

  const maxImpact = Math.max(...sortedFeatures.map(([, val]) => Math.abs(val)), 0.05);

  return (
    <div className="clinical-card p-5 sm:p-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-6 pb-3 border-b border-slate-100">
        <div>
          <h3 className="text-sm font-bold uppercase tracking-wider text-slate-900 flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-teal-600" /> Explainable AI: Biomarker Risk Attributions (SHAP)
          </h3>
          <p className="text-xs text-slate-500 mt-0.5">
            Features ranked by directional contribution to the predicted cardiovascular risk.
          </p>
        </div>
        <div className="flex items-center gap-4 text-xs">
          <span className="flex items-center gap-1.5 text-rose-700 font-medium">
            <span className="w-3 h-3 rounded-xs bg-rose-500"></span> Increases Risk
          </span>
          <span className="flex items-center gap-1.5 text-emerald-700 font-medium">
            <span className="w-3 h-3 rounded-xs bg-emerald-500"></span> Decreases Risk
          </span>
        </div>
      </div>

      <div className="space-y-4">
        {sortedFeatures.map(([featKey, shapVal]) => {
          const isIncreasing = shapVal > 0;
          const absVal = Math.abs(shapVal);
          const barWidth = Math.min(100, Math.max(6, (absVal / maxImpact) * 100));
          const recordedVal = featureValues ? featureValues[featKey] : undefined;
          const adviceInfo = SHAP_ADVICE[featKey] || {
            title: featKey,
            context: 'Biomarker influence calculated by LinearExplainer.',
            advice: 'Consult physician for personalized guidance.',
          };

          return (
            <div
              key={featKey}
              className="p-3.5 rounded-lg border border-slate-100 bg-slate-50/50 hover:bg-white hover:border-slate-200 transition-all shadow-2xs"
            >
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 mb-2">
                <div className="flex items-center gap-2">
                  <span className="font-bold text-xs text-slate-900">{adviceInfo.title}</span>
                  {recordedVal !== undefined && (
                    <span className="text-[11px] font-mono text-slate-500 bg-slate-200/70 px-1.5 py-0.5 rounded">
                      Value: {typeof recordedVal === 'number' ? recordedVal.toFixed(1) : recordedVal}
                    </span>
                  )}
                </div>

                <div className="flex items-center gap-2 text-xs font-mono font-semibold">
                  <span className={isIncreasing ? 'text-rose-700' : 'text-emerald-700'}>
                    {isIncreasing ? '+' : ''}{shapVal.toFixed(4)}
                  </span>
                  <span className={`text-[10px] uppercase font-sans font-bold px-1.5 py-0.5 rounded ${
                    isIncreasing ? 'bg-rose-100 text-rose-800' : 'bg-emerald-100 text-emerald-800'
                  }`}>
                    {isIncreasing ? 'Raises Risk' : 'Protective'}
                  </span>
                </div>
              </div>

              {/* Visual Horizontal Centered Impact Bar */}
              <div className="h-2 w-full bg-slate-200 rounded-full overflow-hidden relative mb-2">
                <div
                  className={`h-full rounded-full transition-all duration-500 ${
                    isIncreasing
                      ? 'bg-rose-500 ml-auto'
                      : 'bg-emerald-500 mr-auto'
                  }`}
                  style={{ width: `${barWidth}%` }}
                />
              </div>

              {/* Clinical Context & Recommendation */}
              <div className="text-xs text-slate-600 leading-relaxed mt-2 pt-2 border-t border-slate-200/60 flex items-start gap-2">
                <Info className="w-3.5 h-3.5 text-slate-400 shrink-0 mt-0.5" />
                <div>
                  <span className="font-semibold text-slate-700">{adviceInfo.context}</span>{' '}
                  <span className="text-slate-500">{adviceInfo.advice}</span>
                </div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
