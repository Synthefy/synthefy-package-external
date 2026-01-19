import React, { useMemo, useState } from 'react';
import type { SynthesisResponse, OneTimeSeries } from '../types';
import { calculateMAPE } from '../utils/api';

// Calculate mean of an array of numbers
function calculateMean(values: (number | null)[]): number | null {
  const valid = values.filter((v): v is number => v !== null && !isNaN(v));
  if (valid.length === 0) return null;
  return valid.reduce((sum, v) => sum + v, 0) / valid.length;
}

// Calculate standard deviation
function calculateStd(values: (number | null)[]): number | null {
  const valid = values.filter((v): v is number => v !== null && !isNaN(v));
  if (valid.length < 2) return null;
  const mean = valid.reduce((sum, v) => sum + v, 0) / valid.length;
  const squaredDiffs = valid.map(v => Math.pow(v - mean, 2));
  const variance = squaredDiffs.reduce((sum, v) => sum + v, 0) / valid.length;
  return Math.sqrt(variance);
}

// Calculate linear trend (slope) using least squares regression
// Returns slope per time unit (positive = increasing, negative = decreasing)
function calculateLinearTrend(values: (number | null)[]): number | null {
  const valid: { x: number; y: number }[] = [];
  values.forEach((v, i) => {
    if (v !== null && !isNaN(v)) {
      valid.push({ x: i, y: v });
    }
  });
  
  if (valid.length < 2) return null;
  
  const n = valid.length;
  const sumX = valid.reduce((sum, p) => sum + p.x, 0);
  const sumY = valid.reduce((sum, p) => sum + p.y, 0);
  const sumXY = valid.reduce((sum, p) => sum + p.x * p.y, 0);
  const sumXX = valid.reduce((sum, p) => sum + p.x * p.x, 0);
  
  const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
  return slope;
}

interface MetricStats {
  name: string;
  mean: number | null;
  std: number | null;
  trend: number | null;
  trendDirection: 'up' | 'down' | 'flat';
  trendPercent: number | null; // Trend as % change over the time window
}

function calculateMetricStats(timeseries: OneTimeSeries[]): MetricStats[] {
  return timeseries.map(ts => {
    const mean = calculateMean(ts.values);
    const std = calculateStd(ts.values);
    const trend = calculateLinearTrend(ts.values);
    
    // Calculate trend as percentage change over the time window
    let trendPercent: number | null = null;
    let trendDirection: 'up' | 'down' | 'flat' = 'flat';
    
    if (trend !== null && mean !== null && mean !== 0) {
      // Total change over the window = slope * (length - 1)
      const totalChange = trend * (ts.values.length - 1);
      trendPercent = (totalChange / Math.abs(mean)) * 100;
      
      // Determine direction (use 1% threshold to avoid noise)
      if (trendPercent > 1) trendDirection = 'up';
      else if (trendPercent < -1) trendDirection = 'down';
    }
    
    return {
      name: ts.name.replace('_synthetic', ''),
      mean,
      std,
      trend,
      trendDirection,
      trendPercent,
    };
  });
}

interface SynthesisPanelProps {
  onSynthesize: () => void;
  isLoading: boolean;
  disabled: boolean;
  baseResult: SynthesisResponse | null;
  modifiedResult: SynthesisResponse | null;
  modificationCount?: number;
}

export const SynthesisPanel: React.FC<SynthesisPanelProps> = ({
  onSynthesize,
  isLoading,
  disabled,
  baseResult,
  modifiedResult,
  modificationCount = 0,
}) => {
  // State for collapsible sections
  const [showStats, setShowStats] = useState(false);

  // Use the most relevant result for display
  const result = modifiedResult || baseResult;
  const hasComparison = baseResult && modifiedResult;

  // Calculate MAPE for each metric when results are available
  const mapeResults = useMemo(() => {
    if (!result) return [];
    return calculateMAPE(result.original_timeseries, result.synthetic_timeseries);
  }, [result]);

  // Calculate statistics for base scenario
  const baseStats = useMemo(() => {
    if (!baseResult?.synthetic_timeseries) return [];
    return calculateMetricStats(baseResult.synthetic_timeseries);
  }, [baseResult]);

  // Calculate statistics for modified scenario
  const modifiedStats = useMemo(() => {
    if (!modifiedResult?.synthetic_timeseries) return [];
    return calculateMetricStats(modifiedResult.synthetic_timeseries);
  }, [modifiedResult]);

  return (
    <div className="card liquid-glass-shine p-6 animate-fade-in">
      <div className="flex items-center gap-3 mb-5">
        <div className="step-indicator">
          <span>4</span>
        </div>
        <div>
          <h2 className="oura-heading text-xl">Predict Outcomes</h2>
          <p className="text-xs" style={{ color: 'rgba(245, 242, 237, 0.5)' }}>See how your body might respond</p>
        </div>
      </div>

      <button
        onClick={onSynthesize}
        disabled={disabled || isLoading}
        className="w-full py-4 rounded-xl font-semibold text-base flex items-center justify-center gap-3 transition-all duration-200 ease-out"
        style={{
          background: disabled || isLoading
            ? 'rgba(60, 55, 50, 0.5)'
            : 'linear-gradient(135deg, #7eb8da 0%, #4a7c9b 100%)',
          color: disabled || isLoading ? 'rgba(245, 242, 237, 0.3)' : 'white',
          cursor: disabled || isLoading ? 'not-allowed' : 'pointer',
          boxShadow: disabled || isLoading ? 'none' : '0 4px 16px rgba(126, 184, 218, 0.3)',
        }}
      >
        {isLoading ? (
          <>
            <svg className="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
            </svg>
            Running Simulation...
          </>
        ) : (
          <>
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            Run Scenario Simulation
          </>
        )}
      </button>

      {result && (
        <div className="mt-5 space-y-4">
          {/* Status header */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2" style={{ color: '#7eb8da' }}>
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <span className="text-sm font-medium">
                {hasComparison ? 'Comparison Ready' : 'Simulation Complete'}
              </span>
            </div>
            {hasComparison && (
              <span 
                className="text-xs px-2 py-0.5 rounded-full"
                style={{ background: 'rgba(126, 184, 218, 0.2)', color: '#7eb8da' }}
              >
                Base vs Modified
              </span>
            )}
            </div>

          {/* MAPE Results - one card per metric */}
          {mapeResults.length > 0 && (
            <div className="space-y-2">
              <p className="text-xs uppercase tracking-wider" style={{ color: 'rgba(245, 242, 237, 0.4)' }}>
                MAPE by Metric
              </p>
              <div className="space-y-1.5">
                {mapeResults.map((metric) => (
                  <div 
                    key={metric.name}
                    className="flex items-center justify-between py-2 px-3 rounded-lg"
                    style={{ background: 'rgba(30, 27, 24, 0.5)' }}
                  >
                    <span 
                      className="text-sm truncate mr-3" 
                      style={{ color: 'rgba(245, 242, 237, 0.7)' }}
                      title={metric.name}
                    >
                      {metric.name}
                    </span>
                    <span 
                      className="text-sm font-mono font-medium flex-shrink-0"
                      style={{ color: '#f5f2ed' }}
                    >
                      {metric.mape.toFixed(1)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Detailed Statistics - Collapsible */}
          {modifiedStats.length > 0 && (
            <div className="space-y-2">
              <button
                onClick={() => setShowStats(!showStats)}
                className="w-full flex items-center justify-between text-xs uppercase tracking-wider py-2 px-1 transition-colors hover:opacity-80"
                style={{ color: 'rgba(245, 242, 237, 0.4)' }}
              >
                <span className="flex items-center gap-2">
                  <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                  </svg>
                  Predicted Trends & Statistics
                </span>
                <svg 
                  className={`w-4 h-4 transition-transform duration-200 ${showStats ? 'rotate-180' : ''}`} 
                  fill="none" 
                  viewBox="0 0 24 24" 
                  stroke="currentColor"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                </svg>
              </button>

              {showStats && (
                <div className="space-y-2 animate-fade-in">
                  <p className="text-xs px-1" style={{ color: 'rgba(245, 242, 237, 0.5)' }}>
                    See how your scenario affects each metric over time
                  </p>
                  
                  {(modifiedStats.length > 0 ? modifiedStats : baseStats).map((stat) => (
                    <div 
                      key={stat.name}
                      className="py-3 px-3 rounded-lg"
                      style={{ background: 'rgba(30, 27, 24, 0.5)' }}
                    >
                      {/* Metric name with trend indicator */}
                      <div className="flex items-center justify-between mb-2">
                        <span 
                          className="text-sm font-medium truncate mr-2" 
                          style={{ color: '#f5f2ed' }}
                          title={stat.name}
                        >
                          {stat.name.replace(/_/g, ' ')}
                        </span>
                        
                        {/* Trend badge */}
                        {stat.trendPercent !== null && (
                          <span 
                            className="flex items-center gap-1 text-xs font-medium px-2 py-0.5 rounded-full flex-shrink-0"
                            style={{
                              background: stat.trendDirection === 'up' 
                                ? 'rgba(7, 173, 152, 0.2)' 
                                : stat.trendDirection === 'down' 
                                  ? 'rgba(229, 115, 115, 0.2)' 
                                  : 'rgba(140, 125, 105, 0.2)',
                              color: stat.trendDirection === 'up' 
                                ? '#07ad98' 
                                : stat.trendDirection === 'down' 
                                  ? '#e57373' 
                                  : 'rgba(245, 242, 237, 0.6)',
                            }}
                          >
                            {stat.trendDirection === 'up' ? '↑' : stat.trendDirection === 'down' ? '↓' : '→'}
                            {Math.abs(stat.trendPercent).toFixed(1)}%
                          </span>
                        )}
                      </div>

                      {/* Stats row */}
                      <div className="grid grid-cols-3 gap-2 text-xs">
                        <div>
                          <span style={{ color: 'rgba(245, 242, 237, 0.4)' }}>Mean</span>
                          <p className="font-mono" style={{ color: 'rgba(245, 242, 237, 0.8)' }}>
                            {stat.mean !== null ? stat.mean.toFixed(1) : '—'}
                          </p>
                        </div>
                        <div>
                          <span style={{ color: 'rgba(245, 242, 237, 0.4)' }}>Std</span>
                          <p className="font-mono" style={{ color: 'rgba(245, 242, 237, 0.8)' }}>
                            {stat.std !== null ? stat.std.toFixed(1) : '—'}
                          </p>
                        </div>
                        <div>
                          <span style={{ color: 'rgba(245, 242, 237, 0.4)' }}>Trend</span>
                          <p className="font-mono" style={{ color: 'rgba(245, 242, 237, 0.8)' }}>
                            {stat.trend !== null ? (stat.trend >= 0 ? '+' : '') + stat.trend.toFixed(2) : '—'}
                            <span style={{ color: 'rgba(245, 242, 237, 0.4)' }}>/day</span>
                          </p>
                        </div>
                      </div>
                    </div>
                  ))}

                  {/* Interpretation helper */}
                  <div 
                    className="p-3 rounded-lg text-xs"
                    style={{ background: 'rgba(126, 184, 218, 0.1)', border: '1px solid rgba(126, 184, 218, 0.2)' }}
                  >
                    <p style={{ color: '#7eb8da' }}>
                      <strong>How to read:</strong> A positive trend (↑) means the metric is predicted to increase over time. 
                      For metrics like HRV where higher is better, ↑ is good. For heart rate where lower is often better, ↓ may be good.
                    </p>
            </div>
            </div>
              )}
          </div>
          )}
        </div>
      )}

      {!result && !disabled && (
        <div className="mt-4">
          {modificationCount > 0 ? (
            <div 
              className="p-3 rounded-xl"
              style={{ background: 'rgba(126, 184, 218, 0.1)', border: '1px solid rgba(126, 184, 218, 0.2)' }}
            >
              <p className="text-sm text-center flex items-center justify-center gap-2" style={{ color: '#7eb8da' }}>
                <span 
                  className="inline-flex items-center justify-center w-6 h-6 text-xs font-bold rounded-full"
                  style={{ background: 'linear-gradient(135deg, #7eb8da 0%, #4a7c9b 100%)', color: 'white' }}
                >
                  {modificationCount}
                </span>
                <span>
                  scenario modification{modificationCount > 1 ? 's' : ''} applied — ready to simulate!
                </span>
              </p>
            </div>
          ) : (
            <p className="text-sm text-center" style={{ color: 'rgba(245, 242, 237, 0.5)' }}>
              ✨ Apply modifications above, then simulate to see predictions
            </p>
          )}
        </div>
      )}
    </div>
  );
};
