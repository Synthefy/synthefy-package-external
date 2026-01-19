import React, { useMemo, useState } from 'react';
import type { SynthesisResponse, OneTimeSeries } from '../types';

// Calculate mean of an array of numbers
function calculateMean(values: (number | null)[]): number | null {
  const valid = values.filter((v): v is number => v !== null && !isNaN(v));
  if (valid.length === 0) return null;
  return valid.reduce((sum, v) => sum + v, 0) / valid.length;
}

// Calculate linear trend (slope) using least squares regression
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
  trend: number | null;
}

function calculateMetricStats(timeseries: OneTimeSeries[]): MetricStats[] {
  return timeseries.map(ts => ({
    name: ts.name.replace('_synthetic', ''),
    mean: calculateMean(ts.values),
    trend: calculateLinearTrend(ts.values),
  }));
}

interface DeltaStats {
  name: string;
  baseMean: number | null;
  modifiedMean: number | null;
  meanDelta: number | null;
  meanDeltaPercent: number | null;
}

function calculateDeltaStats(baseStats: MetricStats[], modifiedStats: MetricStats[]): DeltaStats[] {
  return baseStats.map(baseStat => {
    const modifiedStat = modifiedStats.find(m => m.name === baseStat.name);
    
    let meanDelta: number | null = null;
    let meanDeltaPercent: number | null = null;
    
    if (modifiedStat) {
      if (baseStat.mean !== null && modifiedStat.mean !== null) {
        meanDelta = modifiedStat.mean - baseStat.mean;
        if (baseStat.mean !== 0) {
          meanDeltaPercent = (meanDelta / Math.abs(baseStat.mean)) * 100;
        }
      }
    }
    
    return {
      name: baseStat.name,
      baseMean: baseStat.mean,
      modifiedMean: modifiedStat?.mean ?? null,
      meanDelta,
      meanDeltaPercent,
    };
  });
}

interface DeltaDisplayProps {
  baseResult: SynthesisResponse | null;
  modifiedResult: SynthesisResponse | null;
}

export const DeltaDisplay: React.FC<DeltaDisplayProps> = ({
  baseResult,
  modifiedResult,
}) => {
  const [isCollapsed, setIsCollapsed] = useState(false);
  
  const hasComparison = baseResult && modifiedResult;

  const baseStats = useMemo(() => {
    if (!baseResult?.synthetic_timeseries) return [];
    return calculateMetricStats(baseResult.synthetic_timeseries);
  }, [baseResult]);

  const modifiedStats = useMemo(() => {
    if (!modifiedResult?.synthetic_timeseries) return [];
    return calculateMetricStats(modifiedResult.synthetic_timeseries);
  }, [modifiedResult]);

  const deltaStats = useMemo(() => {
    if (!hasComparison) return [];
    return calculateDeltaStats(baseStats, modifiedStats);
  }, [hasComparison, baseStats, modifiedStats]);

  if (!hasComparison || deltaStats.length === 0) {
    return null;
  }

  return (
    <div 
      className="liquid-glass-card p-4 animate-fade-in"
      style={{ background: 'rgba(30, 27, 24, 0.6)' }}
    >
      <button
        onClick={() => setIsCollapsed(!isCollapsed)}
        className="w-full flex items-center justify-between text-sm font-medium py-1 transition-colors hover:opacity-80"
        style={{ color: '#f5f2ed' }}
      >
        <span className="flex items-center gap-2">
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7h12m0 0l-4-4m4 4l-4 4m0 6H4m0 0l4 4m-4-4l4-4" />
          </svg>
          Impact of Your Changes (Δ)
        </span>
        <svg 
          className={`w-5 h-5 transition-transform duration-200 ${isCollapsed ? '' : 'rotate-180'}`} 
          fill="none" 
          viewBox="0 0 24 24" 
          stroke="currentColor"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {!isCollapsed && (
        <div className="mt-3 grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6 gap-3 animate-fade-in">
          {deltaStats.map((stat) => (
            <div 
              key={stat.name}
              className="p-3 rounded-lg"
              style={{ background: 'rgba(45, 40, 35, 0.6)', border: '1px solid rgba(140, 125, 105, 0.2)' }}
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-medium truncate" style={{ color: '#f5f2ed' }} title={stat.name}>
                  {stat.name.replace(/_/g, ' ')}
                </span>
                {stat.meanDeltaPercent !== null && (
                  <span 
                    className="text-xs font-mono font-bold px-1.5 py-0.5 rounded ml-1 flex-shrink-0"
                    style={{ 
                      background: stat.meanDeltaPercent > 0 
                        ? 'rgba(7, 173, 152, 0.3)' 
                        : stat.meanDeltaPercent < 0 
                          ? 'rgba(229, 115, 115, 0.3)' 
                          : 'rgba(140, 125, 105, 0.3)',
                      color: stat.meanDeltaPercent > 0 
                        ? '#07ad98' 
                        : stat.meanDeltaPercent < 0 
                          ? '#e57373' 
                          : 'rgba(245, 242, 237, 0.7)',
                    }}
                  >
                    {stat.meanDeltaPercent > 0 ? '+' : ''}{stat.meanDeltaPercent.toFixed(1)}%
                  </span>
                )}
              </div>
              <div className="grid grid-cols-3 gap-1 text-xs">
                <div>
                  <span style={{ color: 'rgba(245, 242, 237, 0.4)' }}>Base</span>
                  <p className="font-mono" style={{ color: '#7eb8da' }}>
                    {stat.baseMean !== null ? stat.baseMean.toFixed(1) : '—'}
                  </p>
                </div>
                <div>
                  <span style={{ color: 'rgba(245, 242, 237, 0.4)' }}>Mod</span>
                  <p className="font-mono" style={{ color: '#07ad98' }}>
                    {stat.modifiedMean !== null ? stat.modifiedMean.toFixed(1) : '—'}
                  </p>
                </div>
                <div>
                  <span style={{ color: 'rgba(245, 242, 237, 0.4)' }}>Δ</span>
                  <p className="font-mono" style={{ color: '#f5f2ed' }}>
                    {stat.meanDelta !== null ? (stat.meanDelta > 0 ? '+' : '') + stat.meanDelta.toFixed(1) : '—'}
                  </p>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
