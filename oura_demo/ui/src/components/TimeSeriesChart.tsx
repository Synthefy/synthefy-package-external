import React, { useMemo, useState, useEffect } from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import type { OneTimeSeries } from '../types';

interface TimeSeriesChartProps {
  original?: OneTimeSeries[];  // Ground truth data (actual values)
  baseSynthetic?: OneTimeSeries[];  // Prediction on original/unmodified data
  modifiedSynthetic?: OneTimeSeries[];  // Prediction on modified data
  metadata?: OneTimeSeries[];
  title?: string;
  subtitle?: string;
  changedColumns?: string[];
  flashKey?: number;
  showGroundTruth?: boolean;
  showBaseScenario?: boolean;
  showModifiedScenario?: boolean;
  taskType?: 'synthesis' | 'forecast';
  forecastHorizon?: number;  // Number of timestamps being forecasted (default 96)
}

type ChartTab = 'simulation' | 'anomaly';

interface AnomalyPoint {
  index: number;
  metric: string;
  deviation: number;  // How many std devs from expected
}

// OURA App Exact Colors (from screenshots)
const OURA_COLORS = {
  // Sleep stages (from Time asleep chart)
  awake: '#f5f5f5',        // White/off-white
  rem: '#7eb8da',          // Light blue
  light: '#a8c8dc',        // Lighter blue (from bars)
  deep: '#4a7c9b',         // Dark blue

  // Heart rate
  heart: '#e57373',        // Coral/red

  // HRV
  hrv: '#7eb8da',          // Light blue (same as REM)

  // Other metrics
  teal: '#07ad98',
  amber: '#f5a623',

  // Text
  textPrimary: '#f5f2ed',
  textSecondary: 'rgba(245, 242, 237, 0.7)',
};

// OURA-themed colors for timeseries - using exact app colors
const TIMESERIES_COLORS: Record<string, string> = {
  average_hrv: OURA_COLORS.rem,              // Light blue
  lowest_heart_rate: OURA_COLORS.heart,       // Coral red
  age_cva_diff: OURA_COLORS.amber,            // Amber
  highest_temperature: OURA_COLORS.teal,      // Teal
  stressed_duration: OURA_COLORS.deep,        // Dark blue
  latency: OURA_COLORS.light,                 // Light blue
  BVP: OURA_COLORS.rem,                       // Light blue
};

// Metadata colors - distinct palette that doesn't overlap with timeseries
// Uses earth tones, purples, and muted greens
const METADATA_COLORS: Record<string, string> = {
  bmi: '#9b7ed9',                 // Purple
  hrv_std: '#d4a574',             // Tan/caramel
  awake_mins: '#8fbc8f',          // Dark sea green
  age: '#c9a0dc',                 // Light purple
  readiness_score: '#87ceeb',     // Sky blue (lighter than timeseries blues)
  sleep_score: '#dda0dd',         // Plum
  low_activity_time: '#bc8f8f',   // Rosy brown
  deep_mins: '#6b8e23',           // Olive drab
  sleep_duration: '#20b2aa',      // Light sea green
  restored_duration: '#cd853f',   // Peru/brown
  non_wear_time: '#778899',       // Light slate gray
  rem_mins: '#9370db',            // Medium purple
  steps: '#3cb371',               // Medium sea green
  active_calories: '#db7093',     // Pale violet red
  heart_rate_std: '#b8860b',      // Dark goldenrod
  sedentary_time: '#708090',      // Slate gray
  high_activity_met_minutes: '#8b4513', // Saddle brown
  hours_after_oura_sleep_start: '#556b2f', // Dark olive green
  light_mins: '#66cdaa',          // Medium aquamarine
  medium_activity_met_minutes: '#cd5c5c', // Indian red
  gender_male: '#9acd32',         // Yellow green
};

// Generate a color based on string hash
function hashColor(str: string, isMetadata: boolean): string {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = str.charCodeAt(i) + ((hash << 5) - hash);
  }
  if (isMetadata) {
    // Metadata uses purple/green hue range (270-330 and 90-150) to avoid overlap
    const hueOptions = [270, 290, 310, 90, 110, 130, 150];
    const h = hueOptions[Math.abs(hash) % hueOptions.length];
    return `hsl(${h}, 50%, 55%)`;
  }
  // Timeseries uses blue hues (190-230)
  const h = 190 + (Math.abs(hash) % 40);
  return `hsl(${h}, 45%, 60%)`;
}

function getColor(name: string, isMetadata: boolean): string {
  if (isMetadata) {
    return METADATA_COLORS[name] || hashColor(name, true);
  }
  return TIMESERIES_COLORS[name] || hashColor(name, false);
}

// Get a contrasting lighter color for synthetic/predicted lines
function getSyntheticColor(originalColor: string): string {
  // Convert hex to HSL and make it lighter/more saturated for better contrast
  const hex = originalColor.replace('#', '');
  const r = parseInt(hex.substring(0, 2), 16) / 255;
  const g = parseInt(hex.substring(2, 4), 16) / 255;
  const b = parseInt(hex.substring(4, 6), 16) / 255;

  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  let h = 0, s = 0;
  const l = (max + min) / 2;

  if (max !== min) {
    const d = max - min;
    s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
    switch (max) {
      case r: h = ((g - b) / d + (g < b ? 6 : 0)) / 6; break;
      case g: h = ((b - r) / d + 2) / 6; break;
      case b: h = ((r - g) / d + 4) / 6; break;
    }
  }

  // Shift hue slightly (toward cyan/lighter) and increase lightness
  const newH = (h * 360 + 15) % 360; // Shift hue by 15 degrees
  const newS = Math.min(s * 100 + 15, 80); // Boost saturation
  const newL = Math.min(l * 100 + 20, 85); // Make it lighter

  return `hsl(${newH}, ${newS}%, ${newL}%)`;
}

function normalizeValues(values: (number | null)[]): (number | null)[] {
  const validValues = values.filter((v): v is number => v !== null && !isNaN(v));
  if (validValues.length === 0) return values;

  const min = Math.min(...validValues);
  const max = Math.max(...validValues);
  const range = max - min;

  if (range === 0) return values.map(v => v !== null ? 50 : null);

  return values.map(v => {
    if (v === null || isNaN(v)) return null;
    return ((v - min) / range) * 100;
  });
}

function normalizeValuesWithRange(
  values: (number | null)[],
  min: number,
  max: number
): (number | null)[] {
  const range = max - min;
  if (range === 0) return values.map(v => v !== null ? 50 : null);

  return values.map(v => {
    if (v === null || isNaN(v)) return null;
    return ((v - min) / range) * 100;
  });
}

function getMinMax(values: (number | null)[]): { min: number; max: number } | null {
  const validValues = values.filter((v): v is number => v !== null && !isNaN(v));
  if (validValues.length === 0) return null;
  return {
    min: Math.min(...validValues),
    max: Math.max(...validValues),
  };
}

function formatName(name: string): string {
  return name
    .replace(/_/g, ' ')
    .replace(/\b\w/g, l => l.toUpperCase());
}

interface SeriesToggleProps {
  name: string;
  color: string;
  enabled: boolean;
  onToggle: () => void;
}

const SeriesToggle: React.FC<SeriesToggleProps> = ({ name, color, enabled, onToggle }) => (
  <button
    onClick={onToggle}
    className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs font-medium transition-all duration-150"
    style={{
      background: enabled ? 'rgba(140, 125, 105, 0.4)' : 'rgba(60, 55, 50, 0.4)',
      border: enabled ? '1px solid rgba(180, 165, 145, 0.3)' : '1px solid transparent',
      opacity: enabled ? 1 : 0.5,
    }}
  >
    <span
      className="w-2.5 h-2.5 rounded-full flex-shrink-0"
      style={{ backgroundColor: enabled ? color : 'rgba(140, 125, 105, 0.4)' }}
    />
    <span className="truncate max-w-[100px]" style={{ color: '#f5f2ed' }}>{formatName(name)}</span>
  </button>
);

export const TimeSeriesChart: React.FC<TimeSeriesChartProps> = ({
  original,
  baseSynthetic,
  modifiedSynthetic,
  metadata,
  title = 'Time Series',
  subtitle,
  changedColumns = [],
  flashKey = 0,
  showGroundTruth = true,
  showBaseScenario = true,
  showModifiedScenario = true,
  taskType,
  forecastHorizon = 96,
}) => {
  const hasBaseScenario = baseSynthetic && baseSynthetic.length > 0;
  const hasModifiedScenario = modifiedSynthetic && modifiedSynthetic.length > 0;
  const hasAnyScenario = hasBaseScenario || hasModifiedScenario;

  // Get timeseries names from any available data source
  const timeseriesNames = useMemo(() => {
    if (original && original.length > 0) return original.map(ts => ts.name);
    if (baseSynthetic && baseSynthetic.length > 0) return baseSynthetic.map(ts => ts.name.replace('_synthetic', ''));
    if (modifiedSynthetic && modifiedSynthetic.length > 0) return modifiedSynthetic.map(ts => ts.name.replace('_synthetic', ''));
    return [];
  }, [original, baseSynthetic, modifiedSynthetic]);

  const [visibleTimeseries, setVisibleTimeseries] = useState<Set<string>>(
    new Set(timeseriesNames)
  );
  const [visibleMetadata, setVisibleMetadata] = useState<Set<string>>(new Set());
  const [showMetadata, setShowMetadata] = useState(false);
  const [normalize, setNormalize] = useState(true);
  const [flashingColumns, setFlashingColumns] = useState<Set<string>>(new Set());

  // Tab state
  const [activeTab, setActiveTab] = useState<ChartTab>('simulation');

  // Zoom state - get total points from any available data
  const totalPoints = useMemo(() => {
    if (original && original[0]) return original[0].values.length;
    if (baseSynthetic && baseSynthetic[0]) return baseSynthetic[0].values.length;
    if (modifiedSynthetic && modifiedSynthetic[0]) return modifiedSynthetic[0].values.length;
    return 0;
  }, [original, baseSynthetic, modifiedSynthetic]);

  const [zoomRange, setZoomRange] = useState<[number, number]>([0, totalPoints]);

  // Reset zoom when data changes
  useEffect(() => {
    const newTotal = totalPoints;
    setZoomRange([0, newTotal]);
  }, [original]);

  // Calculate forecast separator position
  const forecastStartIndex = totalPoints - forecastHorizon;
  const showForecastSeparator = taskType === 'forecast' && hasAnyScenario && forecastStartIndex > 0;

  // Calculate anomalies - using conservative z-score threshold (3+ std deviations)
  // Compare ground truth (original) with predictions (base or modified synthetic)
  const anomalies = useMemo((): AnomalyPoint[] => {
    if (!original || original.length === 0) return [];

    // Use modified synthetic if available, otherwise base synthetic
    const synthetic = modifiedSynthetic || baseSynthetic;
    if (!synthetic || synthetic.length === 0) return [];

    const anomalyPoints: AnomalyPoint[] = [];
    const THRESHOLD = 3.0; // 3 standard deviations - very conservative

    for (const origTs of original) {
      const synthTs = synthetic.find(
        s => s.name === origTs.name || s.name === `${origTs.name}_synthetic`
      );
      if (!synthTs) continue;

      // Calculate differences
      const differences: number[] = [];
      const minLength = Math.min(origTs.values.length, synthTs.values.length);

      for (let i = 0; i < minLength; i++) {
        const orig = origTs.values[i];
        const synth = synthTs.values[i];
        if (orig !== null && synth !== null && orig !== 0) {
          // Absolute percentage difference
          differences.push(Math.abs((orig - synth) / orig));
        }
      }

      if (differences.length < 5) continue; // Need enough data

      // Calculate mean and std of differences
      const mean = differences.reduce((a, b) => a + b, 0) / differences.length;
      const variance = differences.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / differences.length;
      const std = Math.sqrt(variance);

      if (std === 0) continue; // No variation

      // Find anomalies
      for (let i = 0; i < minLength; i++) {
        const orig = origTs.values[i];
        const synth = synthTs.values[i];
        if (orig !== null && synth !== null && orig !== 0) {
          const diff = Math.abs((orig - synth) / orig);
          const zScore = (diff - mean) / std;

          if (zScore > THRESHOLD) {
            anomalyPoints.push({
              index: i,
              metric: origTs.name,
              deviation: Math.round(zScore * 10) / 10,
            });
          }
        }
      }
    }

    // Deduplicate by index (keep highest deviation)
    const byIndex = new Map<number, AnomalyPoint>();
    for (const ap of anomalyPoints) {
      const existing = byIndex.get(ap.index);
      if (!existing || ap.deviation > existing.deviation) {
        byIndex.set(ap.index, ap);
      }
    }

    return Array.from(byIndex.values()).sort((a, b) => a.index - b.index);
  }, [original, baseSynthetic, modifiedSynthetic]);

  useEffect(() => {
    if (changedColumns.length > 0 && flashKey > 0) {
      setFlashingColumns(new Set(changedColumns));
      const timer = setTimeout(() => {
        setFlashingColumns(new Set());
      }, 2000);
      return () => clearTimeout(timer);
    } else {
      setFlashingColumns(new Set());
    }
  }, [flashKey, changedColumns]);

  const toggleTimeseries = (name: string) => {
    setVisibleTimeseries(prev => {
      const next = new Set(prev);
      if (next.has(name)) {
        next.delete(name);
      } else {
        next.add(name);
      }
      return next;
    });
  };

  const toggleMetadata = (name: string) => {
    setVisibleMetadata(prev => {
      const next = new Set(prev);
      if (next.has(name)) {
        next.delete(name);
      } else {
        next.add(name);
      }
      return next;
    });
  };

  const toggleAllTimeseries = () => {
    if (visibleTimeseries.size === timeseriesNames.length) {
      setVisibleTimeseries(new Set());
    } else {
      setVisibleTimeseries(new Set(timeseriesNames));
    }
  };

  const toggleAllMetadata = () => {
    if (!metadata) return;
    if (visibleMetadata.size === metadata.length) {
      setVisibleMetadata(new Set());
    } else {
      setVisibleMetadata(new Set(metadata.map(ts => ts.name)));
    }
  };

  const chartData = useMemo(() => {
    if (totalPoints === 0) return [];

    const data: Record<string, number | null>[] = [];

    // Normalize all data sources together for consistent scaling
    const normalizedOriginal = original && normalize
      ? original.map(ts => ({ ...ts, values: normalizeValues(ts.values) }))
      : original;

    const normalizedBaseSynthetic = baseSynthetic && normalize
      ? baseSynthetic.map(ts => ({ ...ts, values: normalizeValues(ts.values) }))
      : baseSynthetic;

    const normalizedModifiedSynthetic = modifiedSynthetic && normalize
      ? modifiedSynthetic.map(ts => ({ ...ts, values: normalizeValues(ts.values) }))
      : modifiedSynthetic;

    const normalizedMetadata = metadata && normalize
      ? metadata.map(ts => ({ ...ts, values: normalizeValues(ts.values) }))
      : metadata;

    for (let i = 0; i < totalPoints; i++) {
      const point: Record<string, number | null> = { index: i };

      // Ground truth (original data)
      if (normalizedOriginal && showGroundTruth) {
        for (const ts of normalizedOriginal) {
          if (visibleTimeseries.has(ts.name)) {
            point[ts.name] = ts.values[i];
          }
        }
      }

      // Base scenario predictions
      if (normalizedBaseSynthetic && showBaseScenario) {
        for (const ts of normalizedBaseSynthetic) {
          const baseName = ts.name.replace('_synthetic', '');
          if (visibleTimeseries.has(baseName)) {
            point[`${baseName}_base`] = ts.values[i];
          }
        }
      }

      // Modified scenario predictions
      if (normalizedModifiedSynthetic && showModifiedScenario) {
        for (const ts of normalizedModifiedSynthetic) {
          const baseName = ts.name.replace('_synthetic', '');
          if (visibleTimeseries.has(baseName)) {
            point[`${baseName}_modified`] = ts.values[i];
          }
        }
      }

      // Metadata
      if (normalizedMetadata && showMetadata) {
        for (const ts of normalizedMetadata) {
          if (visibleMetadata.has(ts.name)) {
            point[`meta_${ts.name}`] = ts.values[i];
          }
        }
      }

      data.push(point);
    }

    return data;
  }, [original, baseSynthetic, modifiedSynthetic, metadata, visibleTimeseries, visibleMetadata, showMetadata, normalize, showGroundTruth, showBaseScenario, showModifiedScenario, totalPoints]);

  // Apply zoom filter to chart data
  const zoomedChartData = useMemo(() => {
    return chartData.slice(zoomRange[0], zoomRange[1]);
  }, [chartData, zoomRange]);

  const originalStats = useMemo(() => {
    const stats: Record<string, { min: number; max: number; avg: number }> = {};
    const dataSource = original || baseSynthetic || modifiedSynthetic || [];
    for (const ts of dataSource) {
      const baseName = ts.name.replace('_synthetic', '');
      const validValues = ts.values.filter((v): v is number => v !== null && !isNaN(v));
      if (validValues.length > 0) {
        stats[baseName] = {
          min: Math.min(...validValues),
          max: Math.max(...validValues),
          avg: validValues.reduce((a, b) => a + b, 0) / validValues.length,
        };
      }
    }
    return stats;
  }, [original, baseSynthetic, modifiedSynthetic]);

  if (chartData.length === 0) {
    return (
      <div className="card liquid-glass-chart p-6 animate-fade-in">
        <h3 className="oura-heading text-xl">{title}</h3>
        <div className="h-64 flex items-center justify-center" style={{ color: 'rgba(245, 242, 237, 0.5)' }}>
          No data to display
        </div>
      </div>
    );
  }

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (!active || !payload || payload.length === 0) return null;

    return (
      <div
        className="rounded-xl shadow-lg p-3 max-w-xs"
        style={{
          background: 'rgba(45, 40, 35, 0.95)',
          backdropFilter: 'blur(12px)',
          border: '1px solid rgba(140, 125, 105, 0.3)',
        }}
      >
        <div className="text-xs font-medium mb-2" style={{ color: 'rgba(245, 242, 237, 0.6)' }}>Day {label}</div>
        <div className="space-y-1.5 max-h-48 overflow-y-auto">
          {payload.map((entry: any, idx: number) => {
            if (entry.value == null || typeof entry.value !== 'number') return null;

            const dataKey = entry.dataKey || '';
            const isMetadata = dataKey.startsWith('meta_');
            const isBase = dataKey.endsWith('_base');
            const isModified = dataKey.endsWith('_modified');
            const isPrediction = isBase || isModified;

            // Extract base metric name
            const baseName = dataKey
              .replace('meta_', '')
              .replace('_base', '')
              .replace('_modified', '')
              .replace('_synthetic', '');

            // Determine display color
            let displayColor = entry.color;
            if (isBase) {
              displayColor = getColor(baseName, false);
            } else if (isModified) {
              displayColor = '#07ad98'; // Teal for modified
            }

            let displayValue: string;
            try {
              if (normalize && originalStats[baseName]) {
                displayValue = ((entry.value / 100) * (originalStats[baseName].max - originalStats[baseName].min) + originalStats[baseName].min).toFixed(1);
              } else {
                displayValue = entry.value.toFixed(1);
              }
            } catch {
              displayValue = String(entry.value);
            }

            // Determine label suffix
            let labelSuffix = '';
            if (isBase) labelSuffix = ' (base pred)';
            else if (isModified) labelSuffix = ' (modified pred)';
            else if (isMetadata) labelSuffix = ' ⓜ';

            return (
              <div key={idx} className="flex items-center gap-2.5 text-xs">
                <div
                  className="w-5 h-0.5 flex-shrink-0 rounded-full"
                  style={{
                    backgroundColor: isPrediction ? 'transparent' : displayColor,
                    backgroundImage: isPrediction
                      ? `repeating-linear-gradient(90deg, ${displayColor} 0, ${displayColor} 4px, transparent 4px, transparent 6px)`
                      : undefined,
                  }}
                />
                <span className="flex-1" style={{ color: '#f5f2ed' }}>
                  {formatName(baseName)}
                  <span style={{ color: 'rgba(245, 242, 237, 0.5)' }}>{labelSuffix}</span>
                </span>
                <span
                  className="font-semibold tabular-nums"
                  style={{ color: isModified ? '#07ad98' : isBase ? OURA_COLORS.rem : '#f5f2ed' }}
                >
                  {displayValue}
                </span>
              </div>
            );
          }).filter(Boolean)}
        </div>
      </div>
    );
  };

  return (
    <div className="card liquid-glass-chart p-6 animate-fade-in">
      {/* Tabs - show when we have any predictions */}
      {hasAnyScenario && (
        <div className="flex items-center gap-1 mb-5 p-1 rounded-xl" style={{ background: 'rgba(30, 27, 24, 0.6)' }}>
          <button
            onClick={() => setActiveTab('simulation')}
            className="flex-1 py-2.5 px-4 rounded-lg text-sm font-medium transition-all duration-200 flex items-center justify-center gap-2"
            style={{
              background: activeTab === 'simulation' ? 'rgba(126, 184, 218, 0.2)' : 'transparent',
              color: activeTab === 'simulation' ? '#7eb8da' : 'rgba(245, 242, 237, 0.5)',
              border: activeTab === 'simulation' ? '1px solid rgba(126, 184, 218, 0.3)' : '1px solid transparent',
            }}
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
            Scenario Simulation
          </button>
          <button
            onClick={() => setActiveTab('anomaly')}
            className="flex-1 py-2.5 px-4 rounded-lg text-sm font-medium transition-all duration-200 flex items-center justify-center gap-2"
            style={{
              background: activeTab === 'anomaly' ? 'rgba(180, 180, 180, 0.15)' : 'transparent',
              color: activeTab === 'anomaly' ? '#c0c0c0' : 'rgba(245, 242, 237, 0.5)',
              border: activeTab === 'anomaly' ? '1px solid rgba(180, 180, 180, 0.3)' : '1px solid transparent',
            }}
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
            Anomaly Detection
            {anomalies.length > 0 && (
              <span
                className="px-1.5 py-0.5 rounded-full text-xs font-bold"
                style={{ background: 'rgba(180, 180, 180, 0.8)', color: '#1a1815' }}
              >
                {anomalies.length}
              </span>
            )}
          </button>
        </div>
      )}

      {/* Header */}
      <div className="flex items-start justify-between mb-5">
        <div>
          <h3 className="oura-heading text-xl">
            {activeTab === 'anomaly' ? 'Anomaly Detection' : title}
          </h3>
          <p className="text-sm mt-1" style={{ color: 'rgba(245, 242, 237, 0.6)' }}>
            {activeTab === 'anomaly'
              ? anomalies.length > 0
                ? `${anomalies.length} anomal${anomalies.length === 1 ? 'y' : 'ies'} detected (>3σ deviation)`
                : 'No significant anomalies detected'
              : subtitle
            }
          </p>
        </div>

        {/* Controls */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => setNormalize(!normalize)}
            className="px-3 py-1.5 rounded-lg text-xs font-medium transition-all"
            style={{
              background: normalize ? 'rgba(126, 184, 218, 0.2)' : 'rgba(60, 55, 50, 0.5)',
              color: normalize ? OURA_COLORS.rem : 'rgba(245, 242, 237, 0.6)',
              border: normalize ? '1px solid rgba(126, 184, 218, 0.3)' : '1px solid transparent',
            }}
          >
            {normalize ? '📊 Normalized' : '📈 Raw Scale'}
          </button>

          {metadata && metadata.length > 0 && (
            <button
              onClick={() => setShowMetadata(!showMetadata)}
              className="px-3 py-1.5 rounded-lg text-xs font-medium transition-all"
              style={{
                background: showMetadata ? 'rgba(245, 166, 35, 0.2)' : 'rgba(60, 55, 50, 0.5)',
                color: showMetadata ? OURA_COLORS.amber : 'rgba(245, 242, 237, 0.6)',
                border: showMetadata ? '1px solid rgba(245, 166, 35, 0.3)' : '1px solid transparent',
              }}
            >
              {showMetadata ? '📋 Metadata On' : '📋 Metadata Off'}
            </button>
          )}
        </div>
      </div>

      {/* Timeseries toggles */}
      <div className="mb-3">
        <div className="flex items-center gap-2 mb-2">
          <span className="text-xs font-medium uppercase tracking-wider" style={{ color: 'rgba(245, 242, 237, 0.5)' }}>
            Time Series
          </span>
          <button
            onClick={toggleAllTimeseries}
            className="text-xs font-medium transition-colors"
            style={{ color: OURA_COLORS.rem }}
          >
            {visibleTimeseries.size === timeseriesNames.length ? 'Hide All' : 'Show All'}
          </button>
        </div>
        <div className="flex flex-wrap gap-1.5">
          {timeseriesNames.map(name => (
            <SeriesToggle
              key={name}
              name={name}
              color={getColor(name, false)}
              enabled={visibleTimeseries.has(name)}
              onToggle={() => toggleTimeseries(name)}
            />
          ))}
        </div>
      </div>

      {/* Metadata toggles */}
      {showMetadata && metadata && metadata.length > 0 && (
        <div className="mb-3 pt-2" style={{ borderTop: '1px solid rgba(140, 125, 105, 0.2)' }}>
          <div className="flex items-center gap-2 mb-2">
            <span className="text-xs font-medium uppercase tracking-wider" style={{ color: 'rgba(245, 242, 237, 0.5)' }}>
              Metadata
            </span>
            <button
              onClick={toggleAllMetadata}
              className="text-xs font-medium transition-colors"
              style={{ color: OURA_COLORS.amber }}
            >
              {visibleMetadata.size === metadata.length ? 'Hide All' : 'Show All'}
            </button>
          </div>
          <div className="flex flex-wrap gap-1.5">
            {metadata.map(ts => (
              <SeriesToggle
                key={ts.name}
                name={ts.name}
                color={getColor(ts.name, true)}
                enabled={visibleMetadata.has(ts.name)}
                onToggle={() => toggleMetadata(ts.name)}
              />
            ))}
          </div>
        </div>
      )}

      {/* Chart */}
      <div className="mt-4">
        <ResponsiveContainer width="100%" height={500}>
          <LineChart data={zoomedChartData} margin={{ top: 10, right: 10, bottom: 10, left: 0 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(140, 125, 105, 0.15)" vertical={false} />

            <XAxis
              dataKey="index"
              axisLine={false}
              tickLine={false}
              tick={{ fill: 'rgba(245, 242, 237, 0.5)', fontSize: 11 }}
              tickFormatter={(v) => `${v}`}
            />

            <YAxis
              axisLine={false}
              tickLine={false}
              tick={{ fill: 'rgba(245, 242, 237, 0.5)', fontSize: 11 }}
              width={45}
              domain={normalize ? [0, 100] : ['auto', 'auto']}
              tickFormatter={(v) => normalize ? `${v}%` : v.toLocaleString()}
            />

            {normalize && (
              <>
                <ReferenceLine y={25} stroke="rgba(140, 125, 105, 0.1)" strokeDasharray="2 2" />
                <ReferenceLine y={50} stroke="rgba(140, 125, 105, 0.15)" strokeDasharray="2 2" />
                <ReferenceLine y={75} stroke="rgba(140, 125, 105, 0.1)" strokeDasharray="2 2" />
              </>
            )}

            <Tooltip content={<CustomTooltip />} />

            {/* Forecast separator line */}
            {showForecastSeparator && forecastStartIndex >= zoomRange[0] && forecastStartIndex < zoomRange[1] && (
              <ReferenceLine
                x={forecastStartIndex}
                stroke="#f5a623"
                strokeWidth={2}
                strokeDasharray="8 4"
                label={{
                  value: 'Forecast →',
                  position: 'insideTopRight',
                  fill: '#f5a623',
                  fontSize: 11,
                  fontWeight: 600,
                }}
              />
            )}

            {/* Anomaly vertical lines - grey to avoid confusion with data */}
            {activeTab === 'anomaly' && anomalies
              .filter(a => a.index >= zoomRange[0] && a.index < zoomRange[1])
              .map(anomaly => (
                <ReferenceLine
                  key={`anomaly-${anomaly.index}`}
                  x={anomaly.index}
                  stroke="rgba(180, 180, 180, 0.6)"
                  strokeWidth={1.5}
                  strokeDasharray="4 2"
                  label={{
                    value: `▼`,
                    position: 'top',
                    fill: 'rgba(200, 200, 200, 0.8)',
                    fontSize: 10,
                  }}
                />
              ))
            }

            {/* Ground Truth lines (original data) - solid lines */}
            {showGroundTruth && original
              ?.filter(ts => visibleTimeseries.has(ts.name))
              .map(ts => {
                const isFlashing = flashingColumns.has(ts.name);
                const color = getColor(ts.name, false);
                return (
                  <Line
                    key={ts.name}
                    type="monotone"
                    dataKey={ts.name}
                    stroke={color}
                    strokeWidth={isFlashing ? 5 : 2.5}
                    dot={false}
                    activeDot={{ r: 5, strokeWidth: 0, fill: color }}
                    style={isFlashing ? {
                      filter: `drop-shadow(0 0 8px ${color}) drop-shadow(0 0 16px ${color})`,
                    } : undefined}
                    isAnimationActive={false}
                  />
                );
              })}

            {/* Base scenario lines - semi-transparent with light dashes */}
            {showBaseScenario && baseSynthetic
              ?.filter(ts => {
                const baseName = ts.name.replace('_synthetic', '');
                return visibleTimeseries.has(baseName);
              })
              .map(ts => {
                const baseName = ts.name.replace('_synthetic', '');
                const color = getColor(baseName, false);
                return (
                  <Line
                    key={`${baseName}_base`}
                    type="monotone"
                    dataKey={`${baseName}_base`}
                    stroke={color}
                    strokeWidth={2}
                    strokeDasharray="4 2"
                    dot={false}
                    activeDot={{ r: 4, strokeWidth: 2, fill: 'rgba(45, 40, 35, 0.9)', stroke: color }}
                    opacity={0.6}
                    isAnimationActive={false}
                  />
                );
              })}

            {/* Modified scenario lines - teal/green with longer dashes */}
            {showModifiedScenario && modifiedSynthetic
              ?.filter(ts => {
                const baseName = ts.name.replace('_synthetic', '');
                return visibleTimeseries.has(baseName);
              })
              .map(ts => {
                const baseName = ts.name.replace('_synthetic', '');
                // Use teal for modified scenario to distinguish from base
                return (
                  <Line
                    key={`${baseName}_modified`}
                    type="monotone"
                    dataKey={`${baseName}_modified`}
                    stroke="#07ad98"
                    strokeWidth={2.5}
                    strokeDasharray="8 4"
                    dot={false}
                    activeDot={{ r: 5, strokeWidth: 2, fill: 'rgba(45, 40, 35, 0.9)', stroke: '#07ad98' }}
                    opacity={1}
                    isAnimationActive={false}
                  />
                );
              })}

            {/* Metadata lines */}
            {showMetadata && metadata
              ?.filter(ts => visibleMetadata.has(ts.name))
              .map(ts => {
                const isFlashing = flashingColumns.has(ts.name);
                const color = getColor(ts.name, true);
                return (
                  <Line
                    key={`meta_${ts.name}`}
                    type="monotone"
                    dataKey={`meta_${ts.name}`}
                    stroke={color}
                    strokeWidth={isFlashing ? 3.5 : 1.5}
                    strokeDasharray="2 2"
                    dot={false}
                    activeDot={{ r: 3, strokeWidth: 0, fill: color }}
                    opacity={isFlashing ? 1 : 0.6}
                    style={isFlashing ? {
                      filter: `drop-shadow(0 0 6px ${color}) drop-shadow(0 0 12px ${color})`,
                    } : undefined}
                    isAnimationActive={false}
                  />
                );
              })}
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Zoom Slider */}
      {totalPoints > 10 && (
        <div className="mt-4 pt-4" style={{ borderTop: '1px solid rgba(140, 125, 105, 0.15)' }}>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <span className="text-xs" style={{ color: 'rgba(245, 242, 237, 0.5)' }}>From:</span>
              <input
                type="range"
                min={0}
                max={totalPoints - 5}
                value={zoomRange[0]}
                onChange={(e) => {
                  const val = parseInt(e.target.value);
                  if (val < zoomRange[1] - 5) {
                    setZoomRange([val, zoomRange[1]]);
                  }
                }}
                className="w-24 h-1 rounded-full cursor-pointer"
                style={{ background: `linear-gradient(to right, rgba(60, 55, 50, 0.6) 0%, #7eb8da ${(zoomRange[0] / (totalPoints - 5)) * 100}%, rgba(60, 55, 50, 0.6) 100%)` }}
              />
              <span className="text-xs font-mono w-8" style={{ color: '#f5f2ed' }}>{zoomRange[0]}</span>
            </div>

            <div className="flex items-center gap-2">
              <span className="text-xs" style={{ color: 'rgba(245, 242, 237, 0.5)' }}>To:</span>
              <input
                type="range"
                min={5}
                max={totalPoints}
                value={zoomRange[1]}
                onChange={(e) => {
                  const val = parseInt(e.target.value);
                  if (val > zoomRange[0] + 5) {
                    setZoomRange([zoomRange[0], val]);
                  }
                }}
                className="w-24 h-1 rounded-full cursor-pointer"
                style={{ background: `linear-gradient(to right, rgba(60, 55, 50, 0.6) 0%, #7eb8da ${((zoomRange[1] - 5) / (totalPoints - 5)) * 100}%, rgba(60, 55, 50, 0.6) 100%)` }}
              />
              <span className="text-xs font-mono w-8" style={{ color: '#f5f2ed' }}>{zoomRange[1]}</span>
            </div>

            <div className="flex-1" />

            <span className="text-xs" style={{ color: 'rgba(245, 242, 237, 0.4)' }}>
              Showing {zoomRange[1] - zoomRange[0]} of {totalPoints} points
            </span>

            {(zoomRange[0] > 0 || zoomRange[1] < totalPoints) && (
              <button
                onClick={() => setZoomRange([0, totalPoints])}
                className="text-xs px-3 py-1.5 rounded-lg transition-colors flex items-center gap-1"
                style={{ background: 'rgba(60, 55, 50, 0.5)', color: '#7eb8da' }}
              >
                <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
                Reset
              </button>
            )}
          </div>
        </div>
      )}

      {/* Legend - OURA style (Simulation tab) */}
      {hasAnyScenario && activeTab === 'simulation' && (
        <div className="flex items-center justify-center gap-8 mt-4 pt-4" style={{ borderTop: '1px solid rgba(140, 125, 105, 0.15)' }}>
          {showGroundTruth && (
            <div className="flex items-center gap-2 text-xs">
              <div className="w-8 h-0.5 rounded" style={{ background: '#f5f2ed' }} />
              <span style={{ color: '#f5f2ed' }} className="font-medium">Ground Truth</span>
            </div>
          )}
          {showBaseScenario && hasBaseScenario && (
            <div className="flex items-center gap-2 text-xs">
              <div className="w-8 h-0.5 rounded" style={{ backgroundImage: `repeating-linear-gradient(90deg, ${OURA_COLORS.rem} 0, ${OURA_COLORS.rem} 3px, transparent 3px, transparent 5px)`, opacity: 0.6 }} />
              <span style={{ color: '#7eb8da' }} className="font-medium">Base Prediction</span>
            </div>
          )}
          {showModifiedScenario && hasModifiedScenario && (
            <div className="flex items-center gap-2 text-xs">
              <div className="w-8 h-0.5 rounded" style={{ backgroundImage: `repeating-linear-gradient(90deg, #07ad98 0, #07ad98 4px, transparent 4px, transparent 7px)` }} />
              <span style={{ color: '#07ad98' }} className="font-medium">Modified Prediction</span>
            </div>
          )}
        </div>
      )}

      {/* Anomaly Details (Anomaly tab) */}
      {activeTab === 'anomaly' && hasAnyScenario && (
        <div className="mt-4 pt-4" style={{ borderTop: '1px solid rgba(140, 125, 105, 0.15)' }}>
          {anomalies.length > 0 ? (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-xs">
                  <div className="w-4 h-4 rounded flex items-center justify-center" style={{ background: 'rgba(180, 180, 180, 0.2)' }}>
                    <svg className="w-2.5 h-2.5" style={{ color: '#b0b0b0' }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01" />
                    </svg>
                  </div>
                  <span style={{ color: 'rgba(245, 242, 237, 0.6)' }}>Anomalies are points where prediction differs significantly from actual ({">"} 3σ)</span>
                </div>
              </div>

              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-2">
                {anomalies.slice(0, 12).map((anomaly) => (
                  <button
                    key={`detail-${anomaly.index}`}
                    onClick={() => {
                      // Zoom to show anomaly with context
                      const start = Math.max(0, anomaly.index - 20);
                      const end = Math.min(totalPoints, anomaly.index + 20);
                      setZoomRange([start, end]);
                    }}
                    className="p-2 rounded-lg text-left transition-all duration-200 hover:scale-105"
                    style={{
                      background: 'rgba(180, 180, 180, 0.08)',
                      border: '1px solid rgba(180, 180, 180, 0.2)',
                    }}
                  >
                    <div className="text-xs font-mono" style={{ color: '#c0c0c0' }}>
                      Day {anomaly.index}
                    </div>
                    <div className="text-xs mt-0.5 truncate" style={{ color: 'rgba(245, 242, 237, 0.5)' }}>
                      {anomaly.metric}
                    </div>
                    <div className="text-xs mt-0.5" style={{ color: 'rgba(245, 242, 237, 0.4)' }}>
                      {anomaly.deviation}σ deviation
                    </div>
                  </button>
                ))}
              </div>

              {anomalies.length > 12 && (
                <p className="text-xs" style={{ color: 'rgba(245, 242, 237, 0.4)' }}>
                  +{anomalies.length - 12} more anomalies
                </p>
              )}
            </div>
          ) : (
            <div className="text-center py-6">
              <div className="w-12 h-12 mx-auto mb-3 rounded-xl flex items-center justify-center" style={{ background: 'rgba(7, 173, 152, 0.15)' }}>
                <svg className="w-6 h-6" style={{ color: '#07ad98' }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <p className="text-sm font-medium" style={{ color: '#07ad98' }}>All Clear</p>
              <p className="text-xs mt-1" style={{ color: 'rgba(245, 242, 237, 0.5)' }}>
                Predictions closely match actual values. No anomalies detected.
              </p>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
