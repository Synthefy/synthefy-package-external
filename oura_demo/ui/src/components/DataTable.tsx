import React, { useState, useMemo, useEffect } from 'react';
import type { DataFrameModel } from '../types';

interface DataTableProps {
  data: DataFrameModel;
  title?: string;
  highlightColumns?: string[];
  changedCells?: string[];  // Format: "columnName:index"
  flashKey?: number;
  timeseriesColumns?: string[];
  metadataColumns?: string[];
  requiredColumns?: string[];
}

// OURA colors
const OURA_COLORS = {
  rem: '#7eb8da',
  light: '#a8c8dc',
  deep: '#4a7c9b',
  heart: '#e57373',
  amber: '#f5a623',
  textPrimary: '#f5f2ed',
  textSecondary: 'rgba(245, 242, 237, 0.7)',
  textMuted: 'rgba(245, 242, 237, 0.5)',
};

// Calculate mean and std for an array of values
function calculateStats(values: (number | string | null)[]): { mean: number | null; std: number | null } {
  const numericValues = values.filter((v): v is number => typeof v === 'number' && !isNaN(v));
  
  if (numericValues.length === 0) {
    return { mean: null, std: null };
  }
  
  const mean = numericValues.reduce((sum, v) => sum + v, 0) / numericValues.length;
  
  if (numericValues.length < 2) {
    return { mean, std: null };
  }
  
  const squaredDiffs = numericValues.map(v => Math.pow(v - mean, 2));
  const variance = squaredDiffs.reduce((sum, v) => sum + v, 0) / numericValues.length;
  const std = Math.sqrt(variance);
  
  return { mean, std };
}

export const DataTable: React.FC<DataTableProps> = ({
  data,
  title = 'Data Preview',
  highlightColumns = [],
  changedCells = [],
  flashKey = 0,
  timeseriesColumns = [],
  metadataColumns = [],
  requiredColumns = [],
}) => {
  const [flashingCells, setFlashingCells] = useState<Set<string>>(new Set());
  const [showStats, setShowStats] = useState(false);

  useEffect(() => {
    if (changedCells.length > 0 && flashKey > 0) {
      setFlashingCells(new Set(changedCells));
      const timer = setTimeout(() => {
        setFlashingCells(new Set());
      }, 2000);
      return () => clearTimeout(timer);
    } else {
      setFlashingCells(new Set());
    }
  }, [flashKey, changedCells]);

  const transposedData = useMemo(() => {
    const columns = Object.keys(data.columns);
    if (columns.length === 0) return [];

    // Filter to only show required columns if specified
    const columnsToShow = requiredColumns.length > 0
      ? columns.filter(col => requiredColumns.includes(col))
      : columns;

    return columnsToShow.map((columnName) => ({
      columnName,
      values: data.columns[columnName],
    }));
  }, [data, requiredColumns]);

  // Calculate stats for each column
  const columnStats = useMemo(() => {
    const stats: Record<string, { mean: number | null; std: number | null }> = {};
    transposedData.forEach(row => {
      stats[row.columnName] = calculateStats(row.values);
    });
    return stats;
  }, [transposedData]);

  const numRows = transposedData[0]?.values.length || 0;

  const getColumnType = (columnName: string): 'timeseries' | 'metadata' | 'other' => {
    if (timeseriesColumns.includes(columnName)) {
      return 'timeseries';
    }
    if (metadataColumns.includes(columnName)) {
      return 'metadata';
    }
    return 'other';
  };

  if (transposedData.length === 0) {
    return (
      <div className="card liquid-glass-chart p-6 animate-fade-in">
        <h3 className="oura-heading text-xl mb-4">{title}</h3>
        <div className="h-32 flex items-center justify-center" style={{ color: OURA_COLORS.textMuted }}>
          No data available
        </div>
      </div>
    );
  }

  return (
    <div className="card liquid-glass-chart animate-fade-in overflow-hidden">
      <div className="p-5" style={{ borderBottom: '1px solid rgba(140, 125, 105, 0.15)' }}>
        <div className="flex items-center justify-between">
          <h3 className="oura-heading text-xl">{title}</h3>
          <div className="flex items-center gap-4 text-xs">
            {/* Stats toggle */}
            <button
              onClick={() => setShowStats(!showStats)}
              className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg transition-all duration-200"
              style={{
                background: showStats ? 'rgba(126, 184, 218, 0.2)' : 'rgba(60, 55, 50, 0.5)',
                color: showStats ? OURA_COLORS.rem : OURA_COLORS.textSecondary,
                border: showStats ? '1px solid rgba(126, 184, 218, 0.3)' : '1px solid transparent',
              }}
            >
              <svg className="w-3.5 h-3.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
              {showStats ? 'Hide Stats' : 'Show Stats'}
            </button>
            
            <div className="flex items-center gap-2">
              <span className="w-3 h-3 rounded-full" style={{ background: OURA_COLORS.rem }} />
              <span style={{ color: OURA_COLORS.textSecondary }}>Time Series</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="w-3 h-3 rounded-full" style={{ background: OURA_COLORS.amber }} />
              <span style={{ color: OURA_COLORS.textSecondary }}>Metadata</span>
            </div>
          </div>
        </div>
      </div>

      <div className="flex">
        {/* Fixed column names */}
        <div 
          className="flex-shrink-0 transition-all duration-200"
          style={{ 
            width: showStats ? '320px' : '220px',
            boxShadow: '4px 0 12px -2px rgba(0, 0, 0, 0.4)',
            zIndex: 10,
          }}
        >
          <table className="w-full" style={{ tableLayout: 'fixed' }}>
            <thead>
              <tr>
                <th 
                  className="px-4 py-3 text-left text-xs font-medium uppercase tracking-wider"
                  style={{ 
                    background: 'rgba(30, 27, 24, 0.98)', 
                    color: OURA_COLORS.textMuted,
                  }}
                >
                  Column
                </th>
              </tr>
            </thead>
            <tbody>
              {transposedData.map((row, rowIdx) => {
                const columnType = getColumnType(row.columnName);
                const isTimeseries = columnType === 'timeseries';
                const isMetadata = columnType === 'metadata';
                const stats = columnStats[row.columnName];

                return (
                  <tr
                    key={row.columnName}
                    style={{ 
                      borderTop: rowIdx > 0 ? '1px solid rgba(140, 125, 105, 0.1)' : 'none',
                    }}
                  >
                    <td 
                      className="px-4 py-3 text-sm font-semibold"
                      style={{
                        background: isTimeseries
                          ? 'rgba(30, 50, 60, 0.98)'
                          : isMetadata
                            ? 'rgba(50, 40, 30, 0.98)'
                            : 'rgba(30, 27, 24, 0.98)',
                        color: isTimeseries
                          ? OURA_COLORS.rem
                          : isMetadata
                            ? OURA_COLORS.amber
                            : OURA_COLORS.textPrimary,
                        wordBreak: 'break-word',
                      }}
                    >
                      {row.columnName}
                      {showStats && stats && (stats.mean !== null || stats.std !== null) && (
                        <span 
                          className="ml-2 text-xs font-normal font-mono"
                          style={{ color: 'rgba(245, 242, 237, 0.45)' }}
                        >
                          ({stats.mean !== null ? `μ${stats.mean.toFixed(1)}` : ''}
                          {stats.mean !== null && stats.std !== null ? ' ' : ''}
                          {stats.std !== null ? `σ${stats.std.toFixed(1)}` : ''})
                        </span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Scrollable values */}
        <div className="flex-1 overflow-x-auto">
          <table className="w-full" style={{ tableLayout: 'fixed' }}>
            <thead>
              <tr>
                {Array.from({ length: numRows }, (_, i) => (
                  <th
                    key={i}
                    className="px-4 py-3 text-center text-xs font-medium uppercase tracking-wider"
                    style={{ 
                      color: OURA_COLORS.textMuted, 
                      background: 'rgba(30, 27, 24, 0.6)',
                      width: '100px',
                    }}
                  >
                    #{i}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {transposedData.map((row, rowIdx) => {
                const columnType = getColumnType(row.columnName);
                const isTimeseries = columnType === 'timeseries';
                const isMetadata = columnType === 'metadata';

                return (
                  <tr
                    key={row.columnName}
                    className="transition-colors duration-150"
                    style={{ 
                      borderTop: rowIdx > 0 ? '1px solid rgba(140, 125, 105, 0.1)' : 'none',
                    }}
                  >
                    {row.values.map((value, idx) => {
                      const cellKey = `${row.columnName}:${idx}`;
                      const isFlashing = flashingCells.has(cellKey);

                      return (
                        <td
                          key={`${cellKey}-${isFlashing ? flashKey : 0}`}
                          className="px-4 py-3 text-sm font-mono text-center transition-all duration-200"
                          style={{
                            width: '100px',
                            ...(isFlashing ? {
                              animation: 'pulse-highlight 0.5s ease-in-out infinite',
                              backgroundColor: OURA_COLORS.rem,
                              color: 'white',
                              fontWeight: 700,
                              boxShadow: `0 0 12px 4px rgba(126, 184, 218, 0.6), inset 0 0 8px 2px rgba(126, 184, 218, 0.4)`,
                              borderRadius: '0.5rem',
                            } : {
                              backgroundColor: isTimeseries
                                ? 'rgba(126, 184, 218, 0.05)'
                                : isMetadata
                                  ? 'rgba(245, 166, 35, 0.05)'
                                  : 'transparent',
                              color: OURA_COLORS.textSecondary,
                            }),
                          }}
                        >
                          {typeof value === 'number'
                            ? (value as number).toFixed(2)
                            : value ?? <span style={{ color: 'rgba(245, 242, 237, 0.3)' }}>null</span>}
                        </td>
                      );
                    })}
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
