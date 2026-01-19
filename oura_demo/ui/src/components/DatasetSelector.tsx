import React, { useState } from 'react';
import type { DatasetName, ConfigResponse } from '../types';

interface DatasetSelectorProps {
  selectedDataset: DatasetName | null;
  config: ConfigResponse | null;
  availableDatasets: string[];
  onSelect: (dataset: DatasetName) => void;
  isLoading: boolean;
}

export const DatasetSelector: React.FC<DatasetSelectorProps> = ({
  selectedDataset,
  config,
  availableDatasets,
  onSelect,
  isLoading,
}) => {
  const [showDetails, setShowDetails] = useState(false);

  return (
    <div className="card p-4 animate-fade-in">
      <div className="flex items-center gap-2 mb-3">
        <div 
          className="w-6 h-6 rounded-md flex items-center justify-center text-xs font-bold"
          style={{ background: 'rgba(126, 184, 218, 0.2)', color: '#7eb8da' }}
        >
          1
        </div>
        <h2 className="text-sm font-medium" style={{ color: '#f5f2ed' }}>Dataset</h2>
      </div>

      <select
        value={selectedDataset || ''}
        onChange={(e) => onSelect(e.target.value as DatasetName)}
        disabled={isLoading}
        className="select text-sm py-2"
      >
        <option value="">Choose dataset...</option>
        {availableDatasets.map((dataset) => (
          <option key={dataset} value={dataset}>
            {dataset}
          </option>
        ))}
      </select>

      {isLoading && (
        <div className="mt-3 flex items-center gap-2 text-xs" style={{ color: 'rgba(245, 242, 237, 0.6)' }}>
          <svg className="w-3 h-3 animate-spin" style={{ color: '#7eb8da' }} fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
          </svg>
          Loading...
        </div>
      )}

      {config && !isLoading && (
        <div className="mt-3">
          <button
            onClick={() => setShowDetails(!showDetails)}
            className="w-full flex items-center justify-between text-xs py-2 px-3 rounded-lg transition-colors"
            style={{ 
              background: 'rgba(30, 27, 24, 0.6)', 
              color: 'rgba(245, 242, 237, 0.6)',
            }}
          >
            <span>
              {config.window_size} pts × {config.num_channels} channels
            </span>
            <svg 
              className={`w-3 h-3 transition-transform ${showDetails ? 'rotate-180' : ''}`} 
              fill="none" viewBox="0 0 24 24" stroke="currentColor"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>
          
          {showDetails && (
            <div className="mt-2 p-3 rounded-lg space-y-2" style={{ background: 'rgba(30, 27, 24, 0.6)' }}>
              <div className="flex flex-wrap gap-1">
                {config.required_columns.timeseries.map((col) => (
                  <span key={col} className="tag text-xs py-0.5 px-2">
                    {col}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
