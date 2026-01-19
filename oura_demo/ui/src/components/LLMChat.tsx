import React, { useState } from 'react';

interface ModificationHistoryItem {
  query: string;
  explanation: string;
  timestamp: Date;
}

interface LLMChatProps {
  onSubmit: (query: string) => void;
  onReset?: () => void;
  isLoading: boolean;
  disabled: boolean;
  hasModifications?: boolean;
  lastResponse?: {
    code_executed: string;
    explanation: string;
  };
  modificationHistory?: ModificationHistoryItem[];
}

const SCENARIO_EXAMPLES = [
  "What if I sleep 8 hours every night?",
  "Show me my HRV if I exercise 20% more",
  "What happens to my heart rate with better sleep?",
  "Simulate reducing my stress by half",
];

export const LLMChat: React.FC<LLMChatProps> = ({
  onSubmit,
  onReset,
  isLoading,
  disabled,
  hasModifications = false,
  lastResponse,
  modificationHistory = [],
}) => {
  const [query, setQuery] = useState('');
  const [showCode, setShowCode] = useState(false);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (query.trim() && !isLoading && !disabled) {
      onSubmit(query.trim());
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (query.trim() && !isLoading && !disabled) {
        onSubmit(query.trim());
      }
    }
  };

  const handleExampleClick = (example: string) => {
    setQuery(example);
  };

  return (
    <div className="card liquid-glass-shine p-6 animate-fade-in hover-glow" style={{ border: '2px solid rgba(126, 184, 218, 0.2)' }}>
      <div className="flex items-center justify-between mb-5">
        <div className="flex items-center gap-3">
          <div className="step-indicator-active" style={{ width: '2.5rem', height: '2.5rem', borderRadius: '0.75rem', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            </svg>
          </div>
          <div>
            <h2 className="oura-heading text-xl">Scenario Simulation</h2>
            <p className="text-sm" style={{ color: 'rgba(245, 242, 237, 0.6)' }}>Ask "what if" questions about your health data</p>
          </div>
        </div>
        {hasModifications && onReset && (
          <button
            onClick={onReset}
            disabled={disabled || isLoading}
            className="btn-secondary text-sm flex items-center gap-2"
            title="Reset all modifications to original data"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
            </svg>
            Reset
          </button>
        )}
      </div>

      <form onSubmit={handleSubmit}>
        <div className="relative">
          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={disabled
              ? 'Upload your wearable data first...'
              : 'Ask a what-if question... e.g., "What if I improved my sleep quality?"'
            }
            disabled={disabled || isLoading}
            rows={4}
            className="input resize-none pr-14 text-base overflow-y-auto"
            style={{ fontFamily: 'DM Sans, sans-serif', minHeight: '120px', maxHeight: '200px' }}
          />
          <button
            type="submit"
            disabled={disabled || isLoading || !query.trim()}
            className="absolute right-3 bottom-3 p-3 rounded-xl transition-all duration-200"
            style={{
              background: disabled || isLoading || !query.trim() ? 'rgba(60, 55, 50, 0.5)' : 'linear-gradient(135deg, #7eb8da 0%, #4a7c9b 100%)',
              color: disabled || isLoading || !query.trim() ? 'rgba(245, 242, 237, 0.3)' : 'white',
              cursor: disabled || isLoading || !query.trim() ? 'not-allowed' : 'pointer',
              boxShadow: disabled || isLoading || !query.trim() ? 'none' : '0 2px 8px rgba(126, 184, 218, 0.3)',
            }}
          >
            {isLoading ? (
              <svg className="w-5 h-5 animate-spin" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
              </svg>
            ) : (
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            )}
          </button>
        </div>
      </form>

      {/* Scenario examples */}
      <div className="mt-4">
        <p className="text-xs mb-2 font-medium uppercase tracking-wider" style={{ color: 'rgba(245, 242, 237, 0.5)' }}>Try asking:</p>
        <div className="flex flex-wrap gap-2">
          {SCENARIO_EXAMPLES.map((example) => (
            <button
              key={example}
              onClick={() => handleExampleClick(example)}
              disabled={disabled}
              className="text-sm px-4 py-2 rounded-xl border transition-all duration-200"
              style={{
                background: disabled ? 'rgba(40, 35, 30, 0.4)' : 'rgba(60, 55, 50, 0.5)',
                color: disabled ? 'rgba(245, 242, 237, 0.3)' : '#f5f2ed',
                borderColor: disabled ? 'rgba(140, 125, 105, 0.1)' : 'rgba(140, 125, 105, 0.25)',
                cursor: disabled ? 'not-allowed' : 'pointer',
              }}
            >
              {example}
            </button>
          ))}
        </div>
      </div>

      {/* Modification History */}
      {modificationHistory.length > 0 && (
        <div className="mt-5 space-y-3">
          <div className="flex items-center justify-between">
            <p className="text-xs font-medium uppercase tracking-wider flex items-center gap-2" style={{ color: 'rgba(245, 242, 237, 0.5)' }}>
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              Modification History ({modificationHistory.length})
            </p>
          </div>

          <div className="space-y-2 max-h-64 overflow-y-auto">
            {modificationHistory.map((item, index) => (
              <div
                key={index}
                className="p-4 rounded-xl border transition-all duration-300 animate-fade-in"
                style={{
                  background: index === modificationHistory.length - 1
                    ? 'linear-gradient(135deg, rgba(126, 184, 218, 0.15) 0%, rgba(74, 124, 155, 0.1) 100%)'
                    : 'rgba(40, 35, 30, 0.5)',
                  borderColor: index === modificationHistory.length - 1
                    ? 'rgba(126, 184, 218, 0.3)'
                    : 'rgba(140, 125, 105, 0.15)',
                }}
              >
                <div className="flex items-start gap-3">
                  <div
                    className="w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0 text-xs font-bold"
                    style={{
                      background: index === modificationHistory.length - 1 ? 'linear-gradient(135deg, #7eb8da 0%, #4a7c9b 100%)' : 'rgba(140, 125, 105, 0.4)',
                      color: 'white',
                    }}
                  >
                    {index + 1}
                  </div>
                  <div className="flex-1 min-w-0">
                    <p
                      className="text-sm font-medium break-words whitespace-pre-wrap"
                      style={{ color: index === modificationHistory.length - 1 ? '#7eb8da' : '#f5f2ed' }}
                    >
                      "{item.query}"
                    </p>
                    <p
                      className="text-xs mt-1"
                      style={{ color: index === modificationHistory.length - 1 ? 'rgba(126, 184, 218, 0.8)' : 'rgba(245, 242, 237, 0.5)' }}
                    >
                      {item.explanation}
                    </p>
                    <p className="text-xs mt-1" style={{ color: 'rgba(245, 242, 237, 0.4)' }}>
                      {item.timestamp.toLocaleTimeString()}
                    </p>
                  </div>
                  {index === modificationHistory.length - 1 && (
                    <span
                      className="px-2 py-1 text-xs font-medium rounded-full flex-shrink-0"
                      style={{ background: 'rgba(126, 184, 218, 0.2)', color: '#7eb8da' }}
                    >
                      Latest
                    </span>
                  )}
                </div>
              </div>
            ))}
          </div>

          {/* Summary */}
          <div
            className="p-3 rounded-xl"
            style={{ background: 'rgba(245, 166, 35, 0.15)', border: '1px solid rgba(245, 166, 35, 0.25)' }}
          >
            <p className="text-xs flex items-center gap-2" style={{ color: '#f5a623' }}>
              <svg className="w-4 h-4 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <span>
                <strong>{modificationHistory.length} modification{modificationHistory.length > 1 ? 's' : ''}</strong> applied.
                Click "Run Scenario Simulation" to see predicted outcomes with these changes.
              </span>
            </p>
          </div>
        </div>
      )}

      {/* Show code toggle for last response */}
      {lastResponse && (
        <div className="mt-3">
          <button
            onClick={() => setShowCode(!showCode)}
            className="text-xs font-medium flex items-center gap-1 transition-colors"
            style={{ color: '#7eb8da' }}
          >
            <svg className={`w-3 h-3 transition-transform ${showCode ? 'rotate-90' : ''}`} fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
            {showCode ? 'Hide last transformation code' : 'View last transformation code'}
          </button>
          {showCode && (
            <pre className="mt-2 p-4 rounded-xl text-xs overflow-x-auto font-mono" style={{ background: 'rgba(15, 14, 12, 0.8)', color: '#7eb8da', border: '1px solid rgba(140, 125, 105, 0.2)' }}>
              {lastResponse.code_executed}
            </pre>
          )}
        </div>
      )}
    </div>
  );
};
