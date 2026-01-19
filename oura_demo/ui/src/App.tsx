import { useCallback, useEffect, useState, useRef } from 'react';
import { DatasetSelector } from './components/DatasetSelector';
import { FileUploader } from './components/FileUploader';
import { TimeSeriesChart } from './components/TimeSeriesChart';
import { DataTable } from './components/DataTable';
import { LLMChat } from './components/LLMChat';
import { SynthesisPanel } from './components/SynthesisPanel';
import { DeltaDisplay } from './components/DeltaDisplay';
import { getConfig, getDatasets, uploadFile, modifyWithLLM, synthesize } from './utils/api';
import type {
  DatasetName,
  ModelType,
  TaskType,
  ConfigResponse,
  DataFrameModel,
  ColumnValidationResult,
  SynthesisResponse,
  OneTimeSeries,
} from './types';

function App() {
  // State
  const [availableDatasets, setAvailableDatasets] = useState<string[]>([]);
  const [selectedDataset, setSelectedDataset] = useState<DatasetName | null>(null);
  const [config, setConfig] = useState<ConfigResponse | null>(null);
  const [uploadedData, setUploadedData] = useState<DataFrameModel | null>(null);
  const [modifiedData, setModifiedData] = useState<DataFrameModel | null>(null);
  const [previousData, setPreviousData] = useState<DataFrameModel | null>(null);
  const [changedCells, setChangedCells] = useState<string[]>([]);  // Format: "columnName:index"
  const [flashKey, setFlashKey] = useState(0);
  const [validation, setValidation] = useState<ColumnValidationResult | null>(null);
  const [uploadedFileName, setUploadedFileName] = useState<string | null>(null);

  // Scenario results - separate base (original data) and modified scenarios
  const [baseScenarioResult, setBaseScenarioResult] = useState<SynthesisResponse | null>(null);
  const [modifiedScenarioResult, setModifiedScenarioResult] = useState<SynthesisResponse | null>(null);

  // Visibility toggles for chart overlays
  const [showGroundTruth, setShowGroundTruth] = useState(true);
  const [showBaseScenario, setShowBaseScenario] = useState(true);
  const [showModifiedScenario, setShowModifiedScenario] = useState(true);

  const [selectedModelType, setSelectedModelType] = useState<ModelType>('flexible');
  const [selectedTaskType, setSelectedTaskType] = useState<TaskType>('synthesis');
  const [numSamples, setNumSamples] = useState<number>(2);
  const [groundTruthPrefixEnabled, setGroundTruthPrefixEnabled] = useState<boolean>(false);
  const [groundTruthPrefixLength, setGroundTruthPrefixLength] = useState<number>(48);
  const [forecastLength, setForecastLength] = useState<number>(96);
  const [lastLLMResponse, setLastLLMResponse] = useState<{
    code_executed: string;
    explanation: string;
  } | null>(null);

  // Track modification history
  const [modificationHistory, setModificationHistory] = useState<Array<{
    query: string;
    explanation: string;
    timestamp: Date;
  }>>([]);

  // Loading states
  const [isLoadingConfig, setIsLoadingConfig] = useState(false);
  const [isLoadingUpload, setIsLoadingUpload] = useState(false);
  const [isLoadingLLM, setIsLoadingLLM] = useState(false);
  const [isLoadingSynthesis, setIsLoadingSynthesis] = useState(false);

  // Error state
  const [error, setError] = useState<string | null>(null);

  // Side panel state
  const [isDataPanelOpen, setIsDataPanelOpen] = useState(true);

  // Track if we've auto-selected "oura"
  const hasAutoSelected = useRef(false);

  // Load available datasets on mount and auto-select "oura"
  useEffect(() => {
    if (hasAutoSelected.current) return;

    let isMounted = true;

    getDatasets()
      .then((datasets) => {
        if (!isMounted || hasAutoSelected.current) return;
        setAvailableDatasets(datasets);
        // Auto-select "oura" if available
        if (datasets.includes('oura')) {
          hasAutoSelected.current = true;
          setIsLoadingConfig(true);
          getConfig('oura' as DatasetName)
            .then((configData) => {
              if (!isMounted) return;
              setSelectedDataset('oura' as DatasetName);
              setConfig(configData);
            })
            .catch((err: unknown) => {
              if (!isMounted) return;
              const message = err instanceof Error ? err.message : 'Unknown error';
              setError(`Failed to load config: ${message}`);
            })
            .finally(() => {
              if (isMounted) {
                setIsLoadingConfig(false);
              }
            });
        }
      })
      .catch((err) => {
        if (isMounted) {
          setError(`Failed to load datasets: ${err.message}`);
        }
      });

    return () => {
      isMounted = false;
    };
  }, []);

  // Handle dataset selection
  const handleDatasetSelect = useCallback(async (dataset: DatasetName) => {
    setSelectedDataset(dataset);
    setConfig(null);
    setUploadedData(null);
    setModifiedData(null);
    setPreviousData(null);
    setChangedCells([]);
    setFlashKey(0);
    setValidation(null);
    setUploadedFileName(null);
    setBaseScenarioResult(null);
    setModifiedScenarioResult(null);
    setShowGroundTruth(true);
    setShowBaseScenario(true);
    setShowModifiedScenario(true);
    setLastLLMResponse(null);
    setError(null);
    setModificationHistory([]);

    setIsLoadingConfig(true);
    try {
      const configData = await getConfig(dataset);
      setConfig(configData);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      setError(`Failed to load config: ${message}`);
    } finally {
      setIsLoadingConfig(false);
    }
  }, []);

  // Handle file upload
  const handleFileUpload = useCallback(async (file: File) => {
    if (!selectedDataset) return;

    setIsLoadingUpload(true);
    setError(null);

    try {
      const response = await uploadFile(selectedDataset, file);
      setUploadedData(response.data);
      setModifiedData(response.data);
      setPreviousData(null);
      setChangedCells([]);
      setFlashKey(0);
      setValidation(response.validation);
      setUploadedFileName(file.name);
      // Clear scenario results on new upload (base will be re-computed)
      setBaseScenarioResult(null);
      setModifiedScenarioResult(null);
      setShowGroundTruth(true);
      setShowBaseScenario(true);
      setShowModifiedScenario(true);
      setLastLLMResponse(null);
      setModificationHistory([]); // Clear history on new upload
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      setError(`Upload failed: ${message}`);
    } finally {
      setIsLoadingUpload(false);
    }
  }, [selectedDataset]);

  // Compare two DataFrameModels and return specific cells that changed
  // Returns array of "columnName:index" strings
  const findChangedCells = useCallback((
    oldData: DataFrameModel | null,
    newData: DataFrameModel
  ): string[] => {
    if (!oldData) return [];

    const changed: string[] = [];
    const columns = Object.keys(newData.columns);

    for (const col of columns) {
      const oldCol = oldData.columns[col];
      const newCol = newData.columns[col];
      if (!oldCol || !newCol) continue;

      const length = Math.max(oldCol.length, newCol.length);
      for (let i = 0; i < length; i++) {
        const oldVal = oldCol[i];
        const newVal = newCol[i];

        // Compare values
        let isChanged = false;
        if (oldVal !== newVal) {
          if (typeof oldVal === 'number' && typeof newVal === 'number') {
            if (Math.abs(oldVal - newVal) > 1e-10) {
              isChanged = true;
            }
          } else {
            isChanged = true;
          }
        }

        if (isChanged) {
          changed.push(`${col}:${i}`);
        }
      }
    }

    return changed;
  }, []);

  // Handle LLM modification
  const handleLLMModify = useCallback(async (query: string) => {
    if (!selectedDataset || !modifiedData) return;

    setIsLoadingLLM(true);
    setError(null);

    // Clear previous changes immediately so glow doesn't show during loading
    setChangedCells([]);

    try {
      // Store current data as previous before modification
      const currentData = modifiedData;

      const response = await modifyWithLLM({
        dataset_name: selectedDataset,
        data: modifiedData,
        user_query: query,
      });

      // Find changed cells (specific column:index pairs)
      const changed = findChangedCells(currentData, response.modified_data);
      setChangedCells(changed);
      setFlashKey(prev => prev + 1);  // Increment to trigger animation

      setPreviousData(currentData);
      setModifiedData(response.modified_data);
      setLastLLMResponse({
        code_executed: response.code_executed,
        explanation: response.explanation,
      });
      // Add to modification history
      setModificationHistory(prev => [...prev, {
        query,
        explanation: response.explanation,
        timestamp: new Date(),
      }]);
      // Clear modified scenario result (base scenario stays cached)
      setModifiedScenarioResult(null);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      setError(`LLM modification failed: ${message}`);
    } finally {
      setIsLoadingLLM(false);
    }
  }, [selectedDataset, modifiedData, findChangedCells]);

  // Handle reset to original data
  const handleResetModifications = useCallback(() => {
    if (!uploadedData) return;

    setModifiedData(uploadedData);
    setPreviousData(null);
    setChangedCells([]);
    setFlashKey(0);
    setModificationHistory([]);
    setLastLLMResponse(null);
    // Clear modified scenario, keep base scenario cached
    setModifiedScenarioResult(null);
  }, [uploadedData]);

  // Check if data has been modified
  const hasModifications = modificationHistory.length > 0;

  // Handle synthesis - runs base + modified scenarios in parallel when modifications exist
  const handleSynthesize = useCallback(async () => {
    if (!selectedDataset || !modifiedData || !uploadedData) return;

    setIsLoadingSynthesis(true);
    setError(null);

    try {
      if (hasModifications) {
        // Run both base and modified scenarios in parallel
        const baseRequest = {
          dataset_name: selectedDataset,
          data: uploadedData, // Original unmodified data
          model_type: selectedModelType,
          task_type: selectedTaskType,
          num_samples: numSamples,
          ground_truth_prefix_length: selectedTaskType === 'synthesis' && groundTruthPrefixEnabled ? groundTruthPrefixLength : 0,
          ...(selectedTaskType === 'forecast' ? { forecast_length: forecastLength } : {}),
        };

        const modifiedRequest = {
          dataset_name: selectedDataset,
          data: modifiedData, // Modified data
          model_type: selectedModelType,
          task_type: selectedTaskType,
          num_samples: numSamples,
          ground_truth_prefix_length: selectedTaskType === 'synthesis' && groundTruthPrefixEnabled ? groundTruthPrefixLength : 0,
          ...(selectedTaskType === 'forecast' ? { forecast_length: forecastLength } : {}),
        };

        // Check if we need to re-run base scenario (only if not cached or settings changed)
        const needsBaseScenario = !baseScenarioResult ||
          baseScenarioResult.model_type !== selectedModelType ||
          baseScenarioResult.task_type !== selectedTaskType ||
          (selectedTaskType === 'forecast' && baseScenarioResult.forecast_horizon !== forecastLength);

        if (needsBaseScenario) {
          // Run both in parallel
          const [baseResponse, modifiedResponse] = await Promise.all([
            synthesize(baseRequest),
            synthesize(modifiedRequest),
          ]);
          setBaseScenarioResult(baseResponse);
          setModifiedScenarioResult(modifiedResponse);
        } else {
          // Only run modified scenario (base is cached)
          const modifiedResponse = await synthesize(modifiedRequest);
          setModifiedScenarioResult(modifiedResponse);
        }

        // Show both scenarios by default
        setShowBaseScenario(true);
        setShowModifiedScenario(true);
      } else {
        // No modifications - just run base scenario
        const response = await synthesize({
          dataset_name: selectedDataset,
          data: uploadedData,
          model_type: selectedModelType,
          task_type: selectedTaskType,
          num_samples: numSamples,
          ground_truth_prefix_length: selectedTaskType === 'synthesis' && groundTruthPrefixEnabled ? groundTruthPrefixLength : 0,
          ...(selectedTaskType === 'forecast' ? { forecast_length: forecastLength } : {}),
        });
        setBaseScenarioResult(response);
        setModifiedScenarioResult(null);
        setShowBaseScenario(true);
      }
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Unknown error';
      setError(`Synthesis failed: ${message}`);
    } finally {
      setIsLoadingSynthesis(false);
    }
  }, [selectedDataset, modifiedData, uploadedData, selectedModelType, selectedTaskType, numSamples, groundTruthPrefixEnabled, groundTruthPrefixLength, forecastLength, hasModifications, baseScenarioResult]);

  // Convert data to timeseries for chart
  const getTimeseriesFromData = useCallback((data: DataFrameModel | null): OneTimeSeries[] => {
    if (!data || !config) return [];

    return config.required_columns.timeseries
      .filter((col) => col in data.columns)
      .map((col) => ({
        name: col,
        values: data.columns[col] as (number | null)[],
      }));
  }, [config]);

  // Get metadata (continuous + discrete) for chart
  const getMetadataFromData = useCallback((data: DataFrameModel | null): OneTimeSeries[] => {
    if (!data || !config) return [];

    const metadataCols = [
      ...config.required_columns.continuous,
      ...config.required_columns.discrete,
    ];

    return metadataCols
      .filter((col) => col in data.columns)
      .map((col) => ({
        name: col,
        values: data.columns[col] as (number | null)[],
      }));
  }, [config]);

  const currentTimeseries = getTimeseriesFromData(modifiedData);
  const currentMetadata = getMetadataFromData(modifiedData);

  return (
    <div className="min-h-screen">
      {/* OURA-style Header with Liquid Glass */}
      <header className="sticky top-0 z-50 liquid-glass-header">
        <div className="max-w-[1600px] mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              {/* OURA-style logo */}
              <div className="flex items-center gap-3">
                <div className="oura-ring-icon">
                  <svg className="w-5 h-5" style={{ color: '#f5f2ed' }} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <circle cx="12" cy="12" r="8" />
                    <circle cx="12" cy="12" r="4" />
                  </svg>
                </div>
                <span className="text-xl font-medium tracking-wide" style={{ fontFamily: 'DM Sans, sans-serif', color: '#f5f2ed' }}>ŌURA x Synthefy</span>
              </div>
            </div>
            <div className="flex items-center gap-3">
              {selectedDataset && (
                <span className="tag tag-accent">
                  <svg className="w-3.5 h-3.5 mr-1.5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  {selectedDataset}
                </span>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Error banner - Dark OURA style */}
      {error && (
        <div className="max-w-[1600px] mx-auto px-6 mt-4 animate-fade-in">
          <div className="px-5 py-4 rounded-xl flex justify-between items-center" style={{ background: 'rgba(229, 115, 115, 0.15)', border: '1px solid rgba(229, 115, 115, 0.3)' }}>
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ background: 'rgba(229, 115, 115, 0.2)' }}>
                <svg className="w-5 h-5" style={{ color: '#e57373' }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <span style={{ color: '#f5f2ed' }}>{error}</span>
            </div>
            <button
              onClick={() => setError(null)}
              className="p-2 rounded-lg transition-colors"
              style={{ color: '#e57373' }}
            >
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>
      )}

      {/* Main content with collapsible side panel */}
      <div className="flex min-h-[calc(100vh-80px)]">
        {/* Collapsible Data Panel (Dataset & Upload) */}
        <aside
          className={`
            transition-all duration-300 ease-in-out flex-shrink-0
            ${isDataPanelOpen ? 'w-80' : 'w-0'}
            overflow-hidden
          `}
          style={{
            background: 'rgba(15, 14, 12, 0.6)',
            borderRight: isDataPanelOpen ? '1px solid rgba(140, 125, 105, 0.15)' : 'none',
          }}
        >
          <div className="w-80 p-4 space-y-4 h-full overflow-y-auto">
            {/* Panel Header */}
            <div className="flex items-center justify-between mb-2">
              <h2 className="text-sm font-medium uppercase tracking-wider" style={{ color: 'rgba(245, 242, 237, 0.5)' }}>
                Data Source
              </h2>
              <button
                onClick={() => setIsDataPanelOpen(false)}
                className="p-1.5 rounded-lg transition-colors"
                style={{ color: 'rgba(245, 242, 237, 0.5)' }}
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 19l-7-7 7-7m8 14l-7-7 7-7" />
                </svg>
              </button>
            </div>

            <DatasetSelector
              selectedDataset={selectedDataset}
              config={config}
              availableDatasets={availableDatasets}
              onSelect={handleDatasetSelect}
              isLoading={isLoadingConfig}
            />

            <FileUploader
              onUpload={handleFileUpload}
              isLoading={isLoadingUpload}
              disabled={!selectedDataset}
              validation={validation}
              uploadedFileName={uploadedFileName}
            />

            {/* Model Settings */}
            {uploadedData && (
              <div className="liquid-glass-card p-4 space-y-4">
                <h3 className="text-sm font-medium uppercase tracking-wider" style={{ color: 'rgba(245, 242, 237, 0.5)' }}>
                  Model Settings
                </h3>

                {/* Model Type */}
                <div className="space-y-2">
                  <span className="text-xs" style={{ color: 'rgba(245, 242, 237, 0.5)' }}>Model</span>
                  <div className="flex gap-1">
                    <button
                      onClick={() => setSelectedModelType('flexible')}
                      disabled={isLoadingSynthesis}
                      className="flex-1 px-3 py-2 rounded-lg text-xs font-medium transition-all duration-200"
                      style={{
                        background: selectedModelType === 'flexible' ? 'linear-gradient(135deg, #7eb8da 0%, #4a7c9b 100%)' : 'rgba(60, 55, 50, 0.5)',
                        color: selectedModelType === 'flexible' ? 'white' : 'rgba(245, 242, 237, 0.7)',
                        opacity: isLoadingSynthesis ? 0.5 : 1,
                      }}
                    >
                      Flexible
                    </button>
                    <button
                      onClick={() => setSelectedModelType('standard')}
                      disabled={isLoadingSynthesis}
                      className="flex-1 px-3 py-2 rounded-lg text-xs font-medium transition-all duration-200"
                      style={{
                        background: selectedModelType === 'standard' ? 'linear-gradient(135deg, #7eb8da 0%, #4a7c9b 100%)' : 'rgba(60, 55, 50, 0.5)',
                        color: selectedModelType === 'standard' ? 'white' : 'rgba(245, 242, 237, 0.7)',
                        opacity: isLoadingSynthesis ? 0.5 : 1,
                      }}
                    >
                      Standard
                    </button>
                  </div>
                </div>

                {/* Task Type */}
                <div className="space-y-2">
                  <span className="text-xs" style={{ color: 'rgba(245, 242, 237, 0.5)' }}>Task</span>
                  <div className="flex gap-1">
                    <button
                      onClick={() => setSelectedTaskType('synthesis')}
                      disabled={isLoadingSynthesis}
                      className="flex-1 px-3 py-2 rounded-lg text-xs font-medium transition-all duration-200"
                      style={{
                        background: selectedTaskType === 'synthesis' ? 'linear-gradient(135deg, #e57373 0%, #c75050 100%)' : 'rgba(60, 55, 50, 0.5)',
                        color: selectedTaskType === 'synthesis' ? 'white' : 'rgba(245, 242, 237, 0.7)',
                        opacity: isLoadingSynthesis ? 0.5 : 1,
                      }}
                    >
                      Synthesis
                    </button>
                    <button
                      onClick={() => setSelectedTaskType('forecast')}
                      disabled={isLoadingSynthesis}
                      className="flex-1 px-3 py-2 rounded-lg text-xs font-medium transition-all duration-200"
                      style={{
                        background: selectedTaskType === 'forecast' ? 'linear-gradient(135deg, #e57373 0%, #c75050 100%)' : 'rgba(60, 55, 50, 0.5)',
                        color: selectedTaskType === 'forecast' ? 'white' : 'rgba(245, 242, 237, 0.7)',
                        opacity: isLoadingSynthesis ? 0.5 : 1,
                      }}
                    >
                      Forecast
                    </button>
                  </div>
                </div>

                {/* Num Samples */}
                <div className="space-y-2">
                  <span className="text-xs" style={{ color: 'rgba(245, 242, 237, 0.5)' }}>Samples to Average</span>
                  <div className="flex items-center gap-2">
                    <input
                      type="number"
                      min={1}
                      max={100}
                      value={numSamples}
                      onChange={(e) => {
                        const val = parseInt(e.target.value, 10);
                        if (!isNaN(val) && val >= 1 && val <= 100) {
                          setNumSamples(val);
                        }
                      }}
                      disabled={isLoadingSynthesis}
                      className="flex-1 px-3 py-2 rounded-lg text-xs font-medium text-center"
                      style={{
                        background: 'linear-gradient(135deg, #f5a623 0%, #d4920f 100%)',
                        color: 'white',
                        border: 'none',
                        opacity: isLoadingSynthesis ? 0.5 : 1,
                      }}
                    />
                    <span className="text-xs" style={{ color: 'rgba(245, 242, 237, 0.4)' }}>(1-100)</span>
                  </div>
                </div>

                {/* Ground Truth Prefix - Only for Synthesis */}
                {selectedTaskType === 'synthesis' && (
                  <div className="space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-xs" style={{ color: 'rgba(245, 242, 237, 0.5)' }}>Ground Truth Prefix</span>
                      <label className="flex items-center cursor-pointer">
                        <input
                          type="checkbox"
                          checked={groundTruthPrefixEnabled}
                          onChange={(e) => setGroundTruthPrefixEnabled(e.target.checked)}
                          disabled={isLoadingSynthesis}
                          className="sr-only"
                        />
                        <div
                          className="w-10 h-5 rounded-full transition-all relative"
                          style={{
                            background: groundTruthPrefixEnabled
                              ? 'linear-gradient(135deg, #07ad98 0%, #059681 100%)'
                              : 'rgba(60, 55, 50, 0.6)',
                          }}
                        >
                          <div
                            className="w-4 h-4 rounded-full absolute top-0.5 transition-all"
                            style={{
                              background: 'white',
                              left: groundTruthPrefixEnabled ? '22px' : '2px',
                            }}
                          />
                        </div>
                      </label>
                    </div>
                    {groundTruthPrefixEnabled && (
                      <div className="flex items-center gap-2">
                        <input
                          type="number"
                          min={1}
                          max={config?.window_size || 192}
                          value={groundTruthPrefixLength}
                          onChange={(e) => {
                            const val = parseInt(e.target.value, 10);
                            const maxVal = config?.window_size || 192;
                            if (!isNaN(val) && val >= 1 && val <= maxVal) {
                              setGroundTruthPrefixLength(val);
                            }
                          }}
                          disabled={isLoadingSynthesis}
                          className="flex-1 px-3 py-2 rounded-lg text-xs font-medium text-center"
                          style={{
                            background: 'linear-gradient(135deg, #07ad98 0%, #059681 100%)',
                            color: 'white',
                            border: 'none',
                            opacity: isLoadingSynthesis ? 0.5 : 1,
                          }}
                        />
                        <span className="text-xs" style={{ color: 'rgba(245, 242, 237, 0.4)' }}>pts</span>
                      </div>
                    )}
                    <p className="text-xs" style={{ color: 'rgba(245, 242, 237, 0.4)' }}>
                      First N points kept as ground truth
                    </p>
                  </div>
                )}

                {/* Forecast Length - Only for Forecast */}
                {selectedTaskType === 'forecast' && (
                  <div className="space-y-2">
                    <span className="text-xs" style={{ color: 'rgba(245, 242, 237, 0.5)' }}>Forecast Length</span>
                    <div className="flex items-center gap-2">
                      <input
                        type="number"
                        min={1}
                        max={config?.window_size || 192}
                        value={forecastLength}
                        onChange={(e) => {
                          const val = parseInt(e.target.value, 10);
                          const maxVal = config?.window_size || 192;
                          if (!isNaN(val) && val >= 1 && val <= maxVal) {
                            setForecastLength(val);
                          }
                        }}
                        disabled={isLoadingSynthesis}
                        className="flex-1 px-3 py-2 rounded-lg text-xs font-medium text-center"
                        style={{
                          background: 'linear-gradient(135deg, #e57373 0%, #c75050 100%)',
                          color: 'white',
                          border: 'none',
                          opacity: isLoadingSynthesis ? 0.5 : 1,
                        }}
                      />
                      <span className="text-xs" style={{ color: 'rgba(245, 242, 237, 0.4)' }}>steps</span>
                    </div>
                    <p className="text-xs" style={{ color: 'rgba(245, 242, 237, 0.4)' }}>
                      Number of future time steps to forecast
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>
        </aside>

        {/* Toggle button when panel is closed */}
        {!isDataPanelOpen && (
          <button
            onClick={() => setIsDataPanelOpen(true)}
            className="fixed left-0 top-1/2 -translate-y-1/2 z-40 p-3 rounded-r-xl transition-all duration-200"
            style={{
              background: 'rgba(140, 125, 105, 0.4)',
              backdropFilter: 'blur(12px)',
              border: '1px solid rgba(140, 125, 105, 0.3)',
              borderLeft: 'none',
            }}
          >
            <svg className="w-5 h-5" style={{ color: '#f5f2ed' }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 5l7 7-7 7M5 5l7 7-7 7" />
            </svg>
          </button>
        )}

        {/* Main content area */}
        <main className="flex-1 overflow-y-auto">
          <div className="max-w-[1600px] mx-auto px-6 py-6">
            <div className="grid grid-cols-1 xl:grid-cols-12 gap-6">
              {/* Left column - Scenario Simulation & Synthesis */}
              <div className="xl:col-span-3 space-y-5">
                {/* Data status indicator */}
                {uploadedFileName && (
                  <div
                    className="p-3 rounded-xl flex items-center gap-3"
                    style={{
                      background: 'rgba(7, 173, 152, 0.15)',
                      border: '1px solid rgba(7, 173, 152, 0.25)'
                    }}
                  >
                    <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ background: 'rgba(7, 173, 152, 0.2)' }}>
                      <svg className="w-4 h-4" style={{ color: '#07ad98' }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                      </svg>
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium truncate" style={{ color: '#f5f2ed' }}>{uploadedFileName}</p>
                      <p className="text-xs" style={{ color: 'rgba(245, 242, 237, 0.5)' }}>Data loaded</p>
                    </div>
                    {!isDataPanelOpen && (
                      <button
                        onClick={() => setIsDataPanelOpen(true)}
                        className="p-2 rounded-lg transition-colors"
                        style={{ color: 'rgba(245, 242, 237, 0.5)' }}
                        title="Open data panel"
                      >
                        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                        </svg>
                      </button>
                    )}
                  </div>
                )}

                {/* Chart Visibility Toggles - show when we have results */}
                {(baseScenarioResult || modifiedScenarioResult) && (
                  <div
                    className="p-4 rounded-xl space-y-3"
                    style={{
                      background: 'rgba(45, 40, 35, 0.6)',
                      border: '1px solid rgba(140, 125, 105, 0.2)'
                    }}
                  >
                    <p className="text-xs font-medium uppercase tracking-wider" style={{ color: 'rgba(245, 242, 237, 0.5)' }}>
                      Chart Layers
                    </p>

                    {/* Ground Truth Toggle */}
                    <label className="flex items-center gap-3 cursor-pointer group">
                      <input
                        type="checkbox"
                        checked={showGroundTruth}
                        onChange={(e) => setShowGroundTruth(e.target.checked)}
                        className="sr-only"
                      />
                      <div
                        className="w-5 h-5 rounded flex items-center justify-center transition-all"
                        style={{
                          background: showGroundTruth ? '#f5f2ed' : 'rgba(60, 55, 50, 0.6)',
                          border: '2px solid ' + (showGroundTruth ? '#f5f2ed' : 'rgba(140, 125, 105, 0.4)'),
                        }}
                      >
                        {showGroundTruth && (
                          <svg className="w-3 h-3" style={{ color: '#1a1815' }} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
                            <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                          </svg>
                        )}
                      </div>
                      <div className="flex items-center gap-2">
                        <div className="w-4 h-0.5 rounded" style={{ background: '#f5f2ed' }} />
                        <span className="text-sm" style={{ color: showGroundTruth ? '#f5f2ed' : 'rgba(245, 242, 237, 0.5)' }}>
                          Ground Truth
                        </span>
                      </div>
                    </label>

                    {/* Base Scenario Toggle */}
                    {baseScenarioResult && (
                      <label className="flex items-center gap-3 cursor-pointer group">
                        <input
                          type="checkbox"
                          checked={showBaseScenario}
                          onChange={(e) => setShowBaseScenario(e.target.checked)}
                          className="sr-only"
                        />
                        <div
                          className="w-5 h-5 rounded flex items-center justify-center transition-all"
                          style={{
                            background: showBaseScenario ? '#7eb8da' : 'rgba(60, 55, 50, 0.6)',
                            border: '2px solid ' + (showBaseScenario ? '#7eb8da' : 'rgba(140, 125, 105, 0.4)'),
                          }}
                        >
                          {showBaseScenario && (
                            <svg className="w-3 h-3" style={{ color: '#1a1815' }} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
                              <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                            </svg>
                          )}
                        </div>
                        <div className="flex items-center gap-2">
                          <div className="w-4 h-0.5 rounded" style={{ background: '#7eb8da', opacity: 0.7 }} />
                          <span className="text-sm" style={{ color: showBaseScenario ? '#f5f2ed' : 'rgba(245, 242, 237, 0.5)' }}>
                            Base Prediction
                          </span>
                        </div>
                      </label>
                    )}

                    {/* Modified Scenario Toggle */}
                    {modifiedScenarioResult && (
                      <label className="flex items-center gap-3 cursor-pointer group">
                        <input
                          type="checkbox"
                          checked={showModifiedScenario}
                          onChange={(e) => setShowModifiedScenario(e.target.checked)}
                          className="sr-only"
                        />
                        <div
                          className="w-5 h-5 rounded flex items-center justify-center transition-all"
                          style={{
                            background: showModifiedScenario ? '#07ad98' : 'rgba(60, 55, 50, 0.6)',
                            border: '2px solid ' + (showModifiedScenario ? '#07ad98' : 'rgba(140, 125, 105, 0.4)'),
                          }}
                        >
                          {showModifiedScenario && (
                            <svg className="w-3 h-3" style={{ color: '#1a1815' }} fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={3}>
                              <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                            </svg>
                          )}
                        </div>
                        <div className="flex items-center gap-2">
                          <div
                            className="w-4 h-0.5 rounded"
                            style={{
                              background: 'repeating-linear-gradient(90deg, #07ad98 0, #07ad98 3px, transparent 3px, transparent 5px)',
                            }}
                          />
                          <span className="text-sm" style={{ color: showModifiedScenario ? '#f5f2ed' : 'rgba(245, 242, 237, 0.5)' }}>
                            Modified Prediction
                          </span>
                        </div>
                      </label>
                    )}
                  </div>
                )}

                <LLMChat
                  onSubmit={handleLLMModify}
                  onReset={handleResetModifications}
                  isLoading={isLoadingLLM}
                  disabled={!modifiedData}
                  hasModifications={modificationHistory.length > 0 || lastLLMResponse !== null}
                  lastResponse={lastLLMResponse ?? undefined}
                  modificationHistory={modificationHistory}
                />

                <SynthesisPanel
                  onSynthesize={handleSynthesize}
                  isLoading={isLoadingSynthesis}
                  disabled={!modifiedData || !validation?.valid}
                  baseResult={baseScenarioResult}
                  modifiedResult={modifiedScenarioResult}
                  modificationCount={modificationHistory.length}
                />
              </div>

              {/* Right column - Visualizations */}
              <div className="xl:col-span-9 space-y-6">
                {/* Empty state - Dark OURA style */}
                {!uploadedData && (
                  <div className="card p-12 text-center animate-fade-in-up">
                    <div className="w-24 h-24 mx-auto mb-8 rounded-2xl flex items-center justify-center" style={{ background: 'linear-gradient(135deg, rgba(126, 184, 218, 0.15) 0%, rgba(74, 124, 155, 0.1) 100%)' }}>
                      <svg className="w-12 h-12" style={{ color: '#7eb8da' }} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                        <circle cx="12" cy="12" r="8" />
                        <circle cx="12" cy="12" r="4" />
                        <path d="M12 4v2M12 18v2M4 12h2M18 12h2" />
                      </svg>
                    </div>
                    <h3 className="oura-heading text-3xl mb-3">
                      Explore Your Health Potential
                    </h3>
                    <p className="text-lg max-w-lg mx-auto mb-8" style={{ color: 'rgba(245, 242, 237, 0.6)' }}>
                      Upload your wearable data and ask "what if" questions. See how lifestyle changes could impact your heart rate, HRV, sleep, and more.
                    </p>
                    <div className="flex flex-wrap justify-center gap-3">
                      <span className="tag">
                        <svg className="w-4 h-4 mr-1.5" style={{ color: '#7eb8da' }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
                        </svg>
                        Sleep
                      </span>
                      <span className="tag">
                        <svg className="w-4 h-4 mr-1.5" style={{ color: '#e57373' }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
                        </svg>
                        Heart Rate
                      </span>
                      <span className="tag">
                        <svg className="w-4 h-4 mr-1.5" style={{ color: '#f5a623' }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                        </svg>
                        Activity
                      </span>
                      <span className="tag">
                        <svg className="w-4 h-4 mr-1.5" style={{ color: '#a8c8dc' }} fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                        </svg>
                        HRV
                      </span>
                    </div>
                  </div>
                )}

                {/* Delta Display - Show impact of changes above the chart */}
                <DeltaDisplay
                  baseResult={baseScenarioResult}
                  modifiedResult={modifiedScenarioResult}
                />

                {/* Main Chart: Shows current data, optionally with simulation overlays - Liquid Glass */}
                {currentTimeseries.length > 0 && (
                  <TimeSeriesChart
                    original={showGroundTruth ? currentTimeseries : undefined}
                    baseSynthetic={showBaseScenario && baseScenarioResult ? baseScenarioResult.synthetic_timeseries : undefined}
                    modifiedSynthetic={showModifiedScenario && modifiedScenarioResult ? modifiedScenarioResult.synthetic_timeseries : undefined}
                    metadata={currentMetadata}
                    title={
                      modifiedScenarioResult
                        ? modifiedScenarioResult.task_type === 'forecast'
                          ? "Time Series Forecast Comparison"
                          : "Scenario Comparison"
                        : baseScenarioResult
                          ? baseScenarioResult.task_type === 'forecast'
                            ? "Time Series Forecast"
                            : "Base Scenario Prediction"
                          : "Your Health Data"
                    }
                    subtitle={
                      modifiedScenarioResult
                        ? "Compare base vs modified predictions"
                        : baseScenarioResult
                          ? "Prediction based on original data"
                          : lastLLMResponse
                            ? "Scenario applied ✨ — run simulation to see predictions"
                            : "Original wearable metrics"
                    }
                    changedColumns={[...new Set(changedCells.map(c => c.split(':')[0]))]}
                    flashKey={flashKey}
                    showGroundTruth={showGroundTruth}
                    showBaseScenario={showBaseScenario}
                    showModifiedScenario={showModifiedScenario}
                    taskType={modifiedScenarioResult?.task_type || baseScenarioResult?.task_type}
                    forecastHorizon={modifiedScenarioResult?.forecast_horizon || baseScenarioResult?.forecast_horizon || forecastLength}
                  />
                )}

                {/* Data table */}
                {modifiedData && config && (
                  <DataTable
                    data={modifiedData}
                    title="Data Preview"
                    highlightColumns={config.required_columns.timeseries}
                    changedCells={changedCells}
                    flashKey={flashKey}
                    timeseriesColumns={config.required_columns.timeseries}
                    metadataColumns={[
                      ...config.required_columns.discrete,
                      ...config.required_columns.continuous,
                    ]}
                    requiredColumns={[
                      ...config.required_columns.timeseries,
                      ...config.required_columns.discrete,
                      ...config.required_columns.continuous,
                    ]}
                  />
                )}
              </div>
            </div>
          </div>
        </main>
      </div>

      {/* OURA-style footer - Dark */}
      <footer className="border-t mt-12" style={{ borderColor: 'rgba(140, 125, 105, 0.15)' }}>
        <div className="max-w-[1600px] mx-auto px-6 py-6">
          <div className="flex items-center justify-between text-sm" style={{ color: 'rgba(245, 242, 237, 0.5)' }}>
            <div className="flex items-center gap-2">
              <div className="w-6 h-6 rounded-md flex items-center justify-center" style={{ background: 'rgba(140, 125, 105, 0.4)' }}>
                <svg className="w-3.5 h-3.5" style={{ color: '#f5f2ed' }} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="12" r="8" />
                  <circle cx="12" cy="12" r="4" />
                </svg>
              </div>
              <span>ŌURA Health Scenario Lab</span>
            </div>
            <span>Powered by AI Synthesis</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
