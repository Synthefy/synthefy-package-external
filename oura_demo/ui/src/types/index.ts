/**
 * TypeScript types for Oura Demo UI
 * Mirrors the Pydantic models from the backend
 */

export type DatasetName = 'oura' | 'oura_subset' | 'ppg';

export type FileType = 'parquet' | 'csv';

export type ModelType = 'standard' | 'flexible';

export type TaskType = 'synthesis' | 'forecast';

// DataFrame-like structure
export interface DataFrameModel {
  columns: Record<string, (number | string | null)[]>;
}

// Config types
export interface RequiredColumns {
  timeseries: string[];
  discrete: string[];
  continuous: string[];
  group_labels: string[];
}

export interface ConfigResponse {
  dataset_name: DatasetName;
  required_columns: RequiredColumns;
  window_size: number;
  num_channels: number;
  available_datasets: string[];
}

// Validation types
export interface ColumnValidationResult {
  valid: boolean;
  missing_columns: string[];
  extra_columns: string[];
}

// Upload types
export interface UploadResponse {
  data: DataFrameModel;
  window_size: number;
  validation: ColumnValidationResult;
  file_type: FileType;
}

// LLM types
export interface LLMModifyRequest {
  dataset_name: DatasetName;
  data: DataFrameModel;
  user_query: string;
}

export interface LLMModifyResponse {
  modified_data: DataFrameModel;
  code_executed: string;
  explanation: string;
}

// Synthesis types
export interface OneTimeSeries {
  name: string;
  values: (number | null)[];
}

export interface SynthesisRequest {
  dataset_name: DatasetName;
  data: DataFrameModel;
  model_type?: ModelType;  // defaults to 'flexible' on backend
  task_type?: TaskType;    // defaults to 'synthesis' on backend
  num_samples?: number;    // defaults to 2 on backend (1-100)
  ground_truth_prefix_length?: number;  // For synthesis: keep first N points as ground truth (0 = disabled)
  forecast_length?: number;  // For forecast: number of time steps to forecast (default: 96)
}

export interface SynthesisResponse {
  original_timeseries: OneTimeSeries[];
  synthetic_timeseries: OneTimeSeries[];
  window_size: number;
  num_channels: number;
  dataset_name: DatasetName;
  model_type: ModelType;
  task_type: TaskType;
  forecast_horizon?: number;  // Number of time steps in forecast (for forecast task)
}

// MAPE (Mean Absolute Percentage Error) result for a single metric
export interface MetricMAPE {
  name: string;
  mape: number;  // Percentage value (0-100+)
}

// App state
export interface AppState {
  selectedDataset: DatasetName | null;
  selectedModelType: ModelType;
  selectedTaskType: TaskType;
  config: ConfigResponse | null;
  uploadedData: DataFrameModel | null;
  modifiedData: DataFrameModel | null;
  synthesisResult: SynthesisResponse | null;
  isLoading: boolean;
  error: string | null;
}
