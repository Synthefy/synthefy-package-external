/**
 * API client for Oura Demo backend
 */

import axios from 'axios';
import type {
  ConfigResponse,
  DataFrameModel,
  DatasetName,
  LLMModifyRequest,
  LLMModifyResponse,
  MetricMAPE,
  OneTimeSeries,
  SynthesisRequest,
  SynthesisResponse,
  UploadResponse,
} from '../types';

const API_BASE = '/api';

// Create axios instance
const api = axios.create({
  baseURL: API_BASE,
  headers: {
    'Content-Type': 'application/json',
  },
});

/**
 * Get list of available datasets
 */
export async function getDatasets(): Promise<string[]> {
  const response = await api.get<string[]>('/config/datasets');
  return response.data;
}

/**
 * Get config for a specific dataset
 */
export async function getConfig(datasetName: DatasetName): Promise<ConfigResponse> {
  const response = await api.get<ConfigResponse>(`/config/${datasetName}`);
  return response.data;
}

/**
 * Upload a file for a specific dataset
 */
export async function uploadFile(
  datasetName: DatasetName,
  file: File
): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const response = await api.post<UploadResponse>(
    `/upload/${datasetName}`,
    formData,
    {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    }
  );
  return response.data;
}

/**
 * Modify data using LLM
 */
export async function modifyWithLLM(
  request: LLMModifyRequest
): Promise<LLMModifyResponse> {
  const response = await api.post<LLMModifyResponse>('/llm/modify', request);
  return response.data;
}

/**
 * Run synthesis on data using the real model inference.
 *
 * The backend defaults to real model inference. To use mock mode (for UI dev),
 * set USE_MOCK_SYNTHESIS=true env var on the backend or pass ?mock=true.
 */
export async function synthesize(
  request: SynthesisRequest
): Promise<SynthesisResponse> {
  const response = await api.post<SynthesisResponse>('/synthesize', request);
  return response.data;
}

/**
 * Convert DataFrameModel to array of row objects for display
 */
export function dataFrameToRows(
  data: DataFrameModel
): Record<string, number | string | null>[] {
  const columns = Object.keys(data.columns);
  if (columns.length === 0) return [];

  const numRows = data.columns[columns[0]].length;
  const rows: Record<string, number | string | null>[] = [];

  for (let i = 0; i < numRows; i++) {
    const row: Record<string, number | string | null> = { _index: i };
    for (const col of columns) {
      row[col] = data.columns[col][i];
    }
    rows.push(row);
  }

  return rows;
}

/**
 * Convert timeseries to chart data format
 */
export function timeseriesToChartData(
  timeseries: { name: string; values: (number | null)[] }[]
): { index: number; [key: string]: number | null }[] {
  if (timeseries.length === 0) return [];

  const numPoints = timeseries[0].values.length;
  const chartData: { index: number; [key: string]: number | null }[] = [];

  for (let i = 0; i < numPoints; i++) {
    const point: { index: number; [key: string]: number | null } = { index: i };
    for (const ts of timeseries) {
      point[ts.name] = ts.values[i];
    }
    chartData.push(point);
  }

  return chartData;
}

/**
 * Calculate MAPE (Mean Absolute Percentage Error) between original and synthetic timeseries.
 * MAPE = (1/n) * Σ|actual - predicted| / |actual| * 100
 * 
 * Note: We compare the modified/scenario data (original_timeseries from synthesis response, 
 * which is the input that was modified) against the synthetic_timeseries (model predictions).
 */
export function calculateMAPE(
  original: OneTimeSeries[],
  synthetic: OneTimeSeries[]
): MetricMAPE[] {
  const results: MetricMAPE[] = [];

  for (const origTs of original) {
    // Find matching synthetic timeseries (may have _synthetic suffix)
    const synthTs = synthetic.find(
      s => s.name === origTs.name || s.name === `${origTs.name}_synthetic`
    );

    if (!synthTs) continue;

    // Calculate MAPE
    let sumAPE = 0;
    let validCount = 0;

    const minLength = Math.min(origTs.values.length, synthTs.values.length);

    for (let i = 0; i < minLength; i++) {
      const actual = origTs.values[i];
      const predicted = synthTs.values[i];

      // Skip if either value is null or actual is 0 (to avoid division by zero)
      if (actual === null || predicted === null || actual === 0) continue;

      const ape = Math.abs((actual - predicted) / actual);
      sumAPE += ape;
      validCount++;
    }

    if (validCount > 0) {
      const mape = (sumAPE / validCount) * 100;
      results.push({
        name: origTs.name,
        mape: Math.round(mape * 100) / 100,  // Round to 2 decimal places
      });
    }
  }

  return results;
}
