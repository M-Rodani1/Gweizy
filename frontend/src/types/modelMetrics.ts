/**
 * Types for model metrics and accuracy tracking
 */

export interface HorizonMetrics {
  mae: number | null;
  rmse: number | null;
  r2: number | null;
  directional_accuracy: number | null;
  n: number;
}

export interface DriftInfo {
  is_drifting: boolean;
  drift_ratio: number;
  mae_current: number;
  mae_baseline: number;
}

export interface DriftResponse {
  success: boolean;
  should_retrain: boolean;
  drift: Record<string, DriftInfo>;
  horizons_drifting: string[];
}

export interface MetricsResponse {
  success: boolean;
  metrics: Record<string, HorizonMetrics>;
}

export interface ValidationTrends {
  dates: string[];
  mae_trend: number[];
  accuracy_trend: number[];
  horizon: string;
}

export type MainTab = 'overview' | 'metrics' | 'trends';
export type Horizon = '1h' | '4h' | '24h';
