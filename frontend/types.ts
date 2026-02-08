import React from 'react';

// ========================================
// API Response Wrapper Types
// ========================================

/**
 * Standard API response wrapper for successful responses
 */
export interface ApiResponse<T> {
  data: T;
  success: true;
  timestamp: string;
}

/**
 * Standard API error response
 */
export interface ApiErrorResponse {
  success: false;
  error: string;
  code?: string;
  details?: Record<string, unknown>;
}

/**
 * Union type for API responses that can be either success or error
 */
export type ApiResult<T> = ApiResponse<T> | ApiErrorResponse;

/**
 * Type guard to check if response is successful
 */
export function isApiSuccess<T>(result: ApiResult<T>): result is ApiResponse<T> {
  return result.success === true;
}

/**
 * Type guard to check if response is an error
 */
export function isApiError<T>(result: ApiResult<T>): result is ApiErrorResponse {
  return result.success === false;
}

// ========================================
// Domain Types
// ========================================

// Bias correction info from backend
export interface BiasCorrection {
  applied: boolean;
  type?: 'time_period' | 'overall';
  period?: 'night' | 'morning' | 'afternoon' | 'evening';
  correction?: number;
  original?: number;
  corrected?: number;
}

// Graph data point
export interface GraphDataPoint {
  time: string;
  gwei?: number;
  predictedGwei?: number;
  lowerBound?: number;
  upperBound?: number;
  confidence?: number;
  confidenceLevel?: 'high' | 'medium' | 'low';
  confidenceEmoji?: string;
  confidenceColor?: string;
  trend_signal_4h?: number; // The "Macro" signal (-1.0 to 1.0)
  bias_correction?: BiasCorrection;
  classification?: {
    class?: string;
    emoji?: string;
    color?: string;
    probabilities?: {
      wait?: number;
      normal?: number;
      urgent?: number;
      // Alternative format from backend
      elevated?: number;
      spike?: number;
    };
    trend_signal_4h?: number;
  };
  probabilities?: {
    wait?: number;
    normal?: number;
    urgent?: number;
  };
}

// Current gas data from backend
export interface CurrentGasData {
  timestamp: string;
  current_gas: number;
  base_fee: number;
  priority_fee: number;
  block_number: number;
}

// Predictions response
export interface PredictionsResponse {
  current: CurrentGasData;
  predictions: {
    '1h': GraphDataPoint[];
    '4h': GraphDataPoint[];
    '24h': GraphDataPoint[];
    historical: GraphDataPoint[];
  };
  model_info?: {
    [key: string]: {
      name: string;
      mae: number;
    };
  };
}

// Transaction data
export interface TableRowData {
  txHash: string;
  method: string;
  age: string;
  gasUsed: number;
  gasPrice: number;
  timestamp: number;
}

// Historical data response
export interface HistoricalResponse {
  data: Array<{
    time: string;
    gwei: number;
    baseFee: number;
    priorityFee: number;
  }>;
  count: number;
  timeframe: string;
}

// Leaderboard item (keeping for UI)
export interface LeaderboardItem {
  rank: number;
  name: string;
  price: number;
  icon: React.ComponentType<{ className?: string }>;
}

// API error
export interface APIError {
  error: string;
  message?: string;
}

// Hybrid Model Prediction
export interface HybridPrediction {
  action: 'WAIT' | 'NORMAL' | 'URGENT';
  confidence: number; // 0.0 to 1.0
  trend_signal_4h: number; // The "Macro" signal (-1.0 to 1.0)
  probabilities: {
    wait: number;
    normal: number;
    urgent: number;
  };
}

// Block Pulse (Mempool Monitor)
export interface BlockPulse {
  block_number: number;
  utilization: number; // 0.0 to 1.0 (e.g. 0.85 for 85%)
  gas_used: number;
  base_fee: number;
  timestamp: string;
}

// Platform config response
export interface ConfigResponse {
  chains: Array<{
    id: number;
    name: string;
    enabled: boolean;
  }>;
  features: {
    [key: string]: boolean;
  };
  version: string;
}

// Model accuracy metrics
export interface AccuracyResponse {
  horizons: {
    [horizon: string]: {
      mae: number;
      rmse?: number;
      r2?: number;
      directional_accuracy?: number;
      n_samples: number;
    };
  };
  overall: {
    mae: number;
    rmse?: number;
    r2?: number;
  };
  updated_at: string;
}

// User transaction history
export interface UserHistoryResponse {
  address: string;
  transactions: Array<{
    hash: string;
    timestamp: string;
    gas_used: number;
    gas_price: number;
    savings?: number;
  }>;
  total_savings: number;
  total_transactions: number;
}

// Leaderboard response
export interface LeaderboardResponse {
  entries: Array<{
    rank: number;
    address: string;
    display_name?: string;
    total_savings: number;
    transaction_count: number;
  }>;
  updated_at: string;
}

// Global stats response
export interface GlobalStatsResponse {
  total_users: number;
  total_transactions: number;
  total_savings_usd: number;
  predictions_made: number;
  average_accuracy: number;
  active_chains: number;
}
