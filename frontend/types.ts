import React from 'react';

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
