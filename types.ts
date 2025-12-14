import React from 'react';

// Graph data point
export interface GraphDataPoint {
  time: string;
  gwei?: number;
  predictedGwei?: number;
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
