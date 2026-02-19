/**
 * MSW Request Handlers
 *
 * Mock API responses for testing. These handlers intercept fetch requests
 * and return mock data, enabling more realistic integration tests.
 */

import { http, HttpResponse } from 'msw';

const API_BASE = '/api';

// Mock data
const mockCurrentGas = {
  timestamp: new Date().toISOString(),
  current_gas: 0.00125,
  base_fee: 0.001,
  priority_fee: 0.00025,
  block_number: 12345678,
};

const mockPredictions = {
  current: mockCurrentGas,
  predictions: {
    '1h': [
      {
        time: '1h',
        predictedGwei: 0.0015,
        confidence: 0.85,
        lowerBound: 0.001,
        upperBound: 0.002,
        confidenceLevel: 'high',
      },
    ],
    '4h': [
      {
        time: '4h',
        predictedGwei: 0.0018,
        confidence: 0.75,
        lowerBound: 0.0012,
        upperBound: 0.0024,
        confidenceLevel: 'medium',
      },
    ],
    '24h': [
      {
        time: '24h',
        predictedGwei: 0.002,
        confidence: 0.65,
        lowerBound: 0.0015,
        upperBound: 0.0025,
        confidenceLevel: 'medium',
      },
    ],
    historical: [],
  },
  model_info: {
    '1h': { name: 'GradientBoosting', mae: 0.0001 },
    '4h': { name: 'GradientBoosting', mae: 0.0002 },
    '24h': { name: 'GradientBoosting', mae: 0.0003 },
  },
};

const mockHybridPrediction = {
  action: 'WAIT',
  confidence: 0.85,
  trend_signal_4h: -0.2,
  probabilities: {
    wait: 0.6,
    normal: 0.3,
    urgent: 0.1,
  },
};

const mockAccuracy = {
  horizons: {
    '1h': { mae: 0.0001, rmse: 0.00015, r2: 0.92, directional_accuracy: 0.85, n_samples: 1000 },
    '4h': { mae: 0.0002, rmse: 0.00025, r2: 0.88, directional_accuracy: 0.8, n_samples: 800 },
    '24h': { mae: 0.0003, rmse: 0.00035, r2: 0.82, directional_accuracy: 0.75, n_samples: 500 },
  },
  overall: {
    mae: 0.0002,
    rmse: 0.00025,
    r2: 0.87,
  },
  updated_at: new Date().toISOString(),
};

const mockConfig = {
  chains: [
    { id: 8453, name: 'Base', enabled: true },
    { id: 1, name: 'Ethereum', enabled: true },
    { id: 42161, name: 'Arbitrum', enabled: true },
  ],
  features: {
    predictions: true,
    multichain: true,
    scheduler: true,
    alerts: true,
  },
  version: '1.0.0',
};

const mockGlobalStats = {
  total_users: 15234,
  total_transactions: 523456,
  total_savings_usd: 1250000,
  predictions_made: 2500000,
  average_accuracy: 0.87,
  active_chains: 5,
};

const mockLeaderboard = {
  entries: [
    { rank: 1, address: '0x1234...abcd', display_name: 'GasSaver1', total_savings: 5000, transaction_count: 150 },
    { rank: 2, address: '0x5678...efgh', display_name: 'OptimizeKing', total_savings: 4500, transaction_count: 120 },
    { rank: 3, address: '0x9abc...ijkl', display_name: 'FeeMinimizer', total_savings: 4000, transaction_count: 100 },
  ],
  updated_at: new Date().toISOString(),
};

const mockHistorical = {
  data: Array.from({ length: 168 }, (_, i) => ({
    timestamp: new Date(Date.now() - i * 3600000).toISOString(),
    gwei: 0.001 + Math.random() * 0.002,
    block_number: 12345678 - i * 1800,
  })),
};

const mockTransactions = {
  transactions: [
    { hash: '0xabc123', timestamp: new Date().toISOString(), gas_used: 21000, gas_price: 0.001 },
    { hash: '0xdef456', timestamp: new Date().toISOString(), gas_used: 65000, gas_price: 0.0012 },
    { hash: '0xghi789', timestamp: new Date().toISOString(), gas_used: 150000, gas_price: 0.0015 },
  ],
};

// Handlers
export const handlers = [
  // Health check
  http.get(`${API_BASE}/health`, () => {
    return HttpResponse.json({ status: 'ok' });
  }),

  // Current gas
  http.get(`${API_BASE}/current`, () => {
    return HttpResponse.json(mockCurrentGas);
  }),

  // Predictions
  http.get(`${API_BASE}/predictions`, () => {
    return HttpResponse.json(mockPredictions);
  }),

  // Hybrid prediction
  http.get(`${API_BASE}/predictions/hybrid`, () => {
    return HttpResponse.json(mockHybridPrediction);
  }),

  // Historical data
  http.get(`${API_BASE}/historical`, () => {
    return HttpResponse.json(mockHistorical);
  }),

  // Transactions
  http.get(`${API_BASE}/transactions`, () => {
    return HttpResponse.json(mockTransactions);
  }),

  // Config
  http.get(`${API_BASE}/config`, () => {
    return HttpResponse.json(mockConfig);
  }),

  // Accuracy
  http.get(`${API_BASE}/accuracy`, () => {
    return HttpResponse.json(mockAccuracy);
  }),

  // Stats
  http.get(`${API_BASE}/stats`, () => {
    return HttpResponse.json(mockGlobalStats);
  }),

  // Leaderboard
  http.get(`${API_BASE}/leaderboard`, () => {
    return HttpResponse.json(mockLeaderboard);
  }),

  // User history (dynamic)
  http.get(`${API_BASE}/user-history/:address`, ({ params }) => {
    const { address } = params;
    return HttpResponse.json({
      address,
      transactions: mockTransactions.transactions,
      total_savings: 150,
      total_transactions: 25,
    });
  }),
];

// Error handlers for testing error scenarios
export const errorHandlers = [
  http.get(`${API_BASE}/current`, () => {
    return HttpResponse.json({ error: 'Server error' }, { status: 500 });
  }),

  http.get(`${API_BASE}/predictions`, () => {
    return HttpResponse.json({ error: 'Server error' }, { status: 500 });
  }),
];

// Delay handlers for testing loading states
export const delayHandlers = [
  http.get(`${API_BASE}/current`, async () => {
    await new Promise((resolve) => setTimeout(resolve, 2000));
    return HttpResponse.json(mockCurrentGas);
  }),
];

// Export mock data for assertions
export const mockData = {
  currentGas: mockCurrentGas,
  predictions: mockPredictions,
  hybridPrediction: mockHybridPrediction,
  accuracy: mockAccuracy,
  config: mockConfig,
  globalStats: mockGlobalStats,
  leaderboard: mockLeaderboard,
  historical: mockHistorical,
  transactions: mockTransactions,
};
