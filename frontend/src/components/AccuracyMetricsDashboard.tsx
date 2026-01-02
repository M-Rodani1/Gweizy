/**
 * Real-Time Accuracy Metrics Dashboard
 * Displays live model accuracy metrics and prediction performance
 */

import React, { useEffect, useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { fetchAccuracyMetrics, fetchAccuracyHistory } from '../api/gasApi';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts';

interface AccuracyMetrics {
  '1h': {
    mae?: number;
    rmse?: number;
    r2?: number;
    directional_accuracy?: number;
    n?: number;
  };
  '4h': {
    mae?: number;
    rmse?: number;
    r2?: number;
    directional_accuracy?: number;
    n?: number;
  };
  '24h': {
    mae?: number;
    rmse?: number;
    r2?: number;
    directional_accuracy?: number;
    n?: number;
  };
}

interface AccuracyHistoryPoint {
  timestamp: string;
  mae: number;
  rmse: number;
  r2: number;
  directional_accuracy: number;
  n: number;
}

const AccuracyMetricsDashboard: React.FC = () => {
  const [selectedHorizon, setSelectedHorizon] = useState<'1h' | '4h' | '24h'>('1h');

  // Fetch current metrics
  const { data: metrics, isLoading: metricsLoading } = useQuery<AccuracyMetrics>({
    queryKey: ['accuracyMetrics'],
    queryFn: async () => {
      const response = await fetch('https://basegasfeesml-production.up.railway.app/api/accuracy/metrics');
      if (!response.ok) throw new Error('Failed to fetch metrics');
      const data = await response.json();
      return data.metrics || {};
    },
    refetchInterval: 60000, // Refresh every minute
  });

  // Fetch historical accuracy
  const { data: history, isLoading: historyLoading } = useQuery<Record<string, AccuracyHistoryPoint[]>>({
    queryKey: ['accuracyHistory', selectedHorizon],
    queryFn: async () => {
      const response = await fetch(
        `https://basegasfeesml-production.up.railway.app/api/accuracy/history?hours_back=168&resolution=hourly`
      );
      if (!response.ok) throw new Error('Failed to fetch history');
      const data = await response.json();
      return data.history || {};
    },
    refetchInterval: 300000, // Refresh every 5 minutes
  });

  const currentMetrics = metrics?.[selectedHorizon];
  const historyData = history?.[selectedHorizon] || [];

  const formatMetric = (value: number | undefined, decimals: number = 4): string => {
    if (value === undefined || value === null) return 'N/A';
    return value.toFixed(decimals);
  };

  const formatPercent = (value: number | undefined): string => {
    if (value === undefined || value === null) return 'N/A';
    return `${(value * 100).toFixed(2)}%`;
  };

  const getR2Color = (r2: number | undefined): string => {
    if (r2 === undefined || r2 === null) return 'text-gray-500';
    if (r2 > 0.7) return 'text-green-400';
    if (r2 > 0.4) return 'text-yellow-400';
    return 'text-red-400';
  };

  const getDirectionalAccuracyColor = (da: number | undefined): string => {
    if (da === undefined || da === null) return 'text-gray-500';
    if (da > 0.7) return 'text-green-400';
    if (da > 0.6) return 'text-yellow-400';
    return 'text-red-400';
  };

  if (metricsLoading) {
    return (
      <div className="bg-slate-800 rounded-lg p-6 border border-slate-700">
        <div className="animate-pulse space-y-4">
          <div className="h-6 bg-slate-700 rounded w-1/3"></div>
          <div className="h-32 bg-slate-700 rounded"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-slate-800 rounded-lg p-6 border border-slate-700 space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold text-white">Model Accuracy Metrics</h2>
        <div className="flex gap-2">
          {(['1h', '4h', '24h'] as const).map((horizon) => (
            <button
              key={horizon}
              onClick={() => setSelectedHorizon(horizon)}
              className={`px-3 py-1 rounded-lg text-sm font-medium transition-colors ${
                selectedHorizon === horizon
                  ? 'bg-cyan-500 text-white'
                  : 'bg-slate-700 text-gray-300 hover:bg-slate-600'
              }`}
            >
              {horizon}
            </button>
          ))}
        </div>
      </div>

      {/* Current Metrics Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-slate-900 rounded-lg p-4 border border-slate-700">
          <div className="text-sm text-gray-400 mb-1">R² Score</div>
          <div className={`text-2xl font-bold ${getR2Color(currentMetrics?.r2)}`}>
            {formatPercent(currentMetrics?.r2)}
          </div>
          <div className="text-xs text-gray-500 mt-1">
            {currentMetrics?.n ? `${currentMetrics.n} predictions` : 'No data'}
          </div>
        </div>

        <div className="bg-slate-900 rounded-lg p-4 border border-slate-700">
          <div className="text-sm text-gray-400 mb-1">Directional Accuracy</div>
          <div className={`text-2xl font-bold ${getDirectionalAccuracyColor(currentMetrics?.directional_accuracy)}`}>
            {formatPercent(currentMetrics?.directional_accuracy)}
          </div>
          <div className="text-xs text-gray-500 mt-1">Direction prediction</div>
        </div>

        <div className="bg-slate-900 rounded-lg p-4 border border-slate-700">
          <div className="text-sm text-gray-400 mb-1">MAE</div>
          <div className="text-2xl font-bold text-cyan-400">
            {formatMetric(currentMetrics?.mae)}
          </div>
          <div className="text-xs text-gray-500 mt-1">Mean Absolute Error</div>
        </div>

        <div className="bg-slate-900 rounded-lg p-4 border border-slate-700">
          <div className="text-sm text-gray-400 mb-1">RMSE</div>
          <div className="text-2xl font-bold text-cyan-400">
            {formatMetric(currentMetrics?.rmse)}
          </div>
          <div className="text-xs text-gray-500 mt-1">Root Mean Squared</div>
        </div>
      </div>

      {/* Historical Accuracy Chart */}
      {historyData.length > 0 && (
        <div className="bg-slate-900 rounded-lg p-4 border border-slate-700">
          <h3 className="text-lg font-semibold text-white mb-4">Accuracy Over Time (Last 7 Days)</h3>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={historyData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis
                dataKey="timestamp"
                stroke="#9CA3AF"
                tickFormatter={(value) => {
                  const date = new Date(value);
                  return `${date.getMonth() + 1}/${date.getDate()} ${date.getHours()}:00`;
                }}
              />
              <YAxis stroke="#9CA3AF" />
              <Tooltip
                contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151', borderRadius: '8px' }}
                labelStyle={{ color: '#F3F4F6' }}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="r2"
                name="R² Score"
                stroke="#06B6D4"
                strokeWidth={2}
                dot={{ r: 3 }}
              />
              <Line
                type="monotone"
                dataKey="directional_accuracy"
                name="Directional Accuracy"
                stroke="#10B981"
                strokeWidth={2}
                dot={{ r: 3 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Error Distribution */}
      {historyData.length > 0 && (
        <div className="bg-slate-900 rounded-lg p-4 border border-slate-700">
          <h3 className="text-lg font-semibold text-white mb-4">Error Distribution</h3>
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={historyData.slice(-24)}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis
                dataKey="timestamp"
                stroke="#9CA3AF"
                tickFormatter={(value) => {
                  const date = new Date(value);
                  return `${date.getHours()}:00`;
                }}
              />
              <YAxis stroke="#9CA3AF" />
              <Tooltip
                contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151', borderRadius: '8px' }}
                labelStyle={{ color: '#F3F4F6' }}
              />
              <Bar dataKey="mae" name="MAE" fill="#06B6D4" />
              <Bar dataKey="rmse" name="RMSE" fill="#8B5CF6" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {!currentMetrics && (
        <div className="text-center py-8 text-gray-400">
          <p>No accuracy data available yet. Metrics will appear as predictions are validated.</p>
        </div>
      )}
    </div>
  );
};

export default AccuracyMetricsDashboard;

