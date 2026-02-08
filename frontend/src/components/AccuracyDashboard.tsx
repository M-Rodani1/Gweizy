import React, { useState, useEffect } from 'react';
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  ComposedChart, Area, AreaChart
} from 'recharts';
import { TrendingUp, TrendingDown, Activity } from 'lucide-react';
import { API_CONFIG, getApiUrl } from '../config/api';

interface MetricsData {
  horizon: string;
  mae_7d: number;
  mae_30d: number;
  mae_90d: number;
  success_rate_7d: number;
  success_rate_30d: number;
  success_rate_90d: number;
  predictions_count: number;
}

interface TrendData {
  date: string;
  mae: number;
  success_rate: number;
  predictions: number;
}

interface AccuracyDashboardProps {
  selectedChain?: string;
}

const AccuracyDashboard: React.FC<AccuracyDashboardProps> = ({ selectedChain = 'base' }) => {
  const [metrics, setMetrics] = useState<Record<string, MetricsData>>({});
  const [trends, setTrends] = useState<Record<string, TrendData[]>>({});
  const [loading, setLoading] = useState(true);
  const [selectedHorizon, setSelectedHorizon] = useState<'1h' | '4h' | '24h'>('1h');
  const [selectedPeriod, setSelectedPeriod] = useState<7 | 30 | 90>(7);

  useEffect(() => {
    fetchAccuracyData();
  }, [selectedChain, selectedHorizon]);

  const fetchAccuracyData = async () => {
    try {
      setLoading(true);

      // Fetch metrics for all horizons
      const metricsRes = await fetch(
        getApiUrl(`${API_CONFIG.ENDPOINTS.ANALYTICS}/performance?days=90`)
      );
      const metricsData = await metricsRes.json();

      if (metricsData.metrics) {
        // Transform metrics data
        const transformedMetrics: Record<string, MetricsData> = {};
        Object.entries(metricsData.metrics).forEach(([horizon, data]: [string, any]) => {
          transformedMetrics[horizon] = {
            horizon,
            mae_7d: data['7d']?.mae || 0,
            mae_30d: data['30d']?.mae || 0,
            mae_90d: data['90d']?.mae || 0,
            success_rate_7d: (data['7d']?.success_rate || 0) * 100,
            success_rate_30d: (data['30d']?.success_rate || 0) * 100,
            success_rate_90d: (data['90d']?.success_rate || 0) * 100,
            predictions_count: data['90d']?.count || 0
          };
        });
        setMetrics(transformedMetrics);
      }

      // Fetch trends
      const trendsRes = await fetch(
        getApiUrl(`${API_CONFIG.ENDPOINTS.ANALYTICS}/trends?horizon=${selectedHorizon}&days=90`)
      );
      const trendsData = await trendsRes.json();

      if (trendsData.trends) {
        setTrends({ [selectedHorizon]: trendsData.trends });
      }

      setLoading(false);
    } catch (error) {
      console.error('Failed to fetch accuracy data:', error);
      setLoading(false);
    }
  };

  const formatMetric = (value: number, decimals = 2) => {
    return value.toFixed(decimals);
  };

  const getPeriodKey = (days: number): 'mae_7d' | 'mae_30d' | 'mae_90d' => {
    return `mae_${days}d` as any;
  };

  const getSuccessRateKey = (days: number): 'success_rate_7d' | 'success_rate_30d' | 'success_rate_90d' => {
    return `success_rate_${days}d` as any;
  };

  if (loading) {
    return (
      <div className="bg-gray-800 rounded-lg p-6 mb-6 text-center">
        <div className="inline-block">
          <Activity className="w-8 h-8 text-cyan-400 animate-spin" />
        </div>
        <p className="text-gray-400 mt-2">Loading accuracy metrics...</p>
      </div>
    );
  }

  const currentMetrics = metrics[selectedHorizon];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-slate-800/50 to-slate-900/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6 shadow-xl">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h2 className="text-2xl font-bold text-white">Model Accuracy Dashboard</h2>
            <p className="text-sm text-gray-400">Historical performance tracking across time horizons</p>
          </div>
          <div className="flex gap-2">
            {(['1h', '4h', '24h'] as const).map((horizon) => (
              <button
                key={horizon}
                onClick={() => setSelectedHorizon(horizon)}
                className={`px-4 py-2 rounded-lg font-medium transition-colors ${
                  selectedHorizon === horizon
                    ? 'bg-cyan-500 text-white'
                    : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                }`}
              >
                {horizon}
              </button>
            ))}
          </div>
        </div>

        {/* Time Period Selector */}
        <div className="flex gap-2">
          {([7, 30, 90] as const).map((days) => (
            <button
              key={days}
              onClick={() => setSelectedPeriod(days)}
              className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
                selectedPeriod === days
                  ? 'bg-cyan-500/30 text-cyan-300 border border-cyan-400'
                  : 'bg-gray-700/50 text-gray-400 border border-gray-600'
              }`}
            >
              {days}d
            </button>
          ))}
        </div>
      </div>

      {/* Key Metrics Cards */}
      {currentMetrics && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* MAE Card */}
          <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-gray-400 text-sm font-medium">Mean Absolute Error</span>
              {selectedPeriod === 7 && currentMetrics.mae_7d < currentMetrics.mae_30d ? (
                <TrendingDown className="w-4 h-4 text-green-400" />
              ) : (
                <TrendingUp className="w-4 h-4 text-red-400" />
              )}
            </div>
            <div className="text-3xl font-bold text-white mb-1">
              {formatMetric(
                selectedPeriod === 7 ? currentMetrics.mae_7d :
                selectedPeriod === 30 ? currentMetrics.mae_30d :
                currentMetrics.mae_90d
              )}
            </div>
            <p className="text-xs text-gray-500">
              Gwei {selectedPeriod}d period
            </p>
          </div>

          {/* Success Rate Card */}
          <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-gray-400 text-sm font-medium">Success Rate</span>
              {selectedPeriod === 7 && currentMetrics.success_rate_7d > currentMetrics.success_rate_30d ? (
                <TrendingUp className="w-4 h-4 text-green-400" />
              ) : (
                <TrendingDown className="w-4 h-4 text-red-400" />
              )}
            </div>
            <div className="text-3xl font-bold text-white mb-1">
              {formatMetric(
                selectedPeriod === 7 ? currentMetrics.success_rate_7d :
                selectedPeriod === 30 ? currentMetrics.success_rate_30d :
                currentMetrics.success_rate_90d,
                1
              )}%
            </div>
            <p className="text-xs text-gray-500">
              Predictions beating baseline
            </p>
          </div>

          {/* Predictions Count Card */}
          <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
            <div className="flex items-center justify-between mb-2">
              <span className="text-gray-400 text-sm font-medium">Predictions</span>
              <Activity className="w-4 h-4 text-cyan-400" />
            </div>
            <div className="text-3xl font-bold text-white mb-1">
              {currentMetrics.predictions_count.toLocaleString()}
            </div>
            <p className="text-xs text-gray-500">
              Total validated {selectedHorizon}
            </p>
          </div>
        </div>
      )}

      {/* Trends Chart */}
      {trends[selectedHorizon] && trends[selectedHorizon].length > 0 && (
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white mb-4">MAE Trend</h3>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={trends[selectedHorizon]}>
              <defs>
                <linearGradient id="colorMae" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.8} />
                  <stop offset="95%" stopColor="#06b6d4" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis
                dataKey="date"
                stroke="#9ca3af"
                style={{ fontSize: '12px' }}
              />
              <YAxis stroke="#9ca3af" style={{ fontSize: '12px' }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1f2937',
                  border: '1px solid #374151',
                  borderRadius: '8px'
                }}
                labelStyle={{ color: '#9ca3af' }}
              />
              <Area
                type="monotone"
                dataKey="mae"
                stroke="#06b6d4"
                fillOpacity={1}
                fill="url(#colorMae)"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Success Rate Chart */}
      {trends[selectedHorizon] && trends[selectedHorizon].length > 0 && (
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white mb-4">Success Rate Trend</h3>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={trends[selectedHorizon]}>
              <defs>
                <linearGradient id="colorSuccess" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#10b981" stopOpacity={0.8} />
                  <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis
                dataKey="date"
                stroke="#9ca3af"
                style={{ fontSize: '12px' }}
              />
              <YAxis stroke="#9ca3af" style={{ fontSize: '12px' }} domain={[0, 100]} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1f2937',
                  border: '1px solid #374151',
                  borderRadius: '8px'
                }}
                labelStyle={{ color: '#9ca3af' }}
                formatter={(value: any) => `${(value * 100).toFixed(1)}%`}
              />
              <Area
                type="monotone"
                dataKey="success_rate"
                stroke="#10b981"
                fillOpacity={1}
                fill="url(#colorSuccess)"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Comparison Across Horizons */}
      {metrics && Object.keys(metrics).length > 0 && (
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white mb-4">MAE Comparison by Horizon</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart
              data={[
                {
                  horizon: '1h',
                  mae: metrics['1h']?.[getPeriodKey(selectedPeriod)] || 0
                },
                {
                  horizon: '4h',
                  mae: metrics['4h']?.[getPeriodKey(selectedPeriod)] || 0
                },
                {
                  horizon: '24h',
                  mae: metrics['24h']?.[getPeriodKey(selectedPeriod)] || 0
                }
              ]}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="horizon" stroke="#9ca3af" style={{ fontSize: '12px' }} />
              <YAxis stroke="#9ca3af" style={{ fontSize: '12px' }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1f2937',
                  border: '1px solid #374151',
                  borderRadius: '8px'
                }}
                labelStyle={{ color: '#9ca3af' }}
              />
              <Bar dataKey="mae" fill="#06b6d4" radius={[8, 8, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Success Rate by Horizon */}
      {metrics && Object.keys(metrics).length > 0 && (
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white mb-4">Success Rate by Horizon</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart
              data={[
                {
                  horizon: '1h',
                  successRate: metrics['1h']?.[getSuccessRateKey(selectedPeriod)] || 0
                },
                {
                  horizon: '4h',
                  successRate: metrics['4h']?.[getSuccessRateKey(selectedPeriod)] || 0
                },
                {
                  horizon: '24h',
                  successRate: metrics['24h']?.[getSuccessRateKey(selectedPeriod)] || 0
                }
              ]}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="horizon" stroke="#9ca3af" style={{ fontSize: '12px' }} />
              <YAxis stroke="#9ca3af" style={{ fontSize: '12px' }} domain={[0, 100]} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1f2937',
                  border: '1px solid #374151',
                  borderRadius: '8px'
                }}
                labelStyle={{ color: '#9ca3af' }}
                formatter={(value: any) => `${value.toFixed(1)}%`}
              />
              <Bar dataKey="successRate" fill="#10b981" radius={[8, 8, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Info Box */}
      <div className="bg-cyan-500/10 border border-cyan-500/30 rounded-lg p-4">
        <p className="text-sm text-cyan-300">
          <strong>💡 What these metrics mean:</strong>
          <br />
          • <strong>MAE:</strong> Average prediction error in gwei (lower is better)
          <br />
          • <strong>Success Rate:</strong> % of predictions that beat the baseline model
          <br />
          • <strong>Predictions:</strong> Number of validated predictions in this period
        </p>
      </div>
    </div>
  );
};

export default AccuracyDashboard;
