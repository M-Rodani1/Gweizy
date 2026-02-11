import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Area,
  AreaChart,
} from 'recharts';
import type { MetricsData, TrendData } from './AccuracyDashboard';

type PeriodKey = 'mae_7d' | 'mae_30d' | 'mae_90d';
type SuccessRateKey = 'success_rate_7d' | 'success_rate_30d' | 'success_rate_90d';

type AccuracyDashboardChartsProps = {
  metrics: Record<string, MetricsData>;
  trends: Record<string, TrendData[]>;
  selectedHorizon: '1h' | '4h' | '24h';
  selectedPeriod: 7 | 30 | 90;
  getPeriodKey: (days: number) => PeriodKey;
  getSuccessRateKey: (days: number) => SuccessRateKey;
};

const AccuracyDashboardCharts: React.FC<AccuracyDashboardChartsProps> = ({
  metrics,
  trends,
  selectedHorizon,
  selectedPeriod,
  getPeriodKey,
  getSuccessRateKey,
}) => {
  const selectedTrends = trends[selectedHorizon];
  const hasTrends = selectedTrends && selectedTrends.length > 0;
  const hasMetrics = metrics && Object.keys(metrics).length > 0;

  return (
    <>
      {/* Trends Chart */}
      {hasTrends && (
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white mb-4">MAE Trend</h3>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={selectedTrends}>
              <defs>
                <linearGradient id="colorMae" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.8} />
                  <stop offset="95%" stopColor="#06b6d4" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="date" stroke="#9ca3af" style={{ fontSize: '12px' }} />
              <YAxis stroke="#9ca3af" style={{ fontSize: '12px' }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1f2937',
                  border: '1px solid #374151',
                  borderRadius: '8px',
                }}
                labelStyle={{ color: '#9ca3af' }}
              />
              <Area type="monotone" dataKey="mae" stroke="#06b6d4" fillOpacity={1} fill="url(#colorMae)" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Success Rate Chart */}
      {hasTrends && (
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white mb-4">Success Rate Trend</h3>
          <ResponsiveContainer width="100%" height={300}>
            <AreaChart data={selectedTrends}>
              <defs>
                <linearGradient id="colorSuccess" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#10b981" stopOpacity={0.8} />
                  <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="date" stroke="#9ca3af" style={{ fontSize: '12px' }} />
              <YAxis stroke="#9ca3af" style={{ fontSize: '12px' }} domain={[0, 100]} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1f2937',
                  border: '1px solid #374151',
                  borderRadius: '8px',
                }}
                labelStyle={{ color: '#9ca3af' }}
                formatter={(value: any) => `${(value * 100).toFixed(1)}%`}
              />
              <Area type="monotone" dataKey="success_rate" stroke="#10b981" fillOpacity={1} fill="url(#colorSuccess)" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Comparison Across Horizons */}
      {hasMetrics && (
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white mb-4">MAE Comparison by Horizon</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart
              data={[
                { horizon: '1h', mae: metrics['1h']?.[getPeriodKey(selectedPeriod)] || 0 },
                { horizon: '4h', mae: metrics['4h']?.[getPeriodKey(selectedPeriod)] || 0 },
                { horizon: '24h', mae: metrics['24h']?.[getPeriodKey(selectedPeriod)] || 0 },
              ]}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="horizon" stroke="#9ca3af" style={{ fontSize: '12px' }} />
              <YAxis stroke="#9ca3af" style={{ fontSize: '12px' }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1f2937',
                  border: '1px solid #374151',
                  borderRadius: '8px',
                }}
                labelStyle={{ color: '#9ca3af' }}
              />
              <Bar dataKey="mae" fill="#06b6d4" radius={[8, 8, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Success Rate by Horizon */}
      {hasMetrics && (
        <div className="bg-gray-800 border border-gray-700 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white mb-4">Success Rate by Horizon</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart
              data={[
                { horizon: '1h', successRate: metrics['1h']?.[getSuccessRateKey(selectedPeriod)] || 0 },
                { horizon: '4h', successRate: metrics['4h']?.[getSuccessRateKey(selectedPeriod)] || 0 },
                { horizon: '24h', successRate: metrics['24h']?.[getSuccessRateKey(selectedPeriod)] || 0 },
              ]}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="horizon" stroke="#9ca3af" style={{ fontSize: '12px' }} />
              <YAxis stroke="#9ca3af" style={{ fontSize: '12px' }} domain={[0, 100]} />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#1f2937',
                  border: '1px solid #374151',
                  borderRadius: '8px',
                }}
                labelStyle={{ color: '#9ca3af' }}
                formatter={(value: any) => `${value.toFixed(1)}%`}
              />
              <Bar dataKey="successRate" fill="#10b981" radius={[8, 8, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </>
  );
};

export default AccuracyDashboardCharts;
