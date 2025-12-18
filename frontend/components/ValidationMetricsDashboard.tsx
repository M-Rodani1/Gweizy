import React, { useState, useEffect } from 'react';
import { Activity, TrendingUp, Target, CheckCircle, AlertTriangle } from 'lucide-react';

interface ValidationMetrics {
  mae: number;
  rmse: number;
  r2_score: number;
  directional_accuracy: number;
  total_predictions: number;
  validated_predictions: number;
}

interface ValidationSummary {
  '1h': ValidationMetrics;
  '4h': ValidationMetrics;
  '24h': ValidationMetrics;
  overall: ValidationMetrics;
}

interface ValidationHealth {
  status: 'healthy' | 'warning' | 'degraded';
  issues: string[];
  last_validation: string;
  predictions_pending: number;
}

interface ValidationTrends {
  dates: string[];
  mae_trend: number[];
  accuracy_trend: number[];
  horizon: string;
}

const ValidationMetricsDashboard: React.FC = () => {
  const [summary, setSummary] = useState<ValidationSummary | null>(null);
  const [health, setHealth] = useState<ValidationHealth | null>(null);
  const [trends, setTrends] = useState<ValidationTrends | null>(null);
  const [selectedHorizon, setSelectedHorizon] = useState<'1h' | '4h' | '24h'>('1h');
  const [trendPeriod, setTrendPeriod] = useState<number>(7);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isExpanded, setIsExpanded] = useState(true);

  const API_BASE = import.meta.env.VITE_API_URL || 'https://basegasfeesml.onrender.com';

  useEffect(() => {
    fetchValidationData();
    const interval = setInterval(fetchValidationData, 60000); // Refresh every minute
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    if (selectedHorizon) {
      fetchTrends(selectedHorizon, trendPeriod);
    }
  }, [selectedHorizon, trendPeriod]);

  const fetchValidationData = async () => {
    try {
      setLoading(true);
      setError(null);

      const [metricsRes, healthRes] = await Promise.all([
        fetch(`${API_BASE}/api/validation/metrics?horizon=all`),
        fetch(`${API_BASE}/api/validation/health`)
      ]);

      // Handle 503 (no data yet) gracefully
      if (metricsRes.status === 503 || healthRes.status === 503) {
        setError('No validation data available yet. System is still collecting initial data.');
        setLoading(false);
        return;
      }

      if (!metricsRes.ok || !healthRes.ok) {
        throw new Error('Failed to fetch validation data');
      }

      const metricsData = await metricsRes.json();
      const healthData = await healthRes.json();

      // Transform metrics API response (horizons) to component format
      if (metricsData.horizons) {
        const transformedSummary: ValidationSummary = {
          '1h': metricsData.horizons['1h'] || { mae: 0, rmse: 0, r2_score: 0, directional_accuracy: 0, total_predictions: 0, validated_predictions: 0 },
          '4h': metricsData.horizons['4h'] || { mae: 0, rmse: 0, r2_score: 0, directional_accuracy: 0, total_predictions: 0, validated_predictions: 0 },
          '24h': metricsData.horizons['24h'] || { mae: 0, rmse: 0, r2_score: 0, directional_accuracy: 0, total_predictions: 0, validated_predictions: 0 },
          overall: { mae: 0, rmse: 0, r2_score: 0, directional_accuracy: 0, total_predictions: 0, validated_predictions: 0 } as ValidationMetrics
        };
        setSummary(transformedSummary);
      }

      setHealth(healthData);
      setLoading(false);
    } catch (err) {
      console.error('Error fetching validation data:', err);
      setError(err instanceof Error ? err.message : 'Failed to load validation data');
      setLoading(false);
    }
  };

  const fetchTrends = async (horizon: string, days: number) => {
    try {
      const response = await fetch(`${API_BASE}/api/validation/trends?horizon=${horizon}&days=${days}`);
      if (!response.ok) {
        console.warn('Trends not available yet');
        return;
      }

      const data = await response.json();
      setTrends(data);
    } catch (err) {
      console.error('Error fetching trends:', err);
    }
  };

  const getHealthColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'text-green-400 bg-green-500/20';
      case 'warning': return 'text-yellow-400 bg-yellow-500/20';
      case 'degraded': return 'text-red-400 bg-red-500/20';
      default: return 'text-gray-400 bg-gray-500/20';
    }
  };

  const getHealthIcon = (status: string) => {
    switch (status) {
      case 'healthy': return <CheckCircle className="w-5 h-5" />;
      case 'warning': return <AlertTriangle className="w-5 h-5" />;
      case 'degraded': return <AlertTriangle className="w-5 h-5" />;
      default: return <Activity className="w-5 h-5" />;
    }
  };

  const formatPercentage = (value: number | undefined) => {
    if (value === undefined || value === null) return 'N/A';
    return `${(value * 100).toFixed(1)}%`;
  };
  const formatMetric = (value: number | undefined) => {
    if (value === undefined || value === null) return 'N/A';
    return value.toFixed(6);
  };

  if (loading && !summary) {
    return (
      <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
        <div className="flex items-center justify-center h-48">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-cyan-400"></div>
        </div>
      </div>
    );
  }

  if (error && !summary) {
    return (
      <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
        <div className="text-center text-red-400">
          <AlertTriangle className="w-8 h-8 mx-auto mb-2" />
          <p>{error}</p>
          <button
            onClick={fetchValidationData}
            className="mt-4 px-4 py-2 bg-cyan-500/20 hover:bg-cyan-500/30 text-cyan-400 rounded-lg transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl overflow-hidden">
      {/* Header */}
      <div
        className="p-4 border-b border-slate-700 cursor-pointer hover:bg-slate-700/30 transition-colors"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-cyan-500/20 rounded-lg">
              <Activity className="w-5 h-5 text-cyan-400" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-white">Prediction Validation</h3>
              <p className="text-sm text-gray-400">Real-time accuracy tracking</p>
            </div>
          </div>
          <div className="flex items-center gap-4">
            {health && (
              <div className={`flex items-center gap-2 px-3 py-1.5 rounded-lg ${getHealthColor(health.status)}`}>
                {getHealthIcon(health.status)}
                <span className="text-sm font-medium capitalize">{health.status}</span>
              </div>
            )}
            <svg
              className={`w-5 h-5 text-gray-400 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </div>
        </div>
      </div>

      {isExpanded && summary && (
        <div className="p-6 space-y-6">
          {/* Health Status */}
          {health && health.issues.length > 0 && (
            <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-4">
              <div className="flex items-start gap-3">
                <AlertTriangle className="w-5 h-5 text-yellow-400 mt-0.5" />
                <div className="flex-1">
                  <h4 className="text-sm font-medium text-yellow-400 mb-2">Performance Issues Detected</h4>
                  <ul className="space-y-1">
                    {health.issues.map((issue, idx) => (
                      <li key={idx} className="text-sm text-yellow-300/80">{issue}</li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          )}

          {/* Overall Metrics */}
          <div>
            <h4 className="text-sm font-medium text-gray-400 mb-3">Overall Performance</h4>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div className="bg-slate-700/50 rounded-lg p-4">
                <div className="flex items-center gap-2 mb-2">
                  <Target className="w-4 h-4 text-cyan-400" />
                  <p className="text-xs text-gray-400">Directional Accuracy</p>
                </div>
                <p className="text-2xl font-bold text-white">
                  {formatPercentage(summary.overall.directional_accuracy)}
                </p>
                <p className="text-xs text-gray-500 mt-1">
                  {summary.overall.validated_predictions} predictions
                </p>
              </div>

              <div className="bg-slate-700/50 rounded-lg p-4">
                <div className="flex items-center gap-2 mb-2">
                  <TrendingUp className="w-4 h-4 text-green-400" />
                  <p className="text-xs text-gray-400">R² Score</p>
                </div>
                <p className="text-2xl font-bold text-white">
                  {formatPercentage(summary.overall.r2_score)}
                </p>
                <p className="text-xs text-gray-500 mt-1">
                  Model fit quality
                </p>
              </div>

              <div className="bg-slate-700/50 rounded-lg p-4">
                <div className="flex items-center gap-2 mb-2">
                  <Activity className="w-4 h-4 text-blue-400" />
                  <p className="text-xs text-gray-400">MAE</p>
                </div>
                <p className="text-2xl font-bold text-white">
                  {formatMetric(summary.overall.mae)}
                </p>
                <p className="text-xs text-gray-500 mt-1">
                  Gwei average error
                </p>
              </div>

              <div className="bg-slate-700/50 rounded-lg p-4">
                <div className="flex items-center gap-2 mb-2">
                  <Activity className="w-4 h-4 text-purple-400" />
                  <p className="text-xs text-gray-400">RMSE</p>
                </div>
                <p className="text-2xl font-bold text-white">
                  {formatMetric(summary.overall.rmse)}
                </p>
                <p className="text-xs text-gray-500 mt-1">
                  Root mean squared error
                </p>
              </div>
            </div>
          </div>

          {/* Horizon Selector */}
          <div>
            <div className="flex items-center justify-between mb-3">
              <h4 className="text-sm font-medium text-gray-400">Performance by Horizon</h4>
              <div className="flex gap-2">
                {(['1h', '4h', '24h'] as const).map((horizon) => (
                  <button
                    key={horizon}
                    onClick={() => setSelectedHorizon(horizon)}
                    className={`px-3 py-1 rounded-lg text-xs font-medium transition-colors ${
                      selectedHorizon === horizon
                        ? 'bg-cyan-500 text-white'
                        : 'bg-slate-700/50 text-gray-400 hover:bg-slate-700'
                    }`}
                  >
                    {horizon}
                  </button>
                ))}
              </div>
            </div>

            {/* Selected Horizon Metrics */}
            <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
              <div className="bg-slate-700/30 rounded-lg p-3 border border-slate-600">
                <p className="text-xs text-gray-400 mb-1">Directional Accuracy</p>
                <p className="text-xl font-bold text-cyan-400">
                  {formatPercentage(summary[selectedHorizon].directional_accuracy)}
                </p>
              </div>
              <div className="bg-slate-700/30 rounded-lg p-3 border border-slate-600">
                <p className="text-xs text-gray-400 mb-1">R² Score</p>
                <p className="text-xl font-bold text-green-400">
                  {formatPercentage(summary[selectedHorizon].r2_score)}
                </p>
              </div>
              <div className="bg-slate-700/30 rounded-lg p-3 border border-slate-600">
                <p className="text-xs text-gray-400 mb-1">MAE</p>
                <p className="text-xl font-bold text-blue-400">
                  {formatMetric(summary[selectedHorizon].mae)}
                </p>
              </div>
              <div className="bg-slate-700/30 rounded-lg p-3 border border-slate-600">
                <p className="text-xs text-gray-400 mb-1">RMSE</p>
                <p className="text-xl font-bold text-purple-400">
                  {formatMetric(summary[selectedHorizon].rmse)}
                </p>
              </div>
              <div className="bg-slate-700/30 rounded-lg p-3 border border-slate-600">
                <p className="text-xs text-gray-400 mb-1">Validated</p>
                <p className="text-xl font-bold text-white">
                  {summary[selectedHorizon].validated_predictions}
                </p>
              </div>
            </div>
          </div>

          {/* Trends */}
          {trends && trends.dates.length > 0 && (
            <div>
              <div className="flex items-center justify-between mb-3">
                <h4 className="text-sm font-medium text-gray-400">Performance Trends</h4>
                <select
                  value={trendPeriod}
                  onChange={(e) => setTrendPeriod(Number(e.target.value))}
                  className="px-3 py-1 bg-slate-700 text-white text-xs rounded-lg border border-slate-600 focus:outline-none focus:border-cyan-500"
                >
                  <option value={7}>Last 7 days</option>
                  <option value={30}>Last 30 days</option>
                  <option value={90}>Last 90 days</option>
                </select>
              </div>

              <div className="bg-slate-700/30 rounded-lg p-4 border border-slate-600">
                <div className="space-y-4">
                  {/* MAE Trend */}
                  <div>
                    <p className="text-xs text-gray-400 mb-2">MAE Trend</p>
                    <div className="flex items-end gap-1 h-16">
                      {trends.mae_trend.map((value, idx) => {
                        const maxVal = Math.max(...trends.mae_trend);
                        const height = (value / maxVal) * 100;
                        return (
                          <div
                            key={idx}
                            className="flex-1 bg-blue-500/50 rounded-t hover:bg-blue-500/70 transition-colors"
                            style={{ height: `${height}%` }}
                            title={`${trends.dates[idx]}: ${formatMetric(value)}`}
                          />
                        );
                      })}
                    </div>
                  </div>

                  {/* Accuracy Trend */}
                  <div>
                    <p className="text-xs text-gray-400 mb-2">Directional Accuracy Trend</p>
                    <div className="flex items-end gap-1 h-16">
                      {trends.accuracy_trend.map((value, idx) => {
                        const height = value * 100;
                        return (
                          <div
                            key={idx}
                            className="flex-1 bg-cyan-500/50 rounded-t hover:bg-cyan-500/70 transition-colors"
                            style={{ height: `${height}%` }}
                            title={`${trends.dates[idx]}: ${formatPercentage(value)}`}
                          />
                        );
                      })}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Footer Stats */}
          {health && (
            <div className="flex items-center justify-between text-xs text-gray-400 pt-4 border-t border-slate-700">
              <div>
                Last validation: <span className="text-gray-300">{new Date(health.last_validation).toLocaleString()}</span>
              </div>
              <div>
                Pending: <span className="text-gray-300">{health.predictions_pending} predictions</span>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ValidationMetricsDashboard;
