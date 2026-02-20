/**
 * ModelMetricsPanel - Unified Model Metrics Component
 * Consolidates AccuracyMetricsCard, ModelHealthCard, ValidationMetricsDashboard, and ModelAccuracy
 * into a single tabbed interface.
 */

import React, { useState, useEffect, useCallback, memo } from 'react';
import {
  Brain,
  RefreshCw,
  AlertTriangle,
  CheckCircle2,
  TrendingDown,
  TrendingUp,
  Activity,
  Target,
  Compass,
  BarChart3,
  ExternalLink,
  ChevronDown
} from 'lucide-react';
import { fetchAccuracyDrift, fetchAccuracyMetrics, fetchValidationTrends } from '../api/gasApi';
import { REFRESH_INTERVALS } from '../constants';
import { withTimeout } from '../utils/withTimeout';
import type {
  DriftResponse,
  MetricsResponse,
  ValidationTrends,
  MainTab,
  Horizon
} from '../types/modelMetrics';

interface ModelMetricsPanelProps {
  compact?: boolean;
  defaultTab?: MainTab;
}

const ModelMetricsPanel: React.FC<ModelMetricsPanelProps> = ({
  compact = false,
  defaultTab = 'overview'
}) => {
  // State
  const [activeTab, setActiveTab] = useState<MainTab>(defaultTab);
  const [selectedHorizon, setSelectedHorizon] = useState<Horizon>('1h');
  const [metricsData, setMetricsData] = useState<MetricsResponse | null>(null);
  const [driftData, setDriftData] = useState<DriftResponse | null>(null);
  const [trends, setTrends] = useState<ValidationTrends | null>(null);
  const [trendPeriod, setTrendPeriod] = useState<number>(7);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<string | null>(null);
  const [isExpanded, setIsExpanded] = useState(!compact);
  const REQUEST_TIMEOUT_MS = 12000;

  const fetchWithTimeout = useCallback(
    async <T,>(promise: Promise<T>, label: string): Promise<T> =>
      withTimeout(promise, REQUEST_TIMEOUT_MS, `Request timed out: ${label}`),
    []
  );

  // Fetch all data
  const fetchData = useCallback(async () => {
    try {
      setError(null);
      const [metricsReq, driftReq] = await Promise.allSettled([
        fetchWithTimeout(fetchAccuracyMetrics(), 'accuracy metrics'),
        fetchWithTimeout(fetchAccuracyDrift(), 'accuracy drift')
      ]);
      const metricsRes = metricsReq.status === 'fulfilled' ? metricsReq.value : null;
      const driftRes = driftReq.status === 'fulfilled' ? driftReq.value : null;

      if (metricsRes) {
        const data = metricsRes as unknown as MetricsResponse;
        if ((data as { success?: boolean }).success && (data as { metrics?: unknown }).metrics) {
          setMetricsData(data);
        }
      }

      if (driftRes) {
        setDriftData(driftRes as unknown as DriftResponse);
      }

      if (!metricsRes && !driftRes && !metricsData && !driftData) {
        throw new Error('Failed to load model metrics');
      }

      setLastUpdated(new Date().toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit'
      }));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load metrics');
    } finally {
      setLoading(false);
    }
  }, [driftData, fetchWithTimeout, metricsData]);

  // Fetch trends for selected horizon
  const fetchTrends = useCallback(async () => {
    try {
      const data = await fetchValidationTrends(selectedHorizon, trendPeriod);
      setTrends(data as unknown as ValidationTrends);
    } catch (err) {
      console.warn('Trends not available');
    }
  }, [selectedHorizon, trendPeriod]);

  useEffect(() => {
    void fetchData();
    const refreshIfVisible = () => {
      if (typeof document === 'undefined' || document.visibilityState === 'visible') {
        void fetchData();
      }
    };
    const interval = setInterval(refreshIfVisible, REFRESH_INTERVALS.API_HEALTH);
    const handleVisibilityChange = () => {
      if (document.visibilityState === 'visible') {
        void fetchData();
      }
    };
    document.addEventListener('visibilitychange', handleVisibilityChange);

    return () => {
      clearInterval(interval);
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, [fetchData]);

  useEffect(() => {
    if (activeTab === 'trends') {
      fetchTrends();
    }
  }, [activeTab, fetchTrends]);

  // Health status helpers
  const getHealthStatus = () => {
    if (!driftData) return 'unknown';
    if (driftData.should_retrain) return 'needs_attention';
    if (driftData.horizons_drifting?.length > 0) return 'warning';
    return 'healthy';
  };

  const healthStatus = getHealthStatus();

  const healthConfig = {
    healthy: {
      icon: <CheckCircle2 className="w-5 h-5 text-emerald-400" />,
      text: 'Models Healthy',
      bgClass: 'bg-emerald-500/10 border-emerald-500/30',
      textClass: 'text-emerald-400'
    },
    warning: {
      icon: <AlertTriangle className="w-5 h-5 text-amber-400" />,
      text: 'Minor Drift Detected',
      bgClass: 'bg-amber-500/10 border-amber-500/30',
      textClass: 'text-amber-400'
    },
    needs_attention: {
      icon: <AlertTriangle className="w-5 h-5 text-red-400" />,
      text: 'Retrain Recommended',
      bgClass: 'bg-red-500/10 border-red-500/30',
      textClass: 'text-red-400'
    },
    unknown: {
      icon: <Brain className="w-5 h-5 text-gray-400" />,
      text: 'Status Unknown',
      bgClass: 'bg-gray-500/10 border-gray-500/30',
      textClass: 'text-gray-400'
    }
  };

  const health = healthConfig[healthStatus];

  // Metric formatting helpers
  const formatPercent = (value: number | null | undefined): string => {
    if (value === null || value === undefined || isNaN(value)) return '--';
    return `${(value * 100).toFixed(1)}%`;
  };

  const formatMetric = (value: number | null): string => {
    if (value === null) return '--';
    if (value < 0.001) return value.toExponential(2);
    return value.toFixed(6);
  };

  const getR2Color = (r2: number | null): string => {
    if (r2 === null) return 'text-gray-400';
    if (r2 >= 0.7) return 'text-emerald-400';
    if (r2 >= 0.4) return 'text-amber-400';
    return 'text-red-400';
  };

  const getDAColor = (da: number | null): string => {
    if (da === null) return 'text-gray-400';
    if (da >= 0.6) return 'text-emerald-400';
    if (da >= 0.5) return 'text-amber-400';
    return 'text-red-400';
  };

  // Get current horizon metrics
  const currentMetrics = metricsData?.metrics?.[selectedHorizon];

  // Loading state
  if (loading && !metricsData && !driftData) {
    return (
      <div className="bg-gray-900/50 border border-gray-800 rounded-2xl p-6 shadow-xl">
        <div className="flex items-center gap-2 mb-4">
          <Brain className="w-5 h-5 text-cyan-400" />
          <h3 className="font-semibold text-white">Model Performance</h3>
        </div>
        <div className="flex items-center justify-center h-40">
          <RefreshCw className="w-6 h-6 text-gray-500 animate-spin" />
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gray-900/50 border border-gray-800 rounded-2xl shadow-xl overflow-hidden">
      {/* Header */}
      <div
        className={`p-4 border-b border-gray-800 ${compact ? 'cursor-pointer hover:bg-gray-800/30' : ''}`}
        onClick={compact ? () => setIsExpanded(!isExpanded) : undefined}
      >
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-cyan-500/20 rounded-lg">
              <Brain className="w-5 h-5 text-cyan-400" />
            </div>
            <div>
              <h3 className="font-semibold text-white">Model Performance</h3>
              <p className="text-xs text-gray-400">Accuracy tracking & health status</p>
            </div>
          </div>
          <div className="flex items-center gap-3">
            {/* Health Badge */}
            <div className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border ${health.bgClass}`}>
              {health.icon}
              <span className={`text-xs font-medium ${health.textClass}`}>{health.text}</span>
            </div>
            {/* Refresh Button */}
            <button
              onClick={(e) => { e.stopPropagation(); fetchData(); }}
              disabled={loading}
              className="text-xs text-gray-400 hover:text-white p-2 rounded-lg hover:bg-gray-800 transition-colors"
            >
              <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
            </button>
            {/* Expand/Collapse for compact mode */}
            {compact && (
              <ChevronDown className={`w-4 h-4 text-gray-400 transition-transform ${isExpanded ? 'rotate-180' : ''}`} />
            )}
          </div>
        </div>
      </div>

      {/* Content (collapsible in compact mode) */}
      {isExpanded && (
        <div className="p-4">
          {/* Main Tabs */}
          <div className="flex gap-1 mb-4 bg-gray-800/50 rounded-lg p-1">
            {(['overview', 'metrics', 'trends'] as MainTab[]).map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`flex-1 py-2 text-xs font-medium rounded-md transition-colors capitalize ${
                  activeTab === tab
                    ? 'bg-cyan-500 text-white'
                    : 'text-gray-400 hover:text-white hover:bg-gray-700/50'
                }`}
              >
                {tab}
              </button>
            ))}
          </div>

          {/* Overview Tab */}
          {activeTab === 'overview' && (
            <div className="space-y-4">
              {/* Health Status */}
              <div className={`rounded-xl p-4 border ${health.bgClass}`}>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    {health.icon}
                    <span className={`font-medium ${health.textClass}`}>{health.text}</span>
                  </div>
                  {driftData?.horizons_drifting && driftData.horizons_drifting.length > 0 && (
                    <span className="text-xs text-gray-400">
                      {driftData.horizons_drifting.length} horizon(s) drifting
                    </span>
                  )}
                </div>
              </div>

              {/* Drift Details */}
              {driftData?.horizons_drifting && driftData.horizons_drifting.length > 0 && (
                <div className="p-3 bg-amber-500/10 border border-amber-500/30 rounded-xl">
                  <div className="flex items-center gap-2 mb-2">
                    <TrendingDown className="w-4 h-4 text-amber-400" />
                    <span className="text-sm font-medium text-amber-400">Drift Detected</span>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    {driftData.horizons_drifting.map(horizon => {
                      const drift = driftData.drift[horizon];
                      return (
                        <span key={horizon} className="text-xs px-2 py-1 bg-amber-500/20 text-amber-300 rounded">
                          {horizon}: +{((drift?.drift_ratio || 0) * 100).toFixed(0)}% error
                        </span>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Quick Metrics Summary */}
              {metricsData?.metrics && (
                <div className="space-y-2">
                  <h4 className="text-xs font-medium text-gray-400 uppercase tracking-wider">Quick Summary</h4>
                  {Object.entries(metricsData.metrics).map(([horizon, metrics]) => (
                    <div key={horizon} className="flex items-center justify-between p-3 bg-gray-800/40 rounded-lg border border-gray-700/50">
                      <span className="text-sm text-gray-300 font-medium">{horizon}</span>
                      <div className="flex gap-4 text-xs">
                        <span className={getR2Color(metrics.r2)}>R²: {formatPercent(metrics.r2)}</span>
                        <span className={getDAColor(metrics.directional_accuracy)}>Dir: {formatPercent(metrics.directional_accuracy)}</span>
                        <span className="text-gray-400">{metrics.n} samples</span>
                      </div>
                    </div>
                  ))}
                </div>
              )}

              {/* Validation Latency Info */}
              <div className="p-4 bg-cyan-500/10 border border-cyan-500/30 rounded-xl">
                <div className="flex items-start gap-3">
                  <Activity className="w-5 h-5 text-cyan-400 mt-0.5" />
                  <div className="flex-1">
                    <p className="text-sm text-cyan-300 font-medium">About Metrics</p>
                    <p className="text-xs text-gray-400 mt-1">
                      Predictions are validated after their horizon passes (≈80% of time). 1h metrics update ~48 min later, 4h after ~3.2 hrs, 24h after ~19 hrs.
                    </p>
                  </div>
                </div>
              </div>

              {/* Colab Training Info */}
              <div className="p-4 bg-cyan-500/10 border border-cyan-500/30 rounded-xl">
                <div className="flex items-start gap-3">
                  <Activity className="w-5 h-5 text-cyan-400 mt-0.5" />
                  <div className="flex-1">
                    <p className="text-sm text-cyan-300 font-medium">Training via Colab</p>
                    <p className="text-xs text-gray-400 mt-1">
                      Model training is done using the Colab notebook for better GPU access.
                    </p>
                    <a
                      href="https://colab.research.google.com"
                      target="_blank"
                      rel="noopener noreferrer"
                      className="inline-flex items-center gap-1 text-xs text-cyan-400 hover:text-cyan-300 mt-2 transition-colors"
                    >
                      Open train_all_models.ipynb
                      <ExternalLink className="w-3 h-3" />
                    </a>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Metrics Tab */}
          {activeTab === 'metrics' && (
            <div className="space-y-4">
              {/* Horizon Tabs */}
              <div className="flex gap-1 bg-gray-800/30 rounded-lg p-1">
                {(['1h', '4h', '24h'] as Horizon[]).map((horizon) => (
                  <button
                    key={horizon}
                    onClick={() => setSelectedHorizon(horizon)}
                    className={`flex-1 py-2 text-xs font-medium rounded-md transition-colors ${
                      selectedHorizon === horizon
                        ? 'bg-cyan-500 text-white'
                        : 'text-gray-400 hover:text-white'
                    }`}
                  >
                    {horizon}
                  </button>
                ))}
              </div>

              {/* Detailed Metrics Grid */}
              {currentMetrics ? (
                <div className="grid grid-cols-2 gap-3">
                  {/* R² Score */}
                  <div className="bg-gray-800/40 rounded-xl p-4 border border-gray-700/50">
                    <div className="flex items-center gap-1.5 mb-2">
                      <Target className="w-4 h-4 text-cyan-400" />
                      <span className="text-xs text-gray-400">R² Score</span>
                    </div>
                    <div className={`text-2xl font-bold ${getR2Color(currentMetrics.r2)}`}>
                      {formatPercent(currentMetrics.r2)}
                    </div>
                    <div className="text-xs text-gray-500 mt-1">Variance explained</div>
                  </div>

                  {/* Directional Accuracy */}
                  <div className="bg-gray-800/40 rounded-xl p-4 border border-gray-700/50">
                    <div className="flex items-center gap-1.5 mb-2">
                      <Compass className="w-4 h-4 text-cyan-400" />
                      <span className="text-xs text-gray-400">Direction</span>
                    </div>
                    <div className={`text-2xl font-bold ${getDAColor(currentMetrics.directional_accuracy)}`}>
                      {formatPercent(currentMetrics.directional_accuracy)}
                    </div>
                    <div className="text-xs text-gray-500 mt-1">Trend prediction</div>
                  </div>

                  {/* MAE */}
                  <div className="bg-gray-800/40 rounded-xl p-4 border border-gray-700/50">
                    <div className="flex items-center gap-1.5 mb-2">
                      <TrendingUp className="w-4 h-4 text-amber-400" />
                      <span className="text-xs text-gray-400">MAE</span>
                    </div>
                    <div className="text-xl font-bold text-white font-mono">
                      {formatMetric(currentMetrics.mae)}
                    </div>
                    <div className="text-xs text-gray-500 mt-1">Mean absolute error</div>
                  </div>

                  {/* RMSE */}
                  <div className="bg-gray-800/40 rounded-xl p-4 border border-gray-700/50">
                    <div className="flex items-center gap-1.5 mb-2">
                      <Activity className="w-4 h-4 text-cyan-400" />
                      <span className="text-xs text-gray-400">RMSE</span>
                    </div>
                    <div className="text-xl font-bold text-white font-mono">
                      {formatMetric(currentMetrics.rmse)}
                    </div>
                    <div className="text-xs text-gray-500 mt-1">Root mean squared error</div>
                  </div>

                  {/* Sample Count - Full Width */}
                  <div className="col-span-2 bg-gray-800/40 rounded-xl p-4 border border-gray-700/50">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-1.5">
                        <BarChart3 className="w-4 h-4 text-emerald-400" />
                        <span className="text-xs text-gray-400">Validated Predictions</span>
                      </div>
                      <div className="text-xl font-bold text-white">{currentMetrics.n}</div>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="flex items-center justify-center py-8 text-gray-500">
                  <AlertTriangle className="w-4 h-4 mr-2" />
                  <span className="text-sm">No metrics available for {selectedHorizon}</span>
                </div>
              )}
            </div>
          )}

          {/* Trends Tab */}
          {activeTab === 'trends' && (
            <div className="space-y-4">
              {/* Period Selector */}
              <div className="flex items-center justify-between">
                <div className="flex gap-1 bg-gray-800/30 rounded-lg p-1">
                  {(['1h', '4h', '24h'] as Horizon[]).map((horizon) => (
                    <button
                      key={horizon}
                      onClick={() => setSelectedHorizon(horizon)}
                      className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors ${
                        selectedHorizon === horizon
                          ? 'bg-cyan-500 text-white'
                          : 'text-gray-400 hover:text-white'
                      }`}
                    >
                      {horizon}
                    </button>
                  ))}
                </div>
                <select
                  value={trendPeriod}
                  onChange={(e) => setTrendPeriod(Number(e.target.value))}
                  className="px-3 py-1.5 bg-gray-800 text-white text-xs rounded-lg border border-gray-700 focus:outline-none focus:border-cyan-500"
                >
                  <option value={7}>Last 7 days</option>
                  <option value={30}>Last 30 days</option>
                  <option value={90}>Last 90 days</option>
                </select>
              </div>

              {/* Trend Charts */}
              {trends && trends.dates?.length > 0 ? (
                <div className="space-y-4">
                  {/* MAE Trend */}
                  <div className="bg-gray-800/40 rounded-xl p-4 border border-gray-700/50">
                    <p className="text-xs text-gray-400 mb-3">MAE Trend (lower is better)</p>
                    <div className="flex items-end gap-1 h-20">
                      {trends.mae_trend.map((value, idx) => {
                        const maxVal = Math.max(...trends.mae_trend);
                        const height = maxVal > 0 ? (value / maxVal) * 100 : 0;
                        return (
                          <div
                            key={idx}
                            className="flex-1 bg-cyan-500/50 rounded-t hover:bg-cyan-500/70 transition-colors cursor-pointer"
                            style={{ height: `${height}%`, minHeight: '4px' }}
                            title={`${trends.dates[idx]}: ${formatMetric(value)}`}
                          />
                        );
                      })}
                    </div>
                  </div>

                  {/* Accuracy Trend */}
                  <div className="bg-gray-800/40 rounded-xl p-4 border border-gray-700/50">
                    <p className="text-xs text-gray-400 mb-3">Directional Accuracy Trend</p>
                    <div className="flex items-end gap-1 h-20">
                      {trends.accuracy_trend.map((value, idx) => {
                        const height = value * 100;
                        return (
                          <div
                            key={idx}
                            className="flex-1 bg-cyan-500/50 rounded-t hover:bg-cyan-500/70 transition-colors cursor-pointer"
                            style={{ height: `${height}%`, minHeight: '4px' }}
                            title={`${trends.dates[idx]}: ${formatPercent(value)}`}
                          />
                        );
                      })}
                    </div>
                  </div>
                </div>
              ) : (
                <div className="flex items-center justify-center py-12 text-gray-500">
                  <BarChart3 className="w-5 h-5 mr-2" />
                  <span className="text-sm">No trend data available yet</span>
                </div>
              )}
            </div>
          )}

          {/* Error Display */}
          {error && (
            <div className="mt-4 p-3 bg-red-500/10 border border-red-500/30 rounded-xl">
              <p className="text-sm text-red-400">{error}</p>
            </div>
          )}

          {/* Footer */}
          <div className="mt-4 pt-4 border-t border-gray-800/50 flex items-center justify-between text-xs text-gray-500">
            <span>Checked: {lastUpdated || '--'} (metrics may be delayed)</span>
            <span>Auto-refresh: 1 min</span>
          </div>
        </div>
      )}
    </div>
  );
};

export default memo(ModelMetricsPanel);
