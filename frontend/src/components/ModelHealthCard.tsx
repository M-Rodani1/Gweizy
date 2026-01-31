/**
 * Model Health Card Component
 * Displays ML model health status and accuracy metrics
 * Training is done via Colab notebook - this is read-only status display
 */

import React, { useState, useEffect, useCallback, memo } from 'react';
import {
  Brain,
  RefreshCw,
  AlertTriangle,
  CheckCircle2,
  TrendingDown,
  Activity,
  ExternalLink,
  Database
} from 'lucide-react';
import { API_CONFIG, getApiUrl } from '../config/api';

interface DriftInfo {
  is_drifting: boolean;
  drift_ratio: number;
  mae_current: number;
  mae_baseline: number;
}

interface DriftResponse {
  success: boolean;
  should_retrain: boolean;
  drift: Record<string, DriftInfo>;
  horizons_drifting: string[];
}

interface AccuracyMetrics {
  mae: number;
  rmse: number;
  r2: number;
  direction_accuracy: number;
  predictions_count: number;
}

interface MetricsResponse {
  success: boolean;
  metrics: Record<string, AccuracyMetrics>;
}

const ModelHealthCard: React.FC = () => {
  const [driftData, setDriftData] = useState<DriftResponse | null>(null);
  const [metricsData, setMetricsData] = useState<MetricsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    try {
      setError(null);
      const [driftRes, metricsRes] = await Promise.all([
        fetch(getApiUrl(API_CONFIG.ENDPOINTS.ACCURACY_DRIFT)),
        fetch(getApiUrl(API_CONFIG.ENDPOINTS.ACCURACY_METRICS))
      ]);

      if (driftRes.ok) {
        const drift = await driftRes.json();
        setDriftData(drift);
      }

      if (metricsRes.ok) {
        const metrics = await metricsRes.json();
        setMetricsData(metrics);
      }

      setLastUpdated(new Date().toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit'
      }));
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load model health');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 60000); // Refresh every minute
    return () => clearInterval(interval);
  }, [fetchData]);

  const getOverallHealth = () => {
    if (!driftData) return 'unknown';
    if (driftData.should_retrain) return 'needs_attention';
    if (driftData.horizons_drifting?.length > 0) return 'warning';
    return 'healthy';
  };

  const health = getOverallHealth();

  const getHealthDisplay = () => {
    switch (health) {
      case 'healthy':
        return {
          icon: <CheckCircle2 className="w-5 h-5 text-emerald-400" />,
          text: 'Models Healthy',
          bgColor: 'bg-emerald-500/10 border-emerald-500/30',
          textColor: 'text-emerald-400'
        };
      case 'warning':
        return {
          icon: <AlertTriangle className="w-5 h-5 text-amber-400" />,
          text: 'Minor Drift Detected',
          bgColor: 'bg-amber-500/10 border-amber-500/30',
          textColor: 'text-amber-400'
        };
      case 'needs_attention':
        return {
          icon: <AlertTriangle className="w-5 h-5 text-red-400" />,
          text: 'Retrain Recommended',
          bgColor: 'bg-red-500/10 border-red-500/30',
          textColor: 'text-red-400'
        };
      default:
        return {
          icon: <Brain className="w-5 h-5 text-gray-400" />,
          text: 'Status Unknown',
          bgColor: 'bg-gray-500/10 border-gray-500/30',
          textColor: 'text-gray-400'
        };
    }
  };

  const healthDisplay = getHealthDisplay();

  if (loading && !driftData && !metricsData) {
    return (
      <div className="bg-gray-900/50 border border-gray-800 rounded-2xl p-6 shadow-xl h-full">
        <div className="flex items-center gap-2 mb-4">
          <Brain className="w-5 h-5 text-purple-400" />
          <h3 className="font-semibold text-white">Model Health</h3>
        </div>
        <div className="flex items-center justify-center h-40">
          <RefreshCw className="w-6 h-6 text-gray-500 animate-spin" />
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gray-900/50 border border-gray-800 rounded-2xl p-6 shadow-xl h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Brain className="w-5 h-5 text-purple-400" />
          <h3 className="font-semibold text-white">Model Health</h3>
        </div>
        <button
          onClick={fetchData}
          disabled={loading}
          className="text-xs text-gray-400 hover:text-white px-2 py-1 rounded-lg hover:bg-gray-800 transition-colors flex items-center gap-1"
        >
          <RefreshCw className={`w-3 h-3 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {/* Overall Status */}
      <div className={`rounded-xl p-4 mb-4 border ${healthDisplay.bgColor}`}>
        <div className="flex items-center gap-2">
          {healthDisplay.icon}
          <span className={`font-medium ${healthDisplay.textColor}`}>
            {healthDisplay.text}
          </span>
        </div>
      </div>

      {/* Accuracy Metrics */}
      {metricsData?.metrics && Object.keys(metricsData.metrics).length > 0 && (
        <div className="space-y-3 flex-1">
          <h4 className="text-xs font-medium text-gray-400 uppercase tracking-wider">
            Current Accuracy
          </h4>

          {Object.entries(metricsData.metrics).map(([horizon, metrics]) => (
            <div key={horizon} className="bg-gray-800/40 rounded-xl p-3 border border-gray-700/50">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm text-gray-300 font-medium">{horizon} Horizon</span>
                <span className="text-xs text-gray-500">
                  {metrics.predictions_count?.toLocaleString() || 0} predictions
                </span>
              </div>
              <div className="grid grid-cols-3 gap-2">
                <div className="text-center">
                  <p className="text-xs text-gray-500">MAE</p>
                  <p className="text-sm font-mono text-cyan-400">
                    {metrics.mae?.toFixed(4) || 'N/A'}
                  </p>
                </div>
                <div className="text-center">
                  <p className="text-xs text-gray-500">R²</p>
                  <p className="text-sm font-mono text-purple-400">
                    {metrics.r2 !== undefined ? `${(metrics.r2 * 100).toFixed(1)}%` : 'N/A'}
                  </p>
                </div>
                <div className="text-center">
                  <p className="text-xs text-gray-500">Direction</p>
                  <p className="text-sm font-mono text-emerald-400">
                    {metrics.direction_accuracy !== undefined
                      ? `${(metrics.direction_accuracy * 100).toFixed(1)}%`
                      : 'N/A'}
                  </p>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Drift Details */}
      {driftData?.horizons_drifting && driftData.horizons_drifting.length > 0 && (
        <div className="mt-4 p-3 bg-amber-500/10 border border-amber-500/30 rounded-xl">
          <div className="flex items-center gap-2 mb-2">
            <TrendingDown className="w-4 h-4 text-amber-400" />
            <span className="text-sm font-medium text-amber-400">Drift Detected</span>
          </div>
          <div className="flex flex-wrap gap-2">
            {driftData.horizons_drifting.map(horizon => {
              const drift = driftData.drift[horizon];
              return (
                <span
                  key={horizon}
                  className="text-xs px-2 py-1 bg-amber-500/20 text-amber-300 rounded"
                >
                  {horizon}: +{((drift?.drift_ratio || 0) * 100).toFixed(0)}% error
                </span>
              );
            })}
          </div>
        </div>
      )}

      {/* Training Info */}
      <div className="mt-4 p-4 bg-purple-500/10 border border-purple-500/30 rounded-xl">
        <div className="flex items-start gap-3">
          <Database className="w-5 h-5 text-purple-400 mt-0.5" />
          <div className="flex-1">
            <p className="text-sm text-purple-300 font-medium">Training via Colab</p>
            <p className="text-xs text-gray-400 mt-1">
              Model training is done using the Colab notebook for better GPU access and control.
            </p>
            <a
              href="https://colab.research.google.com"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1 text-xs text-purple-400 hover:text-purple-300 mt-2 transition-colors"
            >
              Open notebooks/train_all_models.ipynb
              <ExternalLink className="w-3 h-3" />
            </a>
          </div>
        </div>
      </div>

      {/* Error Message */}
      {error && (
        <div className="mt-4 p-3 bg-red-500/10 border border-red-500/30 rounded-xl">
          <p className="text-sm text-red-400">{error}</p>
        </div>
      )}

      {/* Footer */}
      <div className="mt-auto pt-4 border-t border-gray-800/50">
        <div className="text-xs text-gray-500">
          Last check: {lastUpdated || '...'}
        </div>
      </div>
    </div>
  );
};

export default memo(ModelHealthCard);
