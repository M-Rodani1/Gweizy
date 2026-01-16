/**
 * AdvancedAnalyticsPanel Component
 *
 * A comprehensive dashboard panel that displays four key analytics metrics
 * for gas price monitoring and prediction quality assessment.
 *
 * @component
 * @example
 * ```tsx
 * <AdvancedAnalyticsPanel />
 * ```
 *
 * ## Features
 *
 * ### Gas Volatility Index (GVI)
 * VIX-style indicator (0-100) showing current gas price stability.
 * - 0-20: Very Low (green) - stable prices
 * - 20-40: Low (green) - relatively stable
 * - 40-60: Moderate (yellow) - normal fluctuations
 * - 60-80: High (orange) - volatile, time carefully
 * - 80-100: Extreme (red) - high uncertainty
 *
 * ### Whale Activity Monitor
 * Tracks large transactions (>500k gas) that may impact prices.
 * Shows current whale count, activity level, and estimated price impact.
 *
 * ### Anomaly Detection
 * Z-score based analysis identifying unusual price movements:
 * - Spikes: Sudden price increases
 * - Drops: Sudden price decreases
 * - Volatility anomalies: Unusual variance patterns
 *
 * ### Model Ensemble Status
 * Shows health of the ML prediction system:
 * - Which models are loaded and active
 * - Overall health percentage
 * - Current prediction mode (ML vs fallback)
 *
 * ## Data Refresh
 * - Auto-refreshes every 60 seconds
 * - Manual refresh via button
 * - 15 second fetch timeout
 *
 * @see {@link https://docs.gweizy.com/analytics} for API documentation
 */

import React, { useEffect, useState, memo } from 'react';
import {
  Activity,
  AlertTriangle,
  TrendingUp,
  TrendingDown,
  Minus,
  RefreshCw,
  Gauge,
  Fish,
  Zap,
  Layers,
  CheckCircle,
  XCircle,
  AlertCircle
} from 'lucide-react';
import { API_CONFIG, getApiUrl } from '../config/api';

// Types
interface VolatilityData {
  available: boolean;
  volatility_index: number;
  level: string;
  description: string;
  color: string;
  trend: string;
  metrics: {
    current_price: number;
    avg_price: number;
  };
}

interface WhaleData {
  available: boolean;
  current: {
    whale_count: number;
    activity_level: string;
    description: string;
    estimated_price_impact_pct: number;
    impact: string;
  };
}

interface AnomalyData {
  available: boolean;
  status: string;
  status_color: string;
  anomaly_count: number;
  anomalies: Array<{
    type: string;
    severity: string;
    message: string;
  }>;
  current_analysis: {
    price: number;
    z_score: number;
    vs_average_pct: number;
  };
}

interface EnsembleData {
  available: boolean;
  health: {
    status: string;
    color: string;
    loaded_models: number;
    total_models: number;
    health_pct: number;
  };
  primary_model: string;
  prediction_mode: string;
  models: Array<{
    name: string;
    type: string;
    loaded: boolean;
  }>;
}

const AdvancedAnalyticsPanel: React.FC = () => {
  const [volatility, setVolatility] = useState<VolatilityData | null>(null);
  const [whales, setWhales] = useState<WhaleData | null>(null);
  const [anomalies, setAnomalies] = useState<AnomalyData | null>(null);
  const [ensemble, setEnsemble] = useState<EnsembleData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchAllData = async () => {
    try {
      setLoading(true);
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 15000);

      const [volRes, whaleRes, anomalyRes, ensembleRes] = await Promise.allSettled([
        fetch(getApiUrl(API_CONFIG.ENDPOINTS.ANALYTICS_VOLATILITY), { signal: controller.signal }),
        fetch(getApiUrl(API_CONFIG.ENDPOINTS.ANALYTICS_WHALES), { signal: controller.signal }),
        fetch(getApiUrl(API_CONFIG.ENDPOINTS.ANALYTICS_ANOMALIES), { signal: controller.signal }),
        fetch(getApiUrl(API_CONFIG.ENDPOINTS.ANALYTICS_ENSEMBLE), { signal: controller.signal })
      ]);

      clearTimeout(timeoutId);

      if (volRes.status === 'fulfilled' && volRes.value.ok) {
        setVolatility(await volRes.value.json());
      }
      if (whaleRes.status === 'fulfilled' && whaleRes.value.ok) {
        setWhales(await whaleRes.value.json());
      }
      if (anomalyRes.status === 'fulfilled' && anomalyRes.value.ok) {
        setAnomalies(await anomalyRes.value.json());
      }
      if (ensembleRes.status === 'fulfilled' && ensembleRes.value.ok) {
        setEnsemble(await ensembleRes.value.json());
      }

      setError(null);
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') {
        setError('Request timed out');
      } else {
        setError('Failed to load analytics');
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchAllData();
    const interval = setInterval(fetchAllData, 60000);
    return () => clearInterval(interval);
  }, []);

  const getVolatilityColor = (color: string) => {
    switch (color) {
      case 'green': return 'text-green-400';
      case 'yellow': return 'text-yellow-400';
      case 'orange': return 'text-orange-400';
      case 'red': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  const getVolatilityBg = (color: string) => {
    switch (color) {
      case 'green': return 'bg-green-400/20';
      case 'yellow': return 'bg-yellow-400/20';
      case 'orange': return 'bg-orange-400/20';
      case 'red': return 'bg-red-400/20';
      default: return 'bg-gray-400/20';
    }
  };

  if (loading && !volatility && !whales && !anomalies && !ensemble) {
    return (
      <div className="bg-gray-800/50 rounded-xl p-6 border border-gray-700">
        <div className="flex items-center gap-2 mb-4">
          <Activity className="w-5 h-5 text-cyan-400" />
          <h3 className="text-lg font-semibold text-white">Advanced Analytics</h3>
        </div>
        <div className="flex items-center justify-center h-48">
          <RefreshCw className="w-6 h-6 text-gray-500 animate-spin" />
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gray-800/50 rounded-xl p-6 border border-gray-700">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-2">
          <Activity className="w-5 h-5 text-cyan-400" />
          <h3 className="text-lg font-semibold text-white">Advanced Analytics</h3>
        </div>
        <button
          onClick={fetchAllData}
          className="p-1.5 rounded-lg hover:bg-gray-700 transition-colors"
          title="Refresh analytics"
        >
          <RefreshCw className={`w-4 h-4 text-gray-400 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      <div className="grid grid-cols-2 gap-4">
        {/* Volatility Index */}
        <div className="bg-gray-700/50 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-3">
            <Gauge className="w-4 h-4 text-purple-400" />
            <span className="text-sm font-medium text-gray-300">Gas Volatility Index</span>
          </div>
          {volatility?.available ? (
            <>
              <div className="flex items-center gap-3 mb-2">
                <span className={`text-3xl font-bold ${getVolatilityColor(volatility.color)}`}>
                  {volatility.volatility_index}
                </span>
                <div className={`px-2 py-1 rounded text-xs font-medium ${getVolatilityBg(volatility.color)} ${getVolatilityColor(volatility.color)}`}>
                  {volatility.level.toUpperCase()}
                </div>
              </div>
              <p className="text-xs text-gray-400 mb-2">{volatility.description}</p>
              <div className="flex items-center gap-1 text-xs text-gray-500">
                <span>Trend:</span>
                {volatility.trend === 'increasing' && <TrendingUp className="w-3 h-3 text-red-400" />}
                {volatility.trend === 'decreasing' && <TrendingDown className="w-3 h-3 text-green-400" />}
                {volatility.trend === 'stable' && <Minus className="w-3 h-3 text-gray-400" />}
                <span className="capitalize">{volatility.trend}</span>
              </div>
            </>
          ) : (
            <p className="text-sm text-gray-500">Insufficient data</p>
          )}
        </div>

        {/* Whale Activity */}
        <div className="bg-gray-700/50 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-3">
            <Fish className="w-4 h-4 text-blue-400" />
            <span className="text-sm font-medium text-gray-300">Whale Activity</span>
          </div>
          {whales?.available ? (
            <>
              <div className="flex items-center gap-3 mb-2">
                <span className="text-3xl font-bold text-white">
                  {whales.current.whale_count}
                </span>
                <div className={`px-2 py-1 rounded text-xs font-medium ${
                  whales.current.activity_level === 'high' ? 'bg-red-400/20 text-red-400' :
                  whales.current.activity_level === 'moderate' ? 'bg-yellow-400/20 text-yellow-400' :
                  whales.current.activity_level === 'low' ? 'bg-blue-400/20 text-blue-400' :
                  'bg-gray-400/20 text-gray-400'
                }`}>
                  {whales.current.activity_level.toUpperCase()}
                </div>
              </div>
              <p className="text-xs text-gray-400 mb-2">{whales.current.description}</p>
              {whales.current.estimated_price_impact_pct > 0 && (
                <div className="text-xs text-orange-400">
                  Est. impact: +{whales.current.estimated_price_impact_pct}% on gas
                </div>
              )}
            </>
          ) : (
            <p className="text-sm text-gray-500">Monitoring inactive</p>
          )}
        </div>

        {/* Anomaly Detection */}
        <div className="bg-gray-700/50 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-3">
            <Zap className="w-4 h-4 text-yellow-400" />
            <span className="text-sm font-medium text-gray-300">Anomaly Detection</span>
          </div>
          {anomalies?.available ? (
            <>
              <div className="flex items-center gap-3 mb-2">
                {anomalies.status === 'normal' && <CheckCircle className="w-6 h-6 text-green-400" />}
                {anomalies.status === 'warning' && <AlertCircle className="w-6 h-6 text-yellow-400" />}
                {anomalies.status === 'alert' && <AlertTriangle className="w-6 h-6 text-red-400" />}
                <span className={`text-lg font-semibold ${
                  anomalies.status === 'normal' ? 'text-green-400' :
                  anomalies.status === 'warning' ? 'text-yellow-400' : 'text-red-400'
                }`}>
                  {anomalies.status === 'normal' ? 'Normal' : `${anomalies.anomaly_count} Alert${anomalies.anomaly_count > 1 ? 's' : ''}`}
                </span>
              </div>
              {anomalies.anomalies.length > 0 ? (
                <div className="space-y-1">
                  {anomalies.anomalies.slice(0, 2).map((a, i) => (
                    <p key={i} className={`text-xs ${a.severity === 'high' ? 'text-red-400' : 'text-yellow-400'}`}>
                      {a.message}
                    </p>
                  ))}
                </div>
              ) : (
                <p className="text-xs text-gray-400">All metrics within normal range</p>
              )}
              <div className="mt-2 text-xs text-gray-500">
                Z-score: {anomalies.current_analysis.z_score} | vs avg: {anomalies.current_analysis.vs_average_pct > 0 ? '+' : ''}{anomalies.current_analysis.vs_average_pct}%
              </div>
            </>
          ) : (
            <p className="text-sm text-gray-500">Insufficient data</p>
          )}
        </div>

        {/* Model Ensemble */}
        <div className="bg-gray-700/50 rounded-lg p-4">
          <div className="flex items-center gap-2 mb-3">
            <Layers className="w-4 h-4 text-indigo-400" />
            <span className="text-sm font-medium text-gray-300">Model Ensemble</span>
          </div>
          {ensemble?.available ? (
            <>
              <div className="flex items-center gap-3 mb-2">
                <span className={`text-lg font-semibold ${
                  ensemble.health.color === 'green' ? 'text-green-400' :
                  ensemble.health.color === 'yellow' ? 'text-yellow-400' : 'text-red-400'
                }`}>
                  {ensemble.health.status.charAt(0).toUpperCase() + ensemble.health.status.slice(1)}
                </span>
                <span className="text-sm text-gray-400">
                  {ensemble.health.loaded_models}/{ensemble.health.total_models} models
                </span>
              </div>
              <p className="text-xs text-gray-400 mb-2">{ensemble.prediction_mode}</p>
              <div className="flex flex-wrap gap-1">
                {ensemble.models.slice(0, 4).map((m, i) => (
                  <div
                    key={i}
                    className={`flex items-center gap-1 px-2 py-0.5 rounded text-xs ${
                      m.loaded ? 'bg-green-400/20 text-green-400' : 'bg-gray-600/50 text-gray-500'
                    }`}
                  >
                    {m.loaded ? <CheckCircle className="w-3 h-3" /> : <XCircle className="w-3 h-3" />}
                    {m.name.replace('_', ' ').slice(0, 10)}
                  </div>
                ))}
              </div>
            </>
          ) : (
            <p className="text-sm text-gray-500">Loading model info...</p>
          )}
        </div>
      </div>

      {error && (
        <p className="mt-4 text-sm text-red-400 text-center">{error}</p>
      )}
    </div>
  );
};

export default memo(AdvancedAnalyticsPanel);
