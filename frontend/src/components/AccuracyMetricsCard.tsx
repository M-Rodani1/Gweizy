/**
 * AccuracyMetricsCard Component
 *
 * Displays ML model accuracy metrics for gas price predictions.
 * Shows key performance indicators including R² score, directional accuracy,
 * MAE, and sample count for different prediction horizons (1h, 4h, 24h).
 *
 * @module components/AccuracyMetricsCard
 */

import React, { useEffect, useState } from 'react';
import { TrendingUp, Target, Compass, BarChart3, RefreshCw, AlertCircle, RotateCcw } from 'lucide-react';
import { API_CONFIG, getApiUrl } from '../config/api';

/**
 * Metrics for a specific prediction horizon.
 */
interface HorizonMetrics {
  mae: number | null;
  rmse: number | null;
  r2: number | null;
  directional_accuracy: number | null;
  n: number;
}

interface MetricsData {
  '1h': HorizonMetrics;
  '4h': HorizonMetrics;
  '24h': HorizonMetrics;
}

/**
 * Card component displaying model accuracy metrics.
 *
 * Features:
 * - Tabbed interface for 1h, 4h, and 24h horizons
 * - Color-coded metrics (green for good, yellow for moderate, red for poor)
 * - Auto-refresh every 5 minutes
 * - Manual refresh button
 * - Graceful fallback to mock data on API errors
 *
 * Metrics Displayed:
 * - **R² Score**: Coefficient of determination (variance explained)
 *   - Green (≥70%): Good model fit
 *   - Yellow (≥40%): Acceptable
 *   - Red (<40%): Poor fit
 *
 * - **Directional Accuracy**: Percentage of correct trend predictions
 *   - Green (≥60%): Better than random
 *   - Yellow (≥50%): Near random
 *   - Red (<50%): Worse than random
 *
 * - **MAE**: Mean Absolute Error in gwei
 * - **Samples**: Number of validated predictions
 *
 * @returns {JSX.Element} The accuracy metrics card component
 *
 * @example
 * ```tsx
 * // Basic usage in a dashboard
 * <AccuracyMetricsCard />
 * ```
 */
const AccuracyMetricsCard: React.FC = () => {
  const [metrics, setMetrics] = useState<MetricsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [resetting, setResetting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedHorizon, setSelectedHorizon] = useState<'1h' | '4h' | '24h'>('1h');

  const resetMetrics = async () => {
    if (!confirm('Reset all accuracy data? This will clear bad metrics and seed fresh data.')) {
      return;
    }
    try {
      setResetting(true);
      const response = await fetch(getApiUrl(API_CONFIG.ENDPOINTS.ACCURACY_RESET), {
        method: 'POST',
      });
      const data = await response.json();
      if (data.success && data.metrics) {
        setMetrics(data.metrics);
        setError(null);
      }
      await fetchMetrics();
    } catch (err) {
      setError('Failed to reset metrics');
    } finally {
      setResetting(false);
    }
  };

  const fetchMetrics = async () => {
    try {
      setLoading(true);
      const response = await fetch(getApiUrl(API_CONFIG.ENDPOINTS.ACCURACY_METRICS));

      if (!response.ok) {
        throw new Error('Failed to fetch metrics');
      }

      const data = await response.json();
      if (data.success && data.metrics) {
        // Check if metrics have actual data (at least one horizon has n > 0)
        const hasData = Object.values(data.metrics).some((m: any) => m && m.n > 0);
        if (hasData) {
          setMetrics(data.metrics);
          setError(null);
        } else {
          // Metrics exist but no data yet
          setError('No metrics available - waiting for prediction data');
          setMetrics(null);
        }
      } else {
        setError(data.error || 'No metrics available');
        setMetrics(null);
      }
    } catch (err) {
      setError('Could not load accuracy metrics');
      // Use fallback mock data for demo
      setMetrics({
        '1h': { mae: 0.000275, rmse: 0.000442, r2: 0.071, directional_accuracy: 0.598, n: 100 },
        '4h': { mae: 0.000312, rmse: 0.000521, r2: 0.063, directional_accuracy: 0.572, n: 50 },
        '24h': { mae: 0.000498, rmse: 0.000712, r2: 0.045, directional_accuracy: 0.541, n: 20 }
      });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMetrics();
    const interval = setInterval(fetchMetrics, 300000); // Refresh every 5 minutes
    return () => clearInterval(interval);
  }, []);

  const formatPercent = (value: number | null | undefined): string => {
    if (value === null || value === undefined || isNaN(value)) return '--';
    return `${(value * 100).toFixed(1)}%`;
  };

  const formatScientific = (value: number | null): string => {
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

  const currentMetrics = metrics?.[selectedHorizon];

  return (
    <div className="bg-gray-900/50 border border-gray-800 rounded-2xl p-6 shadow-xl widget-glow h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <BarChart3 className="w-4 h-4 text-purple-400" />
          <h3 className="font-semibold text-white">Model Accuracy</h3>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={resetMetrics}
            className="text-xs text-red-400 hover:text-red-300 px-2 py-1.5 rounded-lg flex items-center gap-1 font-medium disabled:opacity-50 border border-red-500/30 hover:border-red-500/50 transition-colors"
            disabled={resetting || loading}
            title="Reset metrics data"
          >
            <RotateCcw className={`w-3 h-3 ${resetting ? 'animate-spin' : ''}`} />
            Reset
          </button>
          <button
            onClick={fetchMetrics}
            className="btn-gradient-secondary text-xs text-white px-3 py-1.5 rounded-lg flex items-center gap-1.5 font-medium disabled:opacity-50"
            disabled={loading}
          >
            <RefreshCw className={`w-3 h-3 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        </div>
      </div>

      {/* Horizon Tabs */}
      <div className="flex gap-1 mb-4 bg-gray-800/50 rounded-lg p-1">
        {(['1h', '4h', '24h'] as const).map((horizon) => (
          <button
            key={horizon}
            onClick={() => setSelectedHorizon(horizon)}
            className={`flex-1 py-1.5 text-xs font-medium rounded-md transition-colors ${
              selectedHorizon === horizon
                ? 'bg-purple-500 text-white'
                : 'text-gray-400 hover:text-white'
            }`}
          >
            {horizon}
          </button>
        ))}
      </div>

      {/* Metrics Grid */}
      {loading && !metrics ? (
        <div className="flex items-center justify-center py-8">
          <RefreshCw className="w-5 h-5 text-gray-500 animate-spin" />
        </div>
      ) : currentMetrics ? (
        <div className="grid grid-cols-2 gap-3">
          {/* R² Score */}
          <div className="bg-gray-800/40 rounded-xl p-3 border border-gray-700/50">
            <div className="flex items-center gap-1.5 mb-1">
              <Target className="w-3.5 h-3.5 text-purple-400" />
              <span className="text-xs text-gray-400">R² Score</span>
            </div>
            <div className={`text-xl font-bold ${getR2Color(currentMetrics.r2)}`}>
              {formatPercent(currentMetrics.r2)}
            </div>
            <div className="text-[10px] text-gray-500 mt-0.5">Variance explained</div>
          </div>

          {/* Directional Accuracy */}
          <div className="bg-gray-800/40 rounded-xl p-3 border border-gray-700/50">
            <div className="flex items-center gap-1.5 mb-1">
              <Compass className="w-3.5 h-3.5 text-cyan-400" />
              <span className="text-xs text-gray-400">Direction</span>
            </div>
            <div className={`text-xl font-bold ${getDAColor(currentMetrics.directional_accuracy)}`}>
              {formatPercent(currentMetrics.directional_accuracy)}
            </div>
            <div className="text-[10px] text-gray-500 mt-0.5">Trend prediction</div>
          </div>

          {/* MAE */}
          <div className="bg-gray-800/40 rounded-xl p-3 border border-gray-700/50">
            <div className="flex items-center gap-1.5 mb-1">
              <TrendingUp className="w-3.5 h-3.5 text-amber-400" />
              <span className="text-xs text-gray-400">MAE</span>
            </div>
            <div className="text-lg font-bold text-white font-mono">
              {formatScientific(currentMetrics.mae)}
            </div>
            <div className="text-[10px] text-gray-500 mt-0.5">Mean abs. error</div>
          </div>

          {/* Sample Count */}
          <div className="bg-gray-800/40 rounded-xl p-3 border border-gray-700/50">
            <div className="flex items-center gap-1.5 mb-1">
              <BarChart3 className="w-3.5 h-3.5 text-emerald-400" />
              <span className="text-xs text-gray-400">Samples</span>
            </div>
            <div className="text-lg font-bold text-white">
              {currentMetrics.n}
            </div>
            <div className="text-[10px] text-gray-500 mt-0.5">Predictions tracked</div>
          </div>
        </div>
      ) : (
        <div className="flex items-center justify-center py-8 text-gray-500">
          <AlertCircle className="w-4 h-4 mr-2" />
          <span className="text-sm">No metrics available</span>
        </div>
      )}

      {/* Info Footer */}
      {error && (
        <div className="mt-3 text-xs text-amber-400 flex items-center gap-1">
          <AlertCircle className="w-3 h-3" />
          {error}
        </div>
      )}
    </div>
  );
};

export default AccuracyMetricsCard;
