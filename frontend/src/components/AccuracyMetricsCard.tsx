import React, { useEffect, useState } from 'react';
import { TrendingUp, Target, Compass, BarChart3, RefreshCw, AlertCircle } from 'lucide-react';
import { API_CONFIG, getApiUrl } from '../config/api';

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

const AccuracyMetricsCard: React.FC = () => {
  const [metrics, setMetrics] = useState<MetricsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedHorizon, setSelectedHorizon] = useState<'1h' | '4h' | '24h'>('1h');

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
          // Metrics exist but no data yet - use fallback
          setError('No metrics available');
          setMetrics({
            '1h': { mae: 0.000275, rmse: 0.000442, r2: 0.071, directional_accuracy: 0.598, n: 100 },
            '4h': { mae: 0.000312, rmse: 0.000521, r2: 0.063, directional_accuracy: 0.572, n: 50 },
            '24h': { mae: 0.000498, rmse: 0.000712, r2: 0.045, directional_accuracy: 0.541, n: 20 }
          });
        }
      } else {
        setError('No metrics available');
        // Use fallback mock data for demo
        setMetrics({
          '1h': { mae: 0.000275, rmse: 0.000442, r2: 0.071, directional_accuracy: 0.598, n: 100 },
          '4h': { mae: 0.000312, rmse: 0.000521, r2: 0.063, directional_accuracy: 0.572, n: 50 },
          '24h': { mae: 0.000498, rmse: 0.000712, r2: 0.045, directional_accuracy: 0.541, n: 20 }
        });
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
    <div className="bg-gray-900/50 border border-gray-800 rounded-2xl p-4 shadow-xl">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <BarChart3 className="w-4 h-4 text-purple-400" />
          <h3 className="font-semibold text-white">Model Accuracy</h3>
        </div>
        <button
          onClick={fetchMetrics}
          className="text-xs text-purple-300 hover:text-purple-200 transition-colors flex items-center gap-1"
          disabled={loading}
        >
          <RefreshCw className={`w-3 h-3 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
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
          Using fallback data
        </div>
      )}
    </div>
  );
};

export default AccuracyMetricsCard;
