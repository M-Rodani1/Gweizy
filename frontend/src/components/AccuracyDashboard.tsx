import React, { useState, useEffect, lazy, Suspense } from 'react';
import { TrendingUp, TrendingDown, Activity } from 'lucide-react';
import { API_CONFIG, getApiUrl } from '../config/api';

export interface MetricsData {
  horizon: string;
  mae_7d: number;
  mae_30d: number;
  mae_90d: number;
  success_rate_7d: number;
  success_rate_30d: number;
  success_rate_90d: number;
  predictions_count: number;
}

export interface TrendData {
  date: string;
  mae: number;
  success_rate: number;
  predictions: number;
}

interface AccuracyDashboardProps {
  selectedChain?: string;
}

const AccuracyDashboardCharts = lazy(() => import('./AccuracyDashboardCharts'));

const AccuracyDashboard: React.FC<AccuracyDashboardProps> = ({ selectedChain = 'base' }) => {
  const [metrics, setMetrics] = useState<Record<string, MetricsData>>({});
  const [trends, setTrends] = useState<Record<string, TrendData[]>>({});
  const [loading, setLoading] = useState(true);
  const [selectedHorizon, setSelectedHorizon] = useState<'1h' | '4h' | '24h'>('1h');
  const [selectedPeriod, setSelectedPeriod] = useState<7 | 30 | 90>(7);
  const REQUEST_TIMEOUT_MS = 12000;

  useEffect(() => {
    fetchAccuracyData();
  }, [selectedChain, selectedHorizon]);

  const fetchAccuracyData = async () => {
    try {
      setLoading(true);
      const fetchWithTimeout = async (url: string): Promise<Response> => {
        return Promise.race([
          fetch(url),
          new Promise<Response>((_, reject) => {
            setTimeout(() => reject(new Error(`Request timed out: ${url}`)), REQUEST_TIMEOUT_MS);
          })
        ]);
      };

      const [metricsReq, trendsReq] = await Promise.allSettled([
        fetchWithTimeout(getApiUrl(`${API_CONFIG.ENDPOINTS.ANALYTICS}/performance?days=90`)),
        fetchWithTimeout(getApiUrl(`${API_CONFIG.ENDPOINTS.ANALYTICS}/trends?horizon=${selectedHorizon}&days=90`))
      ]);
      let gotAnyData = false;

      if (metricsReq.status === 'fulfilled' && metricsReq.value.ok) {
        const metricsData = await metricsReq.value.json();
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
          gotAnyData = true;
        }
      }

      if (trendsReq.status === 'fulfilled' && trendsReq.value.ok) {
        const trendsData = await trendsReq.value.json();
        if (trendsData.trends) {
          setTrends({ [selectedHorizon]: trendsData.trends });
          gotAnyData = true;
        }
      }

      if (!gotAnyData && Object.keys(metrics).length === 0) {
        throw new Error('Failed to fetch accuracy data');
      }
    } catch (error) {
      console.error('Failed to fetch accuracy data:', error);
    } finally {
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

      <Suspense
        fallback={(
          <div className="bg-gray-800 border border-gray-700 rounded-lg p-6 text-sm text-gray-400">
            Loading charts...
          </div>
        )}
      >
        <AccuracyDashboardCharts
          metrics={metrics}
          trends={trends}
          selectedHorizon={selectedHorizon}
          selectedPeriod={selectedPeriod}
          getPeriodKey={getPeriodKey}
          getSuccessRateKey={getSuccessRateKey}
        />
      </Suspense>

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
