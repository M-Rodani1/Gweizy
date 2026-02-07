import React, { useEffect, useState, useMemo, memo, useCallback } from 'react';
import { Layers, RefreshCw, Info, ChevronDown, ChevronUp } from 'lucide-react';
import { API_CONFIG, getApiUrl } from '../config/api';

interface FeatureData {
  feature: string;
  importance: number;
  selected: boolean;
  rank: number;
}

const FeatureImportanceChart: React.FC = () => {
  const [features, setFeatures] = useState<FeatureData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expanded, setExpanded] = useState(false);

  const fetchFeatures = useCallback(async () => {
    try {
      setLoading(true);
      const response = await fetch(
        getApiUrl(API_CONFIG.ENDPOINTS.ACCURACY_FEATURES) + '/importance?top_n=15'
      );

      if (!response.ok) {
        throw new Error('Failed to fetch features');
      }

      const data = await response.json();
      if (data.success && data.importance_report) {
        setFeatures(data.importance_report);
        setError(null);
      } else {
        throw new Error('No feature data');
      }
    } catch (err) {
      setError('Could not load features');
      // Fallback mock data for demo
      setFeatures([
        { feature: 'gas_rolling_mean_1h', importance: 0.142, selected: true, rank: 1 },
        { feature: 'hour_sin', importance: 0.098, selected: true, rank: 2 },
        { feature: 'gas_lag_1h', importance: 0.087, selected: true, rank: 3 },
        { feature: 'volatility_1h', importance: 0.076, selected: true, rank: 4 },
        { feature: 'momentum_1h', importance: 0.065, selected: true, rank: 5 },
        { feature: 'hour_cos', importance: 0.054, selected: true, rank: 6 },
        { feature: 'gas_change_1h', importance: 0.048, selected: true, rank: 7 },
        { feature: 'dow_sin', importance: 0.041, selected: true, rank: 8 },
        { feature: 'gas_rolling_std_1h', importance: 0.038, selected: true, rank: 9 },
        { feature: 'pending_tx_count', importance: 0.032, selected: true, rank: 10 },
      ]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchFeatures();
  }, [fetchFeatures]);

  // Memoize expensive calculations
  const maxImportance = useMemo(
    () => Math.max(...features.map(f => f.importance), 0.001),
    [features]
  );

  const formatFeatureName = useCallback((name: string): string => {
    return name
      .replace(/_/g, ' ')
      .replace(/\b\w/g, c => c.toUpperCase())
      .replace(/(\d+)h/g, '$1H')
      .replace(/Gas /g, '')
      .replace(/Rolling /g, 'Roll. ')
      .substring(0, 20);
  }, []);

  const getBarColor = useCallback((rank: number): string => {
    if (rank <= 3) return 'bg-gradient-to-r from-purple-500 to-purple-400';
    if (rank <= 6) return 'bg-gradient-to-r from-cyan-500 to-cyan-400';
    return 'bg-gradient-to-r from-gray-500 to-gray-400';
  }, []);

  const displayFeatures = useMemo(
    () => expanded ? features : features.slice(0, 5),
    [expanded, features]
  );

  // Generate screen reader summary
  const featureSummary = useMemo(() => {
    if (features.length === 0) return 'No feature importance data available.';

    const topFeatures = features.slice(0, 3);
    const topList = topFeatures.map((f, i) =>
      `${i + 1}. ${formatFeatureName(f.feature)} at ${(f.importance * 100).toFixed(1)}%`
    ).join('; ');

    return `Feature importance chart showing ${features.length} features. ` +
      `Top 3 features: ${topList}.`;
  }, [features]);

  return (
    <div className="bg-gray-900/50 border border-gray-800 rounded-2xl p-6 shadow-xl widget-glow h-full flex flex-col">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Layers className="w-4 h-4 text-purple-400" aria-hidden="true" />
          <h3 className="font-semibold text-white">Feature Importance</h3>
          <div className="group relative">
            <button
              type="button"
              className="focus:outline-none focus:ring-2 focus:ring-purple-500 rounded"
              aria-label="Information about feature importance"
            >
              <Info className="w-3.5 h-3.5 text-gray-500 cursor-help" aria-hidden="true" />
            </button>
            <div
              role="tooltip"
              className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-xs text-gray-300 w-48 opacity-0 group-hover:opacity-100 group-focus-within:opacity-100 transition-opacity pointer-events-none z-10"
            >
              SHAP-based feature importance shows which inputs most influence predictions
            </div>
          </div>
        </div>
        <button
          onClick={fetchFeatures}
          aria-label="Refresh feature importance"
          className="btn-gradient-secondary text-xs text-white px-3 py-1.5 rounded-lg flex items-center gap-1.5 font-medium disabled:opacity-50 focus:outline-none focus:ring-2 focus:ring-purple-500"
          disabled={loading}
        >
          <RefreshCw className={`w-3 h-3 ${loading ? 'animate-spin' : ''}`} aria-hidden="true" />
        </button>
      </div>

      {/* Screen reader summary */}
      <p className="sr-only">{featureSummary}</p>

      {/* Chart */}
      {loading && features.length === 0 ? (
        <div className="flex items-center justify-center py-8" role="status" aria-label="Loading">
          <RefreshCw className="w-5 h-5 text-gray-500 animate-spin" aria-hidden="true" />
          <span className="sr-only">Loading feature importance...</span>
        </div>
      ) : (
        <div className="space-y-2">
          {displayFeatures.map((feature, index) => (
            <div key={feature.feature} className="group">
              <div className="flex items-center justify-between text-xs sm:text-xs mb-1 gap-2">
                <span className="text-gray-400 truncate max-w-[100px] sm:max-w-[140px]" title={feature.feature}>
                  {formatFeatureName(feature.feature)}
                </span>
                <span className="text-gray-500 font-mono shrink-0">
                  {(feature.importance * 100).toFixed(1)}%
                </span>
              </div>
              <div className="h-1.5 sm:h-2 bg-gray-800 rounded-full overflow-hidden">
                <div
                  className={`h-full rounded-full transition-all duration-500 ${getBarColor(feature.rank)}`}
                  style={{
                    width: `${(feature.importance / maxImportance) * 100}%`,
                    animationDelay: `${index * 50}ms`
                  }}
                />
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Expand/Collapse */}
      {features.length > 5 && (
        <button
          onClick={() => setExpanded(!expanded)}
          aria-expanded={expanded}
          className="w-full mt-3 pt-2 border-t border-gray-800 flex items-center justify-center gap-1 text-xs text-gray-400 hover:text-gray-300 transition-colors focus:outline-none focus:ring-2 focus:ring-purple-500 rounded"
        >
          {expanded ? (
            <>
              <ChevronUp className="w-3 h-3" aria-hidden="true" />
              Show less
            </>
          ) : (
            <>
              <ChevronDown className="w-3 h-3" aria-hidden="true" />
              Show {features.length - 5} more
            </>
          )}
        </button>
      )}

      {/* Footer */}
      {error && (
        <div className="mt-3 text-xs text-amber-400/80">
          Using demo data
        </div>
      )}
    </div>
  );
};

export default memo(FeatureImportanceChart);
