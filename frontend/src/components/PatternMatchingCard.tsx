/**
 * PatternMatchingCard Component
 *
 * Displays historical pattern matching analysis for gas price predictions.
 * Shows similar patterns found in history and what happened after them.
 *
 * @module components/PatternMatchingCard
 */

import React, { useEffect, useState, memo } from 'react';
import { Search, TrendingUp, TrendingDown, Minus, RefreshCw, Clock, Target } from 'lucide-react';
import { API_CONFIG, getApiUrl } from '../config/api';

interface PatternMatch {
  timestamp: string;
  correlation: number;
  time_similarity: number;
  combined_score: number;
  outcome: {
    '1h_change': number;
    '4h_change': number;
  };
}

interface PatternPrediction {
  predicted_change: number;
  predicted_price: number;
  std_dev: number;
}

interface PatternData {
  available: boolean;
  current_price: number;
  data_points: number;
  search_period_hours: number;
  match_count: number;
  predictions: {
    available: boolean;
    match_count: number;
    avg_correlation: number;
    confidence: number;
    '1h': PatternPrediction;
    '4h': PatternPrediction;
    '24h': PatternPrediction;
    pattern_insight: string;
  };
  top_matches: PatternMatch[];
  timestamp: string;
}

const PatternMatchingCard: React.FC = () => {
  const [data, setData] = useState<PatternData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchPatterns = async () => {
    try {
      setLoading(true);

      // Use AbortController for timeout
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 15000); // 15s timeout

      const response = await fetch(
        getApiUrl(API_CONFIG.ENDPOINTS.PATTERNS, { hours: 72 }), // 72 hours for sufficient data points
        { signal: controller.signal }
      );

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error('Failed to fetch patterns');
      }

      const result = await response.json();
      setData(result);
      setError(null);
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') {
        setError('Request timed out');
      } else {
        setError('Pattern analysis unavailable');
      }
      setData(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPatterns();
    const interval = setInterval(fetchPatterns, 60000); // Refresh every minute
    return () => clearInterval(interval);
  }, []);

  const getTrendIcon = (change: number) => {
    if (change > 2) return <TrendingUp className="w-4 h-4 text-red-500" />;
    if (change < -2) return <TrendingDown className="w-4 h-4 text-green-500" />;
    return <Minus className="w-4 h-4 text-gray-500" />;
  };

  const getChangeColor = (change: number) => {
    if (change > 2) return 'text-red-500';
    if (change < -2) return 'text-green-500';
    return 'text-gray-400';
  };

  const formatChange = (change: number) => {
    const sign = change > 0 ? '+' : '';
    return `${sign}${(change * 100).toFixed(1)}%`;
  };

  if (loading) {
    return (
      <div className="bg-gray-800/50 rounded-xl p-6 border border-gray-700">
        <div className="flex items-center gap-2 mb-4">
          <Search className="w-5 h-5 text-blue-400" />
          <h3 className="text-lg font-semibold text-white">Pattern Analysis</h3>
        </div>
        <div className="flex items-center justify-center h-32">
          <RefreshCw className="w-6 h-6 text-gray-500 animate-spin" />
        </div>
      </div>
    );
  }

  if (error || !data?.available) {
    return (
      <div className="bg-gray-800/50 rounded-xl p-6 border border-gray-700">
        <div className="flex items-center gap-2 mb-4">
          <Search className="w-5 h-5 text-blue-400" />
          <h3 className="text-lg font-semibold text-white">Pattern Analysis</h3>
        </div>
        <p className="text-gray-400 text-sm">
          {error || data?.reason || 'Pattern analysis unavailable'}
        </p>
      </div>
    );
  }

  const { predictions, top_matches } = data;

  return (
    <div className="bg-gray-800/50 rounded-xl p-6 border border-gray-700">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Search className="w-5 h-5 text-blue-400" />
          <h3 className="text-lg font-semibold text-white">Pattern Analysis</h3>
        </div>
        <button
          onClick={fetchPatterns}
          className="p-1.5 rounded-lg hover:bg-gray-700 transition-colors"
          title="Refresh patterns"
        >
          <RefreshCw className="w-4 h-4 text-gray-400" />
        </button>
      </div>

      {/* Pattern insight */}
      {predictions?.pattern_insight && (
        <p className="text-sm text-gray-300 mb-4 bg-gray-700/50 p-3 rounded-lg">
          {predictions.pattern_insight}
        </p>
      )}

      {/* Prediction summary */}
      {predictions?.available && (
        <div className="grid grid-cols-3 gap-3 mb-4">
          {['1h', '4h', '24h'].map((horizon) => {
            const pred = predictions[horizon as '1h' | '4h' | '24h'];
            if (!pred) return null;
            return (
              <div key={horizon} className="bg-gray-700/50 rounded-lg p-3">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-xs text-gray-400">{horizon}</span>
                  {getTrendIcon(pred.predicted_change)}
                </div>
                <div className={`text-lg font-semibold ${getChangeColor(pred.predicted_change)}`}>
                  {formatChange(pred.predicted_change)}
                </div>
                <div className="text-xs text-gray-500">
                  → {pred.predicted_price.toFixed(6)} gwei
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Confidence & Stats */}
      <div className="flex items-center gap-4 mb-4 text-sm">
        <div className="flex items-center gap-1">
          <Target className="w-4 h-4 text-cyan-400" />
          <span className="text-gray-400">Confidence:</span>
          <span className="text-white font-medium">
            {((predictions?.confidence || 0) * 100).toFixed(0)}%
          </span>
        </div>
        <div className="flex items-center gap-1">
          <Clock className="w-4 h-4 text-blue-400" />
          <span className="text-gray-400">Matches:</span>
          <span className="text-white font-medium">{data.match_count}</span>
        </div>
      </div>

      {/* Top matches */}
      {top_matches && top_matches.length > 0 && (
        <div>
          <h4 className="text-xs font-medium text-gray-400 uppercase tracking-wider mb-2">
            Similar Historical Patterns
          </h4>
          <div className="space-y-2">
            {top_matches.slice(0, 3).map((match, idx) => (
              <div
                key={idx}
                className="flex items-center justify-between bg-gray-700/30 rounded-lg px-3 py-2"
              >
                <div className="flex items-center gap-2">
                  <span className="text-xs text-gray-500">
                    {new Date(match.timestamp).toLocaleDateString()}
                  </span>
                  <span className="text-xs px-1.5 py-0.5 bg-blue-500/20 text-blue-400 rounded">
                    {(match.correlation * 100).toFixed(0)}% match
                  </span>
                </div>
                <div className="flex items-center gap-2 text-xs">
                  <span className={getChangeColor(match.outcome['1h_change'] / 100)}>
                    1h: {match.outcome['1h_change'] > 0 ? '+' : ''}{match.outcome['1h_change'].toFixed(1)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default memo(PatternMatchingCard);
