/**
 * MempoolStatusCard Component
 *
 * Displays real-time mempool status and congestion metrics.
 * Shows pending transactions, gas price momentum, and congestion state.
 *
 * @module components/MempoolStatusCard
 */

import React, { useEffect, useState } from 'react';
import { Activity, TrendingUp, TrendingDown, Minus, RefreshCw, AlertTriangle, CheckCircle } from 'lucide-react';
import { API_CONFIG, getApiUrl } from '../config/api';

interface MempoolMetrics {
  pending_count: number;
  avg_gas_price: number;
  median_gas_price: number;
  p90_gas_price: number;
  gas_price_spread: number;
  large_tx_ratio: number;
  arrival_rate: number;
}

interface MempoolSignals {
  is_congested: boolean;
  congestion_level: 'low' | 'moderate' | 'high' | 'unknown';
  count_momentum: number;
  gas_momentum: number;
}

interface MempoolInterpretation {
  trend: string;
  trend_description: string;
  recommendation: string;
}

interface MempoolResponse {
  status: 'active' | 'inactive' | 'error';
  metrics: MempoolMetrics;
  signals: MempoolSignals;
  interpretation: MempoolInterpretation;
  timestamp: string;
  snapshot_time?: string;
  error?: string;
}

const MempoolStatusCard: React.FC = () => {
  const [data, setData] = useState<MempoolResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchMempoolStatus = async () => {
    try {
      setLoading(true);

      // Use AbortController for timeout
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000); // 10s timeout

      const response = await fetch(
        getApiUrl(API_CONFIG.ENDPOINTS.MEMPOOL_STATUS),
        { signal: controller.signal }
      );

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error('Failed to fetch mempool status');
      }

      const result = await response.json();
      setData(result);
      setError(null);
    } catch (err) {
      if (err instanceof Error && err.name === 'AbortError') {
        setError('Request timed out');
      } else {
        setError('Mempool data unavailable');
      }
      setData(null);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMempoolStatus();
    const interval = setInterval(fetchMempoolStatus, 30000); // Refresh every 30s
    return () => clearInterval(interval);
  }, []);

  const getMomentumIcon = (momentum: number) => {
    if (momentum > 0.05) return <TrendingUp className="w-4 h-4 text-red-400" />;
    if (momentum < -0.05) return <TrendingDown className="w-4 h-4 text-green-400" />;
    return <Minus className="w-4 h-4 text-gray-400" />;
  };

  const getMomentumColor = (momentum: number) => {
    if (momentum > 0.05) return 'text-red-400';
    if (momentum < -0.05) return 'text-green-400';
    return 'text-gray-400';
  };

  const formatMomentum = (momentum: number) => {
    const sign = momentum > 0 ? '+' : '';
    return `${sign}${(momentum * 100).toFixed(1)}%`;
  };

  const getCongestionStatus = (level: string, isCongested: boolean) => {
    if (isCongested || level === 'high') {
      return {
        icon: <AlertTriangle className="w-5 h-5 text-amber-400" />,
        text: 'Congested',
        color: 'text-amber-400',
        bgColor: 'bg-amber-400/10',
        borderColor: 'border-amber-400/30'
      };
    }
    if (level === 'moderate') {
      return {
        icon: <Activity className="w-5 h-5 text-yellow-400" />,
        text: 'Moderate',
        color: 'text-yellow-400',
        bgColor: 'bg-yellow-400/10',
        borderColor: 'border-yellow-400/30'
      };
    }
    return {
      icon: <CheckCircle className="w-5 h-5 text-green-400" />,
      text: 'Clear',
      color: 'text-green-400',
      bgColor: 'bg-green-400/10',
      borderColor: 'border-green-400/30'
    };
  };

  if (loading && !data) {
    return (
      <div className="bg-gray-800/50 rounded-xl p-6 border border-gray-700">
        <div className="flex items-center gap-2 mb-4">
          <Activity className="w-5 h-5 text-purple-400" />
          <h3 className="text-lg font-semibold text-white">Mempool Status</h3>
        </div>
        <div className="flex items-center justify-center h-32">
          <RefreshCw className="w-6 h-6 text-gray-500 animate-spin" />
        </div>
      </div>
    );
  }

  if (error || !data?.metrics) {
    return (
      <div className="bg-gray-800/50 rounded-xl p-6 border border-gray-700">
        <div className="flex items-center gap-2 mb-4">
          <Activity className="w-5 h-5 text-purple-400" />
          <h3 className="text-lg font-semibold text-white">Mempool Status</h3>
        </div>
        <p className="text-gray-400 text-sm">{error || 'Mempool data unavailable'}</p>
      </div>
    );
  }

  const { metrics, signals, interpretation } = data;
  const congestion = getCongestionStatus(signals.congestion_level, signals.is_congested);

  return (
    <div className="bg-gray-800/50 rounded-xl p-6 border border-gray-700">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Activity className="w-5 h-5 text-purple-400" />
          <h3 className="text-lg font-semibold text-white">Mempool Status</h3>
        </div>
        <button
          onClick={fetchMempoolStatus}
          className="p-1.5 rounded-lg hover:bg-gray-700 transition-colors"
          title="Refresh mempool"
        >
          <RefreshCw className={`w-4 h-4 text-gray-400 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {/* Congestion Status Banner */}
      <div className={`flex items-center gap-2 p-3 rounded-lg mb-4 ${congestion.bgColor} border ${congestion.borderColor}`}>
        {congestion.icon}
        <span className={`font-medium ${congestion.color}`}>{congestion.text}</span>
        <span className="text-gray-400 text-sm ml-auto capitalize">
          {signals.congestion_level}
        </span>
      </div>

      {/* Recommendation */}
      {interpretation?.recommendation && (
        <p className="text-sm text-gray-300 mb-4 bg-gray-700/50 p-3 rounded-lg">
          {interpretation.recommendation}
        </p>
      )}

      {/* Key Metrics Grid */}
      <div className="grid grid-cols-2 gap-3 mb-4">
        {/* Pending Transactions */}
        <div className="bg-gray-700/50 rounded-lg p-3">
          <div className="text-xs text-gray-400 mb-1">Pending Txs</div>
          <div className="flex items-center gap-2">
            <span className="text-xl font-semibold text-white">
              {metrics.pending_count?.toLocaleString() || 'N/A'}
            </span>
            {signals.count_momentum !== undefined && (
              <span className={`text-xs ${getMomentumColor(signals.count_momentum)}`}>
                {formatMomentum(signals.count_momentum)}
              </span>
            )}
          </div>
        </div>

        {/* Avg Gas Price */}
        <div className="bg-gray-700/50 rounded-lg p-3">
          <div className="text-xs text-gray-400 mb-1">Avg Gas Price</div>
          <div className="flex items-center gap-2">
            <span className="text-xl font-semibold text-white">
              {metrics.avg_gas_price?.toFixed(4) || 'N/A'}
            </span>
            <span className="text-xs text-gray-500">gwei</span>
          </div>
        </div>

        {/* Gas Price Momentum */}
        <div className="bg-gray-700/50 rounded-lg p-3">
          <div className="text-xs text-gray-400 mb-1">Price Momentum</div>
          <div className="flex items-center gap-2">
            {getMomentumIcon(signals.gas_momentum || 0)}
            <span className={`text-lg font-semibold ${getMomentumColor(signals.gas_momentum || 0)}`}>
              {formatMomentum(signals.gas_momentum || 0)}
            </span>
          </div>
        </div>

        {/* Arrival Rate */}
        <div className="bg-gray-700/50 rounded-lg p-3">
          <div className="text-xs text-gray-400 mb-1">Arrival Rate</div>
          <div className="flex items-center gap-2">
            <span className="text-xl font-semibold text-white">
              {metrics.arrival_rate?.toFixed(1) || '0'}
            </span>
            <span className="text-xs text-gray-500">tx/s</span>
          </div>
        </div>
      </div>

      {/* Additional Stats */}
      <div className="flex items-center justify-between text-xs text-gray-500 pt-2 border-t border-gray-700">
        <span>{interpretation?.trend_description || 'Analyzing...'}</span>
        <span>
          {data.status === 'active' ? (
            <span className="flex items-center gap-1">
              <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></span>
              Live
            </span>
          ) : (
            'Inactive'
          )}
        </span>
      </div>
    </div>
  );
};

export default MempoolStatusCard;
