import React, { useState, useEffect } from 'react';
import { Network, Activity, Zap, TrendingUp, Box, Code, Gauge } from 'lucide-react';
import { API_CONFIG, getApiUrl } from '../config/api';

interface NetworkStateResponse {
  network_state: {
    current_block: number;
    avg_utilization: number;
    avg_tx_count: number;
    avg_base_fee: number;
    base_fee_trend: number;
    is_congested: boolean;
    timestamp: string;
  };
  interpretation: {
    congestion_level: string;
    gas_trend: string;
    recommendation: string;
  };
  timestamp: string;
}

interface NetworkState {
  current_block: number;
  block_time_avg?: number;
  gas_price: number;
  network_congestion: string;
  congestion_score: number;
  tx_per_block_avg: number;
  block_utilization_avg: number;
  contract_call_ratio?: number;
}

interface CongestionHistory {
  timestamps: string[];
  congestion_scores: number[];
  block_utilization: number[];
  tx_counts: number[];
}

const NetworkIntelligencePanel: React.FC = () => {
  const [networkState, setNetworkState] = useState<NetworkState | null>(null);
  const [congestionHistory, setCongestionHistory] = useState<CongestionHistory | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isExpanded, setIsExpanded] = useState(true);
  const REQUEST_TIMEOUT_MS = 12000;

  useEffect(() => {
    fetchNetworkData();
    const interval = setInterval(fetchNetworkData, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, []);

  const fetchNetworkData = async () => {
    try {
      setLoading(true);
      setError(null);
      const fetchWithTimeout = async (url: string): Promise<Response> => {
        return Promise.race([
          fetch(url),
          new Promise<Response>((_, reject) => {
            setTimeout(() => reject(new Error(`Request timed out: ${url}`)), REQUEST_TIMEOUT_MS);
          })
        ]);
      };

      const [stateReq, historyReq] = await Promise.allSettled([
        fetchWithTimeout(getApiUrl(API_CONFIG.ENDPOINTS.ONCHAIN_NETWORK_STATE)),
        fetchWithTimeout(getApiUrl(API_CONFIG.ENDPOINTS.ONCHAIN_CONGESTION_HISTORY, { hours: 24 }))
      ]);
      let gotAnyData = false;

      if (stateReq.status === 'fulfilled' && stateReq.value.ok) {
        const stateData: NetworkStateResponse = await stateReq.value.json();

        // Transform API response to component format
        const transformedState: NetworkState = {
          current_block: stateData.network_state.current_block,
          gas_price: stateData.network_state.avg_base_fee,
          network_congestion: stateData.interpretation.congestion_level,
          congestion_score: stateData.network_state.avg_utilization,
          tx_per_block_avg: stateData.network_state.avg_tx_count,
          block_utilization_avg: stateData.network_state.avg_utilization,
        };

        setNetworkState(transformedState);
        gotAnyData = true;
      }

      if (historyReq.status === 'fulfilled' && historyReq.value.ok) {
        const historyData = await historyReq.value.json();
        setCongestionHistory(historyData);
        gotAnyData = true;
      }

      if (!gotAnyData && !networkState) {
        throw new Error('Failed to fetch network data');
      }
    } catch (err) {
      console.error('Error fetching network data:', err);
      setError(err instanceof Error ? err.message : 'Failed to load network data');
    } finally {
      setLoading(false);
    }
  };

  const getCongestionColor = (level: string | undefined) => {
    if (!level) return 'text-gray-400 bg-gray-500/20';
    switch (level.toLowerCase()) {
      case 'low': return 'text-green-400 bg-green-500/20';
      case 'moderate': return 'text-yellow-400 bg-yellow-500/20';
      case 'high': return 'text-orange-400 bg-orange-500/20';
      case 'critical': return 'text-red-400 bg-red-500/20';
      case 'extreme': return 'text-red-400 bg-red-500/20';
      default: return 'text-gray-400 bg-gray-500/20';
    }
  };

  const getCongestionDot = (level: string | undefined) => {
    if (!level) return 'bg-gray-400';
    switch (level.toLowerCase()) {
      case 'low': return 'bg-green-400';
      case 'moderate': return 'bg-yellow-400';
      case 'high': return 'bg-orange-400';
      case 'critical': return 'bg-red-400';
      case 'extreme': return 'bg-red-400';
      default: return 'bg-gray-400';
    }
  };

  const formatPercentage = (value: number) => `${(value * 100).toFixed(1)}%`;

  if (loading && !networkState) {
    return (
      <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
        <div className="flex items-center justify-center h-48">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-cyan-400"></div>
        </div>
      </div>
    );
  }

  if (error && !networkState) {
    return (
      <div className="bg-slate-800/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6">
        <div className="text-center text-red-400">
          <Activity className="w-8 h-8 mx-auto mb-2" />
          <p>{error}</p>
          <button
            onClick={fetchNetworkData}
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
              <Network className="w-5 h-5 text-cyan-400" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-white">Network Intelligence</h3>
              <p className="text-sm text-gray-400">On-chain metrics that drive predictions</p>
            </div>
          </div>
          <div className="flex items-center gap-4">
            {networkState && (
              <div className={`flex items-center gap-2 px-3 py-1.5 rounded-lg ${getCongestionColor(networkState.network_congestion)}`}>
                <span className={`w-2.5 h-2.5 rounded-full ${getCongestionDot(networkState.network_congestion)}`} />
                <span className="text-sm font-medium capitalize">{networkState.network_congestion}</span>
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

      {isExpanded && networkState && (
        <div className="p-6 space-y-6">
          {/* Congestion Gauge */}
          <div className="bg-gradient-to-br from-cyan-500/10 to-cyan-400/10 border border-cyan-500/30 rounded-lg p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2">
                <Gauge className="w-5 h-5 text-cyan-400" />
                <h4 className="text-sm font-medium text-white">Network Congestion</h4>
              </div>
              <span className={`w-4 h-4 rounded-full ${getCongestionDot(networkState.network_congestion)}`} />
            </div>

            {/* Congestion Score Bar */}
            <div className="relative h-8 bg-slate-700/50 rounded-full overflow-hidden mb-2">
              <div
                className="absolute inset-y-0 left-0 bg-gradient-to-r from-green-500 via-yellow-500 via-orange-500 to-red-500 transition-all duration-500"
                style={{ width: `${networkState.congestion_score * 100}%` }}
              />
              <div className="absolute inset-0 flex items-center justify-center">
                <span className="text-sm font-bold text-white drop-shadow-lg">
                  {formatPercentage(networkState.congestion_score)}
                </span>
              </div>
            </div>

            <div className="flex justify-between text-xs text-gray-400 mt-1">
              <span>Low</span>
              <span>Moderate</span>
              <span>High</span>
              <span>Extreme</span>
            </div>
          </div>

          {/* Key Metrics Grid */}
          <div>
            <h4 className="text-sm font-medium text-gray-400 mb-3">On-Chain Metrics</h4>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              {/* Block Utilization */}
              <div className="bg-slate-700/50 rounded-lg p-4">
                <div className="flex items-center gap-2 mb-2">
                  <Box className="w-4 h-4 text-cyan-400" />
                  <p className="text-xs text-gray-400">Block Utilization</p>
                </div>
                <p className="text-2xl font-bold text-white">
                  {formatPercentage(networkState.block_utilization_avg)}
                </p>
                <div className="mt-2 h-1.5 bg-slate-600 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-cyan-500 transition-all"
                    style={{ width: `${networkState.block_utilization_avg * 100}%` }}
                  />
                </div>
              </div>

              {/* Transactions per Block */}
              <div className="bg-slate-700/50 rounded-lg p-4">
                <div className="flex items-center gap-2 mb-2">
                  <TrendingUp className="w-4 h-4 text-green-400" />
                  <p className="text-xs text-gray-400">Tx per Block</p>
                </div>
                <p className="text-2xl font-bold text-white">
                  {networkState.tx_per_block_avg !== undefined && networkState.tx_per_block_avg !== null ? networkState.tx_per_block_avg.toFixed(1) : 'N/A'}
                </p>
                <p className="text-xs text-gray-500 mt-1">
                  Average transactions
                </p>
              </div>

              {/* Contract Call Ratio */}
              <div className="bg-slate-700/50 rounded-lg p-4">
                <div className="flex items-center gap-2 mb-2">
                  <Code className="w-4 h-4 text-cyan-400" />
                  <p className="text-xs text-gray-400">Contract Calls</p>
                </div>
                <p className="text-2xl font-bold text-white">
                  {networkState.contract_call_ratio !== undefined && networkState.contract_call_ratio !== null ? formatPercentage(networkState.contract_call_ratio) : 'N/A'}
                </p>
                <p className="text-xs text-gray-500 mt-1">
                  of all transactions
                </p>
              </div>

              {/* Current Block */}
              <div className="bg-slate-700/50 rounded-lg p-4">
                <div className="flex items-center gap-2 mb-2">
                  <Activity className="w-4 h-4 text-cyan-400" />
                  <p className="text-xs text-gray-400">Current Block</p>
                </div>
                <p className="text-xl font-bold text-white font-mono">
                  {networkState.current_block.toLocaleString()}
                </p>
                <p className="text-xs text-gray-500 mt-1">
                  Latest on Base
                </p>
              </div>

              {/* Block Time */}
              <div className="bg-slate-700/50 rounded-lg p-4">
                <div className="flex items-center gap-2 mb-2">
                  <Zap className="w-4 h-4 text-yellow-400" />
                  <p className="text-xs text-gray-400">Block Time</p>
                </div>
                <p className="text-2xl font-bold text-white">
                  {networkState.block_time_avg !== undefined && networkState.block_time_avg !== null ? networkState.block_time_avg.toFixed(1) : 'N/A'}s
                </p>
                <p className="text-xs text-gray-500 mt-1">
                  Average interval
                </p>
              </div>

              {/* Gas Price */}
              <div className="bg-slate-700/50 rounded-lg p-4">
                <div className="flex items-center gap-2 mb-2">
                  <Activity className="w-4 h-4 text-orange-400" />
                  <p className="text-xs text-gray-400">Current Gas</p>
                </div>
                <p className="text-2xl font-bold text-white">
                  {networkState.gas_price !== undefined && networkState.gas_price !== null ? networkState.gas_price.toFixed(6) : 'N/A'}
                </p>
                <p className="text-xs text-gray-500 mt-1">
                  Gwei
                </p>
              </div>
            </div>
          </div>

          {/* 24h Congestion History */}
          {congestionHistory && Array.isArray(congestionHistory.timestamps) && congestionHistory.timestamps.length > 0 && (
            <div>
              <h4 className="text-sm font-medium text-gray-400 mb-3">24-Hour Congestion Pattern</h4>
              <div className="bg-slate-700/30 rounded-lg p-4 border border-slate-600">
                <div className="space-y-4">
                  {/* Congestion Score Chart */}
                  <div>
                    <p className="text-xs text-gray-400 mb-2">Congestion Score</p>
                    <div className="flex items-end gap-0.5 h-20">
                      {congestionHistory.congestion_scores && congestionHistory.congestion_scores.length > 0 ? congestionHistory.congestion_scores.map((score, idx) => {
                        let color = 'bg-green-500/50';
                        if (score > 0.7) color = 'bg-red-500/50';
                        else if (score > 0.5) color = 'bg-orange-500/50';
                        else if (score > 0.3) color = 'bg-yellow-500/50';

                        return (
                          <div
                            key={idx}
                            className={`flex-1 ${color} rounded-t hover:opacity-100 transition-opacity`}
                            style={{ height: `${score * 100}%`, opacity: 0.7 }}
                            title={`${new Date(congestionHistory.timestamps[idx]).toLocaleTimeString()}: ${formatPercentage(score)}`}
                          />
                        );
                      }) : <div className="text-xs text-gray-500">No data</div>}
                    </div>
                  </div>

                  {/* Block Utilization Chart */}
                  <div>
                    <p className="text-xs text-gray-400 mb-2">Block Utilization</p>
                    <div className="flex items-end gap-0.5 h-16">
                      {congestionHistory.block_utilization && congestionHistory.block_utilization.length > 0 ? congestionHistory.block_utilization.map((util, idx) => (
                        <div
                          key={idx}
                          className="flex-1 bg-cyan-500/50 rounded-t hover:bg-cyan-500/70 transition-colors"
                          style={{ height: `${util * 100}%` }}
                          title={`${new Date(congestionHistory.timestamps[idx]).toLocaleTimeString()}: ${formatPercentage(util)}`}
                        />
                      )) : <div className="text-xs text-gray-500">No data</div>}
                    </div>
                  </div>

                  {/* Transaction Volume Chart */}
                  <div>
                    <p className="text-xs text-gray-400 mb-2">Transaction Volume</p>
                    <div className="flex items-end gap-0.5 h-16">
                      {congestionHistory.tx_counts && congestionHistory.tx_counts.length > 0 ? congestionHistory.tx_counts.map((count, idx) => {
                        const maxCount = congestionHistory.tx_counts && congestionHistory.tx_counts.length > 0
                          ? Math.max(...congestionHistory.tx_counts)
                          : 1;
                        const height = (count / maxCount) * 100;
                        return (
                          <div
                            key={idx}
                            className="flex-1 bg-cyan-500/50 rounded-t hover:bg-cyan-500/70 transition-colors"
                            style={{ height: `${height}%` }}
                            title={`${new Date(congestionHistory.timestamps[idx]).toLocaleTimeString()}: ${count} tx`}
                          />
                        );
                      }) : <div className="text-xs text-gray-500">No data</div>}
                    </div>
                  </div>
                </div>

                <div className="mt-4 flex justify-between text-xs text-gray-500">
                  <span>24h ago</span>
                  <span>12h ago</span>
                  <span>Now</span>
                </div>
              </div>
            </div>
          )}

          {/* Insights */}
          <div className="bg-cyan-500/10 border border-cyan-500/30 rounded-lg p-4">
            <h4 className="text-sm font-medium text-cyan-400 mb-3">Network Insights</h4>
            <ul className="space-y-2 text-sm text-gray-300">
              <li className="flex items-start gap-2">
                <span className="text-cyan-400 mt-0.5">•</span>
                <span>
                  {networkState.block_utilization_avg > 0.8
                    ? "Blocks are heavily utilized - expect higher gas prices"
                    : networkState.block_utilization_avg > 0.5
                    ? "Moderate block utilization - gas prices may vary"
                    : "Low block utilization - good time for transactions"}
                </span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-cyan-400 mt-0.5">•</span>
                <span>
                  {(networkState.contract_call_ratio !== undefined && networkState.contract_call_ratio !== null && networkState.contract_call_ratio > 0.7)
                    ? "High contract interaction - complex transactions dominate"
                    : "Balanced mix of transfers and contract calls"}
                </span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-cyan-400 mt-0.5">•</span>
                <span>
                  {networkState.tx_per_block_avg > 100
                    ? "High transaction volume per block - network is active"
                    : "Normal transaction volume - steady network activity"}
                </span>
              </li>
            </ul>
          </div>

          {/* Footer */}
          <div className="text-xs text-gray-400 text-center pt-4 border-t border-slate-700">
            Real-time data from Base network • Updates every 30 seconds
          </div>
        </div>
      )}
    </div>
  );
};

export default NetworkIntelligencePanel;
