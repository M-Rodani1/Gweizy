import React, { useState, useEffect, useCallback } from 'react';
import { useChain } from '../contexts/ChainContext';
import { useScheduler } from '../contexts/SchedulerContext';
import { TX_GAS_ESTIMATES, TransactionType } from '../config/chains';
import ScheduleTransactionModal from './ScheduleTransactionModal';
import ConfidenceRing from './ui/ConfidenceRing';
import { formatGwei, formatUsd } from '../utils/formatNumber';

interface AgentRecommendation {
  action: string;
  confidence: number;
  recommended_gas: number;
  expected_savings: number;
  reasoning: string;
  urgency_factor: number;
  wait_time?: number;
}

interface TransactionPilotProps {
  ethPrice?: number;
}

const TX_TYPES: { type: TransactionType; label: string; icon: string }[] = [
  { type: 'swap', label: 'Swap', icon: '🔄' },
  { type: 'bridge', label: 'Bridge', icon: '🌉' },
  { type: 'nftMint', label: 'Mint', icon: '🎨' },
  { type: 'transfer', label: 'Transfer', icon: '📤' },
  { type: 'approve', label: 'Approve', icon: '✅' },
];

const TransactionPilot: React.FC<TransactionPilotProps> = ({ ethPrice = 3000 }) => {
  const { selectedChain, multiChainGas, bestChainForTx, setSelectedChainId } = useChain();
  const { pendingCount, readyCount } = useScheduler();
  const [selectedTxType, setSelectedTxType] = useState<TransactionType>('swap');
  const [urgency, setUrgency] = useState(0.5);
  const [recommendation, setRecommendation] = useState<AgentRecommendation | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [countdown, setCountdown] = useState<number | null>(null);
  const [showScheduleModal, setShowScheduleModal] = useState(false);
  const [showChainToast, setShowChainToast] = useState(false);

  const API_URL = import.meta.env.VITE_API_URL || 'https://basegasfeesml-production.up.railway.app/api';

  const currentGas = multiChainGas[selectedChain.id]?.gasPrice || 0;
  const gasUnits = TX_GAS_ESTIMATES[selectedTxType];
  const estimatedCostEth = (currentGas * gasUnits) / 1e9;
  const estimatedCostUsd = estimatedCostEth * ethPrice;

  // Show chain switch toast when better chain available
  useEffect(() => {
    if (bestChainForTx && bestChainForTx.chainId !== selectedChain.id && bestChainForTx.savings > 20) {
      setShowChainToast(true);
      const timer = setTimeout(() => setShowChainToast(false), 10000);
      return () => clearTimeout(timer);
    }
  }, [bestChainForTx, selectedChain.id]);

  const fetchRecommendation = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await fetch(`${API_URL}/agent/recommend?urgency=${urgency}`, {
        signal: AbortSignal.timeout(10000)
      });
      const data = await response.json();

      if (data.success) {
        setRecommendation(data.recommendation);
        setError(null);

        if (data.recommendation.action === 'WAIT') {
          const waitMinutes = Math.round((1 - data.recommendation.confidence) * 60);
          setCountdown(waitMinutes * 60);
        } else {
          setCountdown(null);
        }
      } else {
        setError('Agent unavailable');
      }
    } catch (err) {
      console.error('Failed to fetch recommendation:', err);
      setError('Failed to connect to agent');
    } finally {
      setLoading(false);
    }
  }, [API_URL, urgency]);

  useEffect(() => {
    fetchRecommendation();
    const interval = setInterval(fetchRecommendation, 30000);
    return () => clearInterval(interval);
  }, [fetchRecommendation]);

  useEffect(() => {
    if (countdown === null || countdown <= 0) return;
    const timer = setInterval(() => {
      setCountdown(prev => (prev !== null && prev > 0 ? prev - 1 : null));
    }, 1000);
    return () => clearInterval(timer);
  }, [countdown]);

  const formatCountdown = (seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getActionConfig = (action: string) => {
    switch (action) {
      case 'WAIT':
        return {
          gradient: 'from-yellow-500 to-orange-500',
          cardClass: 'recommendation-card-yellow',
          text: 'Wait for Better Price',
          subtext: countdown ? `Optimal in ~${formatCountdown(countdown)}` : 'Prices expected to drop',
          buttonText: 'Notify When Ready',
          buttonClass: 'bg-yellow-500 hover:bg-yellow-600 btn-wait-glow',
          confidenceColor: '#eab308'
        };
      case 'SUBMIT_NOW':
        return {
          gradient: 'from-green-500 to-emerald-500',
          cardClass: 'recommendation-card-green',
          text: 'Execute Now',
          subtext: 'Good time to transact',
          buttonText: 'Execute Transaction',
          buttonClass: 'bg-green-500 hover:bg-green-600 btn-execute-glow',
          confidenceColor: '#22c55e'
        };
      case 'SUBMIT_LOW':
        return {
          gradient: 'from-cyan-500 to-blue-500',
          cardClass: 'recommendation-card-cyan',
          text: 'Try Lower Gas',
          subtext: 'Submit 10% below current (~15% fail risk)',
          buttonText: 'Execute Low',
          buttonClass: 'bg-cyan-500 hover:bg-cyan-600',
          confidenceColor: '#06b6d4'
        };
      case 'SUBMIT_HIGH':
        return {
          gradient: 'from-purple-500 to-pink-500',
          cardClass: 'recommendation-card-purple',
          text: 'Priority Submit',
          subtext: 'Faster confirmation guaranteed',
          buttonText: 'Execute Priority',
          buttonClass: 'bg-purple-500 hover:bg-purple-600',
          confidenceColor: '#a855f7'
        };
      default:
        return {
          gradient: 'from-gray-500 to-gray-600',
          cardClass: '',
          text: 'Analyzing...',
          subtext: 'Getting recommendation',
          buttonText: 'Wait',
          buttonClass: 'bg-gray-500',
          confidenceColor: '#6b7280'
        };
    }
  };

  const actionConfig = recommendation ? getActionConfig(recommendation.action) : getActionConfig('');

  const handleSwitchChain = () => {
    if (bestChainForTx) {
      setSelectedChainId(bestChainForTx.chainId);
      setShowChainToast(false);
    }
  };

  return (
    <div className="bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 rounded-2xl border border-gray-700/50 overflow-hidden card-interactive">
      {/* Chain Switch Toast - Improvement #18 */}
      {showChainToast && bestChainForTx && (
        <div className="absolute top-4 right-4 z-50 animate-slide-up">
          <div className="bg-green-500/20 border border-green-500/50 rounded-xl p-4 backdrop-blur-sm shadow-lg max-w-xs">
            <div className="flex items-start gap-3">
              <span className="text-2xl">💡</span>
              <div className="flex-1">
                <div className="font-semibold text-green-400 mb-1">
                  Save {bestChainForTx.savings.toFixed(0)}% on fees!
                </div>
                <div className="text-sm text-gray-300 mb-3">
                  Switch to {bestChainForTx.reason.split(' ')[0]} for cheaper transactions
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={handleSwitchChain}
                    className="px-3 py-1.5 bg-green-500 text-white text-sm rounded-lg hover:bg-green-600 transition-colors"
                  >
                    Switch Now
                  </button>
                  <button
                    onClick={() => setShowChainToast(false)}
                    className="px-3 py-1.5 bg-gray-700 text-gray-300 text-sm rounded-lg hover:bg-gray-600 transition-colors"
                  >
                    Dismiss
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Header */}
      <div className="px-6 py-4 border-b border-gray-700/50 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center shadow-lg shadow-purple-500/20">
            <span className="text-xl">🤖</span>
          </div>
          <div>
            <h2 className="text-lg font-bold text-white">AI Transaction Pilot</h2>
            <p className="text-xs text-gray-400">DQN Neural Network • Real-time optimization</p>
          </div>
        </div>
        <div className="text-right">
          <div className="text-sm text-gray-400">Current on {selectedChain.name}</div>
          <div className="text-lg font-mono font-bold text-cyan-400">
            {formatGwei(currentGas)} gwei
          </div>
        </div>
      </div>

      {/* Two-Column Layout - Improvement #4 */}
      <div className="flex flex-col lg:flex-row">
        {/* Left: Transaction Type Selector */}
        <div className="lg:w-1/3 px-6 py-4 border-b lg:border-b-0 lg:border-r border-gray-700/30">
          <div className="text-sm text-gray-400 mb-3">What do you want to do?</div>
          <div className="flex flex-wrap lg:flex-col gap-2">
            {TX_TYPES.map(({ type, label, icon }) => (
              <button
                key={type}
                onClick={() => setSelectedTxType(type)}
                className={`
                  px-4 py-2.5 rounded-lg flex items-center gap-2 transition-all w-full
                  ${selectedTxType === type
                    ? 'bg-cyan-500/20 border-2 border-cyan-500 text-cyan-400'
                    : 'bg-gray-800/50 border-2 border-transparent text-gray-300 hover:border-gray-600 hover:bg-gray-800'
                  }
                `}
              >
                <span className="text-lg">{icon}</span>
                <span className="font-medium">{label}</span>
              </button>
            ))}
          </div>

          {/* Urgency Slider */}
          <div className="mt-6">
            <div className="flex justify-between text-xs text-gray-400 mb-2">
              <span>Urgency</span>
              <span className="font-mono">{Math.round(urgency * 100)}%</span>
            </div>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={urgency}
              onChange={(e) => setUrgency(parseFloat(e.target.value))}
              className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-purple-500"
            />
            <div className="flex justify-between text-xs text-gray-500 mt-1">
              <span>💰 Save</span>
              <span>⚡ Fast</span>
            </div>
          </div>
        </div>

        {/* Right: Recommendation */}
        <div className="lg:w-2/3 p-6">
          {loading && !recommendation ? (
            <div className="h-48 flex items-center justify-center">
              <div className="text-center">
                <div className="w-12 h-12 border-3 border-gray-600 border-t-cyan-400 rounded-full animate-spin mx-auto mb-4" />
                <div className="text-gray-400">Analyzing network...</div>
              </div>
            </div>
          ) : error || !recommendation ? (
            <div className="relative rounded-xl p-6 bg-gradient-to-r from-gray-700/50 to-gray-800/50 border border-gray-600/50">
              <div className="text-center">
                <div className="text-2xl font-bold text-white mb-2">
                  {currentGas > 0 ? 'Gas Looks Good' : 'Connecting...'}
                </div>
                <div className="text-gray-400 mb-4">
                  {error || 'Agent is analyzing network conditions'}
                </div>
                <div className="grid grid-cols-2 gap-4 mb-4">
                  <div className="bg-black/20 rounded-lg p-3">
                    <div className="text-xs text-gray-500">Current Gas</div>
                    <div className="text-lg font-mono font-bold text-cyan-400">
                      {formatGwei(currentGas)} gwei
                    </div>
                  </div>
                  <div className="bg-black/20 rounded-lg p-3">
                    <div className="text-xs text-gray-500">Est. Cost</div>
                    <div className="text-lg font-mono font-bold text-white">
                      {formatUsd(estimatedCostUsd)}
                    </div>
                  </div>
                </div>
                <button
                  onClick={fetchRecommendation}
                  className="px-6 py-2 bg-cyan-500 hover:bg-cyan-600 text-white rounded-lg transition-colors"
                >
                  Retry Connection
                </button>
              </div>
            </div>
          ) : (
            <div className={`relative rounded-xl p-6 bg-gradient-to-r ${actionConfig.gradient} bg-opacity-10 ${actionConfig.cardClass}`}>
              {/* Confidence Ring - Improvement #9 */}
              <div className="absolute top-4 right-4">
                <ConfidenceRing
                  confidence={recommendation.confidence}
                  size={56}
                  color={actionConfig.confidenceColor}
                />
              </div>

              {/* Main action */}
              <div className="mb-4 pr-16">
                <div className="text-2xl lg:text-3xl font-bold text-white mb-1">{actionConfig.text}</div>
                <div className="text-white/70">{actionConfig.subtext}</div>
              </div>

              {/* Reasoning */}
              {recommendation.reasoning && (
                <div className="mb-4 p-3 bg-black/20 rounded-lg backdrop-blur-sm">
                  <div className="text-xs text-white/50 mb-1">🧠 Agent Reasoning</div>
                  <div className="text-sm text-white/80">{recommendation.reasoning}</div>
                </div>
              )}

              {/* Cost estimate - Improvement #7, #8 */}
              <div className="grid grid-cols-3 gap-3 mb-4">
                <div className="bg-black/20 rounded-lg p-3 backdrop-blur-sm">
                  <div className="text-xs text-white/50">Est. Cost</div>
                  <div className="text-lg font-mono font-bold text-white">
                    {formatUsd(estimatedCostUsd)}
                  </div>
                </div>
                <div className="bg-black/20 rounded-lg p-3 backdrop-blur-sm">
                  <div className="text-xs text-white/50">Gas Units</div>
                  <div className="text-lg font-mono font-bold text-white">
                    {gasUnits.toLocaleString()}
                  </div>
                </div>
                <div className="bg-black/20 rounded-lg p-3 backdrop-blur-sm">
                  <div className="text-xs text-white/50">Savings</div>
                  <div className="text-lg font-mono font-bold text-green-400">
                    {recommendation.expected_savings > 0
                      ? formatUsd(recommendation.expected_savings * estimatedCostUsd)
                      : '-'
                    }
                  </div>
                </div>
              </div>

              {/* Action buttons - Improvement #10 */}
              <div className="flex gap-3">
                <button className={`flex-1 py-3 px-6 rounded-xl font-bold text-white transition-all ${actionConfig.buttonClass}`}>
                  {actionConfig.buttonText}
                </button>
                <button
                  onClick={() => setShowScheduleModal(true)}
                  className="py-3 px-6 rounded-xl font-bold bg-white/10 text-white hover:bg-white/20 transition-all border border-white/10"
                >
                  ⏰ Schedule
                </button>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Footer */}
      <div className="px-6 py-3 bg-gray-800/50 border-t border-gray-700/30 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="text-xs text-gray-500">
            Updates every 30s • {selectedChain.name}
          </div>
          {(pendingCount > 0 || readyCount > 0) && (
            <div className="flex items-center gap-2">
              {pendingCount > 0 && (
                <span className="px-2 py-0.5 text-xs bg-yellow-500/20 text-yellow-400 rounded-full font-mono">
                  {pendingCount} scheduled
                </span>
              )}
              {readyCount > 0 && (
                <span className="px-2 py-0.5 text-xs bg-green-500/20 text-green-400 rounded-full animate-pulse font-mono">
                  {readyCount} ready!
                </span>
              )}
            </div>
          )}
        </div>
        <button
          onClick={fetchRecommendation}
          className="text-xs text-cyan-400 hover:text-cyan-300 transition-colors flex items-center gap-1"
        >
          <svg className="w-3 h-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
          </svg>
          Refresh
        </button>
      </div>

      {/* Schedule Modal */}
      <ScheduleTransactionModal
        isOpen={showScheduleModal}
        onClose={() => setShowScheduleModal(false)}
        defaultTxType={selectedTxType}
        suggestedTargetGas={currentGas * 0.85}
      />
    </div>
  );
};

export default TransactionPilot;
