import React, { useState, useEffect } from 'react';
import { Bot, Brain, Clock, Coins, Lightbulb, RefreshCw, Zap } from 'lucide-react';
import { useChain } from '../contexts/ChainContext';
import { useScheduler } from '../contexts/SchedulerContext';
import { TransactionType } from '../config/chains';
import { TX_TYPE_META, getTxShortLabel } from '../config/transactions';
import { getActionConfig, getPrimaryActionLabel } from '../config/actionConfig';
import ScheduleTransactionModal from './ScheduleTransactionModal';
import ExecuteTransactionModal from './ExecuteTransactionModal';
import ConfidenceRing from './ui/ConfidenceRing';
import { formatGwei, formatUsd } from '../utils/formatNumber';
import { usePreferences } from '../contexts/PreferencesContext';
import { useWalletAddress } from '../hooks/useWalletAddress';
import { useRecommendation } from '../hooks/useRecommendation';
import { calculateGasCost } from '../utils/gasCalculations';

interface TransactionPilotProps {
  ethPrice?: number;
}

const TX_TYPES: { type: TransactionType; label: string; Icon: React.ComponentType<{ className?: string }> }[] = [
  { type: 'swap', label: getTxShortLabel('swap'), Icon: TX_TYPE_META.swap.icon },
  { type: 'bridge', label: getTxShortLabel('bridge'), Icon: TX_TYPE_META.bridge.icon },
  { type: 'nftMint', label: getTxShortLabel('nftMint'), Icon: TX_TYPE_META.nftMint.icon },
  { type: 'transfer', label: getTxShortLabel('transfer'), Icon: TX_TYPE_META.transfer.icon },
  { type: 'approve', label: getTxShortLabel('approve'), Icon: TX_TYPE_META.approve.icon },
];

const TransactionPilot: React.FC<TransactionPilotProps> = ({ ethPrice = 3000 }) => {
  const { selectedChain, multiChainGas, bestChainForTx, setSelectedChainId } = useChain();
  const { pendingCount, readyCount } = useScheduler();
  const { preferences, updatePreferences } = usePreferences();
  const walletAddress = useWalletAddress();
  const [selectedTxType, setSelectedTxType] = useState<TransactionType>(preferences.defaultTxType);
  const [urgency, setUrgency] = useState(preferences.urgency);
  const [showScheduleModal, setShowScheduleModal] = useState(false);
  const [showExecuteModal, setShowExecuteModal] = useState(false);
  const [executeGasGwei, setExecuteGasGwei] = useState<number | null>(null);
  const [showChainToast, setShowChainToast] = useState(false);

  // Use the recommendation hook
  const {
    recommendation,
    loading,
    loadingState,
    error,
    retryCount,
    countdown,
    refresh: fetchRecommendation
  } = useRecommendation(urgency);

  const currentGas = multiChainGas[selectedChain.id]?.gasPrice || 0;
  const { gasUnits, costUsd: estimatedCostUsd } = calculateGasCost(selectedTxType, currentGas, ethPrice);

  // Show chain switch toast when better chain available
  useEffect(() => {
    if (bestChainForTx && bestChainForTx.chainId !== selectedChain.id && bestChainForTx.savings > 20) {
      setShowChainToast(true);
      const timer = setTimeout(() => setShowChainToast(false), 10000);
      return () => clearTimeout(timer);
    }
    return undefined;
  }, [bestChainForTx, selectedChain.id]);

  useEffect(() => {
    setSelectedTxType(preferences.defaultTxType);
  }, [preferences.defaultTxType]);

  useEffect(() => {
    setUrgency(preferences.urgency);
  }, [preferences.urgency]);

  const txShortLabel = getTxShortLabel(selectedTxType);

  const actionConfig = recommendation
    ? getActionConfig(recommendation.action, countdown)
    : getActionConfig('');
  const primaryActionLabel = recommendation
    ? getPrimaryActionLabel(recommendation.action, txShortLabel)
    : actionConfig.buttonText;

  const handleTxTypeSelect = (type: TransactionType) => {
    setSelectedTxType(type);
    updatePreferences({ defaultTxType: type, strategy: 'custom' });
  };

  const handleUrgencyChange = (value: number) => {
    setUrgency(value);
    updatePreferences({ urgency: value, strategy: 'custom' });
  };

  const getActionGasGwei = () => {
    if (recommendation?.recommended_gas) {
      return recommendation.recommended_gas;
    }
    if (recommendation?.action === 'SUBMIT_LOW') {
      return currentGas * 0.9;
    }
    if (recommendation?.action === 'SUBMIT_HIGH') {
      return currentGas * 1.2;
    }
    return currentGas;
  };

  const handlePrimaryAction = () => {
    if (recommendation?.action === 'WAIT') {
      setShowScheduleModal(true);
      return;
    }
    const gas = getActionGasGwei();
    setExecuteGasGwei(gas > 0 ? gas : null);
    setShowExecuteModal(true);
  };

  const handleSwitchChain = () => {
    if (bestChainForTx) {
      setSelectedChainId(bestChainForTx.chainId);
      setShowChainToast(false);
    }
  };

  if (!walletAddress) {
    return (
      <div
        className="bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 rounded-2xl border border-gray-700/50 p-8 card-interactive focus-card"
        role="article"
        aria-label="Transaction pilot onboarding"
        tabIndex={0}
      >
        <div className="flex items-center gap-3 mb-3">
          <div className="w-10 h-10 rounded-xl bg-cyan-500/20 border border-cyan-500/40 flex items-center justify-center">
            <Bot className="w-5 h-5 text-cyan-300" />
          </div>
          <div>
            <h2 className="text-lg font-bold text-white">AI Transaction Pilot</h2>
            <p className="text-xs text-gray-400">Connect your wallet to get personalized actions</p>
          </div>
        </div>
        <div className="bg-black/30 border border-gray-700/60 rounded-xl p-4 mb-4">
          <ul className="space-y-2 text-sm text-gray-200">
            <li>• Get live recommendations for your next transaction</li>
            <li>• Auto-fill gas targets and scheduling options</li>
            <li>• Sync preferences across devices</li>
          </ul>
        </div>
        <div className="text-xs text-gray-400">
          <span>Tip: Use the top-right wallet button to link your account</span>
        </div>
      </div>
    );
  }

  return (
    <div
      className="bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900 rounded-2xl border border-gray-700/50 card-interactive focus-card"
      role="article"
      aria-label="AI transaction pilot"
      tabIndex={0}
    >
      {/* Chain Switch Toast - Improvement #18 */}
      {showChainToast && bestChainForTx && (
        <div className="absolute top-4 right-4 z-50 animate-slide-up">
          <div className="bg-green-500/20 border border-green-500/50 rounded-2xl p-6 backdrop-blur-sm shadow-xl max-w-xs">
            <div className="flex items-start gap-3">
              <Lightbulb className="w-6 h-6 text-green-300" />
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
      <div className="px-8 py-5 border-b border-gray-700/50 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-500 to-emerald-500 flex items-center justify-center shadow-lg shadow-cyan-500/20">
            <Bot className="w-5 h-5 text-white" />
          </div>
          <div>
            <h2 className="text-lg font-bold text-white">AI Transaction Pilot</h2>
            <p className="text-xs text-gray-400">DQN Neural Network • Real-time optimisation</p>
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
        <div className="lg:w-1/3 px-8 py-6 border-b lg:border-b-0 lg:border-r border-gray-700/30">
          <div className="text-sm text-gray-400 mb-3">Choose a transaction</div>
          <div className="flex flex-wrap lg:flex-col gap-2">
            {TX_TYPES.map(({ type, label, Icon }) => (
              <button
                key={type}
                onClick={() => handleTxTypeSelect(type)}
                className={`
                  px-4 py-3 min-h-[44px] rounded-lg flex items-center gap-2 transition-all w-full
                  ${selectedTxType === type
                    ? 'bg-cyan-500/20 border-2 border-cyan-500 text-cyan-400'
                    : 'bg-gray-800/50 border-2 border-transparent text-gray-300 hover:border-gray-600 hover:bg-gray-800'
                  }
                `}
              >
                <Icon className={`w-4 h-4 ${selectedTxType === type ? 'text-cyan-300' : 'text-gray-300'}`} />
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
              onChange={(e) => handleUrgencyChange(parseFloat(e.target.value))}
              className="w-full h-3 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-cyan-400"
            />
            <div className="flex justify-between text-xs text-gray-500 mt-1">
              <span className="flex items-center gap-1">
                <Coins className="w-3 h-3" />
                Save
              </span>
              <span className="flex items-center gap-1">
                <Zap className="w-3 h-3" />
                Fast
              </span>
            </div>
          </div>
        </div>

        {/* Right: Recommendation */}
        <div className="lg:w-2/3 p-8">
          {loading && !recommendation ? (
            <div className="h-48 flex items-center justify-center">
              <div className="text-center">
                <div className="w-12 h-12 border-3 border-gray-600 border-t-cyan-400 rounded-full animate-spin mx-auto mb-4" />
                <div className="text-gray-400">
                  {loadingState === 'timeout' ? 'Still analysing... (complex calculation)' : 'Analysing network...'}
                </div>
                {retryCount > 0 && (
                  <div className="text-xs text-gray-500 mt-2">Attempt {retryCount + 1}</div>
                )}
              </div>
            </div>
          ) : error || !recommendation ? (
            <div className="relative rounded-xl p-8 bg-gradient-to-r from-gray-700/50 to-gray-800/50 border border-gray-600/50">
              <div className="text-center">
                <div className="text-2xl font-bold text-white mb-2">
                  {loadingState === 'timeout' ? 'Agent Busy' :
                   loadingState === 'error' ? 'Using Live Data' :
                   currentGas > 0 ? 'Gas Looks Good' : 'Connecting...'}
                </div>
                <div className="text-gray-400 mb-4">
                  {loadingState === 'timeout' ? 'AI is analyzing 234k+ data points — this can take a moment' :
                   error || 'Agent is analyzing network conditions'}
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
                  disabled={loading}
                  className={`px-6 py-2 min-h-[44px] rounded-lg transition-colors flex items-center justify-center gap-2 mx-auto ${
                    loading ? 'bg-gray-600 cursor-not-allowed' : 'bg-cyan-500 hover:bg-cyan-600'
                  } text-white`}
                >
                  {loading ? (
                    <>
                      <RefreshCw className="w-4 h-4 animate-spin" />
                      Analysing...
                    </>
                  ) : (
                    <>
                      <RefreshCw className="w-4 h-4" />
                      {retryCount > 2 ? 'Try Again' : 'Refresh'}
                    </>
                  )}
                </button>
                {retryCount > 2 && (
                  <div className="text-xs text-gray-500 mt-3">
                    Tip: The agent works fine — it just needs time to crunch the data
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className={`relative rounded-xl p-8 bg-gradient-to-r ${actionConfig.gradient} bg-opacity-10 ${actionConfig.cardClass}`}>
              {/* Confidence Ring - Improvement #9 */}
              <div className="absolute top-4 right-4">
                <ConfidenceRing
                  confidence={recommendation.confidence}
                  size={56}
                  color={actionConfig.confidenceColor}
                />
              </div>

              {/* Main action */}
              <div className="mb-6 pr-16">
                <div className="text-2xl lg:text-3xl font-bold text-white mb-2">{actionConfig.text}</div>
                <div className="text-white/70">{actionConfig.subtext}</div>
              </div>

              {/* Reasoning */}
              {recommendation.reasoning && (
                <div className="mb-6 p-4 bg-black/20 rounded-lg backdrop-blur-sm">
                  <div className="text-xs text-white/50 mb-1 flex items-center gap-2">
                    <Brain className="w-3 h-3" />
                    Agent Reasoning
                  </div>
                  <div className="text-sm text-white/80">{recommendation.reasoning}</div>
                </div>
              )}

              {/* Cost estimate - Improvement #7, #8 */}
              <div className="grid grid-cols-3 gap-4 mb-6">
                <div className="bg-black/20 rounded-lg p-4 backdrop-blur-sm">
                  <div className="text-xs text-white/50 mb-1">Est. Cost</div>
                  <div className="text-lg font-mono font-bold text-white">
                    {formatUsd(estimatedCostUsd)}
                  </div>
                </div>
                <div className="bg-black/20 rounded-lg p-4 backdrop-blur-sm">
                  <div className="text-xs text-white/50 mb-1">Gas Units</div>
                  <div className="text-lg font-mono font-bold text-white">
                    {gasUnits.toLocaleString()}
                  </div>
                </div>
                <div className="bg-black/20 rounded-lg p-4 backdrop-blur-sm">
                  <div className="text-xs text-white/50 mb-1">Savings</div>
                  <div className="text-lg font-mono font-bold text-green-400">
                    {recommendation.expected_savings > 0
                      ? formatUsd(recommendation.expected_savings * estimatedCostUsd)
                      : '-'
                    }
                  </div>
                </div>
              </div>

              {/* Action buttons - Improvement #10 */}
              <div className="flex flex-col sm:flex-row gap-4 sm:static sticky bottom-4 bg-gray-900/80 p-4 rounded-xl border border-white/10 backdrop-blur">
                <button
                  onClick={handlePrimaryAction}
                  className={`flex-1 py-3 px-6 min-h-[44px] rounded-xl font-bold text-white transition-all ${actionConfig.buttonClass}`}
                >
                  {primaryActionLabel}
                </button>
                <button
                  onClick={() => setShowScheduleModal(true)}
                  className="py-3 px-6 min-h-[44px] rounded-xl font-bold bg-white/10 text-white hover:bg-white/20 transition-all border border-white/10 flex items-center justify-center gap-2"
                >
                  <Clock className="w-4 h-4" />
                  {`Schedule ${txShortLabel}`}
                </button>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Footer */}
      <div className="px-8 py-4 bg-gray-800/50 border-t border-gray-700/30 flex items-center justify-between">
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
          <RefreshCw className="w-3 h-3" />
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

      <ExecuteTransactionModal
        isOpen={showExecuteModal}
        onClose={() => setShowExecuteModal(false)}
        chainId={selectedChain.id}
        txType={selectedTxType}
        gasGwei={executeGasGwei}
      />
    </div>
  );
};

export default TransactionPilot;
