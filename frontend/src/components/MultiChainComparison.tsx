import React from 'react';
import { useChain, useChainComparison } from '../contexts/ChainContext';
import { TX_GAS_ESTIMATES, TransactionType } from '../config/chains';
import Sparkline from './ui/Sparkline';
import { formatGwei, formatUsd } from '../utils/formatNumber';

interface MultiChainComparisonProps {
  txType?: TransactionType;
  ethPrice?: number;
}

// Chain gradient icon mapping
const CHAIN_ICON_CLASS: Record<number, string> = {
  1: 'chain-icon-eth',      // Ethereum
  8453: 'chain-icon-base',  // Base
  42161: 'chain-icon-arb',  // Arbitrum
  10: 'chain-icon-op',      // Optimism
  137: 'chain-icon-poly',   // Polygon
};

const MultiChainComparison: React.FC<MultiChainComparisonProps> = ({
  txType = 'swap',
  ethPrice = 3000
}) => {
  const { selectedChainId, setSelectedChainId, bestChainForTx, isLoading } = useChain();
  const chainComparison = useChainComparison();

  const gasUnits = TX_GAS_ESTIMATES[txType];

  // Calculate costs for each chain
  const chainsWithCost = chainComparison.map(({ chain, gas }) => {
    const costEth = ((gas?.gasPrice || 0) * gasUnits) / 1e9;
    const costUsd = costEth * ethPrice;
    return { chain, gas, costEth, costUsd };
  }).sort((a, b) => a.costUsd - b.costUsd);

  const cheapestCost = chainsWithCost[0]?.costUsd || 0;
  const mostExpensiveCost = chainsWithCost[chainsWithCost.length - 1]?.costUsd || 0;

  return (
    <div className="bg-gray-800/50 border border-gray-700 rounded-xl overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 border-b border-gray-700/50 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-lg">⛽</span>
          <h3 className="font-semibold text-white">Multi-Chain Gas</h3>
        </div>
        {isLoading && (
          <div className="w-4 h-4 border-2 border-gray-500 border-t-cyan-400 rounded-full animate-spin" />
        )}
      </div>

      {/* Chain List */}
      <div className="divide-y divide-gray-700/30">
        {chainsWithCost.map(({ chain, gas, costUsd }, index) => {
          const isSelected = chain.id === selectedChainId;
          const isCheapest = index === 0;
          const savingsPercent = mostExpensiveCost > 0
            ? ((mostExpensiveCost - costUsd) / mostExpensiveCost) * 100
            : 0;

          // Generate mock sparkline data based on gas price
          const sparklineData = Array.from({ length: 12 }, (_, i) => {
            const basePrice = gas?.gasPrice || 0.001;
            const variance = basePrice * 0.2;
            return basePrice + (Math.random() - 0.5) * variance;
          });

          return (
            <button
              key={chain.id}
              onClick={() => setSelectedChainId(chain.id)}
              className={`
                w-full px-4 py-3 flex items-center justify-between
                hover:bg-gray-700/30 transition-all card-interactive
                ${isSelected ? 'bg-gray-700/50 ring-1 ring-cyan-500/30' : ''}
              `}
            >
              <div className="flex items-center gap-3">
                {/* Rank with gradient */}
                <div className={`
                  w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold
                  ${isCheapest ? 'bg-gradient-to-br from-green-400 to-emerald-600 text-white shadow-lg shadow-green-500/30' : 'bg-gray-700 text-gray-400'}
                `}>
                  {index + 1}
                </div>

                {/* Chain info with gradient icon */}
                <div className="flex items-center gap-2">
                  <div className={`chain-icon ${CHAIN_ICON_CLASS[chain.id] || 'bg-gray-600'}`}>
                    <span className="text-sm">{chain.icon}</span>
                  </div>
                  <div className="text-left">
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-white">{chain.name}</span>
                      {isCheapest && (
                        <span className="px-1.5 py-0.5 text-xs bg-green-500/20 text-green-400 rounded font-semibold">
                          Best
                        </span>
                      )}
                      {chain.isL2 && (
                        <span className="px-1.5 py-0.5 text-xs bg-purple-500/20 text-purple-400 rounded">
                          L2
                        </span>
                      )}
                    </div>
                    <div className="text-xs text-gray-400 font-mono">
                      {formatGwei(gas?.gasPrice || 0)} gwei
                    </div>
                  </div>
                </div>
              </div>

              {/* Right side: Sparkline + Cost */}
              <div className="flex items-center gap-3">
                {/* Mini sparkline showing trend */}
                <Sparkline
                  data={sparklineData}
                  width={48}
                  height={20}
                  color={isCheapest ? '#22c55e' : '#06b6d4'}
                />

                {/* Cost */}
                <div className="text-right min-w-[70px]">
                  <div className="font-mono font-bold text-white">
                    {formatUsd(costUsd)}
                  </div>
                  {savingsPercent > 0 && !isCheapest && (
                    <div className="text-xs text-red-400 font-mono">
                      +{savingsPercent.toFixed(0)}% more
                    </div>
                  )}
                  {isCheapest && (
                    <div className="text-xs text-green-400 font-semibold">
                      Cheapest
                    </div>
                  )}
                </div>
              </div>
            </button>
          );
        })}
      </div>

      {/* Summary */}
      {chainsWithCost.length >= 2 && (
        <div className="px-4 py-3 bg-gray-800/50 border-t border-gray-700/30">
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-400">Max savings vs expensive:</span>
            <span className="font-mono font-bold text-green-400">
              ${(mostExpensiveCost - cheapestCost).toFixed(4)}
              ({((mostExpensiveCost - cheapestCost) / mostExpensiveCost * 100).toFixed(0)}%)
            </span>
          </div>
        </div>
      )}
    </div>
  );
};

export default MultiChainComparison;
