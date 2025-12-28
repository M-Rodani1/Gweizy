import React from 'react';
import { useChain, useChainComparison } from '../contexts/ChainContext';
import { TX_GAS_ESTIMATES, TransactionType } from '../config/chains';

interface MultiChainComparisonProps {
  txType?: TransactionType;
  ethPrice?: number;
}

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

          return (
            <button
              key={chain.id}
              onClick={() => setSelectedChainId(chain.id)}
              className={`
                w-full px-4 py-3 flex items-center justify-between
                hover:bg-gray-700/30 transition-colors
                ${isSelected ? 'bg-gray-700/50' : ''}
              `}
            >
              <div className="flex items-center gap-3">
                {/* Rank */}
                <div className={`
                  w-6 h-6 rounded-full flex items-center justify-center text-xs font-bold
                  ${isCheapest ? 'bg-green-500 text-white' : 'bg-gray-700 text-gray-400'}
                `}>
                  {index + 1}
                </div>

                {/* Chain info */}
                <div className="flex items-center gap-2">
                  <span className="text-lg">{chain.icon}</span>
                  <div className="text-left">
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-white">{chain.name}</span>
                      {isCheapest && (
                        <span className="px-1.5 py-0.5 text-xs bg-green-500/20 text-green-400 rounded">
                          Best
                        </span>
                      )}
                      {chain.isL2 && (
                        <span className="px-1.5 py-0.5 text-xs bg-purple-500/20 text-purple-400 rounded">
                          L2
                        </span>
                      )}
                    </div>
                    <div className="text-xs text-gray-500">
                      {gas?.gasPrice?.toFixed(4) || '...'} gwei
                    </div>
                  </div>
                </div>
              </div>

              {/* Cost */}
              <div className="text-right">
                <div className="font-mono font-bold text-white">
                  ${costUsd.toFixed(4)}
                </div>
                {savingsPercent > 0 && !isCheapest && (
                  <div className="text-xs text-red-400">
                    +{savingsPercent.toFixed(0)}% more
                  </div>
                )}
                {isCheapest && index > 0 && (
                  <div className="text-xs text-green-400">
                    Cheapest
                  </div>
                )}
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
