import React from 'react';
import { Network } from 'lucide-react';
import { useChain, useChainComparison } from '../contexts/ChainContext';
import { TX_GAS_ESTIMATES, TransactionType } from '../config/chains';
import ChainBadge from './ChainBadge';
import Sparkline from './ui/Sparkline';
import { formatGwei, formatUsd } from '../utils/formatNumber';
import { SkeletonMultiChain } from './ui/Skeleton';
import AnimatedNumber from './ui/AnimatedNumber';

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

  // Show skeleton while initial data is loading
  if (isLoading && chainsWithCost.every(c => !c.gas?.gasPrice)) {
    return <SkeletonMultiChain count={chainComparison.length || 5} className="h-full" />;
  }

  return (
    <div className="bg-gray-800/50 border border-gray-700 rounded-2xl overflow-hidden h-full flex flex-col shadow-xl widget-glow w-full max-w-full">
      {/* Header */}
      <div className="px-6 py-4 border-b border-gray-700/50 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Network className="w-4 h-4 text-cyan-400" />
          <h3 className="font-semibold text-white">Multi-Chain Gas</h3>
        </div>
        {isLoading && (
          <div className="flex items-center gap-1.5">
            <div className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
            <span className="text-xs text-gray-400">Updating</span>
          </div>
        )}
      </div>

      {/* Chain List */}
      <div className="divide-y divide-gray-700/30 flex-1 overflow-y-auto">
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
                w-full px-6 py-4 flex items-center justify-between min-w-0
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
                <div className="flex items-center gap-2 min-w-0 flex-1">
                  <ChainBadge chain={chain} size="sm" />
                  <div className="text-left min-w-0 flex-1">
                    <div className="flex items-center gap-1 sm:gap-2 flex-wrap">
                      <span className="font-medium text-white text-sm sm:text-base truncate">{chain.name}</span>
                      {isCheapest && (
                        <span className="badge badge-success badge-pulse hidden sm:inline whitespace-nowrap">
                          Best
                        </span>
                      )}
                      {chain.isL2 && (
                        <span className="badge badge-info whitespace-nowrap">
                          L2
                        </span>
                      )}
                    </div>
                    <div className="text-[10px] sm:text-xs text-gray-400 font-mono truncate">
                      {formatGwei(gas?.gasPrice || 0)} gwei
                    </div>
                  </div>
                </div>
              </div>

              {/* Right side: Sparkline + Cost */}
              <div className="flex items-center gap-2 sm:gap-3 flex-shrink-0">
                {/* Mini sparkline showing trend - hidden on mobile */}
                <div className="hidden sm:block flex-shrink-0">
                  <Sparkline
                    data={sparklineData}
                    width={48}
                    height={20}
                    color={isCheapest ? '#22c55e' : '#06b6d4'}
                  />
                </div>

                {/* Cost */}
                <div className="text-right min-w-[55px] sm:min-w-[70px] flex-shrink-0">
                  <div className="font-mono font-bold text-white text-sm sm:text-base truncate">
                    {formatUsd(costUsd)}
                  </div>
                  {savingsPercent > 0 && !isCheapest && (
                    <div className="text-[10px] sm:text-xs text-red-400 font-mono">
                      +{savingsPercent.toFixed(0)}%
                    </div>
                  )}
                  {isCheapest && (
                    <div className="badge badge-success badge-pulse text-[10px] sm:text-xs">
                      Best
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
        <div className="px-6 py-4 bg-gray-800/50 border-t border-gray-700/30">
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-400">Max savings vs expensive:</span>
            <span className="font-mono font-bold text-green-400">
              $<AnimatedNumber value={mostExpensiveCost - cheapestCost} decimals={4} />
              {' '}(<AnimatedNumber
                value={mostExpensiveCost > 0 ? ((mostExpensiveCost - cheapestCost) / mostExpensiveCost * 100) : 0}
                decimals={0}
                suffix="%"
              />)
            </span>
          </div>
        </div>
      )}
    </div>
  );
};

export default MultiChainComparison;
