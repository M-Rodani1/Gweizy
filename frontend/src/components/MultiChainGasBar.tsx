import React from 'react';
import { useChain, useChainComparison } from '../contexts/ChainContext';

interface MultiChainGasBarProps {
  onChainSelect?: (chainId: number) => void;
}

const MultiChainGasBar: React.FC<MultiChainGasBarProps> = ({ onChainSelect }) => {
  const { selectedChainId, setSelectedChainId, bestChainForTx, isLoading } = useChain();
  const chainComparison = useChainComparison();

  const handleClick = (chainId: number) => {
    setSelectedChainId(chainId);
    onChainSelect?.(chainId);
  };

  const formatGasPrice = (price: number): string => {
    if (price >= 100) return price.toFixed(0);
    if (price >= 1) return price.toFixed(2);
    if (price >= 0.01) return price.toFixed(4);
    return price.toFixed(6);
  };

  // Find min and max for relative comparison
  const gasPrices = chainComparison.map(c => c.gas?.gasPrice || 0).filter(p => p > 0);
  const minGas = Math.min(...gasPrices);
  const maxGas = Math.max(...gasPrices);

  const getBarColor = (price: number): string => {
    if (maxGas === minGas) return 'bg-cyan-500';
    const ratio = (price - minGas) / (maxGas - minGas);
    if (ratio < 0.33) return 'bg-green-500';
    if (ratio < 0.66) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  return (
    <div className="bg-gray-800/50 border border-gray-700 rounded-xl p-4">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium text-gray-300">Multi-Chain Gas Comparison</h3>
        {isLoading && (
          <div className="w-4 h-4 border-2 border-gray-500 border-t-cyan-400 rounded-full animate-spin" />
        )}
      </div>

      <div className="grid grid-cols-5 gap-2">
        {chainComparison.map(({ chain, gas }) => {
          const isSelected = chain.id === selectedChainId;
          const isBest = bestChainForTx?.chainId === chain.id;
          const gasPrice = gas?.gasPrice || 0;

          return (
            <button
              key={chain.id}
              onClick={() => handleClick(chain.id)}
              className={`
                relative p-3 rounded-lg transition-all duration-200
                ${isSelected
                  ? 'bg-gray-700 ring-2 ring-cyan-500'
                  : 'bg-gray-800 hover:bg-gray-700'}
              `}
            >
              {/* Best indicator */}
              {isBest && (
                <div className="absolute -top-1 -right-1 w-4 h-4 bg-green-500 rounded-full flex items-center justify-center">
                  <span className="text-xs">✓</span>
                </div>
              )}

              {/* Chain icon and name */}
              <div className="text-center mb-2">
                <div className="text-xl mb-1">{chain.icon}</div>
                <div className="text-xs text-gray-400">{chain.shortName}</div>
              </div>

              {/* Gas price */}
              <div className="text-center">
                {gas?.loading ? (
                  <div className="w-3 h-3 border border-gray-500 border-t-cyan-400 rounded-full animate-spin mx-auto" />
                ) : (
                  <>
                    <div className={`font-mono text-sm ${isSelected ? 'text-cyan-400' : 'text-white'}`}>
                      {formatGasPrice(gasPrice)}
                    </div>
                    <div className="text-xs text-gray-500">gwei</div>
                  </>
                )}
              </div>

              {/* Relative bar */}
              {!gas?.loading && gasPrice > 0 && (
                <div className="mt-2 h-1 bg-gray-700 rounded-full overflow-hidden">
                  <div
                    className={`h-full ${getBarColor(gasPrice)} transition-all duration-300`}
                    style={{
                      width: `${maxGas > 0 ? Math.max(10, (gasPrice / maxGas) * 100) : 0}%`
                    }}
                  />
                </div>
              )}
            </button>
          );
        })}
      </div>

      {/* Best chain recommendation */}
      {bestChainForTx && bestChainForTx.savings > 5 && (
        <div className="mt-3 p-2 bg-green-500/10 border border-green-500/30 rounded-lg">
          <div className="text-xs text-green-400 text-center">
            Save up to {bestChainForTx.savings.toFixed(0)}% by using {bestChainForTx.reason.split(' ')[0]}
          </div>
        </div>
      )}
    </div>
  );
};

export default MultiChainGasBar;
