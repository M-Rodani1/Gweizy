/**
 * Chain Comparison Component
 * Displays gas price predictions and comparisons across all supported chains
 */
import React, { useState, useEffect } from 'react';
import { fetchPredictions } from '../api/gasApi';
import { SUPPORTED_CHAINS, getChainById } from '../config/chains';
import { useChain } from '../contexts/ChainContext';
import LoadingSpinner from './LoadingSpinner';

interface ChainPrediction {
  chainId: number;
  chainName: string;
  currentGas: number;
  predictions: {
    '1h': number;
    '4h': number;
    '24h': number;
  };
  loading: boolean;
  error: string | null;
}

const ChainComparison: React.FC = () => {
  const { multiChainGas, selectedChainId, setSelectedChainId } = useChain();
  const [chainPredictions, setChainPredictions] = useState<Record<number, ChainPrediction>>({});
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadAllChainPredictions = async () => {
      setLoading(true);
      const predictions: Record<number, ChainPrediction> = {};

      // Load predictions for all enabled chains
      const enabledChains = Object.values(SUPPORTED_CHAINS).filter(c => c.enabled);
      
      await Promise.all(
        enabledChains.map(async (chain) => {
          try {
            const result = await fetchPredictions(chain.id);
            const currentGas = multiChainGas[chain.id]?.gasPrice || 0;
            
            const preds = {
              '1h': 0,
              '4h': 0,
              '24h': 0
            };

            (['1h', '4h', '24h'] as const).forEach((horizon) => {
              const data = result?.predictions?.[horizon];
              if (Array.isArray(data) && data.length > 0) {
                preds[horizon] = data[0].predictedGwei || 0;
              }
            });

            predictions[chain.id] = {
              chainId: chain.id,
              chainName: chain.name,
              currentGas,
              predictions: preds,
              loading: false,
              error: null
            };
          } catch (err) {
            predictions[chain.id] = {
              chainId: chain.id,
              chainName: chain.name,
              currentGas: multiChainGas[chain.id]?.gasPrice || 0,
              predictions: { '1h': 0, '4h': 0, '24h': 0 },
              loading: false,
              error: err instanceof Error ? err.message : 'Failed to load'
            };
          }
        })
      );

      setChainPredictions(predictions);
      setLoading(false);
    };

    loadAllChainPredictions();
    
    // Refresh every 60 seconds
    const interval = setInterval(loadAllChainPredictions, 60000);
    return () => clearInterval(interval);
  }, [multiChainGas]);

  if (loading) {
    return (
      <div className="bg-gray-800 rounded-lg p-6">
        <LoadingSpinner message="Loading chain comparisons..." />
      </div>
    );
  }

  const chains = Object.values(chainPredictions);
  const cheapest1h = chains.reduce((min, chain) => 
    chain.predictions['1h'] > 0 && (min.predictions['1h'] === 0 || chain.predictions['1h'] < min.predictions['1h'])
      ? chain : min, chains[0] || { predictions: { '1h': 0 } }
  );

  return (
    <div className="bg-gray-800 rounded-lg p-6">
      <h2 className="text-2xl font-bold text-white mb-4">Multi-Chain Gas Comparison</h2>
      <p className="text-gray-400 mb-6">Compare gas prices and predictions across all supported chains</p>

      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="border-b border-gray-700">
              <th className="text-left py-3 px-4 text-gray-300">Chain</th>
              <th className="text-right py-3 px-4 text-gray-300">Current</th>
              <th className="text-right py-3 px-4 text-gray-300">1h Prediction</th>
              <th className="text-right py-3 px-4 text-gray-300">4h Prediction</th>
              <th className="text-right py-3 px-4 text-gray-300">24h Prediction</th>
              <th className="text-center py-3 px-4 text-gray-300">Action</th>
            </tr>
          </thead>
          <tbody>
            {chains.map((chain) => {
              const isSelected = chain.chainId === selectedChainId;
              const isCheapest = chain.chainId === cheapest1h.chainId;
              
              return (
                <tr
                  key={chain.chainId}
                  className={`border-b border-gray-700 hover:bg-gray-700/50 transition ${
                    isSelected ? 'bg-blue-900/20' : ''
                  }`}
                >
                  <td className="py-3 px-4">
                    <div className="flex items-center gap-2">
                      <span className="font-semibold text-white">{chain.chainName}</span>
                      {isCheapest && (
                        <span className="text-xs bg-green-500/20 text-green-400 px-2 py-1 rounded">
                          Cheapest
                        </span>
                      )}
                    </div>
                  </td>
                  <td className="text-right py-3 px-4 text-white">
                    {chain.currentGas > 0 ? `${chain.currentGas.toFixed(4)} gwei` : '-'}
                  </td>
                  <td className="text-right py-3 px-4">
                    {chain.predictions['1h'] > 0 ? (
                      <span className={isCheapest ? 'text-green-400 font-semibold' : 'text-white'}>
                        {chain.predictions['1h'].toFixed(4)} gwei
                      </span>
                    ) : (
                      <span className="text-gray-500">-</span>
                    )}
                  </td>
                  <td className="text-right py-3 px-4 text-white">
                    {chain.predictions['4h'] > 0 ? `${chain.predictions['4h'].toFixed(4)} gwei` : '-'}
                  </td>
                  <td className="text-right py-3 px-4 text-white">
                    {chain.predictions['24h'] > 0 ? `${chain.predictions['24h'].toFixed(4)} gwei` : '-'}
                  </td>
                  <td className="text-center py-3 px-4">
                    {isSelected ? (
                      <span className="text-blue-400 text-sm">Selected</span>
                    ) : (
                      <button
                        onClick={() => setSelectedChainId(chain.chainId)}
                        className="text-blue-400 hover:text-blue-300 text-sm underline"
                      >
                        Select
                      </button>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      {chains.length === 0 && (
        <div className="text-center py-8 text-gray-400">
          No chain data available. Please ensure data collection is running.
        </div>
      )}
    </div>
  );
};

export default ChainComparison;

