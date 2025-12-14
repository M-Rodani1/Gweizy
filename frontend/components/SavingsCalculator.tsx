import React, { useState } from 'react';

interface TransactionType {
  name: string;
  gasLimit: number;
}

const TRANSACTION_TYPES: TransactionType[] = [
  { name: 'Simple Transfer', gasLimit: 21000 },
  { name: 'ERC20 Transfer', gasLimit: 65000 },
  { name: 'Uniswap Swap', gasLimit: 150000 },
  { name: 'NFT Mint', gasLimit: 100000 },
  { name: 'Contract Deploy', gasLimit: 500000 }
];

interface SavingsCalculatorProps {
  currentGas: number;
  predictions: {
    '1h': number;
    '4h': number;
    '24h': number;
  };
  ethPrice?: number; // default 3000
}

const SavingsCalculator: React.FC<SavingsCalculatorProps> = ({
  currentGas,
  predictions,
  ethPrice = 3000
}) => {
  const [selectedType, setSelectedType] = useState<TransactionType>(
    TRANSACTION_TYPES.find(t => t.name === 'Uniswap Swap') || TRANSACTION_TYPES[2]
  );

  // Find the best (lowest) prediction across all horizons
  const findBestPrediction = () => {
    let bestGas = currentGas;
    let bestHorizon = 'now';
    
    (['1h', '4h', '24h'] as const).forEach((horizon) => {
      const predicted = predictions[horizon];
      if (predicted && predicted < bestGas) {
        bestGas = predicted;
        bestHorizon = horizon;
      }
    });
    
    return { bestGas, bestHorizon };
  };

  const { bestGas, bestHorizon } = findBestPrediction();

  const calculateCost = (gasPrice: number, gasLimit: number): number => {
    // Convert gwei to ETH: 1 gwei = 0.000000001 ETH
    const ethCost = (gasPrice * gasLimit) / 1e9;
    return ethCost * ethPrice; // Convert to USD
  };

  const costNow = calculateCost(currentGas, selectedType.gasLimit);
  const costBest = calculateCost(bestGas, selectedType.gasLimit);
  const savings = costNow - costBest;
  const savingsPercent = costNow > 0 ? (savings / costNow) * 100 : 0;

  const formatHorizon = (horizon: string) => {
    if (horizon === 'now') return 'NOW';
    return horizon.toUpperCase();
  };

  const formatTime = (horizon: string) => {
    if (horizon === 'now') return 'now';
    if (horizon === '1h') return '1 hour';
    if (horizon === '4h') return '4 hours';
    if (horizon === '24h') return '24 hours';
    return horizon;
  };

  return (
    <div className="bg-gray-800 p-6 rounded-lg shadow-lg border border-gray-700">
      {/* Header */}
      <div className="flex items-center mb-6 pb-4 border-b border-gray-700">
        <span className="text-2xl mr-2">💰</span>
        <h3 className="text-xl font-bold text-gray-200">SAVINGS CALCULATOR</h3>
      </div>

      {/* Transaction Type Dropdown */}
      <div className="mb-6">
        <label className="block text-sm font-medium text-gray-400 mb-2">
          Transaction Type:
        </label>
        <select
          value={selectedType.name}
          onChange={(e) => {
            const type = TRANSACTION_TYPES.find((t) => t.name === e.target.value);
            if (type) setSelectedType(type);
          }}
          className="w-full bg-gray-700 text-gray-200 border border-gray-600 rounded-md px-4 py-3 text-base focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:border-cyan-500"
        >
          {TRANSACTION_TYPES.map((type) => (
            <option key={type.name} value={type.name}>
              {type.name}
            </option>
          ))}
        </select>
      </div>

      {/* Divider */}
      <div className="mb-6 border-b border-gray-700"></div>

      {/* Cost Comparison */}
      <div className="space-y-4 mb-6">
        <div className="bg-gray-700/50 rounded-lg p-4">
          <div className="text-sm text-gray-400 mb-2">Cost if you transact NOW:</div>
          <div className="text-2xl font-bold text-gray-200">${costNow.toFixed(4)}</div>
        </div>

        <div className="bg-gray-700/50 rounded-lg p-4">
          <div className="text-sm text-gray-400 mb-2">
            Cost if you wait {formatTime(bestHorizon).toUpperCase()}:
          </div>
          <div className="text-2xl font-bold text-green-400">${costBest.toFixed(4)}</div>
        </div>
      </div>

      {/* Divider */}
      <div className="mb-6 border-b border-gray-700"></div>

      {/* Savings Display */}
      <div className="bg-gradient-to-r from-green-500/20 to-cyan-500/20 rounded-lg p-5 border-2 border-green-500/50">
        <div className="mb-3">
          <div className="text-lg font-semibold text-green-400 mb-1">💵 YOU SAVE:</div>
          <div className="text-3xl font-bold text-green-400">
            ${savings.toFixed(4)} ({savingsPercent.toFixed(0)}%)
          </div>
        </div>
        {bestHorizon !== 'now' && (
          <div className="text-sm text-gray-300 mt-2">
            By waiting {formatTime(bestHorizon)}
          </div>
        )}
      </div>
    </div>
  );
};

export default SavingsCalculator;

