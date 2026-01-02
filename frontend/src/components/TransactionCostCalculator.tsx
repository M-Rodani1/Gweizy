import React, { useState } from 'react';
import { ArrowLeftRight, Calculator, Coins, FileCode, Image, Lightbulb, Send, Sparkles, Zap } from 'lucide-react';

interface TransactionType {
  name: string;
  gasEstimate: number;
  Icon: React.ComponentType<{ className?: string }>;
}

const TRANSACTION_TYPES: TransactionType[] = [
  { name: 'Token Transfer', gasEstimate: 21000, Icon: Send },
  { name: 'Token Swap', gasEstimate: 150000, Icon: ArrowLeftRight },
  { name: 'NFT Mint', gasEstimate: 100000, Icon: Sparkles },
  { name: 'NFT Transfer', gasEstimate: 80000, Icon: Image },
  { name: 'Contract Deploy', gasEstimate: 500000, Icon: FileCode },
  { name: 'Token Approve', gasEstimate: 46000, Icon: Coins }
];

interface TransactionCostCalculatorProps {
  currentGas: number;
  ethPrice?: number;
}

const TransactionCostCalculator: React.FC<TransactionCostCalculatorProps> = ({
  currentGas,
  ethPrice = 3000
}) => {
  const [selectedTx, setSelectedTx] = useState(TRANSACTION_TYPES[0]);

  const calculateCost = () => {
    if (currentGas === 0) return { eth: 0, usd: 0 };

    // Cost in ETH = (gas used * gas price in gwei) / 1e9
    const costInEth = (selectedTx.gasEstimate * currentGas) / 1e9;
    const costInUsd = costInEth * ethPrice;

    return { eth: costInEth, usd: costInUsd };
  };

  const cost = calculateCost();

  const getPotentialSavings = () => {
    if (currentGas === 0) return 0;

    // Calculate potential savings if gas drops to 80% of current
    const lowerGasPrice = currentGas * 0.8;
    const currentCostUsd = cost.usd;
    const lowerCostUsd = ((selectedTx.gasEstimate * lowerGasPrice) / 1e9) * ethPrice;

    return currentCostUsd - lowerCostUsd;
  };

  return (
    <div className="bg-gradient-to-br from-slate-800/50 to-slate-900/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6 shadow-xl">
      {/* Header */}
      <div className="flex items-center gap-3 mb-6">
        <div className="p-2 bg-purple-500/20 rounded-lg">
          <Calculator className="w-5 h-5 text-purple-400" />
        </div>
        <div>
          <h3 className="text-lg font-bold text-white">Transaction Cost Calculator</h3>
          <p className="text-xs text-gray-400">Estimate costs for common transactions</p>
        </div>
      </div>

      {/* Transaction Type Selector */}
      <div className="grid grid-cols-2 md:grid-cols-3 gap-3 mb-6">
        {TRANSACTION_TYPES.map(tx => (
          <button
            key={tx.name}
            onClick={() => setSelectedTx(tx)}
            className={`p-4 rounded-lg border transition-all ${
              selectedTx.name === tx.name
                ? 'bg-purple-500/20 border-purple-500/50 text-white shadow-lg shadow-purple-500/20'
                : 'bg-slate-700/30 border-slate-600 text-gray-400 hover:border-slate-500'
            }`}
          >
            <div className="flex justify-center mb-2">
              <tx.Icon className="w-6 h-6" />
            </div>
            <div className="text-sm font-medium">{tx.name}</div>
            <div className="text-xs text-gray-500 mt-1">
              {(tx.gasEstimate / 1000).toFixed(0)}k gas
            </div>
          </button>
        ))}
      </div>

      {/* Cost Display */}
      <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg p-6 mb-4">
        <div className="text-center">
          <div className="flex items-center justify-center gap-2 mb-2">
            <Zap className="w-4 h-4 text-purple-400" />
            <div className="text-sm text-gray-400">Estimated Cost</div>
          </div>

          {currentGas > 0 ? (
            <>
              <div className="text-4xl font-bold text-white mb-2">
                ${cost.usd.toFixed(4)}
              </div>
              <div className="text-sm text-gray-400 mb-4">
                {cost.eth.toFixed(8)} ETH • {currentGas.toFixed(4)} gwei
              </div>

              {/* Breakdown */}
              <div className="grid grid-cols-2 gap-3 text-sm">
                <div className="bg-slate-700/30 rounded-lg p-3 border border-slate-600">
                  <div className="text-gray-400 mb-1">Gas Used</div>
                  <div className="font-bold text-white">
                    {selectedTx.gasEstimate.toLocaleString()}
                  </div>
                </div>
                <div className="bg-slate-700/30 rounded-lg p-3 border border-slate-600">
                  <div className="text-gray-400 mb-1">ETH Price</div>
                  <div className="font-bold text-white">
                    ${ethPrice.toLocaleString()}
                  </div>
                </div>
              </div>
            </>
          ) : (
            <div className="text-gray-400 py-4">
              Waiting for gas price data...
            </div>
          )}
        </div>
      </div>

      {/* Savings Tip */}
      {currentGas > 0 && getPotentialSavings() > 0.001 && (
        <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-4">
          <div className="flex items-start gap-3">
            <Lightbulb className="w-5 h-5 text-green-400 mt-0.5" />
            <div>
              <div className="text-sm font-semibold text-green-400 mb-1">Potential Savings</div>
              <div className="text-xs text-gray-300">
                You could save ~${getPotentialSavings().toFixed(4)} if gas drops by 20%
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default TransactionCostCalculator;
