import React, { useState, useEffect } from 'react';
import { AlertTriangle, Coins, TrendingDown } from 'lucide-react';
import { fetchHistoricalData } from '../api/gasApi';

interface GasWasteCalculatorProps {
  walletAddress?: string | null;
}

const TRANSACTION_TYPES = [
  { name: 'Simple Transfer', gasLimit: 21000 },
  { name: 'ERC20 Transfer', gasLimit: 65000 },
  { name: 'Uniswap Swap', gasLimit: 150000 },
  { name: 'NFT Mint', gasLimit: 100000 },
  { name: 'Contract Deploy', gasLimit: 500000 }
];

const ETH_PRICE = 3000; // USD per ETH

const GasWasteCalculator: React.FC<GasWasteCalculatorProps> = ({ walletAddress }) => {
  const [timePeriod, setTimePeriod] = useState<'week' | 'month' | '3months'>('month');
  const [transactionsPerWeek, setTransactionsPerWeek] = useState<number>(5);
  const [selectedType, setSelectedType] = useState(TRANSACTION_TYPES[2]); // Uniswap Swap default
  const [historicalData, setHistoricalData] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadData = async () => {
      try {
        setLoading(true);
        setError(null);

        const hours = timePeriod === 'week' ? 168 : timePeriod === 'month' ? 720 : 2160;
        const data = await fetchHistoricalData(hours);
        setHistoricalData(data.data || []);
      } catch (err) {
        console.error('Error loading historical data:', err);
        setError(err instanceof Error ? err.message : 'Failed to load data');
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, [timePeriod]);

  const calculateWaste = () => {
    if (!historicalData || historicalData.length === 0) {
      return {
        avgGasPaid: 0,
        optimizedGasCost: 0,
        waste: 0,
        wastePercent: 0,
        annualWaste: 0
      };
    }

    // Calculate average gas price over period - with safety checks
    const gasPrices = historicalData && historicalData.length > 0
      ? historicalData.map(d => d.gwei || 0).filter(p => p > 0)
      : [];
    if (!gasPrices || gasPrices.length === 0) return { avgGasPaid: 0, optimizedGasCost: 0, waste: 0, wastePercent: 0, annualWaste: 0 };

    const avgGasPrice = gasPrices.length > 0
      ? gasPrices.reduce((a, b) => a + b, 0) / gasPrices.length
      : 0;

    // Find optimal (lowest) gas prices (bottom 20th percentile)
    const sortedPrices = gasPrices && gasPrices.length > 0 ? [...gasPrices].sort((a, b) => a - b) : [];
    const optimalGasPrice = sortedPrices && sortedPrices.length > 0
      ? sortedPrices[Math.floor(sortedPrices.length * 0.2)]
      : 0; // 20th percentile

    // Calculate costs
    const costPerTx = (gasPrice: number) => {
      const ethCost = (gasPrice * selectedType.gasLimit) / 1e9;
      return ethCost * ETH_PRICE;
    };

    const avgCostPerTx = costPerTx(avgGasPrice);
    const optimalCostPerTx = costPerTx(optimalGasPrice);

    // Calculate for the period
    const days = timePeriod === 'week' ? 7 : timePeriod === 'month' ? 30 : 90;
    const transactionsInPeriod = (transactionsPerWeek / 7) * days;
    
    const totalPaid = avgCostPerTx * transactionsInPeriod;
    const totalOptimal = optimalCostPerTx * transactionsInPeriod;
    const waste = totalPaid - totalOptimal;
    const wastePercent = totalPaid > 0 ? (waste / totalPaid) * 100 : 0;

    // Annual projection
    const annualTransactions = transactionsPerWeek * 52;
    const annualPaid = avgCostPerTx * annualTransactions;
    const annualOptimal = optimalCostPerTx * annualTransactions;
    const annualWaste = annualPaid - annualOptimal;

    return {
      avgGasPaid: totalPaid,
      optimizedGasCost: totalOptimal,
      waste,
      wastePercent,
      annualWaste
    };
  };

  const results = calculateWaste();
  const coffeePrice = 5; // $5 per coffee
  const coffeesWasted = Math.floor(results.waste / coffeePrice);

  const formatUSD = (amount: number) => amount.toFixed(2);
  const formatPeriod = () => {
    if (timePeriod === 'week') return 'Last 7 days';
    if (timePeriod === 'month') return 'Last 30 days';
    return 'Last 90 days';
  };

  if (loading && historicalData.length === 0) {
    return (
      <div className="bg-gray-800 p-6 rounded-2xl shadow-xl border border-gray-700">
        <h3 className="text-lg font-semibold text-gray-200 mb-4 flex items-center gap-2">
          <Coins className="w-5 h-5 text-cyan-400" />
          Gas Waste Calculator
        </h3>
        <div className="text-gray-400 text-sm">Loading historical data...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-gray-800 p-6 rounded-2xl shadow-xl border border-gray-700">
        <h3 className="text-lg font-semibold text-gray-200 mb-4 flex items-center gap-2">
          <Coins className="w-5 h-5 text-cyan-400" />
          Gas Waste Calculator
        </h3>
        <p className="text-red-400 text-sm mb-2 flex items-center gap-2">
          <AlertTriangle className="w-4 h-4" />
          {error}
        </p>
      </div>
    );
  }

  return (
    <div className="bg-gray-800 p-6 rounded-2xl shadow-xl border border-gray-700">
      <h3 className="text-lg font-semibold text-gray-200 mb-4 flex items-center gap-2">
        <Coins className="w-5 h-5 text-cyan-400" />
        Gas Waste Calculator
      </h3>

      {/* Time Period Selector */}
      <div className="mb-4">
        <label className="block text-sm text-gray-400 mb-2">Time Period</label>
        <div className="flex space-x-2">
          {(['week', 'month', '3months'] as const).map((period) => (
            <button
              key={period}
              onClick={() => setTimePeriod(period)}
              className={`px-3 py-2 text-sm rounded-md transition-colors ${
                timePeriod === period
                  ? 'bg-cyan-500 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              {period === 'week' ? 'Last Week' : period === 'month' ? 'Last Month' : 'Last 3 Months'}
            </button>
          ))}
        </div>
      </div>

      {/* User Inputs */}
      <div className="space-y-4 mb-6">
        <div>
          <label className="block text-sm text-gray-400 mb-2">
            How many transactions do you make per week?
          </label>
          <input
            type="number"
            min="1"
            value={transactionsPerWeek}
            onChange={(e) => setTransactionsPerWeek(parseInt(e.target.value) || 1)}
            className="w-full bg-gray-700 text-gray-200 border border-gray-600 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-cyan-500"
          />
        </div>

        <div>
          <label className="block text-sm text-gray-400 mb-2">Average transaction type</label>
          <select
            value={selectedType.name}
            onChange={(e) => {
              const type = TRANSACTION_TYPES.find((t) => t.name === e.target.value);
              if (type) setSelectedType(type);
            }}
            className="w-full bg-gray-700 text-gray-200 border border-gray-600 rounded-md px-3 py-2 focus:outline-none focus:ring-2 focus:ring-cyan-500"
          >
            {TRANSACTION_TYPES.map((type) => (
              <option key={type.name} value={type.name}>
                {type.name}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Calculation Results */}
      <div className="bg-gray-700/50 rounded-lg p-5 border border-gray-600">
        <div className="flex items-center mb-4 gap-2">
          <TrendingDown className="w-5 h-5 text-red-400" />
          <h4 className="text-lg font-bold text-gray-200">Gas Waste Analysis</h4>
        </div>

        <div className="space-y-3 mb-4">
          <div className="text-sm text-gray-400">{formatPeriod()}:</div>

          <div className="bg-gray-800/50 rounded p-3">
            <div className="text-sm text-gray-400 mb-1">If you transacted randomly:</div>
            <div className="text-xl font-bold text-gray-200">${formatUSD(results.avgGasPaid)}</div>
            <div className="text-xs text-gray-500">Average gas paid</div>
          </div>

          <div className="bg-gray-800/50 rounded p-3">
            <div className="text-sm text-gray-400 mb-1">If you used our predictions:</div>
            <div className="text-xl font-bold text-green-400">${formatUSD(results.optimizedGasCost)}</div>
            <div className="text-xs text-gray-500">Optimised gas cost</div>
          </div>
        </div>

        {/* Waste Display */}
        <div className="bg-red-500/20 border-2 border-red-500/50 rounded-lg p-4 mb-4">
          <div className="flex items-center justify-between mb-2">
            <span className="text-lg font-semibold text-red-400">Estimated Waste:</span>
            <span className="text-2xl font-bold text-red-400">
              ${formatUSD(results.waste)} ({results.wastePercent.toFixed(0)}%)
            </span>
          </div>
          <div className="text-sm text-gray-300">Annual waste: ~${formatUSD(results.annualWaste)}</div>
        </div>

        {/* Visual Impact */}
        {results.waste > 0 && (
          <div className="space-y-3">
            {/* Progress Bar */}
            <div>
              <div className="flex justify-between text-xs text-gray-400 mb-1">
                <span>Wasted</span>
                <span>Saved</span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-3">
                <div
                  className="bg-red-500 h-3 rounded-full transition-all"
                  style={{ width: `${Math.min(results.wastePercent, 100)}%` }}
                ></div>
              </div>
            </div>

            {/* Fun Comparisons */}
            <div className="text-sm text-gray-300 space-y-1">
              {coffeesWasted > 0 && (
                <div>
                  ☕ That's <span className="text-yellow-400 font-bold">{coffeesWasted}</span> cups of coffee!
                </div>
              )}
              <div>
                Network-wide waste: <span className="text-red-400 font-bold">$2.4M</span>/year (estimated)
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default GasWasteCalculator;
