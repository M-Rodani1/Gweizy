import React, { useState, useEffect } from 'react';
import { fetchUserHistory } from '../src/api/gasApi';
import LoadingSpinner from './LoadingSpinner';

interface Transaction {
  hash: string;
  timestamp: number;
  gasUsed: number;
  gasPrice: number;
  value: string;
  from: string;
  to: string;
  method?: string;
}

interface UserHistoryData {
  transactions: Transaction[];
  total_transactions: number;
  total_gas_paid: number;
  potential_savings: number;
  savings_percentage: number;
  recommendations: {
    usual_time?: string;
    best_time?: string;
    avg_savings?: number;
  };
}

interface UserTransactionHistoryProps {
  address: string;
}

const UserTransactionHistory: React.FC<UserTransactionHistoryProps> = ({ address }) => {
  const [data, setData] = useState<UserHistoryData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadData = async () => {
      if (!address) return;

      try {
        setLoading(true);
        setError(null);
        const historyData = await fetchUserHistory(address);
        setData(historyData);
      } catch (err) {
        console.error('Error loading user history:', err);
        setError(err instanceof Error ? err.message : 'Failed to load transaction history');
      } finally {
        setLoading(false);
      }
    };

    loadData();
    const interval = setInterval(loadData, 60000); // Refresh every minute
    return () => clearInterval(interval);
  }, [address]);

  const formatTime = (timestamp: number) => {
    const date = new Date(timestamp * 1000);
    return date.toLocaleString();
  };

  const formatGasPrice = (gasPrice: number) => {
    return (gasPrice / 1e9).toFixed(4);
  };

  const formatUSD = (amount: number) => {
    return amount.toFixed(4);
  };

  if (loading) {
    return (
      <div className="bg-gray-800 p-6 rounded-lg shadow-lg border border-gray-700">
        <LoadingSpinner message="Loading your transaction history..." />
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-gray-800 p-6 rounded-lg shadow-lg border border-gray-700">
        <p className="text-red-400 mb-4">⚠️ {error}</p>
        <button
          onClick={() => window.location.reload()}
          className="px-4 py-2 bg-cyan-500 hover:bg-cyan-600 rounded-md transition-colors text-sm"
        >
          Retry
        </button>
      </div>
    );
  }

  if (!data) return null;

  return (
    <div className="space-y-6">
      {/* Savings Potential Card */}
      <div className="bg-gray-800 p-6 rounded-lg shadow-lg border border-gray-700">
        <div className="flex items-center mb-4">
          <span className="text-2xl mr-2">💰</span>
          <h3 className="text-xl font-bold text-gray-200">YOUR POTENTIAL SAVINGS</h3>
        </div>

        <div className="space-y-3">
          <div className="flex justify-between text-sm">
            <span className="text-gray-400">Last 30 days:</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">• Transactions:</span>
            <span className="text-gray-200 font-medium">{data.total_transactions}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">• Total gas paid:</span>
            <span className="text-gray-200 font-medium">${formatUSD(data.total_gas_paid)}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">• Could have saved:</span>
            <span className="text-green-400 font-bold">${formatUSD(data.potential_savings)}</span>
          </div>

          <div className="mt-4 pt-4 border-t border-gray-700">
            <div className="flex items-center justify-between">
              <span className="text-lg font-semibold text-gray-300">📈 Savings if optimized:</span>
              <span className="text-2xl font-bold text-green-400">
                {data.savings_percentage.toFixed(0)}%
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Recent Transactions */}
      <div className="bg-gray-800 p-6 rounded-lg shadow-lg border border-gray-700">
        <h3 className="text-lg font-semibold text-gray-200 mb-4">
          Your Recent Base Transactions
        </h3>
        
        {data.transactions.length === 0 ? (
          <p className="text-gray-400 text-sm">No transactions found.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-700">
                  <th className="text-left py-2 text-gray-400">Hash</th>
                  <th className="text-left py-2 text-gray-400">Time</th>
                  <th className="text-right py-2 text-gray-400">Gas Price</th>
                  <th className="text-right py-2 text-gray-400">Gas Used</th>
                  <th className="text-right py-2 text-gray-400">Cost</th>
                </tr>
              </thead>
              <tbody>
                {data.transactions.map((tx) => (
                  <tr key={tx.hash} className="border-b border-gray-700/50 hover:bg-gray-700/30">
                    <td className="py-2">
                      <a
                        href={`https://basescan.org/tx/${tx.hash}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-cyan-400 hover:text-cyan-300"
                      >
                        {tx.hash.slice(0, 10)}...{tx.hash.slice(-8)}
                      </a>
                    </td>
                    <td className="py-2 text-gray-300">{formatTime(tx.timestamp)}</td>
                    <td className="py-2 text-right text-gray-300">{formatGasPrice(tx.gasPrice)} gwei</td>
                    <td className="py-2 text-right text-gray-300">{tx.gasUsed.toLocaleString()}</td>
                    <td className="py-2 text-right text-gray-300">
                      ${formatUSD((tx.gasPrice * tx.gasUsed) / 1e9 * 3000)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};

export default UserTransactionHistory;

