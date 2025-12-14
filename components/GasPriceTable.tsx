import React, { useState, useEffect } from 'react';
import { TableRowData } from '../types';
import { fetchTransactions } from '../src/api/gasApi';
import LoadingSpinner from '../components/LoadingSpinner';

const GasPriceTable: React.FC = () => {
  const [transactions, setTransactions] = useState<TableRowData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadData = async () => {
    try {
      setError(null);
      const data = await fetchTransactions(10);
      setTransactions(data);
    } catch (err) {
      console.error('Error loading transactions:', err);
      setError(err instanceof Error ? err.message : 'Failed to load transactions');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
    
    // Auto-refresh every 30 seconds
    const interval = setInterval(loadData, 30000);
    return () => clearInterval(interval);
  }, []);

  if (loading && transactions.length === 0) {
    return (
      <div className="bg-gray-800 p-4 sm:p-6 rounded-lg shadow-lg h-full overflow-x-auto">
        <LoadingSpinner message="Loading transactions..." />
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-gray-800 p-4 sm:p-6 rounded-lg shadow-lg h-full">
        <div className="text-center">
          <p className="text-red-400 mb-4">⚠️ {error}</p>
          <button
            onClick={loadData}
            className="px-4 py-2 bg-cyan-500 hover:bg-cyan-600 rounded-md transition-colors text-sm"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gray-800 p-4 sm:p-6 rounded-lg shadow-lg h-full overflow-x-auto">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-semibold text-gray-200">Recent Base Transactions</h2>
        <button
          onClick={loadData}
          className="text-sm text-cyan-400 hover:text-cyan-300 transition-colors"
          title="Refresh"
        >
          🔄 Refresh
        </button>
      </div>
      
      <table className="w-full text-left table-auto">
        <thead className="border-b-2 border-gray-600">
          <tr className="text-gray-400">
            <th className="p-3">Tx Hash</th>
            <th className="p-3">Method</th>
            <th className="p-3">Age</th>
            <th className="p-3 text-right">Gas Used</th>
            <th className="p-3 text-right">Gas Price (Gwei)</th>
          </tr>
        </thead>
        <tbody>
          {transactions.map((row, index) => (
            <tr key={`${row.txHash}-${index}`} className="border-b border-gray-700 hover:bg-gray-700/50">
              <td className="p-3 font-mono text-sm text-cyan-400">{row.txHash}</td>
              <td className="p-3">
                <span className="bg-purple-600/50 text-purple-200 text-xs font-semibold mr-2 px-2.5 py-0.5 rounded">
                  {row.method}
                </span>
              </td>
              <td className="p-3 text-gray-300">{row.age}</td>
              <td className="p-3 text-right text-gray-300">{row.gasUsed.toLocaleString()}</td>
              <td className="p-3 text-right font-semibold text-teal-300">{row.gasPrice.toFixed(4)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      
      {transactions.length === 0 && !loading && (
        <div className="text-center py-8 text-gray-500">
          No recent transactions found
        </div>
      )}
    </div>
  );
};

export default GasPriceTable;
