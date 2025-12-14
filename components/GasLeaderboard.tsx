import React, { useState, useEffect } from 'react';
import { CurrentGasData } from '../types';
import { fetchCurrentGas } from '../src/api/gasApi';
import LoadingSpinner from '../components/LoadingSpinner';

const GasLeaderboard: React.FC = () => {
  const [currentGas, setCurrentGas] = useState<CurrentGasData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadData = async () => {
    try {
      setError(null);
      const data = await fetchCurrentGas();
      setCurrentGas(data);
    } catch (err) {
      console.error('Error loading current gas:', err);
      setError(err instanceof Error ? err.message : 'Failed to load gas data');
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

  if (loading && !currentGas) {
    return (
      <div className="bg-gray-800 p-4 sm:p-6 rounded-lg shadow-lg h-full">
        <LoadingSpinner message="Loading gas data..." />
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

  if (!currentGas) return null;

  return (
    <div className="bg-gray-800 p-4 sm:p-6 rounded-lg shadow-lg h-full">
      <h2 className="text-xl font-semibold text-gray-200 mb-4">Base Network Gas</h2>
      
      <div className="space-y-4">
        {/* Current Gas Price */}
        <div className="bg-gradient-to-br from-cyan-600/20 to-cyan-800/20 p-4 rounded-lg border border-cyan-500/30">
          <div className="text-sm text-gray-400 mb-1">Current Gas Price</div>
          <div className="text-3xl font-bold text-cyan-400">
            {currentGas.current_gas.toFixed(4)}
            <span className="text-lg text-gray-400 ml-2">Gwei</span>
          </div>
        </div>

        {/* Base Fee */}
        <div className="bg-gray-700/50 p-3 rounded-md">
          <div className="text-sm text-gray-400">Base Fee</div>
          <div className="text-xl font-semibold text-gray-100">
            {currentGas.base_fee.toFixed(4)} Gwei
          </div>
        </div>

        {/* Priority Fee */}
        <div className="bg-gray-700/50 p-3 rounded-md">
          <div className="text-sm text-gray-400">Priority Fee</div>
          <div className="text-xl font-semibold text-gray-100">
            {currentGas.priority_fee.toFixed(4)} Gwei
          </div>
        </div>

        {/* Latest Block */}
        <div className="bg-gray-700/50 p-3 rounded-md">
          <div className="text-sm text-gray-400">Latest Block</div>
          <div className="text-lg font-mono text-gray-100">
            #{currentGas.block_number.toLocaleString()}
          </div>
        </div>

        {/* Last Updated */}
        <div className="text-xs text-gray-500 text-center">
          Updated: {new Date(currentGas.timestamp).toLocaleTimeString()}
        </div>
      </div>
    </div>
  );
};

export default GasLeaderboard;
