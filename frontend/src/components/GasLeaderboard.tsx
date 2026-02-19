import React, { useState, useEffect } from 'react';
import { CurrentGasData } from '../../types';
import { fetchCurrentGas } from '../api/gasApi';
import LoadingSpinner from './LoadingSpinner';
import { REFRESH_INTERVALS } from '../constants';

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
    void loadData();

    // Auto-refresh every 30 seconds
    const refreshIfVisible = () => {
      if (typeof document === 'undefined' || document.visibilityState === 'visible') {
        void loadData();
      }
    };
    const interval = setInterval(refreshIfVisible, REFRESH_INTERVALS.GAS_DATA);
    const handleVisibilityChange = () => {
      if (document.visibilityState === 'visible') {
        void loadData();
      }
    };
    document.addEventListener('visibilitychange', handleVisibilityChange);

    return () => {
      clearInterval(interval);
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, []);

  if (loading && !currentGas) {
    return (
      <div className="bg-gray-800 p-4 sm:p-6 rounded-2xl shadow-xl h-full">
        <LoadingSpinner message="Loading gas data..." />
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-gray-800 p-4 sm:p-6 rounded-2xl shadow-xl h-full">
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
    <div className="bg-gray-800 p-6 rounded-2xl shadow-xl">
      <h2 className="text-lg font-semibold text-gray-200 mb-3">Base Network Gas</h2>

      <div className="space-y-2">
        {/* Current Gas Price */}
        <div className="bg-gradient-to-br from-cyan-600/20 to-cyan-800/20 p-3 rounded-lg border border-cyan-500/30">
          <div className="text-xs text-gray-400 mb-1">Current Gas Price</div>
          <div className="text-2xl font-bold text-cyan-400">
            {currentGas.current_gas !== undefined && currentGas.current_gas !== null ? currentGas.current_gas.toFixed(4) : 'N/A'}
            <span className="text-sm text-gray-400 ml-1">Gwei</span>
          </div>
        </div>

        {/* Base Fee */}
        <div className="bg-gray-700/50 p-2 rounded-md">
          <div className="text-xs text-gray-400">Base Fee</div>
          <div className="text-lg font-semibold text-gray-100">
            {currentGas.base_fee !== undefined && currentGas.base_fee !== null ? currentGas.base_fee.toFixed(4) : 'N/A'} Gwei
          </div>
        </div>

        {/* Priority Fee */}
        <div className="bg-gray-700/50 p-2 rounded-md">
          <div className="text-xs text-gray-400">Priority Fee</div>
          <div className="text-lg font-semibold text-gray-100">
            {currentGas.priority_fee !== undefined && currentGas.priority_fee !== null ? currentGas.priority_fee.toFixed(4) : 'N/A'} Gwei
          </div>
        </div>

        {/* Latest Block */}
        <div className="bg-gray-700/50 p-2 rounded-md">
          <div className="text-xs text-gray-400">Latest Block</div>
          <div className="text-sm font-mono text-gray-100">
            #{currentGas.block_number.toLocaleString()}
          </div>
        </div>
      </div>
    </div>
  );
};

export default GasLeaderboard;
