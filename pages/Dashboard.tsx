import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import Logo from '../components/branding/Logo';
import BestTimeWidget from '../components/BestTimeWidget';
import RelativePriceIndicator from '../components/RelativePriceIndicator';

const Dashboard: React.FC = () => {
  const [currentGas, setCurrentGas] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadGasPrice = async () => {
      try {
        const response = await fetch('https://basegasfeesml.onrender.com/api/current');
        if (!response.ok) {
          throw new Error('Failed to fetch gas price');
        }
        const data = await response.json();
        setCurrentGas(data.current_gas);
        setError(null);
      } catch (err) {
        console.error('Failed to load gas price:', err);
        setError('Unable to load gas data');
      } finally {
        setLoading(false);
      }
    };

    loadGasPrice();
    const interval = setInterval(loadGasPrice, 15000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-gray-900 text-white p-4 sm:p-8">
      <div className="max-w-7xl mx-auto">
        <header className="mb-8">
          <div className="flex items-center justify-between flex-wrap gap-4">
            <div className="flex items-center space-x-3">
              <Logo size="lg" />
              <div>
                <h1 className="text-2xl sm:text-3xl font-bold text-gray-100">Base Gas Optimizer</h1>
                <p className="text-xs sm:text-sm text-gray-400 mt-1">Real-time gas price tracking for Base network</p>
              </div>
            </div>
            <Link
              to="/"
              className="px-4 sm:px-6 py-2 bg-gray-800 text-white rounded-lg font-semibold hover:bg-gray-700 transition text-sm sm:text-base"
            >
              ← Back to Home
            </Link>
          </div>
        </header>

        <main>
          {/* Current Gas Price Card */}
          <div className="bg-gradient-to-br from-cyan-900/30 via-gray-800 to-emerald-900/30 rounded-lg p-6 sm:p-8 border border-gray-700 mb-6">
            <div className="text-center">
              <p className="text-sm text-gray-400 mb-2">Current Base Gas Price</p>
              {loading ? (
                <div className="text-4xl sm:text-5xl font-bold text-cyan-400">Loading...</div>
              ) : error ? (
                <div className="text-2xl sm:text-3xl font-bold text-red-400">{error}</div>
              ) : currentGas !== null ? (
                <div>
                  <div className="text-5xl sm:text-6xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-emerald-400">
                    {currentGas.toFixed(4)} Gwei
                  </div>
                  <p className="text-xs sm:text-sm text-gray-400 mt-2">Updated every 15 seconds</p>
                </div>
              ) : (
                <div className="text-2xl sm:text-3xl font-bold text-red-400">No data available</div>
              )}
            </div>
          </div>

          {/* Relative Price Indicator */}
          {currentGas !== null && !error && (
            <div className="mb-6">
              <RelativePriceIndicator currentGas={currentGas} />
            </div>
          )}

          {/* Best Time Widget */}
          <div className="mb-6">
            <BestTimeWidget currentGas={currentGas || 0} />
          </div>

          {/* Info Cards */}
          <div className="grid md:grid-cols-3 gap-4 sm:gap-6 mb-6">
            <div className="bg-gray-800 rounded-lg p-4 sm:p-6 border border-gray-700">
              <div className="text-3xl sm:text-4xl mb-3">🚦</div>
              <h3 className="text-lg sm:text-xl font-bold text-white mb-2">Smart Timing</h3>
              <p className="text-sm text-gray-400">Our AI analyzes patterns to show you the best times to transact</p>
            </div>
            <div className="bg-gray-800 rounded-lg p-4 sm:p-6 border border-gray-700">
              <div className="text-3xl sm:text-4xl mb-3">📊</div>
              <h3 className="text-lg sm:text-xl font-bold text-white mb-2">Real-Time Data</h3>
              <p className="text-sm text-gray-400">Live gas prices updated every 15 seconds directly from Base</p>
            </div>
            <div className="bg-gray-800 rounded-lg p-4 sm:p-6 border border-gray-700">
              <div className="text-3xl sm:text-4xl mb-3">💰</div>
              <h3 className="text-lg sm:text-xl font-bold text-white mb-2">Save Money</h3>
              <p className="text-sm text-gray-400">Users save up to 40% on transaction costs by timing smartly</p>
            </div>
          </div>

          {/* Coming Soon */}
          <div className="bg-gray-800 rounded-lg p-6 sm:p-8 text-center border border-gray-700">
            <h2 className="text-xl sm:text-2xl font-bold mb-4">Advanced Features</h2>
            <p className="text-sm sm:text-base text-gray-400 mb-6">
              More analytics and prediction features coming soon! Currently showing real-time Base gas prices.
            </p>
            <div className="grid md:grid-cols-3 gap-4 mt-6">
              <div className="bg-gray-900 p-4 sm:p-6 rounded-lg">
                <div className="text-3xl sm:text-4xl mb-2">🚦</div>
                <h3 className="text-sm sm:text-base font-bold mb-2">Price Indicator</h3>
                <p className="text-xs text-gray-400">In development</p>
              </div>
              <div className="bg-gray-900 p-4 sm:p-6 rounded-lg">
                <div className="text-3xl sm:text-4xl mb-2">🗓️</div>
                <h3 className="text-sm sm:text-base font-bold mb-2">24-Hour Heatmap</h3>
                <p className="text-xs text-gray-400">In development</p>
              </div>
              <div className="bg-gray-900 p-4 sm:p-6 rounded-lg">
                <div className="text-3xl sm:text-4xl mb-2">📊</div>
                <h3 className="text-sm sm:text-base font-bold mb-2">Historical Charts</h3>
                <p className="text-xs text-gray-400">In development</p>
              </div>
            </div>
          </div>
        </main>

        <footer className="mt-8 text-center text-gray-500 text-xs sm:text-sm">
          <p>Real-time data for Base network • Chain ID: 8453</p>
        </footer>
      </div>
    </div>
  );
};

export default Dashboard;
