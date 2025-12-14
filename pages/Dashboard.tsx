import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import Logo from '../components/branding/Logo';
import RelativePriceIndicator from '../components/RelativePriceIndicator';
import BestTimeWidget from '../components/BestTimeWidget';
import HourlyHeatmap from '../components/HourlyHeatmap';
import { fetchCurrentGas } from '../src/api/gasApi';

const Dashboard: React.FC = () => {
  const [currentGas, setCurrentGas] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadGasPrice = async () => {
      try {
        const data = await fetchCurrentGas();
        setCurrentGas(data.current_gas);
      } catch (error) {
        console.error('Failed to load gas price:', error);
      } finally {
        setLoading(false);
      }
    };

    loadGasPrice();
    const interval = setInterval(loadGasPrice, 15000); // Refresh every 15 seconds
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-gray-900 text-white p-8">
      <div className="max-w-7xl mx-auto">
        <header className="mb-8">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Logo size="lg" />
              <div>
                <h1 className="text-3xl font-bold text-gray-100">Base Gas Optimizer</h1>
                <p className="text-sm text-gray-400 mt-1">Know the best times to transact on Base network</p>
              </div>
            </div>
            <Link
              to="/"
              className="px-6 py-2 bg-gray-800 text-white rounded-lg font-semibold hover:bg-gray-700 transition"
            >
              ← Back to Home
            </Link>
          </div>
        </header>

        <main>
          {/* Current Gas Price Card */}
          <div className="bg-gradient-to-br from-cyan-900/30 via-gray-800 to-emerald-900/30 rounded-lg p-8 border border-gray-700 mb-6">
            <div className="text-center">
              <p className="text-sm text-gray-400 mb-2">Current Base Gas Price</p>
              {loading ? (
                <div className="text-5xl font-bold text-cyan-400">Loading...</div>
              ) : currentGas !== null ? (
                <div>
                  <div className="text-6xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-emerald-400">
                    {currentGas.toFixed(4)} Gwei
                  </div>
                  <p className="text-sm text-gray-400 mt-2">Updated every 15 seconds</p>
                </div>
              ) : (
                <div className="text-3xl font-bold text-red-400">Unable to load</div>
              )}
            </div>
          </div>

          {/* Relative Price Indicator */}
          {currentGas !== null && (
            <div className="mb-6">
              <RelativePriceIndicator currentGas={currentGas} />
            </div>
          )}

          {/* Best Time Widget */}
          <div className="mb-6">
            <BestTimeWidget currentGas={currentGas || 0} />
          </div>

          {/* Hourly Heatmap */}
          <div className="mb-6">
            <HourlyHeatmap />
          </div>

          {/* Placeholder for more features */}
          <div className="bg-gray-800 rounded-lg p-8 text-center border border-gray-700">
            <h2 className="text-2xl font-bold mb-4">More Features Coming Soon</h2>
            <p className="text-gray-400 mb-6">
              Advanced analytics, historical charts, and savings tracking are on the way!
            </p>
            <div className="grid md:grid-cols-2 gap-4 mt-8">
              <div className="bg-gray-900 p-6 rounded-lg">
                <div className="text-4xl mb-2">📊</div>
                <h3 className="font-bold mb-2">Historical Charts</h3>
                <p className="text-sm text-gray-400">Next up</p>
              </div>
              <div className="bg-gray-900 p-6 rounded-lg">
                <div className="text-4xl mb-2">💰</div>
                <h3 className="font-bold mb-2">Savings Tracker</h3>
                <p className="text-sm text-gray-400">Next up</p>
              </div>
            </div>
          </div>
        </main>

        <footer className="mt-8 text-center text-gray-500 text-sm">
          <p>Pattern-based guidance for Base network • Chain ID: 8453</p>
        </footer>
      </div>
    </div>
  );
};

export default Dashboard;
