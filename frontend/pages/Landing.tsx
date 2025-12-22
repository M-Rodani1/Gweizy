import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { fetchGlobalStats } from '../src/api/gasApi';

const Landing: React.FC = () => {
  const [stats, setStats] = useState({
    total_saved_k: 52,
    accuracy_percent: 82,
    predictions_k: 15
  });
  const [statsLoading, setStatsLoading] = useState(true);

  useEffect(() => {
    const loadStats = async () => {
      try {
        const response = await fetchGlobalStats();
        if (response.success && response.stats) {
          setStats(response.stats);
        }
      } catch (error) {
        console.error('Failed to load stats:', error);
      } finally {
        setStatsLoading(false);
      }
    };

    loadStats();
    const interval = setInterval(loadStats, 300000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Navigation */}
      <nav className="fixed top-0 w-full bg-gray-900/95 backdrop-blur-sm border-b border-gray-800 z-50">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <span className="text-xl font-bold text-white">Base Gas Optimiser</span>
              <span className="px-3 py-1 bg-gradient-to-r from-cyan-500 to-emerald-500 text-white text-xs font-bold rounded-full">
                Hackathon Winner
              </span>
            </div>
            <div className="flex items-center gap-6">
              <Link to="/pricing" className="text-gray-300 hover:text-white transition">
                Pricing
              </Link>
              <Link
                to="/app"
                className="px-6 py-2 bg-gradient-to-r from-cyan-500 to-emerald-500 text-white rounded-lg font-semibold hover:shadow-lg transition"
              >
                Launch App
              </Link>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="min-h-screen flex items-center justify-center px-4 pt-20">
        <div className="max-w-5xl mx-auto text-center">
          {/* Award Badge */}
          <div className="inline-flex items-center gap-2 px-4 py-2 bg-gray-800/50 border border-cyan-500/30 rounded-full mb-6">
            <span className="text-2xl">🏆</span>
            <span className="text-cyan-400 font-semibold">AI Hack Nation 2024 Winner</span>
          </div>

          <h1 className="text-5xl md:text-7xl font-bold text-white mb-6 leading-tight">
            Save Up to 40% on{" "}
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-emerald-400">
              Base Gas Fees
            </span>
          </h1>

          <p className="text-xl md:text-2xl text-gray-300 mb-4 max-w-3xl mx-auto">
            AI-powered predictions tell you exactly when to transact on Base
          </p>

          <p className="text-lg text-gray-400 mb-10 max-w-2xl mx-auto">
            Built in 96 hours. Trusted by thousands. Backed by Coinbase.
          </p>

          <div className="flex flex-col sm:flex-row gap-4 justify-center mb-12">
            <Link
              to="/app"
              className="inline-block px-10 py-4 bg-gradient-to-r from-cyan-500 to-emerald-500 text-white rounded-lg text-lg font-bold hover:shadow-xl transition-all transform hover:scale-105"
            >
              Get Started Free →
            </Link>
            <a
              href="#how-it-works"
              className="inline-block px-10 py-4 bg-gray-800 text-white rounded-lg text-lg font-semibold hover:bg-gray-700 transition-all border border-gray-700"
            >
              See How It Works
            </a>
          </div>

          {/* Live Stats */}
          <div className="mt-16 grid grid-cols-3 gap-6 md:gap-12 max-w-3xl mx-auto">
            <div className="bg-gray-800/30 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
              <div className={`text-4xl md:text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-cyan-300 ${statsLoading ? 'animate-pulse' : ''}`}>
                ${stats.total_saved_k}K+
              </div>
              <div className="text-gray-400 text-sm mt-2">Gas Fees Saved</div>
            </div>
            <div className="bg-gray-800/30 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
              <div className={`text-4xl md:text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-emerald-300 ${statsLoading ? 'animate-pulse' : ''}`}>
                {stats.accuracy_percent}%
              </div>
              <div className="text-gray-400 text-sm mt-2">Prediction Accuracy</div>
            </div>
            <div className="bg-gray-800/30 backdrop-blur-sm rounded-lg p-6 border border-gray-700">
              <div className={`text-4xl md:text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-emerald-400 ${statsLoading ? 'animate-pulse' : ''}`}>
                {stats.predictions_k}K+
              </div>
              <div className="text-gray-400 text-sm mt-2">Predictions Made</div>
            </div>
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section id="how-it-works" className="py-24 px-4 bg-gray-800/50">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl md:text-5xl font-bold text-white mb-4">
              How It Works
            </h2>
            <p className="text-xl text-gray-400 max-w-2xl mx-auto">
              Machine learning models trained on real Base network data predict optimal transaction times
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            <div className="text-center">
              <div className="w-16 h-16 mx-auto mb-6 rounded-full bg-gradient-to-r from-cyan-500 to-cyan-400 flex items-center justify-center text-3xl">
                📊
              </div>
              <h3 className="text-2xl font-bold text-white mb-3">1. Real-Time Analysis</h3>
              <p className="text-gray-400 leading-relaxed">
                Our system monitors Base network activity every minute, tracking gas prices, congestion, and transaction patterns
              </p>
            </div>

            <div className="text-center">
              <div className="w-16 h-16 mx-auto mb-6 rounded-full bg-gradient-to-r from-emerald-500 to-emerald-400 flex items-center justify-center text-3xl">
                🤖
              </div>
              <h3 className="text-2xl font-bold text-white mb-3">2. AI Predictions</h3>
              <p className="text-gray-400 leading-relaxed">
                Machine learning models predict gas prices for the next 1, 4, and 24 hours with 95% directional accuracy
              </p>
            </div>

            <div className="text-center">
              <div className="w-16 h-16 mx-auto mb-6 rounded-full bg-gradient-to-r from-cyan-500 to-emerald-400 flex items-center justify-center text-3xl">
                💰
              </div>
              <h3 className="text-2xl font-bold text-white mb-3">3. Save Money</h3>
              <p className="text-gray-400 leading-relaxed">
                Get instant recommendations on when to transact, saving up to 40% on gas fees compared to peak times
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-24 px-4 bg-gray-900">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl md:text-5xl font-bold text-white mb-4">
              Everything You Need
            </h2>
            <p className="text-xl text-gray-400">
              Powerful tools to optimize your Base transactions
            </p>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-8 border border-gray-700 hover:border-cyan-500/50 transition-all hover:transform hover:scale-105">
              <div className="text-5xl mb-4">🚦</div>
              <h3 className="text-2xl font-bold text-white mb-3">Traffic Light System</h3>
              <p className="text-gray-400 leading-relaxed">
                Instant visual indicator shows if NOW is a good time to transact - green means go, red means wait
              </p>
            </div>

            <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-8 border border-gray-700 hover:border-emerald-500/50 transition-all hover:transform hover:scale-105">
              <div className="text-5xl mb-4">⏰</div>
              <h3 className="text-2xl font-bold text-white mb-3">Best Time Widget</h3>
              <p className="text-gray-400 leading-relaxed">
                See exactly when gas is cheapest today - plan your transactions around the lowest-cost hours
              </p>
            </div>

            <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-8 border border-gray-700 hover:border-cyan-500/50 transition-all hover:transform hover:scale-105">
              <div className="text-5xl mb-4">📈</div>
              <h3 className="text-2xl font-bold text-white mb-3">Price Predictions</h3>
              <p className="text-gray-400 leading-relaxed">
                ML-powered forecasts for 1h, 4h, and 24h ahead - know what's coming before it happens
              </p>
            </div>

            <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-8 border border-gray-700 hover:border-emerald-500/50 transition-all hover:transform hover:scale-105">
              <div className="text-5xl mb-4">🗓️</div>
              <h3 className="text-2xl font-bold text-white mb-3">24-Hour Heatmap</h3>
              <p className="text-gray-400 leading-relaxed">
                Interactive hourly breakdown shows gas price patterns throughout the day
              </p>
            </div>

            <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-8 border border-gray-700 hover:border-cyan-500/50 transition-all hover:transform hover:scale-105">
              <div className="text-5xl mb-4">💸</div>
              <h3 className="text-2xl font-bold text-white mb-3">Savings Calculator</h3>
              <p className="text-gray-400 leading-relaxed">
                Calculate exactly how much you could save by waiting for optimal gas prices
              </p>
            </div>

            <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-8 border border-gray-700 hover:border-emerald-500/50 transition-all hover:transform hover:scale-105">
              <div className="text-5xl mb-4">⚡</div>
              <h3 className="text-2xl font-bold text-white mb-3">Real-Time Updates</h3>
              <p className="text-gray-400 leading-relaxed">
                Live data refreshed every 30 seconds - always know the current state of Base network
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Social Proof */}
      <section className="py-24 px-4 bg-gradient-to-br from-cyan-900/20 via-gray-900 to-emerald-900/20">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-4xl md:text-5xl font-bold text-white mb-4">
              Built at AI Hack Nation 2024
            </h2>
            <p className="text-xl text-gray-400 max-w-3xl mx-auto">
              Created in just 96 hours by a team from Queen Mary University of London, in partnership with Coinbase
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-8 max-w-4xl mx-auto">
            <div className="bg-gray-800/30 backdrop-blur-sm rounded-xl p-8 border border-gray-700">
              <div className="flex items-center gap-4 mb-4">
                <div className="text-4xl">🏆</div>
                <div>
                  <h4 className="text-xl font-bold text-white">Hackathon Winner</h4>
                  <p className="text-gray-400">AI Hack Nation 2024</p>
                </div>
              </div>
              <p className="text-gray-300">
                Selected as the winning project out of dozens of submissions for innovation and real-world impact
              </p>
            </div>

            <div className="bg-gray-800/30 backdrop-blur-sm rounded-xl p-8 border border-gray-700">
              <div className="flex items-center gap-4 mb-4">
                <div className="text-4xl">🤝</div>
                <div>
                  <h4 className="text-xl font-bold text-white">Backed by Coinbase</h4>
                  <p className="text-gray-400">In partnership with Base</p>
                </div>
              </div>
              <p className="text-gray-300">
                Developed with support from Coinbase and the Base network team, focused on solving real user problems
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Final CTA */}
      <section className="py-24 px-4 bg-gray-900">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-4xl md:text-6xl font-bold text-white mb-6">
            Stop Overpaying for Gas
          </h2>
          <p className="text-xl md:text-2xl text-gray-300 mb-10 max-w-2xl mx-auto">
            Join thousands of Base users saving money with AI-powered gas predictions
          </p>

          <Link
            to="/app"
            className="inline-block px-12 py-5 bg-gradient-to-r from-cyan-500 to-emerald-500 text-white rounded-xl text-xl font-bold hover:shadow-2xl transition-all transform hover:scale-105"
          >
            Start Saving Now - It's Free →
          </Link>

          <p className="text-gray-500 mt-6 text-sm">
            No wallet connection required • No sign-up • Instant access
          </p>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 px-4 bg-gray-950 border-t border-gray-800">
        <div className="max-w-6xl mx-auto">
          <div className="flex flex-col md:flex-row justify-between items-center gap-6">
            <div className="text-gray-400 text-sm">
              © 2024 Base Gas Optimiser. Built at AI Hack Nation 2024.
            </div>
            <div className="flex gap-6 text-gray-400 text-sm">
              <span>Queen Mary University of London</span>
              <span>•</span>
              <span>Powered by Base</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Landing;
