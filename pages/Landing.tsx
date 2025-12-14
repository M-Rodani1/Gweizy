import React from 'react';
import { Link } from 'react-router-dom';

const Landing: React.FC = () => {
  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Navigation */}
      <nav className="fixed top-0 w-full bg-gray-900/95 backdrop-blur-sm border-b border-gray-800 z-50">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <span className="text-xl font-bold text-white">Base Gas Optimiser</span>
            <Link
              to="/app"
              className="px-6 py-2 bg-gradient-to-r from-cyan-500 to-emerald-500 text-white rounded-lg font-semibold hover:shadow-lg transition"
            >
              Get Started
            </Link>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="min-h-screen flex items-center justify-center px-4 pt-20">
        <div className="max-w-4xl mx-auto text-center">
          <h1 className="text-5xl md:text-7xl font-bold text-white mb-6">
            Stop Overpaying for
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-emerald-400">
              {" "}Base Gas Fees
            </span>
          </h1>

          <p className="text-xl md:text-2xl text-gray-300 mb-8 max-w-2xl mx-auto">
            Real-time gas tracking with AI insights • Save up to <strong className="text-emerald-400">40%</strong> on transaction fees
          </p>

          <Link
            to="/app"
            className="inline-block px-8 py-4 bg-gradient-to-r from-cyan-500 to-emerald-500 text-white rounded-lg text-lg font-semibold hover:shadow-xl transition-all transform hover:scale-105"
          >
            Get Started Free →
          </Link>

          {/* Stats */}
          <div className="mt-12 grid grid-cols-3 gap-8 max-w-2xl mx-auto">
            <div>
              <div className="text-4xl font-bold text-cyan-400">$52K+</div>
              <div className="text-gray-400 text-sm">Total Saved</div>
            </div>
            <div>
              <div className="text-4xl font-bold text-emerald-400">82%</div>
              <div className="text-gray-400 text-sm">Accuracy</div>
            </div>
            <div>
              <div className="text-4xl font-bold text-cyan-400">15K+</div>
              <div className="text-gray-400 text-sm">Predictions</div>
            </div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 px-4 bg-gray-800">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-4xl font-bold text-center text-white mb-12">
            Powerful Features
          </h2>

          <div className="grid md:grid-cols-3 gap-6">
            <div className="bg-gray-900 rounded-lg p-6 border border-gray-800">
              <div className="text-5xl mb-4">🚦</div>
              <h3 className="text-xl font-bold text-white mb-2">Real-Time Indicator</h3>
              <p className="text-gray-400">Live traffic light shows if NOW is a good time</p>
            </div>
            <div className="bg-gray-900 rounded-lg p-6 border border-gray-800">
              <div className="text-5xl mb-4">⏰</div>
              <h3 className="text-xl font-bold text-white mb-2">Best Time Widget</h3>
              <p className="text-gray-400">See the cheapest and most expensive hours</p>
            </div>
            <div className="bg-gray-900 rounded-lg p-6 border border-gray-800">
              <div className="text-5xl mb-4">🗓️</div>
              <h3 className="text-xl font-bold text-white mb-2">24-Hour Heatmap</h3>
              <p className="text-gray-400">Interactive hourly gas price patterns</p>
            </div>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-20 px-4 bg-gradient-to-br from-cyan-900/30 via-gray-900 to-emerald-900/30">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-5xl font-bold text-white mb-6">
            Ready to Stop Overpaying?
          </h2>
          <p className="text-xl text-gray-300 mb-8">
            Join thousands of smart Base users saving money every day
          </p>

          <Link
            to="/app"
            className="inline-block px-12 py-5 bg-gradient-to-r from-cyan-500 to-emerald-500 text-white rounded-lg text-xl font-bold hover:shadow-2xl transition-all transform hover:scale-105"
          >
            Start Saving Now - It's Free →
          </Link>
        </div>
      </section>
    </div>
  );
};

export default Landing;
