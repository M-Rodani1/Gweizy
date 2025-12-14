import React from 'react';
import { Link } from 'react-router-dom';
import { Logo } from '../components/branding/Logo';

const Docs: React.FC = () => {
  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Navigation */}
      <nav className="fixed top-0 w-full bg-gray-900/95 backdrop-blur-sm border-b border-gray-800 z-50">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <Link to="/" className="flex items-center space-x-2">
              <Logo />
              <span className="text-lg sm:text-xl font-bold text-white">Base Gas Optimiser</span>
            </Link>
            <Link to="/app" className="px-6 py-2 bg-gradient-to-r from-cyan-500 to-emerald-500 text-white rounded-lg font-semibold hover:shadow-lg transition">
              Go to App
            </Link>
          </div>
        </div>
      </nav>

      {/* Content */}
      <div className="max-w-4xl mx-auto px-4 pt-24 pb-12">
        <h1 className="text-4xl font-bold mb-8">Documentation</h1>

        <div className="space-y-8">
          {/* Getting Started */}
          <section className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h2 className="text-2xl font-bold mb-4 text-cyan-400">Getting Started</h2>
            <p className="text-gray-300 mb-4">
              Base Gas Optimiser helps you save money on Base network transactions by showing you when gas prices are typically lowest.
            </p>
            <ol className="list-decimal list-inside space-y-2 text-gray-300">
              <li>Visit the <Link to="/app" className="text-cyan-400 hover:underline">Dashboard</Link></li>
              <li>Check the "Best Time to Transact" widget for cheapest hours</li>
              <li>Use the traffic light indicator to see if NOW is a good time</li>
              <li>View the 24-hour heatmap to understand daily patterns</li>
            </ol>
          </section>

          {/* How It Works */}
          <section className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h2 className="text-2xl font-bold mb-4 text-cyan-400">How It Works</h2>
            <p className="text-gray-300 mb-4">
              Our system analyses 7+ days of historical Base gas price data to identify reliable hourly patterns.
            </p>
            <div className="space-y-4">
              <div>
                <h3 className="font-bold text-white mb-2">📊 Pattern Recognition</h3>
                <p className="text-gray-400">
                  Gas prices follow predictable daily patterns. We've found that prices are typically 127% higher at peak times (10pm-1am UTC) compared to low times (9am-12pm UTC).
                </p>
              </div>
              <div>
                <h3 className="font-bold text-white mb-2">🚦 Real-Time Indicators</h3>
                <p className="text-gray-400">
                  Our traffic light system compares current prices against hourly and 24-hour averages, giving you instant guidance on whether to transact now or wait.
                </p>
              </div>
              <div>
                <h3 className="font-bold text-white mb-2">💰 Savings Calculation</h3>
                <p className="text-gray-400">
                  By transacting during low-price hours instead of peak hours, you can save up to 40% on gas fees.
                </p>
              </div>
            </div>
          </section>

          {/* Features */}
          <section className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h2 className="text-2xl font-bold mb-4 text-cyan-400">Features</h2>
            <div className="space-y-3">
              <div className="flex items-start space-x-3">
                <span className="text-2xl">🟢</span>
                <div>
                  <h3 className="font-bold text-white">Relative Price Indicator</h3>
                  <p className="text-gray-400">See if current gas price is Excellent, Good, Average, High, or Very High compared to typical levels.</p>
                </div>
              </div>
              <div className="flex items-start space-x-3">
                <span className="text-2xl">⏰</span>
                <div>
                  <h3 className="font-bold text-white">Best Time Widget</h3>
                  <p className="text-gray-400">Shows the 3 cheapest and 3 most expensive hours based on historical patterns.</p>
                </div>
              </div>
              <div className="flex items-start space-x-3">
                <span className="text-2xl">🗓️</span>
                <div>
                  <h3 className="font-bold text-white">24-Hour Heatmap</h3>
                  <p className="text-gray-400">Interactive visualisation of gas prices throughout the day with colour coding.</p>
                </div>
              </div>
            </div>
          </section>

          {/* Transparency */}
          <section className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h2 className="text-2xl font-bold mb-4 text-cyan-400">Our Approach to Transparency</h2>
            <p className="text-gray-300 mb-4">
              We believe in being honest about what's predictable and what isn't.
            </p>
            <div className="space-y-4">
              <div className="bg-emerald-900/20 border border-emerald-500/30 rounded p-4">
                <h3 className="font-bold text-emerald-400 mb-2">✅ What We CAN Predict</h3>
                <ul className="text-gray-300 space-y-1 text-sm">
                  <li>• Hourly patterns (some hours are consistently cheaper)</li>
                  <li>• Relative price levels (is current price high or low?)</li>
                  <li>• Best times to transact (based on historical averages)</li>
                  <li>• Potential savings (by waiting for cheaper hours)</li>
                </ul>
              </div>
              <div className="bg-red-900/20 border border-red-500/30 rounded p-4">
                <h3 className="font-bold text-red-400 mb-2">❌ What We CAN'T Predict</h3>
                <ul className="text-gray-300 space-y-1 text-sm">
                  <li>• Exact future gas prices (prices are volatile)</li>
                  <li>• Unexpected network congestion spikes</li>
                  <li>• Impact of major events or launches</li>
                  <li>• Minute-by-minute price movements</li>
                </ul>
              </div>
            </div>
            <p className="text-gray-400 mt-4 text-sm">
              Our approach focuses on what's reliable: historical patterns show that gas is typically cheaper during certain hours. This is more useful than trying to predict exact future prices.
            </p>
          </section>

          {/* FAQ */}
          <section className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h2 className="text-2xl font-bold mb-4 text-cyan-400">Frequently Asked Questions</h2>
            <div className="space-y-4">
              <div>
                <h3 className="font-bold text-white mb-2">How accurate are the patterns?</h3>
                <p className="text-gray-400">
                  Hourly patterns are based on 7+ days of real data and show a consistent 127% difference between peak and low times. However, individual prices can vary due to network activity.
                </p>
              </div>
              <div>
                <h3 className="font-bold text-white mb-2">When is the best time to transact?</h3>
                <p className="text-gray-400">
                  Typically 9am-12pm UTC (morning hours). The worst times are usually 10pm-1am UTC (late evening/night). Check the dashboard for current patterns.
                </p>
              </div>
              <div>
                <h3 className="font-bold text-white mb-2">How much can I save?</h3>
                <p className="text-gray-400">
                  By transacting during cheap hours instead of peak hours, you can save up to 40%. Actual savings depend on your transaction timing and network conditions.
                </p>
              </div>
              <div>
                <h3 className="font-bold text-white mb-2">Do I need to connect my wallet?</h3>
                <p className="text-gray-400">
                  No! All pattern analysis and recommendations are available without connecting a wallet. Wallet connection is optional for future features.
                </p>
              </div>
            </div>
          </section>

          {/* API Access */}
          <section className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h2 className="text-2xl font-bold mb-4 text-cyan-400">API Access</h2>
            <p className="text-gray-300 mb-4">
              Developers can access our gas price data through our API endpoints:
            </p>
            <div className="bg-gray-900 rounded p-4 font-mono text-sm space-y-3">
              <div>
                <div className="text-cyan-400">GET /api/gas-price</div>
                <div className="text-gray-400">Get current Base gas price</div>
              </div>
              <div>
                <div className="text-cyan-400">GET /api/historical?hours=168</div>
                <div className="text-gray-400">Get historical gas prices (last 7 days)</div>
              </div>
              <div>
                <div className="text-cyan-400">GET /api/accuracy</div>
                <div className="text-gray-400">Get model performance metrics</div>
              </div>
            </div>
            <p className="text-gray-400 mt-4 text-sm">
              For Business plan API access with higher rate limits, <Link to="/" className="text-cyan-400 hover:underline">contact sales</Link>.
            </p>
          </section>

          {/* Support */}
          <section className="bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h2 className="text-2xl font-bold mb-4 text-cyan-400">Need Help?</h2>
            <p className="text-gray-300 mb-4">
              We're here to help! Reach out through:
            </p>
            <ul className="space-y-2 text-gray-300">
              <li>🐦 Twitter: @BaseGasOptimiser</li>
              <li>💬 Discord: Join our community</li>
              <li>📧 Email: support@basegasoptimiser.com</li>
            </ul>
          </section>
        </div>

        {/* CTA */}
        <div className="mt-12 text-center">
          <Link
            to="/app"
            className="inline-block px-8 py-4 bg-gradient-to-r from-cyan-500 to-emerald-500 text-white rounded-lg text-lg font-semibold hover:shadow-xl transition-all transform hover:scale-105"
          >
            Try It Now →
          </Link>
        </div>
      </div>
    </div>
  );
};

export default Docs;
