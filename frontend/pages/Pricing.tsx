import React, { useState } from 'react';
import { Link } from 'react-router-dom';

const Pricing: React.FC = () => {
  const [billingPeriod, setBillingPeriod] = useState<'monthly' | 'annual'>('monthly');

  const tiers = {
    free: {
      name: 'Free',
      subtitle: 'Hobbyist',
      price: { monthly: 0, annual: 0 },
      description: 'Perfect for casual Base users',
      features: [
        { text: 'Full predictions (1h, 4h, 24h)', included: true },
        { text: 'Traffic light indicator', included: true },
        { text: 'Basic savings calculator', included: true },
        { text: '50 predictions per day', included: true },
        { text: '7 days historical data', included: true },
        { text: '5-minute data refresh', included: true, note: 'Limited' },
        { text: 'Ad-supported experience', included: true, note: 'Ads shown' },
        { text: 'Extended forecasts', included: false },
        { text: 'Custom alerts', included: false },
        { text: 'Savings tracker', included: false },
      ],
      cta: 'Get Started',
      ctaLink: '/app',
      popular: false,
    },
    pro: {
      name: 'Pro',
      subtitle: 'Active Traders',
      price: { monthly: 9, annual: 90 },
      description: 'For active DeFi users who transact frequently',
      features: [
        { text: 'Everything in Free', included: true, highlight: true },
        { text: 'Remove all ads', included: true },
        { text: 'Extended forecasts (48h, 7-day)', included: true },
        { text: 'Email/Discord/Telegram alerts', included: true },
        { text: 'Personal savings tracker', included: true },
        { text: 'Transaction history (100 txs)', included: true },
        { text: '500 predictions per day', included: true },
        { text: '1-minute data refresh', included: true },
        { text: '90 days historical data', included: true },
        { text: 'CSV export', included: true },
        { text: 'Email support (48h)', included: true },
        { text: 'API access', included: false },
      ],
      cta: 'Start Free Trial',
      ctaLink: '/app?upgrade=pro',
      popular: true,
      badge: 'Most Popular',
      roi: 'Save $20-50/month on gas fees',
    },
    max: {
      name: 'Max',
      subtitle: 'Developers & Bots',
      price: { monthly: 29, annual: 290 },
      description: 'For developers building on Base',
      features: [
        { text: 'Everything in Pro', included: true, highlight: true },
        { text: 'API access (1,000 req/day)', included: true },
        { text: 'Real-time data (30s refresh)', included: true },
        { text: 'Unlimited predictions', included: true },
        { text: 'Advanced analytics', included: true },
        { text: 'Webhook notifications', included: true },
        { text: 'Wallet-specific alerts', included: true },
        { text: 'Priority support (24h)', included: true },
        { text: 'Beta feature access', included: true },
        { text: 'Custom integrations', included: true },
      ],
      cta: 'Get Started',
      ctaLink: '/app?upgrade=max',
      popular: false,
      roi: 'Build profitable bots, save $100+/month',
    },
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Navigation */}
      <nav className="fixed top-0 w-full bg-gray-900/95 backdrop-blur-sm border-b border-gray-800 z-50">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <Link to="/" className="flex items-center gap-3">
              <span className="text-xl font-bold text-white">Base Gas Optimiser</span>
              <span className="px-3 py-1 bg-gradient-to-r from-cyan-500 to-emerald-500 text-white text-xs font-bold rounded-full">
                Hackathon Winner
              </span>
            </Link>
            <div className="flex items-center gap-6">
              <Link to="/" className="text-gray-300 hover:text-white transition">
                Home
              </Link>
              <Link to="/pricing" className="text-white font-semibold">
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
      <section className="pt-32 pb-16 px-4">
        <div className="max-w-7xl mx-auto text-center">
          <h1 className="text-5xl md:text-6xl font-bold text-white mb-6">
            Simple, Transparent Pricing
          </h1>
          <p className="text-xl text-gray-300 mb-8 max-w-3xl mx-auto">
            Save money on every Base transaction. Choose the plan that fits your needs.
          </p>

          {/* Billing Toggle */}
          <div className="inline-flex items-center gap-4 p-2 bg-gray-800 rounded-lg mb-12">
            <button
              onClick={() => setBillingPeriod('monthly')}
              className={`px-6 py-2 rounded-lg font-semibold transition ${
                billingPeriod === 'monthly'
                  ? 'bg-gradient-to-r from-cyan-500 to-emerald-500 text-white'
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              Monthly
            </button>
            <button
              onClick={() => setBillingPeriod('annual')}
              className={`px-6 py-2 rounded-lg font-semibold transition ${
                billingPeriod === 'annual'
                  ? 'bg-gradient-to-r from-cyan-500 to-emerald-500 text-white'
                  : 'text-gray-400 hover:text-white'
              }`}
            >
              Annual
              <span className="ml-2 px-2 py-1 bg-emerald-500 text-white text-xs rounded-full">
                Save 16%
              </span>
            </button>
          </div>
        </div>
      </section>

      {/* Pricing Cards */}
      <section className="pb-24 px-4">
        <div className="max-w-7xl mx-auto grid md:grid-cols-3 gap-8">
          {Object.entries(tiers).map(([key, tier]) => (
            <div
              key={key}
              className={`relative rounded-2xl p-8 ${
                tier.popular
                  ? 'bg-gradient-to-br from-cyan-900/50 via-gray-800 to-emerald-900/50 border-2 border-cyan-500'
                  : 'bg-gray-800/50 border border-gray-700'
              } hover:transform hover:scale-105 transition-all`}
            >
              {/* Popular Badge */}
              {tier.popular && (
                <div className="absolute -top-4 left-1/2 transform -translate-x-1/2">
                  <span className="px-4 py-1 bg-gradient-to-r from-cyan-500 to-emerald-500 text-white text-sm font-bold rounded-full">
                    {tier.badge}
                  </span>
                </div>
              )}

              {/* Header */}
              <div className="text-center mb-6">
                <h3 className="text-2xl font-bold text-white mb-1">{tier.name}</h3>
                <p className="text-gray-400 text-sm mb-4">{tier.subtitle}</p>
                <div className="mb-4">
                  <span className="text-5xl font-bold text-white">
                    ${tier.price[billingPeriod]}
                  </span>
                  {tier.price.monthly > 0 && (
                    <span className="text-gray-400 ml-2">
                      /{billingPeriod === 'monthly' ? 'month' : 'year'}
                    </span>
                  )}
                </div>
                {billingPeriod === 'annual' && tier.price.annual > 0 && (
                  <p className="text-emerald-400 text-sm">
                    ${(tier.price.annual / 12).toFixed(2)}/month billed annually
                  </p>
                )}
                <p className="text-gray-300 text-sm mt-4">{tier.description}</p>
              </div>

              {/* ROI Badge */}
              {tier.roi && (
                <div className="mb-6 p-3 bg-emerald-900/30 border border-emerald-500/30 rounded-lg">
                  <p className="text-emerald-400 text-sm font-semibold text-center">
                    💰 {tier.roi}
                  </p>
                </div>
              )}

              {/* Features */}
              <ul className="space-y-3 mb-8">
                {tier.features.map((feature, idx) => (
                  <li key={idx} className="flex items-start gap-3">
                    {feature.included ? (
                      <svg
                        className="w-5 h-5 text-emerald-400 flex-shrink-0 mt-0.5"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M5 13l4 4L19 7"
                        />
                      </svg>
                    ) : (
                      <svg
                        className="w-5 h-5 text-gray-600 flex-shrink-0 mt-0.5"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M6 18L18 6M6 6l12 12"
                        />
                      </svg>
                    )}
                    <span
                      className={`text-sm ${
                        feature.included ? 'text-gray-200' : 'text-gray-500'
                      } ${feature.highlight ? 'font-semibold' : ''}`}
                    >
                      {feature.text}
                      {feature.note && (
                        <span className="text-gray-500 text-xs ml-2">({feature.note})</span>
                      )}
                    </span>
                  </li>
                ))}
              </ul>

              {/* CTA Button */}
              <Link
                to={tier.ctaLink}
                className={`block w-full py-3 px-6 rounded-lg font-semibold text-center transition ${
                  tier.popular
                    ? 'bg-gradient-to-r from-cyan-500 to-emerald-500 text-white hover:shadow-xl'
                    : 'bg-gray-700 text-white hover:bg-gray-600'
                }`}
              >
                {tier.cta}
              </Link>

              {key === 'pro' && (
                <p className="text-center text-gray-400 text-xs mt-3">
                  7-day free trial • Cancel anytime
                </p>
              )}
            </div>
          ))}
        </div>
      </section>

      {/* FAQ Section */}
      <section className="py-24 px-4 bg-gray-800/50">
        <div className="max-w-4xl mx-auto">
          <h2 className="text-4xl font-bold text-white text-center mb-12">
            Frequently Asked Questions
          </h2>

          <div className="space-y-6">
            <div className="bg-gray-800 rounded-lg p-6">
              <h3 className="text-xl font-bold text-white mb-3">
                How much can I actually save?
              </h3>
              <p className="text-gray-300">
                On average, users save 15-30% on gas fees by timing transactions optimally.
                If you make 10 transactions per month at $10 gas each, that's $15-30 saved.
                Pro tier pays for itself after 1-2 optimized transactions.
              </p>
            </div>

            <div className="bg-gray-800 rounded-lg p-6">
              <h3 className="text-xl font-bold text-white mb-3">
                Can I upgrade or downgrade anytime?
              </h3>
              <p className="text-gray-300">
                Yes! You can upgrade, downgrade, or cancel your subscription at any time.
                Changes take effect at the start of your next billing cycle.
              </p>
            </div>

            <div className="bg-gray-800 rounded-lg p-6">
              <h3 className="text-xl font-bold text-white mb-3">
                Do you offer refunds?
              </h3>
              <p className="text-gray-300">
                We offer a 7-day free trial for Pro tier. If you're not satisfied within the
                first 30 days of any paid subscription, contact us for a full refund.
              </p>
            </div>

            <div className="bg-gray-800 rounded-lg p-6">
              <h3 className="text-xl font-bold text-white mb-3">
                What payment methods do you accept?
              </h3>
              <p className="text-gray-300">
                We accept all major credit cards (Visa, Mastercard, Amex) and cryptocurrency
                payments (ETH, USDC on Base network). Payments are processed securely via Stripe.
              </p>
            </div>

            <div className="bg-gray-800 rounded-lg p-6">
              <h3 className="text-xl font-bold text-white mb-3">
                Is there a limit on API requests for Max tier?
              </h3>
              <p className="text-gray-300">
                Max tier includes 1,000 API requests per day. If you need more, contact us
                about enterprise pricing. We can accommodate high-volume use cases.
              </p>
            </div>

            <div className="bg-gray-800 rounded-lg p-6">
              <h3 className="text-xl font-bold text-white mb-3">
                How accurate are the predictions?
              </h3>
              <p className="text-gray-300">
                Our ML models achieve 75-80% directional accuracy (predicting if gas will go
                up or down) and explain 70%+ of gas price variance. We're constantly improving
                accuracy by retraining models with fresh data.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-24 px-4 bg-gradient-to-br from-cyan-900/20 via-gray-900 to-emerald-900/20">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-4xl md:text-5xl font-bold text-white mb-6">
            Ready to Save on Gas Fees?
          </h2>
          <p className="text-xl text-gray-300 mb-10">
            Join thousands of Base users saving money with AI-powered predictions
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link
              to="/app"
              className="px-12 py-4 bg-gradient-to-r from-cyan-500 to-emerald-500 text-white rounded-xl text-lg font-bold hover:shadow-2xl transition-all transform hover:scale-105"
            >
              Start Free Trial →
            </Link>
            <Link
              to="/"
              className="px-12 py-4 bg-gray-800 text-white rounded-xl text-lg font-semibold hover:bg-gray-700 transition-all border border-gray-700"
            >
              Learn More
            </Link>
          </div>
          <p className="text-gray-500 mt-6 text-sm">
            No credit card required for Free tier • 7-day trial for Pro
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
              <Link to="/pricing" className="hover:text-white transition">
                Pricing
              </Link>
              <span>•</span>
              <Link to="/" className="hover:text-white transition">
                About
              </Link>
              <span>•</span>
              <span>Powered by Base</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Pricing;
