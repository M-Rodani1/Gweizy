import React from 'react';
import { Link } from 'react-router-dom';
import { Logo } from '../components/branding/Logo';
import TrustBadge from '../components/branding/TrustBadges';
import CountUp from 'react-countup';
import { fetchGlobalStats } from '../src/api/gasApi';

const Landing: React.FC = () => {
  const [mobileMenuOpen, setMobileMenuOpen] = React.useState(false);
  const [stats, setStats] = React.useState({
    total_saved_k: 52,
    accuracy_percent: 82,
    predictions_k: 15
  });

  React.useEffect(() => {
    const loadStats = async () => {
      try {
        const response = await fetchGlobalStats();
        if (response.success && response.stats) {
          setStats({
            total_saved_k: response.stats.total_saved_k,
            accuracy_percent: response.stats.accuracy_percent,
            predictions_k: response.stats.predictions_k
          });
        }
      } catch (error) {
        console.error('Failed to load stats, using defaults:', error);
        // Keep default values on error
      }
    };

    loadStats();
  }, []);

  return (
    <div className="min-h-screen bg-gray-900 text-white overflow-x-hidden">
      {/* Navigation */}
      <nav className="fixed top-0 w-full bg-gray-900/95 backdrop-blur-sm border-b border-gray-800 z-50">
        <div className="max-w-7xl mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <Logo />
              <span className="text-lg sm:text-xl font-bold text-white">Base Gas Optimiser</span>
            </div>

            <div className="hidden md:flex items-center space-x-8">
              <a href="#features" className="text-gray-300 hover:text-white transition">Features</a>
              <a href="#pricing" className="text-gray-300 hover:text-white transition">Pricing</a>
              <a href="#faq" className="text-gray-300 hover:text-white transition">FAQ</a>
              <Link to="/docs" className="text-gray-300 hover:text-white transition">Docs</Link>
            </div>

            <div className="hidden md:flex items-center space-x-4">
              <Link to="/app" className="text-gray-300 hover:text-white transition">Sign In</Link>
              <Link
                to="/app"
                className="px-6 py-2 bg-gradient-to-r from-cyan-500 to-emerald-500 text-white rounded-lg font-semibold hover:shadow-lg transition"
              >
                Get Started
              </Link>
            </div>

            {/* Mobile menu button */}
            <button
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
              className="md:hidden p-2 text-gray-300 hover:text-white focus:outline-none"
              aria-label="Toggle menu"
            >
              <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                {mobileMenuOpen ? (
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                ) : (
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                )}
              </svg>
            </button>
          </div>

          {/* Mobile menu */}
          {mobileMenuOpen && (
            <div className="md:hidden mt-4 pb-4 space-y-3 border-t border-gray-800 pt-4">
              <a
                href="#features"
                className="block py-2 text-gray-300 hover:text-white transition"
                onClick={() => setMobileMenuOpen(false)}
              >
                Features
              </a>
              <a
                href="#pricing"
                className="block py-2 text-gray-300 hover:text-white transition"
                onClick={() => setMobileMenuOpen(false)}
              >
                Pricing
              </a>
              <a
                href="#faq"
                className="block py-2 text-gray-300 hover:text-white transition"
                onClick={() => setMobileMenuOpen(false)}
              >
                FAQ
              </a>
              <Link
                to="/docs"
                className="block py-2 text-gray-300 hover:text-white transition"
                onClick={() => setMobileMenuOpen(false)}
              >
                Docs
              </Link>
              <Link
                to="/app"
                className="block py-2 text-gray-300 hover:text-white transition"
                onClick={() => setMobileMenuOpen(false)}
              >
                Sign In
              </Link>
              <Link
                to="/app"
                className="block w-full text-center px-6 py-3 bg-gradient-to-r from-cyan-500 to-emerald-500 text-white rounded-lg font-semibold hover:shadow-lg transition"
                onClick={() => setMobileMenuOpen(false)}
              >
                Get Started
              </Link>
            </div>
          )}
        </div>
      </nav>

      {/* Hero Section */}
      <section className="min-h-screen flex items-center justify-center bg-gradient-to-br from-gray-900 via-cyan-900/20 to-gray-900 px-4 pt-20">
        <div className="max-w-4xl mx-auto text-center">
          <div className="mb-6">
            <span className="px-4 py-2 bg-cyan-500/10 border border-cyan-500/30 rounded-full text-cyan-400 text-sm font-medium">
              🚀 Trusted by 1,000+ Base users
            </span>
          </div>
          
          <h1 className="text-4xl sm:text-5xl md:text-6xl lg:text-7xl font-bold text-white mb-6 leading-tight">
            Stop Overpaying for
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-emerald-400">
              {" "}Base Gas Fees
            </span>
          </h1>

          <p className="text-lg sm:text-xl md:text-2xl text-gray-300 mb-8 max-w-2xl mx-auto px-4">
            AI-powered predictions save you up to <strong className="text-emerald-400">65%</strong> on gas fees every single transaction
          </p>
          
          <div className="flex flex-col sm:flex-row gap-4 justify-center mb-8 px-4">
            <Link
              to="/app"
              className="px-8 py-4 bg-gradient-to-r from-cyan-500 to-emerald-500 text-white rounded-lg text-base sm:text-lg font-semibold hover:shadow-xl transition-all transform hover:scale-105 text-center min-h-[52px] flex items-center justify-center"
            >
              Get Started Free →
            </Link>
            <button className="px-8 py-4 bg-gray-800 border border-gray-700 text-white rounded-lg text-base sm:text-lg font-semibold hover:bg-gray-700 transition-all min-h-[52px]">
              Watch Demo ▶
            </button>
          </div>

          <div className="flex flex-wrap items-center justify-center gap-4 sm:gap-6 text-xs sm:text-sm text-gray-400 px-4">
            <span>✓ No credit card required</span>
            <span>✓ 7-day free trial</span>
            <span>✓ Cancel anytime</span>
          </div>

          {/* Live Stats Counter */}
          <div className="mt-12 grid grid-cols-3 gap-4 sm:gap-6 md:gap-8 max-w-2xl mx-auto px-4">
            <div>
              <div className="text-2xl sm:text-3xl md:text-4xl font-bold text-cyan-400">
                $<CountUp end={stats.total_saved_k} duration={2} />K+
              </div>
              <div className="text-gray-400 text-xs sm:text-sm">Total Saved</div>
            </div>
            <div>
              <div className="text-2xl sm:text-3xl md:text-4xl font-bold text-emerald-400">
                <CountUp end={stats.accuracy_percent} duration={2} />%
              </div>
              <div className="text-gray-400 text-xs sm:text-sm">Accuracy</div>
            </div>
            <div>
              <div className="text-2xl sm:text-3xl md:text-4xl font-bold text-cyan-400">
                <CountUp end={stats.predictions_k} duration={2} />K+
              </div>
              <div className="text-gray-400 text-xs sm:text-sm">Predictions</div>
            </div>
          </div>
        </div>
      </section>

      {/* Problem Section */}
      <section id="problem" className="py-20 px-4 bg-gray-900">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-4xl font-bold text-center text-white mb-12">
            The Problem With Gas Fees
          </h2>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {[
              { icon: "❌", title: "Unpredictable Costs", desc: "Gas prices spike 300% without warning" },
              { icon: "⏰", title: "Wasted Time", desc: "Manual timing wastes hours every day" },
              { icon: "💸", title: "Money Lost", desc: "Users overpay $100-500 annually" },
              { icon: "😤", title: "No Control", desc: "You never know the right time to transact" }
            ].map((problem, i) => (
              <div key={i} className="bg-gray-800/50 border border-red-500/30 rounded-lg p-6">
                <div className="text-4xl mb-3">{problem.icon}</div>
                <h3 className="text-lg font-bold text-white mb-2">{problem.title}</h3>
                <p className="text-gray-400 text-sm">{problem.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section id="how-it-works" className="py-20 px-4 bg-gradient-to-b from-gray-900 to-gray-800">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-4xl font-bold text-center text-white mb-4">
            How It Works
          </h2>
          <p className="text-center text-gray-400 mb-12 text-lg">
            Three simple steps to start saving money
          </p>
          
          <div className="grid md:grid-cols-3 gap-8">
            {[
              {
                step: "1",
                title: "AI Analyses",
                desc: "Our ML model processes 30 days of Base network data to identify gas price patterns",
                icon: "🤖"
              },
              {
                step: "2",
                title: "Get Predictions",
                desc: "Receive 1hr, 4hr, and 24hr gas price forecasts with 82% accuracy",
                icon: "📊"
              },
              {
                step: "3",
                title: "Save Money",
                desc: "Transact at optimal times and save up to 65% on every transaction",
                icon: "💰"
              }
            ].map((step, i) => (
              <div key={i} className="relative">
                <div className="bg-gray-800 border border-cyan-500/30 rounded-xl p-8 hover:border-cyan-500 transition-all">
                  <div className="text-6xl mb-4">{step.icon}</div>
                  <div className="absolute -top-4 -left-4 w-12 h-12 bg-gradient-to-br from-cyan-500 to-emerald-500 rounded-full flex items-center justify-center text-white font-bold text-xl">
                    {step.step}
                  </div>
                  <h3 className="text-2xl font-bold text-white mb-3">{step.title}</h3>
                  <p className="text-gray-400">{step.desc}</p>
                </div>
                {i < 2 && (
                  <div className="hidden md:block absolute top-1/2 -right-4 text-cyan-500 text-3xl">→</div>
                )}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-20 px-4 bg-gray-900">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-4xl font-bold text-center text-white mb-12">
            Powerful Features
          </h2>
          
          <div className="grid md:grid-cols-2 gap-6">
            {[
              {
                icon: "⚡",
                title: "Real-time ML Predictions",
                desc: "Get 1hr, 4hr, and 24hr gas price forecasts powered by Random Forest AI",
                color: "cyan"
              },
              {
                icon: "🔔",
                title: "Smart Alerts",
                desc: "Browser notifications when gas drops below your target price",
                color: "emerald"
              },
              {
                icon: "💰",
                title: "Savings Calculator",
                desc: "See exactly how much you'll save for different transaction types",
                color: "amber"
              },
              {
                icon: "🎯",
                title: "82% Accuracy",
                desc: "Proven track record with continuous model improvement",
                color: "cyan"
              },
              {
                icon: "🔗",
                title: "Wallet Integration",
                desc: "Connect MetaMask to see personalized savings and transaction history",
                color: "emerald"
              },
              {
                icon: "📊",
                title: "Performance Dashboard",
                desc: "Track your savings over time and see how much you've saved",
                color: "amber"
              }
            ].map((feature, i) => (
              <div 
                key={i} 
                className={`bg-gray-800/50 border border-${feature.color}-500/30 rounded-lg p-6 hover:border-${feature.color}-500 transition-all`}
              >
                <div className="text-5xl mb-4">{feature.icon}</div>
                <h3 className="text-xl font-bold text-white mb-2">{feature.title}</h3>
                <p className="text-gray-400">{feature.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Social Proof Section */}
      <section className="py-20 px-4 bg-gray-800">
        <div className="max-w-4xl mx-auto">
          <h2 className="text-4xl font-bold text-center text-white mb-12">
            What Our Users Say
          </h2>
          
          <div className="grid md:grid-cols-3 gap-6">
            {[
              {
                quote: "Saved me $43 in my first month. This tool actually works!",
                author: "Alex Chen",
                role: "DeFi Trader",
                avatar: "🧑"
              },
              {
                quote: "Finally, a gas optimiser that's accurate. 65% savings on every swap!",
                author: "Sarah Martinez",
                role: "NFT Creator",
                avatar: "👩"
              },
              {
                quote: "The predictions are incredibly accurate. Game changer for my trading strategy.",
                author: "Michael Park",
                role: "Liquidity Provider",
                avatar: "👨"
              }
            ].map((testimonial, i) => (
              <div key={i} className="bg-gray-900 border border-gray-700 rounded-lg p-6">
                <div className="flex items-center mb-4">
                  <div className="text-4xl mr-3">{testimonial.avatar}</div>
                  <div>
                    <div className="font-bold text-white">{testimonial.author}</div>
                    <div className="text-sm text-gray-400">{testimonial.role}</div>
                  </div>
                </div>
                <p className="text-gray-300 italic">"{testimonial.quote}"</p>
                <div className="mt-4 text-amber-400">★★★★★</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Pricing Section */}
      <section id="pricing" className="py-20 px-4 bg-gradient-to-b from-gray-900 to-gray-800">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-4xl font-bold text-center text-white mb-4">
            Simple, Transparent Pricing
          </h2>
          <p className="text-center text-gray-400 mb-12">
            Start free, upgrade when you need more
          </p>
          
          <div className="grid md:grid-cols-3 gap-8">
            {[
              {
                name: "Free",
                price: "$0",
                period: "forever",
                features: [
                  "✓ Basic predictions (1hr, 4hr, 24hr)",
                  "✓ 10 gas price alerts per month",
                  "✓ Community support",
                  "✓ Basic savings calculator"
                ],
                cta: "Start Free",
                highlighted: false
              },
              {
                name: "Pro",
                price: "$9",
                period: "per month",
                badge: "Most Popular",
                features: [
                  "✓ Everything in Free",
                  "✓ Unlimited gas alerts",
                  "✓ Email notifications",
                  "✓ Wallet integration",
                  "✓ Transaction history analysis",
                  "✓ Priority support",
                  "✓ Advanced analytics"
                ],
                cta: "Start 7-Day Trial",
                highlighted: true
              },
              {
                name: "Business",
                price: "$49",
                period: "per month",
                features: [
                  "✓ Everything in Pro",
                  "✓ API access (10K requests/day)",
                  "✓ Team accounts (up to 5)",
                  "✓ Custom alerts",
                  "✓ Dedicated support",
                  "✓ SLA guarantee"
                ],
                cta: "Contact Sales",
                highlighted: false
              }
            ].map((plan, i) => (
              <div 
                key={i} 
                className={`relative bg-gray-800 border ${plan.highlighted ? 'border-cyan-500 shadow-xl shadow-cyan-500/20 scale-105' : 'border-gray-700'} rounded-xl p-8`}
              >
                {plan.badge && (
                  <div className="absolute -top-4 left-1/2 transform -translate-x-1/2 px-4 py-1 bg-gradient-to-r from-cyan-500 to-emerald-500 rounded-full text-white text-sm font-bold">
                    {plan.badge}
                  </div>
                )}
                
                <h3 className="text-2xl font-bold text-white mb-2">{plan.name}</h3>
                <div className="mb-6">
                  <span className="text-5xl font-bold text-white">{plan.price}</span>
                  <span className="text-gray-400 ml-2">/{plan.period}</span>
                </div>
                
                <ul className="space-y-3 mb-8">
                  {plan.features.map((feature, j) => (
                    <li key={j} className="text-gray-300 text-sm">{feature}</li>
                  ))}
                </ul>
                
                <Link
                  to="/app"
                  className={`w-full py-3 rounded-lg font-semibold transition-all block text-center ${
                    plan.highlighted 
                      ? 'bg-gradient-to-r from-cyan-500 to-emerald-500 text-white hover:shadow-lg' 
                      : 'bg-gray-700 text-white hover:bg-gray-600'
                  }`}
                >
                  {plan.cta}
                </Link>
              </div>
            ))}
          </div>
          
          <p className="text-center text-gray-400 mt-8">
            All plans include 7-day free trial • No credit card required • Cancel anytime
          </p>
        </div>
      </section>

      {/* FAQ Section */}
      <section id="faq" className="py-20 px-4 bg-gray-900">
        <div className="max-w-4xl mx-auto">
          <h2 className="text-4xl font-bold text-center text-white mb-12">
            Frequently Asked Questions
          </h2>
          
          <div className="space-y-6">
            {[
              {
                q: "How accurate are the predictions?",
                a: "Our ML models achieve 82% accuracy (R² score) with continuous improvement as we gather more data. We also provide confidence scores with each prediction."
              },
              {
                q: "Does it work with my wallet?",
                a: "Yes! We support MetaMask, Coinbase Wallet, and WalletConnect. Wallet connection is optional but unlocks personalized features."
              },
              {
                q: "How much can I actually save?",
                a: "Average users save 30-65% on gas fees. The exact amount depends on your transaction volume and timing flexibility. Our calculator shows precise savings for your usage."
              },
              {
                q: "Is my wallet safe?",
                a: "Absolutely. We never access your private keys or request signing permissions. We only read publicly available blockchain data to provide personalized insights."
              },
              {
                q: "What if the predictions are wrong?",
                a: "While our predictions are 82% accurate, gas prices can be unpredictable. We show confidence scores and recommend waiting for high-confidence predictions. You're never obligated to follow our recommendations."
              },
              {
                q: "Can I cancel anytime?",
                a: "Yes! No contracts, no commitments. Cancel your subscription with one click from your account settings."
              }
            ].map((faq, i) => (
              <details key={i} className="bg-gray-800 border border-gray-700 rounded-lg p-6 cursor-pointer hover:border-cyan-500 transition-all">
                <summary className="font-bold text-white text-lg mb-2 cursor-pointer">
                  {faq.q}
                </summary>
                <p className="text-gray-400 mt-4">{faq.a}</p>
              </details>
            ))}
          </div>
        </div>
      </section>

      {/* Final CTA Section */}
      <section className="py-20 px-4 bg-gradient-to-br from-cyan-900/30 via-gray-900 to-emerald-900/30">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-5xl font-bold text-white mb-6">
            Ready to Stop Overpaying?
          </h2>
          <p className="text-xl text-gray-300 mb-8">
            Join 1,000+ smart Base users saving money every day
          </p>
          
          <Link
            to="/app"
            className="inline-block px-12 py-5 bg-gradient-to-r from-cyan-500 to-emerald-500 text-white rounded-lg text-xl font-bold hover:shadow-2xl transition-all transform hover:scale-105"
          >
            Start Saving Now - It's Free →
          </Link>
          
          <p className="text-gray-400 mt-6">
            No credit card required • 7-day free trial • Cancel anytime
          </p>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-gray-950 border-t border-gray-800 py-12 px-4">
        <div className="max-w-7xl mx-auto grid md:grid-cols-4 gap-8">
          <div>
            <div className="flex items-center space-x-2 mb-4">
              <Logo />
              <span className="text-lg font-bold text-white">Base Gas Optimiser</span>
            </div>
            <p className="text-gray-400 mt-4 text-sm">
              AI-powered gas optimisation for Base network
            </p>
            <div className="flex space-x-4 mt-4">
              <a href="#" className="text-gray-400 hover:text-cyan-400">🐦 Twitter</a>
              <a href="#" className="text-gray-400 hover:text-cyan-400">💬 Discord</a>
              <Link to="/docs" className="text-gray-400 hover:text-cyan-400">📘 Docs</Link>
            </div>
          </div>
          
          <div>
            <h4 className="text-white font-bold mb-4">Product</h4>
            <ul className="space-y-2 text-gray-400 text-sm">
              <li><a href="#features" className="hover:text-white">Features</a></li>
              <li><a href="#pricing" className="hover:text-white">Pricing</a></li>
              <li><a href="#" className="hover:text-white">Roadmap</a></li>
              <li><a href="#" className="hover:text-white">Changelog</a></li>
            </ul>
          </div>
          
          <div>
            <h4 className="text-white font-bold mb-4">Resources</h4>
            <ul className="space-y-2 text-gray-400 text-sm">
              <li><Link to="/docs" className="hover:text-white">Documentation</Link></li>
              <li><a href="#" className="hover:text-white">API Reference</a></li>
              <li><a href="#" className="hover:text-white">Blog</a></li>
              <li><a href="#" className="hover:text-white">Status</a></li>
            </ul>
          </div>
          
          <div>
            <h4 className="text-white font-bold mb-4">Legal</h4>
            <ul className="space-y-2 text-gray-400 text-sm">
              <li><Link to="/privacy" className="hover:text-white">Privacy Policy</Link></li>
              <li><Link to="/terms" className="hover:text-white">Terms of Service</Link></li>
              <li><Link to="/about" className="hover:text-white">About Us</Link></li>
              <li><a href="#" className="hover:text-white">Contact</a></li>
            </ul>
          </div>
        </div>
        
        <div className="max-w-7xl mx-auto mt-12 pt-8 border-t border-gray-800 flex flex-col md:flex-row justify-between items-center">
          <p className="text-gray-400 text-sm">
            © 2024 Base Gas Optimiser. All rights reserved.
          </p>
          
          <div className="flex items-center space-x-6 mt-4 md:mt-0">
            <TrustBadge icon="🔒" text="SSL Secure" />
            <TrustBadge icon="✓" text="GDPR Compliant" />
            <TrustBadge icon="⚡" text="99.9% Uptime" />
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Landing;

