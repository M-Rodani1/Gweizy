import React, { useState } from 'react';
import { Link } from 'react-router-dom';

const Pricing: React.FC = () => {
  const [billingPeriod, setBillingPeriod] = useState<'monthly' | 'annual'>('monthly');
  const [expandedFaq, setExpandedFaq] = useState<number | null>(null);

  const tiers = {
    free: {
      name: 'Free',
      subtitle: 'Hobbyist',
      price: { monthly: 0, annual: 0 },
      description: 'Perfect for casual Base users',
      features: [
        { text: 'Full predictions (1h, 4h, 24h)', included: true, highlight: false },
        { text: 'Traffic light indicator', included: true, highlight: false },
        { text: 'Basic savings calculator', included: true, highlight: false },
        { text: '50 predictions per day', included: true, highlight: false },
        { text: '7 days historical data', included: true, highlight: false },
        { text: 'Extended forecasts', included: false, highlight: false },
        { text: 'Custom alerts', included: false, highlight: false },
        { text: 'API access', included: false, highlight: false },
      ],
      cta: 'Get Started',
      ctaLink: '/app',
      popular: false,
      badge: undefined,
      roi: undefined,
    },
    pro: {
      name: 'Pro',
      subtitle: 'Active Traders',
      price: { monthly: 9, annual: 90 },
      description: 'For active DeFi users who transact frequently',
      features: [
        { text: 'Everything in Free', included: true, highlight: true },
        { text: 'Remove all ads', included: true, highlight: false },
        { text: 'Extended forecasts (48h, 7-day)', included: true, highlight: false },
        { text: 'Email/Discord/Telegram alerts', included: true, highlight: false },
        { text: 'Personal savings tracker', included: true, highlight: false },
        { text: '500 predictions per day', included: true, highlight: false },
        { text: 'Email support (48h)', included: true, highlight: false },
        { text: 'API access', included: false, highlight: false },
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
        { text: 'API access (1,000 req/day)', included: true, highlight: false },
        { text: 'Real-time data (30s refresh)', included: true, highlight: false },
        { text: 'Unlimited predictions', included: true, highlight: false },
        { text: 'Webhook notifications', included: true, highlight: false },
        { text: 'Priority support (24h)', included: true, highlight: false },
        { text: 'Beta feature access', included: true, highlight: false },
      ],
      cta: 'Get Started',
      ctaLink: '/app?upgrade=max',
      popular: false,
      badge: undefined,
      roi: 'Build profitable bots, save $100+/month',
    },
  };

  const faqs = [
    {
      question: 'How much can I actually save?',
      answer: 'On average, users save 15-30% on gas fees by timing transactions optimally. If you make 10 transactions per month at $10 gas each, that\'s $15-30 saved. Pro tier pays for itself after 1-2 optimized transactions.'
    },
    {
      question: 'Can I upgrade or downgrade anytime?',
      answer: 'Yes! You can upgrade, downgrade, or cancel your subscription at any time. Changes take effect at the start of your next billing cycle.'
    },
    {
      question: 'Do you offer refunds?',
      answer: 'We offer a 7-day free trial for Pro tier. If you\'re not satisfied within the first 30 days of any paid subscription, contact us for a full refund.'
    },
    {
      question: 'What payment methods do you accept?',
      answer: 'We accept all major credit cards (Visa, Mastercard, Amex) and cryptocurrency payments (ETH, USDC on Base network). Payments are processed securely via Stripe.'
    },
    {
      question: 'Is there a limit on API requests for Max tier?',
      answer: 'Max tier includes 1,000 API requests per day. If you need more, contact us about enterprise pricing. We can accommodate high-volume use cases.'
    },
    {
      question: 'How accurate are the predictions?',
      answer: 'Our ML models achieve 75-80% directional accuracy (predicting if gas will go up or down) and explain 70%+ of gas price variance. We continuously monitor performance and retrain models when needed to maintain accuracy.'
    }
  ];

  return (
    <div style={{ minHeight: '100vh', background: 'var(--bg)', color: 'var(--text)' }}>
      {/* Navigation */}
      <nav style={{
        position: 'fixed',
        top: 0,
        width: '100%',
        background: 'var(--surface)',
        backdropFilter: 'blur(12px)',
        borderBottom: '1px solid var(--border)',
        zIndex: 50
      }}>
        <div className="container" style={{ padding: 'var(--space-lg) var(--space-xl)' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-md)' }}>
              <Link to="/" style={{ fontSize: '1.25rem', fontWeight: 700, textDecoration: 'none', color: 'var(--text)' }}>
                Base Gas Optimiser
              </Link>
              <span className="badge badge-accent">Coinbase 2025 Hackathon Winner</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-lg)' }}>
              <Link to="/" className="btn btn-ghost">Home</Link>
              <Link to="/app" className="btn btn-primary">Launch App</Link>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="section" style={{ paddingTop: '120px', textAlign: 'center' }}>
        <div className="container">
          <h1 style={{ marginBottom: 'var(--space-lg)' }}>Simple, Transparent Pricing</h1>
          <p style={{ fontSize: '1.25rem', color: 'var(--text-secondary)', marginBottom: 'var(--space-2xl)', maxWidth: '700px', margin: '0 auto' }}>
            Save money on every Base transaction. Choose the plan that fits your needs.
          </p>

          {/* Billing Toggle */}
          <div style={{
            display: 'inline-flex',
            alignItems: 'center',
            gap: 'var(--space-sm)',
            padding: 'var(--space-xs)',
            background: 'var(--surface-2)',
            borderRadius: 'var(--radius-md)',
            marginTop: 'var(--space-2xl)',
            marginBottom: 'var(--space-3xl)'
          }}>
            <button
              onClick={() => setBillingPeriod('monthly')}
              className={billingPeriod === 'monthly' ? 'btn btn-primary' : 'btn btn-ghost'}
              style={{ padding: 'var(--space-md) var(--space-xl)' }}
            >
              Monthly
            </button>
            <button
              onClick={() => setBillingPeriod('annual')}
              className={billingPeriod === 'annual' ? 'btn btn-primary' : 'btn btn-ghost'}
              style={{ padding: 'var(--space-md) var(--space-xl)', display: 'flex', alignItems: 'center', gap: 'var(--space-sm)' }}
            >
              Annual
              <span className="badge badge-success" style={{ fontSize: '0.625rem' }}>
                Save 16%
              </span>
            </button>
          </div>
        </div>
      </section>

      {/* Pricing Cards */}
      <section className="section" style={{ paddingTop: 0 }}>
        <div className="container">
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: 'var(--space-xl)', maxWidth: '1200px', margin: '0 auto' }}>
            {Object.entries(tiers).map(([key, tier]) => (
              <div
                key={key}
                className="card"
                style={{
                  position: 'relative',
                  padding: 'var(--space-xl)',
                  background: tier.popular ? 'var(--surface-2)' : 'var(--surface)',
                  border: tier.popular ? '2px solid var(--accent)' : '1px solid var(--border)',
                  transition: 'transform 0.2s',
                }}
              >
                {/* Popular Badge */}
                {tier.popular && (
                  <div style={{ position: 'absolute', top: '-12px', left: '50%', transform: 'translateX(-50%)' }}>
                    <span className="badge badge-accent">
                      {tier.badge}
                    </span>
                  </div>
                )}

                {/* Header */}
                <div style={{ textAlign: 'center', marginBottom: 'var(--space-lg)' }}>
                  <h3 style={{ marginBottom: 'var(--space-xs)' }}>{tier.name}</h3>
                  <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)', marginBottom: 'var(--space-lg)' }}>{tier.subtitle}</p>

                  <div style={{ marginBottom: 'var(--space-md)' }}>
                    <span style={{ fontSize: '3.5rem', fontWeight: 700, lineHeight: 1 }}>
                      ${tier.price[billingPeriod]}
                    </span>
                    {tier.price.monthly > 0 && (
                      <span style={{ color: 'var(--text-muted)', marginLeft: 'var(--space-sm)' }}>
                        /{billingPeriod === 'monthly' ? 'month' : 'year'}
                      </span>
                    )}
                  </div>

                  {billingPeriod === 'annual' && tier.price.annual > 0 && (
                    <p style={{ fontSize: '0.875rem', color: 'var(--success)', marginBottom: 'var(--space-md)' }}>
                      ${(tier.price.annual / 12).toFixed(2)}/month billed annually
                    </p>
                  )}

                  <p style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}>{tier.description}</p>
                </div>

                {/* ROI Badge */}
                {tier.roi && (
                  <div style={{
                    marginBottom: 'var(--space-lg)',
                    padding: 'var(--space-md)',
                    background: 'var(--success-bg)',
                    border: '1px solid var(--success-border)',
                    borderRadius: 'var(--radius-md)'
                  }}>
                    <p style={{ fontSize: '0.875rem', fontWeight: 600, color: 'var(--success)', textAlign: 'center' }}>
                      💰 {tier.roi}
                    </p>
                  </div>
                )}

                {/* Features */}
                <ul style={{ listStyle: 'none', padding: 0, marginBottom: 'var(--space-xl)' }}>
                  {tier.features.map((feature, idx) => (
                    <li key={idx} style={{ display: 'flex', alignItems: 'flex-start', gap: 'var(--space-md)', marginBottom: 'var(--space-md)' }}>
                      {feature.included ? (
                        <svg
                          style={{ width: '20px', height: '20px', color: 'var(--success)', flexShrink: 0, marginTop: '2px' }}
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
                          style={{ width: '20px', height: '20px', color: 'var(--text-muted)', flexShrink: 0, marginTop: '2px', opacity: 0.3 }}
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
                        style={{
                          fontSize: '0.875rem',
                          color: feature.included ? 'var(--text-secondary)' : 'var(--text-muted)',
                          fontWeight: feature.highlight ? 600 : 400,
                          opacity: feature.included ? 1 : 0.5
                        }}
                      >
                        {feature.text}
                      </span>
                    </li>
                  ))}
                </ul>

                {/* CTA Button */}
                <Link
                  to={tier.ctaLink}
                  className={tier.popular ? 'btn btn-primary' : 'btn btn-secondary'}
                  style={{
                    display: 'block',
                    width: '100%',
                    textAlign: 'center',
                    padding: 'var(--space-md) var(--space-lg)',
                    textDecoration: 'none',
                    fontWeight: 600
                  }}
                >
                  {tier.cta}
                </Link>

                {key === 'pro' && (
                  <p style={{ textAlign: 'center', fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: 'var(--space-md)' }}>
                    7-day free trial • Cancel anytime
                  </p>
                )}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* FAQ Section */}
      <section className="section" style={{ background: 'var(--surface)' }}>
        <div className="container" style={{ maxWidth: '900px' }}>
          <h2 style={{ textAlign: 'center', marginBottom: 'var(--space-3xl)' }}>
            Frequently Asked Questions
          </h2>

          <div style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-md)' }}>
            {faqs.map((faq, idx) => (
              <div key={idx} className="card" style={{ overflow: 'hidden' }}>
                <button
                  onClick={() => setExpandedFaq(expandedFaq === idx ? null : idx)}
                  style={{
                    width: '100%',
                    padding: 'var(--space-lg)',
                    background: 'transparent',
                    border: 'none',
                    color: 'var(--text)',
                    cursor: 'pointer',
                    display: 'flex',
                    justifyContent: 'space-between',
                    alignItems: 'center',
                    textAlign: 'left',
                    fontWeight: 600,
                    fontSize: '1.125rem'
                  }}
                >
                  {faq.question}
                  <span style={{
                    fontSize: '1.5rem',
                    color: 'var(--accent)',
                    transform: expandedFaq === idx ? 'rotate(180deg)' : 'rotate(0deg)',
                    transition: 'transform 0.2s'
                  }}>
                    ▼
                  </span>
                </button>
                {expandedFaq === idx && (
                  <div style={{
                    padding: '0 var(--space-lg) var(--space-lg) var(--space-lg)',
                    color: 'var(--text-secondary)',
                    lineHeight: 1.7
                  }}>
                    {faq.answer}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="section" style={{ textAlign: 'center', background: 'var(--bg)' }}>
        <div className="container" style={{ maxWidth: '800px' }}>
          <h2 style={{ marginBottom: 'var(--space-lg)' }}>
            Ready to Save on Gas Fees?
          </h2>
          <p style={{ fontSize: '1.25rem', color: 'var(--text-secondary)', marginBottom: 'var(--space-2xl)' }}>
            Join thousands of Base users saving money with AI-powered predictions
          </p>
          <div style={{ display: 'flex', gap: 'var(--space-md)', justifyContent: 'center', flexWrap: 'wrap' }}>
            <Link
              to="/app"
              className="btn btn-primary"
              style={{
                padding: '1.25rem 2.5rem',
                fontSize: '1.125rem',
                textDecoration: 'none'
              }}
            >
              Start Free Trial →
            </Link>
            <Link
              to="/"
              className="btn btn-secondary"
              style={{
                padding: '1.25rem 2.5rem',
                fontSize: '1.125rem',
                textDecoration: 'none'
              }}
            >
              Learn More
            </Link>
          </div>
          <p style={{ marginTop: 'var(--space-lg)', fontSize: '0.875rem', color: 'var(--text-muted)' }}>
            No credit card required for Free tier • 7-day trial for Pro
          </p>
        </div>
      </section>

      {/* Footer */}
      <footer style={{ padding: 'var(--space-2xl) 0', background: 'var(--bg)', borderTop: '1px solid var(--border)' }}>
        <div className="container">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 'var(--space-lg)' }}>
            <div style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>
              © 2025 Base Gas Optimiser. Built at Coinbase 2025 Hackathon.
            </div>
            <div style={{ display: 'flex', gap: 'var(--space-lg)', fontSize: '0.875rem', color: 'var(--text-muted)' }}>
              <Link to="/pricing" style={{ color: 'var(--text-muted)', textDecoration: 'none' }}>Pricing</Link>
              <span>•</span>
              <Link to="/" style={{ color: 'var(--text-muted)', textDecoration: 'none' }}>About</Link>
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
