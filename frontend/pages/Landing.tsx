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
              <span style={{ fontSize: '1.25rem', fontWeight: 700 }}>Base Gas Optimiser</span>
              <span className="badge badge-accent">Hackathon Winner</span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-lg)' }}>
              <Link to="/pricing" className="btn btn-ghost">Pricing</Link>
              <Link to="/app" className="btn btn-primary">Launch App</Link>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section - 2 Column Layout */}
      <section className="section" style={{ paddingTop: '120px', minHeight: '90vh', display: 'flex', alignItems: 'center' }}>
        <div className="container">
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(500px, 1fr))', gap: 'var(--space-3xl)', alignItems: 'center' }}>

            {/* Left: Hero Content */}
            <div>
              <div className="badge badge-accent" style={{ marginBottom: 'var(--space-lg)' }}>
                <span>🏆</span> AI Hack Nation 2024 Winner
              </div>

              <h1 style={{ fontSize: '3.5rem', fontWeight: 700, lineHeight: 1.1, marginBottom: 'var(--space-lg)' }}>
                <span style={{ color: 'var(--accent)' }}>AI Transaction Pilot</span>{' '}
                for Base Network
              </h1>

              <p style={{ fontSize: '1.25rem', color: 'var(--text-secondary)', marginBottom: 'var(--space-md)', lineHeight: 1.6 }}>
                DQN Neural Network tells you exactly when to transact. Save up to 40% on gas fees.
              </p>

              <p style={{ fontSize: '1rem', color: 'var(--text-muted)', marginBottom: 'var(--space-2xl)' }}>
                Multi-chain support: Base, Ethereum, Arbitrum, Optimism, Polygon
              </p>

              <div style={{ display: 'flex', gap: 'var(--space-md)', flexWrap: 'wrap', marginBottom: 'var(--space-2xl)' }}>
                <Link to="/app" className="btn btn-primary" style={{ padding: '1rem 2rem', fontSize: '1.125rem', background: 'linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%)' }}>
                  Launch AI Pilot →
                </Link>
                <a href="#how-it-works" className="btn btn-secondary" style={{ padding: '1rem 2rem', fontSize: '1.125rem' }}>
                  See How It Works
                </a>
              </div>

              {/* Trust Indicators */}
              <div style={{ display: 'flex', gap: 'var(--space-xl)', marginTop: 'var(--space-2xl)', paddingTop: 'var(--space-xl)', borderTop: '1px solid var(--border-subtle)' }}>
                <div>
                  <div style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--accent)' }}>
                    {statsLoading ? '...' : `$${stats.total_saved_k}K+`}
                  </div>
                  <div style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>Gas Saved</div>
                </div>
                <div>
                  <div style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--accent)' }}>
                    {statsLoading ? '...' : `${stats.accuracy_percent}%`}
                  </div>
                  <div style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>Accuracy</div>
                </div>
                <div>
                  <div style={{ fontSize: '2rem', fontWeight: 700, color: 'var(--accent)' }}>
                    {statsLoading ? '...' : `${stats.predictions_k}K+`}
                  </div>
                  <div style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>Predictions</div>
                </div>
              </div>
            </div>

            {/* Right: Dashboard Preview in Browser Frame */}
            <div style={{ position: 'relative' }}>
              <div className="card" style={{ padding: 0, overflow: 'hidden', background: 'var(--surface-2)' }}>
                {/* Browser Chrome */}
                <div style={{ padding: 'var(--space-md)', borderBottom: '1px solid var(--border)', display: 'flex', alignItems: 'center', gap: 'var(--space-sm)' }}>
                  <div style={{ display: 'flex', gap: 'var(--space-xs)' }}>
                    <div style={{ width: '12px', height: '12px', borderRadius: '50%', background: '#ef4444' }}></div>
                    <div style={{ width: '12px', height: '12px', borderRadius: '50%', background: '#f59e0b' }}></div>
                    <div style={{ width: '12px', height: '12px', borderRadius: '50%', background: '#10b981' }}></div>
                  </div>
                  <div style={{ flex: 1, background: 'var(--surface)', padding: 'var(--space-xs) var(--space-md)', borderRadius: 'var(--radius-sm)', fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                    basegasfeesml.pages.dev/app
                  </div>
                </div>

                {/* Dashboard Preview Content */}
                <div style={{ padding: 'var(--space-xl)', background: 'var(--bg)' }}>
                  {/* Mini KPI Cards */}
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 'var(--space-md)', marginBottom: 'var(--space-lg)' }}>
                    <div className="card" style={{ padding: 'var(--space-md)' }}>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: 'var(--space-xs)' }}>Current Gas</div>
                      <div style={{ fontSize: '1.5rem', fontWeight: 700 }}>0.0048 gwei</div>
                      <span className="badge badge-success" style={{ marginTop: 'var(--space-xs)', fontSize: '0.625rem' }}>
                        <span className="status-dot success"></span> Low
                      </span>
                    </div>
                    <div className="card" style={{ padding: 'var(--space-md)' }}>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: 'var(--space-xs)' }}>1h Forecast</div>
                      <div style={{ fontSize: '1.5rem', fontWeight: 700 }}>0.0052 gwei</div>
                      <span className="badge badge-warning" style={{ marginTop: 'var(--space-xs)', fontSize: '0.625rem' }}>
                        <span className="status-dot warning"></span> Rising
                      </span>
                    </div>
                  </div>

                  {/* Mini Chart Placeholder */}
                  <div className="card" style={{ padding: 'var(--space-lg)', height: '180px', display: 'flex', alignItems: 'flex-end', gap: '4px' }}>
                    {[40, 60, 55, 70, 45, 50, 35, 60, 55, 48, 42, 38].map((height, i) => (
                      <div key={i} style={{ flex: 1, background: 'var(--accent-light)', borderRadius: '2px', height: `${height}%`, transition: 'all 0.3s' }}></div>
                    ))}
                  </div>

                  {/* Recommendation */}
                  <div className="card" style={{ marginTop: 'var(--space-md)', padding: 'var(--space-md)', background: 'var(--success-bg)', border: '1px solid var(--success-border)' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 'var(--space-sm)' }}>
                      <span style={{ fontSize: '1.5rem' }}>✅</span>
                      <div style={{ fontSize: '0.875rem', fontWeight: 600, color: 'var(--success)' }}>
                        Good time to transact - Gas is 25% below average
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section id="how-it-works" className="section" style={{ background: 'var(--surface)' }}>
        <div className="container">
          <div style={{ textAlign: 'center', marginBottom: 'var(--space-3xl)' }}>
            <h2>How It Works</h2>
            <p style={{ fontSize: '1.25rem', color: 'var(--text-secondary)', maxWidth: '600px', margin: '0 auto', marginTop: 'var(--space-md)' }}>
              Machine learning models trained on real Base network data predict optimal transaction times
            </p>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 'var(--space-xl)' }}>
            <div className="card" style={{ padding: 'var(--space-xl)', textAlign: 'center' }}>
              <div style={{ fontSize: '3rem', marginBottom: 'var(--space-md)' }}>📊</div>
              <h3 style={{ marginBottom: 'var(--space-md)' }}>1. Real-Time Analysis</h3>
              <p style={{ color: 'var(--text-secondary)', lineHeight: 1.7 }}>
                Our system monitors Base network activity every minute, tracking gas prices, congestion, and transaction patterns
              </p>
            </div>

            <div className="card" style={{ padding: 'var(--space-xl)', textAlign: 'center' }}>
              <div style={{ fontSize: '3rem', marginBottom: 'var(--space-md)' }}>🤖</div>
              <h3 style={{ marginBottom: 'var(--space-md)' }}>2. AI Predictions</h3>
              <p style={{ color: 'var(--text-secondary)', lineHeight: 1.7 }}>
                Machine learning models predict gas prices for the next 1, 4, and 24 hours with high accuracy
              </p>
            </div>

            <div className="card" style={{ padding: 'var(--space-xl)', textAlign: 'center' }}>
              <div style={{ fontSize: '3rem', marginBottom: 'var(--space-md)' }}>💰</div>
              <h3 style={{ marginBottom: 'var(--space-md)' }}>3. Save Money</h3>
              <p style={{ color: 'var(--text-secondary)', lineHeight: 1.7 }}>
                Get instant recommendations on when to transact, saving up to 40% on gas fees compared to peak times
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Features Grid - Compact */}
      <section className="section">
        <div className="container">
          <div style={{ textAlign: 'center', marginBottom: 'var(--space-3xl)' }}>
            <h2>Everything You Need</h2>
            <p style={{ fontSize: '1.25rem', color: 'var(--text-secondary)' }}>
              Powerful tools to optimize your Base transactions
            </p>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))', gap: 'var(--space-lg)' }}>
            {[
              { icon: '🚦', title: 'Traffic Light System', desc: 'Instant visual indicator shows if NOW is a good time to transact' },
              { icon: '⏰', title: 'Best Time Widget', desc: 'See exactly when gas is cheapest today' },
              { icon: '📈', title: 'Price Predictions', desc: 'ML-powered forecasts for 1h, 4h, and 24h ahead' },
              { icon: '🗓️', title: '24-Hour Heatmap', desc: 'Interactive hourly breakdown shows gas price patterns' },
              { icon: '💸', title: 'Savings Calculator', desc: 'Calculate exactly how much you could save' },
              { icon: '⚡', title: 'Real-Time Updates', desc: 'Live data refreshed every 30 seconds' }
            ].map((feature, i) => (
              <div key={i} className="card" style={{ padding: 'var(--space-lg)' }}>
                <div style={{ fontSize: '2.5rem', marginBottom: 'var(--space-md)' }}>{feature.icon}</div>
                <h4 style={{ marginBottom: 'var(--space-sm)' }}>{feature.title}</h4>
                <p style={{ color: 'var(--text-secondary)', fontSize: '0.875rem', lineHeight: 1.6 }}>{feature.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Final CTA */}
      <section className="section" style={{ background: 'var(--surface)', textAlign: 'center' }}>
        <div className="container" style={{ maxWidth: '800px' }}>
          <h2 style={{ marginBottom: 'var(--space-lg)' }}>Stop Overpaying for Gas</h2>
          <p style={{ fontSize: '1.25rem', color: 'var(--text-secondary)', marginBottom: 'var(--space-2xl)' }}>
            Join thousands of Base users saving money with AI-powered gas predictions
          </p>
          <Link to="/app" className="btn btn-primary" style={{ padding: '1.25rem 2.5rem', fontSize: '1.25rem' }}>
            Start Saving Now - It's Free →
          </Link>
          <p style={{ marginTop: 'var(--space-lg)', fontSize: '0.875rem', color: 'var(--text-muted)' }}>
            No wallet connection required • No sign-up • Instant access
          </p>
        </div>
      </section>

      {/* Footer */}
      <footer style={{ padding: 'var(--space-2xl) 0', background: 'var(--bg)', borderTop: '1px solid var(--border)' }}>
        <div className="container">
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 'var(--space-lg)' }}>
            <div style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>
              © 2024 Base Gas Optimiser. Built at AI Hack Nation 2024.
            </div>
            <div style={{ display: 'flex', gap: 'var(--space-lg)', fontSize: '0.875rem', color: 'var(--text-muted)' }}>
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
