import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  Activity,
  Brain,
  Calculator,
  CheckCircle,
  Clock,
  Coins,
  Grid3x3,
  TrendingUp,
  Trophy,
  Zap
} from 'lucide-react';
import { fetchGlobalStats } from '../src/api/gasApi';
import Logo from '../src/components/branding/Logo';
import { Button, Badge, Stat, Card } from '../src/components/ui';

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
    <div className="landing-shell min-h-screen">
      {/* Navigation */}
      <nav className="landing-nav">
        <div className="container">
          <div className="landing-nav-content">
            <div className="landing-nav-brand">
              <div className="icon-tile">
                <Logo size="sm" />
              </div>
              <span className="text-[1.1rem] font-semibold tracking-[-0.01em]">
                Gweizy
              </span>
              <Badge variant="accent" icon={<Trophy size={14} />}>Hackathon Winner</Badge>
            </div>
            <div className="landing-nav-links">
              <Link to="/pricing" className="btn btn-ghost">Pricing</Link>
              <Link to="/app">
                <Button variant="primary" size="lg">Launch AI Pilot</Button>
              </Link>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section - 2 Column Layout */}
      <section className="section landing-hero">
        <div className="container">
          <div className="landing-hero-grid">

            {/* Left: Hero Content */}
            <div>
              <Badge variant="accent" className="mb-[var(--space-lg)] inline-flex items-center gap-[0.35rem]" icon={<Trophy size={14} />}>
                Coinbase 2025 Hackathon Winner
              </Badge>

              <h1 className="hero-title mb-[var(--space-lg)]">
                <span className="text-[var(--accent)]">AI Transaction Pilot</span>{' '}
                for Base and beyond
              </h1>

              <p className="hero-subtitle mb-[var(--space-md)]">
                A DQN agent that tells you when to submit, wait, or rebid. Cut gas spend by up to 40%.
              </p>

              <p className="hero-kicker mb-[var(--space-2xl)]">
                Live coverage: Base, Ethereum, Arbitrum, Optimism, Polygon
              </p>

              <div className="flex flex-wrap gap-[var(--space-md)] mb-[var(--space-2xl)]">
                <Link to="/app">
                  <Button size="lg">Launch AI Pilot</Button>
                </Link>
                <a href="#how-it-works">
                  <Button variant="secondary" size="lg">See How It Works</Button>
                </a>
              </div>

              {/* Trust Indicators */}
              <div className="landing-stats-row">
                <Stat label="Gas Saved" value={statsLoading ? '...' : `$${stats.total_saved_k}K+`} helper="vs peak hours" trend="up" />
                <Stat label="Accuracy" value={statsLoading ? '...' : `${stats.accuracy_percent}%`} helper="rolling 30d" />
                <Stat label="Predictions" value={statsLoading ? '...' : `${stats.predictions_k}K+`} helper="served" />
              </div>
            </div>

            {/* Right: Dashboard Preview in Browser Frame */}
            <div className="relative">
              <div className="card browser-frame">
                {/* Browser Chrome */}
                <div className="browser-chrome">
                  <div className="browser-dots">
                    <div className="browser-dot browser-dot-red" />
                    <div className="browser-dot browser-dot-yellow" />
                    <div className="browser-dot browser-dot-green" />
                  </div>
                  <div className="browser-url">basegasfeesml.pages.dev/app</div>
                </div>

                {/* Dashboard Preview Content */}
                <div className="browser-content">
                  {/* Mini KPI Cards */}
                  <div className="grid grid-cols-2 gap-[var(--space-md)] mb-[var(--space-lg)]">
                    <Card padding="sm">
                      <div className="text-[0.75rem] text-[var(--text-muted)] mb-[var(--space-xs)]">Current Gas</div>
                      <div className="text-[1.5rem] font-bold">0.0048 gwei</div>
                      <Badge variant="success" className="mt-[var(--space-xs)] text-[0.625rem]">
                        <span className="status-dot success"></span> Low
                      </Badge>
                    </Card>
                    <Card padding="sm">
                      <div className="text-[0.75rem] text-[var(--text-muted)] mb-[var(--space-xs)]">1h Forecast</div>
                      <div className="text-[1.5rem] font-bold">0.0052 gwei</div>
                      <Badge variant="warning" className="mt-[var(--space-xs)] text-[0.625rem]">
                        <span className="status-dot warning"></span> Rising
                      </Badge>
                    </Card>
                  </div>

                  {/* Mini Chart Placeholder */}
                  <div className="card p-[var(--space-lg)] h-[180px] flex items-end gap-1">
                    {[40, 60, 55, 70, 45, 50, 35, 60, 55, 48, 42, 38].map((height, i) => (
                      <div key={i} className="flex-1 bg-[var(--accent-light)] rounded-sm transition-all duration-300" style={{ height: `${height}%` }}></div>
                    ))}
                  </div>

                  {/* Recommendation */}
                  <div className="card mt-[var(--space-md)] p-[var(--space-md)] recommendation-card-green">
                    <div className="flex items-center gap-[var(--space-sm)]">
                      <CheckCircle size={18} color="var(--success)" />
                      <div className="text-[0.875rem] font-semibold text-[var(--success)]">
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
      <section id="how-it-works" className="section bg-surface">
        <div className="container">
          <div className="landing-section-header">
            <h2>How It Works</h2>
            <p className="landing-section-subtitle">
              Machine learning models trained on real Base network data predict optimal transaction times
            </p>
          </div>

          <div className="landing-features-grid">
            <div className="card landing-feature-card">
              <div className="icon-tile landing-feature-icon">
                <Activity size={22} />
              </div>
              <h3 className="mb-[var(--space-md)]">1. Real-Time Analysis</h3>
              <p className="text-[var(--text-secondary)] leading-[1.7]">
                We track Base network activity every minute, measuring congestion, gas prices, and load.
              </p>
            </div>

            <div className="card landing-feature-card">
              <div className="icon-tile landing-feature-icon">
                <Brain size={22} />
              </div>
              <h3 className="mb-[var(--space-md)]">2. AI Decisions</h3>
              <p className="text-[var(--text-secondary)] leading-[1.7]">
                A reinforcement model predicts the right moment to submit, wait, or bid higher.
              </p>
            </div>

            <div className="card landing-feature-card">
              <div className="icon-tile landing-feature-icon">
                <Coins size={22} />
              </div>
              <h3 className="mb-[var(--space-md)]">3. Save Money</h3>
              <p className="text-[var(--text-secondary)] leading-[1.7]">
                Optimize timing and save up to 40% versus peak-hour submissions.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Features Grid - Compact */}
      <section className="section">
        <div className="container">
          <div className="landing-section-header">
            <h2>Everything You Need</h2>
            <p className="landing-section-subtitle">
              Powerful tools to optimize your Base transactions
            </p>
          </div>

          <div className="landing-features-grid gap-[var(--space-lg)]">
            {[
              { icon: Activity, title: 'Signal Lights', desc: 'Instant visual indicator shows if now is a good time to transact.' },
              { icon: Clock, title: 'Best Time Widget', desc: 'See when gas is cheapest today, not just current price.' },
              { icon: TrendingUp, title: 'Price Predictions', desc: 'ML-powered forecasts for 1h, 4h, and 24h ahead.' },
              { icon: Grid3x3, title: '24-Hour Heatmap', desc: 'Interactive hourly breakdown shows gas patterns.' },
              { icon: Calculator, title: 'Savings Calculator', desc: 'Estimate exactly how much you could save.' },
              { icon: Zap, title: 'Real-Time Updates', desc: 'Live data refreshed every 30 seconds.' }
            ].map((feature, i) => {
              const Icon = feature.icon;
              return (
                <div key={i} className="card p-[var(--space-lg)]">
                  <div className="icon-tile mb-[var(--space-md)]">
                    <Icon size={18} />
                  </div>
                  <h4 className="mb-[var(--space-sm)]">{feature.title}</h4>
                  <p className="text-[var(--text-secondary)] text-[0.875rem] leading-[1.6]">{feature.desc}</p>
                </div>
              );
            })}
          </div>
        </div>
      </section>

      {/* Final CTA */}
      <section className="section bg-surface text-center">
        <div className="container max-w-[800px]">
          <h2 className="mb-[var(--space-lg)]">Stop Overpaying for Gas</h2>
          <p className="text-[1.25rem] text-[var(--text-secondary)] mb-[var(--space-2xl)]">
            Join thousands of Base users saving money with AI-powered gas predictions
          </p>
          <Link to="/app" className="btn btn-primary btn-lg px-10 py-5 text-[1.25rem]">
            Start Saving Now - It's Free →
          </Link>
          <p className="mt-[var(--space-lg)] text-[0.875rem] text-[var(--text-muted)]">
            No wallet connection required • No sign-up • Instant access
          </p>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-[var(--space-2xl)] bg-bg border-t border-[var(--border)]">
        <div className="container">
          <div className="flex justify-between items-center flex-wrap gap-[var(--space-lg)]">
            <div className="text-[0.875rem] text-[var(--text-muted)]">
              © 2025 Base Gas Optimiser. Built at Coinbase 2025 Hackathon.
            </div>
            <div className="flex gap-[var(--space-lg)] text-[0.875rem] text-[var(--text-muted)]">
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
