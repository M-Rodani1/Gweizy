import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  Activity,
  Brain,
  TrendingUp,
  Trophy,
  Zap
} from 'lucide-react';
import { fetchGlobalStats } from '../src/api/gasApi';
import Logo from '../src/components/branding/Logo';
import { Button, Badge, Stat } from '../src/components/ui';
import { trackEvent } from '../src/utils/analytics';

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
              <span className="text-lg font-semibold">
                Gweizy
              </span>
              <Badge variant="accent" icon={<Trophy size={14} />}>Hackathon Winner</Badge>
            </div>
            <div className="landing-nav-links">
                <Link to="/pricing" className="btn btn-ghost" onClick={() => trackEvent('cta_click', { source: 'landing_nav', cta: 'pricing' })}>Pricing</Link>
                <Link to="/app" onClick={() => trackEvent('cta_click', { source: 'landing_nav', cta: 'launch_pilot' })}>
                  <Button variant="primary" size="lg">Launch AI Pilot</Button>
                </Link>
            </div>
          </div>
        </div>
      </nav>

      {/* Hero Section - Centered */}
      <section className="section landing-hero">
        <div className="container">
          <div className="max-w-3xl mx-auto text-center">
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

            <div className="flex flex-wrap justify-center gap-[var(--space-md)] mb-[var(--space-2xl)]">
              <Link to="/app" onClick={() => trackEvent('cta_click', { source: 'hero', cta: 'launch_pilot' })}>
                <Button size="lg">Launch AI Pilot</Button>
              </Link>
              <a href="#how-it-works" onClick={() => trackEvent('cta_click', { source: 'hero', cta: 'see_how' })}>
                <Button variant="secondary" size="lg">See How It Works</Button>
              </a>
            </div>

            {/* Trust Indicators */}
            <div className="landing-stats-row justify-center">
              <Stat label="Gas Saved" value={statsLoading ? '...' : `$${stats.total_saved_k}K+`} helper="vs peak hours" trend="up" />
              <Stat label="Accuracy" value={statsLoading ? '...' : `${stats.accuracy_percent}%`} helper="rolling 30d" />
              <Stat label="Predictions" value={statsLoading ? '...' : `${stats.predictions_k}K+`} helper="served" />
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
              Machine learning models trained on real network data predict optimal transaction times
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-[var(--space-xl)] max-w-4xl mx-auto">
            <div className="card landing-feature-card">
              <div className="icon-tile landing-feature-icon">
                <Activity size={22} />
              </div>
              <h3 className="mb-[var(--space-md)]">1. Real-Time Monitoring</h3>
              <p className="text-[var(--text-secondary)] leading-[1.7]">
                We track network activity every minute across 5 chains, measuring congestion, gas prices, and mempool state.
              </p>
            </div>

            <div className="card landing-feature-card">
              <div className="icon-tile landing-feature-icon">
                <Brain size={22} />
              </div>
              <h3 className="mb-[var(--space-md)]">2. AI Recommends Action</h3>
              <p className="text-[var(--text-secondary)] leading-[1.7]">
                A DQN neural network tells you to submit now, wait for lower prices, or bid higher for faster confirmation.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Features Grid - Top 3 */}
      <section className="section">
        <div className="container">
          <div className="landing-section-header">
            <h2>Key Features</h2>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-[var(--space-lg)] max-w-5xl mx-auto">
            <div className="card p-[var(--space-lg)] text-center">
              <div className="icon-tile mb-[var(--space-md)] mx-auto">
                <TrendingUp size={18} />
              </div>
              <h4 className="mb-[var(--space-sm)]">Price Predictions</h4>
              <p className="text-[var(--text-secondary)] text-sm leading-[1.6]">ML-powered forecasts for 1h, 4h, and 24h ahead.</p>
            </div>
            <div className="card p-[var(--space-lg)] text-center">
              <div className="icon-tile mb-[var(--space-md)] mx-auto">
                <Zap size={18} />
              </div>
              <h4 className="mb-[var(--space-sm)]">Real-Time Updates</h4>
              <p className="text-[var(--text-secondary)] text-sm leading-[1.6]">Live data refreshed every 30 seconds across 5 chains.</p>
            </div>
            <div className="card p-[var(--space-lg)] text-center">
              <div className="icon-tile mb-[var(--space-md)] mx-auto">
                <Brain size={18} />
              </div>
              <h4 className="mb-[var(--space-sm)]">Smart Recommendations</h4>
              <p className="text-[var(--text-secondary)] text-sm leading-[1.6]">AI tells you exactly when to submit your transaction.</p>
            </div>
          </div>
        </div>
      </section>

      {/* Final CTA */}
      <section className="section bg-surface text-center">
        <div className="container max-w-[800px]">
          <h2 className="mb-[var(--space-lg)]">Stop Overpaying for Gas</h2>
          <p className="text-xl text-[var(--text-secondary)] mb-[var(--space-2xl)]">
            Join thousands of Base users saving money with AI-powered gas predictions
          </p>
          <Link to="/app" className="btn btn-primary btn-lg px-10 py-5 text-xl">
            Start Saving Now - It's Free →
          </Link>
          <p className="mt-[var(--space-lg)] text-sm text-[var(--text-muted)]">
            No wallet connection required • No sign-up • Instant access
          </p>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-[var(--space-2xl)] bg-bg border-t border-[var(--border)]">
        <div className="container">
          <div className="flex justify-between items-center flex-wrap gap-[var(--space-lg)]">
            <div className="text-sm text-[var(--text-muted)]">
              © 2025 Base Gas Optimiser. Built at Coinbase 2025 Hackathon.
            </div>
            <div className="flex gap-[var(--space-lg)] text-sm text-[var(--text-muted)]">
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
