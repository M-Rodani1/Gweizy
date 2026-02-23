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
import type { GlobalStatsResponse } from '../types';

const DEFAULT_STATS: GlobalStatsResponse = {
  total_users: 0,
  total_transactions: 0,
  total_savings_usd: 52000,
  predictions_made: 15000,
  average_accuracy: 82,
  active_chains: 5,
};

const toFiniteNumber = (value: unknown, fallback = 0): number => {
  const normalized = typeof value === 'number' ? value : Number(value);
  return Number.isFinite(normalized) ? normalized : fallback;
};

const sanitizeGlobalStats = (stats: Partial<GlobalStatsResponse> | null | undefined): GlobalStatsResponse => ({
  total_users: toFiniteNumber(stats?.total_users, DEFAULT_STATS.total_users),
  total_transactions: toFiniteNumber(stats?.total_transactions, DEFAULT_STATS.total_transactions),
  total_savings_usd: toFiniteNumber(stats?.total_savings_usd, DEFAULT_STATS.total_savings_usd),
  predictions_made: toFiniteNumber(stats?.predictions_made, DEFAULT_STATS.predictions_made),
  average_accuracy: toFiniteNumber(stats?.average_accuracy, DEFAULT_STATS.average_accuracy),
  active_chains: toFiniteNumber(stats?.active_chains, DEFAULT_STATS.active_chains),
});

const Landing: React.FC = () => {
  const [stats, setStats] = useState<GlobalStatsResponse>(DEFAULT_STATS);
  const [statsLoading, setStatsLoading] = useState(true);

  const formatCurrency = (value: number): string => {
    const safeValue = toFiniteNumber(value, 0);
    if (safeValue >= 1_000_000) return `$${(safeValue / 1_000_000).toFixed(1)}M+`;
    if (safeValue >= 1_000) return `$${(safeValue / 1_000).toFixed(0)}K+`;
    return `$${safeValue.toFixed(0)}`;
  };

  const formatCount = (value: number): string => {
    const safeValue = toFiniteNumber(value, 0);
    if (safeValue >= 1_000_000) return `${(safeValue / 1_000_000).toFixed(1)}M+`;
    if (safeValue >= 1_000) return `${(safeValue / 1_000).toFixed(0)}K+`;
    return `${safeValue.toFixed(0)}`;
  };

  const getAccuracyPercent = (value: number | null | undefined): number => {
    const safeValue = toFiniteNumber(value, 0);
    return safeValue <= 1 ? safeValue * 100 : safeValue;
  };

  useEffect(() => {
    const loadStats = async () => {
      try {
        const response = await fetchGlobalStats();
        setStats(sanitizeGlobalStats(response));
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
              <Badge variant="accent" icon={<Trophy className="w-4 h-4" />}>Hackathon Winner</Badge>
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
            <Badge variant="accent" className="mb-6 inline-flex items-center gap-1.5" icon={<Trophy className="w-4 h-4" />}>
              Coinbase 2025 Hackathon Winner
            </Badge>

            <h1 className="hero-title mb-6">
              <span className="text-[var(--accent)]">AI Transaction Pilot</span>{' '}
              for Base and beyond
            </h1>

            <p className="hero-subtitle mb-4">
              A DQN agent that tells you when to submit, wait, or rebid. Cut gas spend by up to 40%.
            </p>

            <p className="hero-kicker mb-12">
              Live coverage: Base, Ethereum, Arbitrum, Optimism, Polygon
            </p>

            <div className="flex flex-wrap justify-center gap-4 mb-12">
              <Link to="/app" onClick={() => trackEvent('cta_click', { source: 'hero', cta: 'launch_pilot' })}>
                <Button size="lg">Launch AI Pilot</Button>
              </Link>
              <a href="#how-it-works" onClick={() => trackEvent('cta_click', { source: 'hero', cta: 'see_how' })}>
                <Button variant="secondary" size="lg">See How It Works</Button>
              </a>
            </div>

            {/* Trust Indicators */}
            <div className="landing-stats-row justify-center">
              <Stat label="Gas Saved" value={statsLoading ? '...' : formatCurrency(stats.total_savings_usd)} helper="vs peak hours" trend="up" />
              <Stat label="Accuracy" value={statsLoading ? '...' : `${toFiniteNumber(getAccuracyPercent(stats.average_accuracy), 0).toFixed(0)}%`} helper="rolling 30d" />
              <Stat label="Predictions" value={statsLoading ? '...' : formatCount(stats.predictions_made)} helper="served" />
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

          <div className="grid grid-cols-1 md:grid-cols-2 gap-8 max-w-4xl mx-auto">
            <div className="card landing-feature-card">
              <div className="icon-tile landing-feature-icon">
                <Activity className="w-6 h-6" />
              </div>
              <h3 className="mb-4">1. Real-Time Monitoring</h3>
              <p className="text-gray-400 leading-relaxed">
                We track network activity every minute across 5 chains, measuring congestion, gas prices, and mempool state.
              </p>
            </div>

            <div className="card landing-feature-card">
              <div className="icon-tile landing-feature-icon">
                <Brain className="w-6 h-6" />
              </div>
              <h3 className="mb-4">2. AI Recommends Action</h3>
              <p className="text-gray-400 leading-relaxed">
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

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 max-w-5xl mx-auto">
            <div className="card p-6 text-center">
              <div className="icon-tile mb-4 mx-auto">
                <TrendingUp className="w-5 h-5" />
              </div>
              <h4 className="mb-2">Price Predictions</h4>
              <p className="text-gray-400 text-sm leading-relaxed">ML-powered forecasts for 1h, 4h, and 24h ahead.</p>
            </div>
            <div className="card p-6 text-center">
              <div className="icon-tile mb-4 mx-auto">
                <Zap className="w-5 h-5" />
              </div>
              <h4 className="mb-2">Real-Time Updates</h4>
              <p className="text-gray-400 text-sm leading-relaxed">Live data refreshed every 30 seconds across 5 chains.</p>
            </div>
            <div className="card p-6 text-center">
              <div className="icon-tile mb-4 mx-auto">
                <Brain className="w-5 h-5" />
              </div>
              <h4 className="mb-2">Smart Recommendations</h4>
              <p className="text-gray-400 text-sm leading-relaxed">AI tells you exactly when to submit your transaction.</p>
            </div>
          </div>
        </div>
      </section>

      {/* Final CTA */}
      <section className="section bg-surface text-center">
        <div className="container max-w-[800px]">
          <h2 className="mb-6">Stop Overpaying for Gas</h2>
          <p className="text-xl text-gray-400 mb-12">
            Join thousands of Base users saving money with AI-powered gas predictions
          </p>
          <Link to="/app" className="btn btn-primary btn-lg px-10 py-5 text-xl">
            Start Saving Now - It's Free →
          </Link>
          <p className="mt-6 text-sm text-gray-500">
            No wallet connection required • No sign-up • Instant access
          </p>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 bg-bg border-t border-gray-800">
        <div className="container">
          <div className="flex justify-between items-center flex-wrap gap-6">
            <div className="text-sm text-gray-500">
              © 2025 Gweizy. Built at Coinbase 2025 Hackathon.
            </div>
            <div className="flex gap-6 text-sm text-gray-500">
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
