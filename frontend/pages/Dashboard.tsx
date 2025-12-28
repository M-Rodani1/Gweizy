import React, { useEffect, useState, lazy } from 'react';
import OnboardingFlow from '../src/components/onboarding/OnboardingFlow';
import StickyHeader from '../src/components/StickyHeader';
import HeroSection from '../src/components/HeroSection';
import GasPriceGraph from '../src/components/GasPriceGraph';
import PredictionCards from '../src/components/PredictionCards';
import RelativePriceIndicator from '../src/components/RelativePriceIndicator';
import ModelStatusWidget from '../src/components/ModelStatusWidget';
import DataCollectionProgress from '../src/components/DataCollectionProgress';
import { LazySection } from '../src/components/LazySection';
import { checkHealth, fetchPredictions } from '../src/api/gasApi';
import { getCurrentAccount, onAccountsChanged } from '../src/utils/wallet';
import { fetchLiveBaseGas } from '../src/utils/baseRpc';
import { useEthPrice } from '../src/hooks/useEthPrice';

// Lazy load heavy components with aggressive code splitting
const GasLeaderboard = lazy(() => import('../src/components/GasLeaderboard'));
const GasPriceTable = lazy(() => import('../src/components/GasPriceTable'));
const SavingsCalculator = lazy(() => import('../src/components/SavingsCalculator'));
const ModelAccuracy = lazy(() => import('../src/components/ModelAccuracy'));
const UserTransactionHistory = lazy(() => import('../src/components/UserTransactionHistory'));
const SavingsLeaderboard = lazy(() => import('../src/components/SavingsLeaderboard'));
const BestTimeWidget = lazy(() => import('../src/components/BestTimeWidget'));
const ValidationMetricsDashboard = lazy(() => import('../src/components/ValidationMetricsDashboard'));
const NetworkIntelligencePanel = lazy(() => import('../src/components/NetworkIntelligencePanel'));
const FarcasterWidget = lazy(() => import('../src/components/FarcasterWidget'));
const SocialProof = lazy(() => import('../src/components/SocialProof'));
const GasAlertSettings = lazy(() => import('../src/components/GasAlertSettings'));
const TransactionCostCalculator = lazy(() => import('../src/components/TransactionCostCalculator'));
const GasPatternHeatmap = lazy(() => import('../src/components/GasPatternHeatmap'));
const TransactionQueue = lazy(() => import('../src/components/TransactionQueue'));
const AgentRecommendation = lazy(() => import('../src/components/AgentRecommendation'));

const Dashboard: React.FC = () => {
  const [showOnboarding, setShowOnboarding] = useState(
    !localStorage.getItem('onboarding_completed')
  );
  const [apiStatus, setApiStatus] = useState<'checking' | 'online' | 'offline'>('checking');
  const [currentGas, setCurrentGas] = useState<number>(0);
  const [predictions, setPredictions] = useState<{
    '1h': number;
    '4h': number;
    '24h': number;
  }>({ '1h': 0, '4h': 0, '24h': 0 });
  const [walletAddress, setWalletAddress] = useState<string | null>(null);

  // Live ETH price from CoinGecko
  const { ethPrice, priceChange24h } = useEthPrice(60000); // Update every minute

  useEffect(() => {
    const checkAPI = async () => {
      const isHealthy = await checkHealth();
      setApiStatus(isHealthy ? 'online' : 'offline');
    };

    const loadCalculatorData = async () => {
      try {
        // Fetch LIVE gas price from Base network RPC
        try {
          const liveGas = await fetchLiveBaseGas();
          if (liveGas && liveGas.gwei !== undefined && liveGas.gwei !== null) {
            setCurrentGas(liveGas.gwei);
            console.log('🔴 LIVE Base gas price:', liveGas.gwei.toFixed(4), 'gwei');
          }
        } catch (rpcErr) {
          console.error('RPC fetch failed, will use API fallback:', rpcErr);
          // Continue to fetch from API
        }

        // Still fetch predictions from API (if available)
        try {
          const predictionsResult = await fetchPredictions();

          // Extract first prediction from each horizon
          const preds: { '1h': number; '4h': number; '24h': number } = {
            '1h': 0,
            '4h': 0,
            '24h': 0
          };

          (['1h', '4h', '24h'] as const).forEach((horizon) => {
            const horizonData = predictionsResult?.predictions?.[horizon];
            if (Array.isArray(horizonData) && horizonData.length > 0 && horizonData[0]?.predictedGwei) {
              preds[horizon] = horizonData[0].predictedGwei;
            }
          });

          setPredictions(preds);
        } catch (apiErr) {
          console.error('API predictions fetch failed:', apiErr);
          // App continues with default predictions
        }
      } catch (err) {
        console.error('Error loading calculator data:', err);
      }
    };

    checkAPI();
    loadCalculatorData();

    // Check wallet connection
    const checkWallet = async () => {
      const account = await getCurrentAccount();
      setWalletAddress(account);
    };
    checkWallet();

    // Listen for wallet account changes
    const unsubscribeAccounts = onAccountsChanged((accounts) => {
      setWalletAddress(accounts.length > 0 ? accounts[0] : null);
    });

    const apiInterval = setInterval(checkAPI, 60000); // Check every minute
    const dataInterval = setInterval(loadCalculatorData, 30000); // Refresh data every 30 seconds

    return () => {
      clearInterval(apiInterval);
      clearInterval(dataInterval);
      unsubscribeAccounts();
    };
  }, []);

  if (showOnboarding) {
    return <OnboardingFlow onComplete={() => setShowOnboarding(false)} />;
  }

  return (
    <div style={{ minHeight: '100vh', background: 'var(--bg)', color: 'var(--text)' }}>
      {/* Sticky Header */}
      <StickyHeader apiStatus={apiStatus} currentGas={currentGas} />

      <div className="container" style={{ padding: 'var(--space-lg) var(--space-xl)', paddingTop: 'var(--space-2xl)' }}>
        {/* Social Proof Banner - Lazy loaded */}
        <LazySection rootMargin="100px">
          <SocialProof />
        </LazySection>

        {/* Hero Section - Critical, load immediately */}
        <HeroSection currentGas={currentGas} predictions={predictions} ethPrice={ethPrice} />

        <main style={{ display: 'grid', gridTemplateColumns: 'repeat(12, 1fr)', gap: 'var(--space-xl)' }}>
          {/* Farcaster Widget - Only shows when in Farcaster context */}
          <LazySection style={{ gridColumn: 'span 12' }} rootMargin="150px">
            <FarcasterWidget />
          </LazySection>

          {/* Week 1 Improvements: Relative Price Indicator + Best Time Widget */}
          <div style={{ gridColumn: 'span 4' }}>
            <RelativePriceIndicator currentGas={currentGas} />
          </div>

          <LazySection style={{ gridColumn: 'span 8' }} rootMargin="150px">
            <BestTimeWidget currentGas={currentGas} />
          </LazySection>

          {/* AI Agent Recommendation */}
          <LazySection style={{ gridColumn: 'span 6' }} rootMargin="150px">
            <AgentRecommendation currentGas={currentGas} />
          </LazySection>

          {/* Transaction Cost Calculator */}
          <LazySection style={{ gridColumn: 'span 6' }} rootMargin="200px">
            <TransactionCostCalculator currentGas={currentGas} ethPrice={ethPrice} />
          </LazySection>

          {/* Gas Price Alerts */}
          <LazySection style={{ gridColumn: 'span 6' }} rootMargin="200px">
            <GasAlertSettings currentGas={currentGas} />
          </LazySection>

          {/* Transaction Queue */}
          <LazySection style={{ gridColumn: 'span 12' }} rootMargin="200px">
            <TransactionQueue currentGas={currentGas} ethPrice={ethPrice} />
          </LazySection>

          {/* Gas Price Graph */}
          <div style={{ gridColumn: 'span 12' }}>
            <GasPriceGraph />
          </div>

          {/* Historical Gas Patterns Heatmap */}
          <LazySection style={{ gridColumn: 'span 12' }} rootMargin="250px">
            <GasPatternHeatmap />
          </LazySection>

          {/* Prediction Cards */}
          <div style={{ gridColumn: 'span 12' }}>
            <PredictionCards />
          </div>

          {/* Data Collection Progress */}
          <div style={{ gridColumn: 'span 12' }}>
            <DataCollectionProgress />
          </div>

          {/* NEW: Enterprise ML Features */}
          {/* Model Status Widget - Compact */}
          <div style={{ gridColumn: 'span 4' }}>
            <ModelStatusWidget />
          </div>

          {/* Network Intelligence Panel */}
          <LazySection style={{ gridColumn: 'span 8' }} rootMargin="300px">
            <NetworkIntelligencePanel />
          </LazySection>

          {/* Validation Metrics Dashboard - Full Width */}
          <LazySection style={{ gridColumn: 'span 12' }} rootMargin="400px">
            <ValidationMetricsDashboard />
          </LazySection>

          {/* Model Accuracy Dashboard */}
          <LazySection style={{ gridColumn: 'span 12' }} rootMargin="400px">
            <ModelAccuracy />
          </LazySection>

          <LazySection style={{ gridColumn: 'span 4' }} rootMargin="500px">
            <div>
              <GasLeaderboard />
              <div style={{ marginTop: 'var(--space-lg)' }}>
                {currentGas > 0 && (
                  <SavingsCalculator
                    currentGas={currentGas}
                    predictions={predictions}
                    ethPrice={ethPrice}
                  />
                )}
              </div>
              {walletAddress && (
                <div style={{ marginTop: 'var(--space-lg)' }}>
                  <UserTransactionHistory address={walletAddress} />
                </div>
              )}
            </div>
          </LazySection>

          <LazySection style={{ gridColumn: 'span 8' }} rootMargin="500px">
            <div>
              <GasPriceTable />
              <div style={{ marginTop: 'var(--space-lg)' }}>
                <SavingsLeaderboard walletAddress={walletAddress} />
              </div>
            </div>
          </LazySection>
        </main>

        <footer style={{ marginTop: 'var(--space-3xl)', padding: 'var(--space-2xl) 0', textAlign: 'center', borderTop: '1px solid var(--border)' }}>
          <p style={{ fontSize: '0.875rem', color: 'var(--text-muted)', marginBottom: 'var(--space-sm)' }}>
            AI-powered gas price predictions for Base network
          </p>
          <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            Chain ID: 8453 • Built with ML • Powered by Base
          </p>
        </footer>
      </div>
    </div>
  );
};

export default Dashboard;
