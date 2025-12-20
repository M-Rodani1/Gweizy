import React, { useEffect, useState } from 'react';
import OnboardingFlow from './components/onboarding/OnboardingFlow';
import StickyHeader from './components/StickyHeader';
import HeroSection from './components/HeroSection';
import GasPriceGraph from './components/GasPriceGraph';
import GasLeaderboard from './components/GasLeaderboard';
import GasPriceTable from './components/GasPriceTable';
import PredictionCards from './components/PredictionCards';
import SavingsCalculator from './components/SavingsCalculator';
import ModelAccuracy from './components/ModelAccuracy';
import UserTransactionHistory from './components/UserTransactionHistory';
import SavingsLeaderboard from './components/SavingsLeaderboard';
import BestTimeWidget from './components/BestTimeWidget';
import RelativePriceIndicator from './components/RelativePriceIndicator';
import ValidationMetricsDashboard from './components/ValidationMetricsDashboard';
import NetworkIntelligencePanel from './components/NetworkIntelligencePanel';
import ModelStatusWidget from './components/ModelStatusWidget';
import FarcasterWidget from './components/FarcasterWidget';
import SocialProof from './components/SocialProof';
import { checkHealth, fetchPredictions } from '../src/api/gasApi';
import { getCurrentAccount, onAccountsChanged } from '../src/utils/wallet';
import { fetchLiveBaseGas } from '../src/utils/baseRpc';

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
    <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-900 to-black text-white font-sans overflow-x-hidden">
      {/* Sticky Header */}
      <StickyHeader apiStatus={apiStatus} currentGas={currentGas} />

      <div className="w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {/* Social Proof Banner */}
        <SocialProof />

        {/* Hero Section */}
        <HeroSection currentGas={currentGas} predictions={predictions} />

        <main className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Farcaster Widget - Only shows when in Farcaster context */}
          <div className="lg:col-span-3">
            <FarcasterWidget />
          </div>

          {/* Week 1 Improvements: Relative Price Indicator + Best Time Widget */}
          <div className="lg:col-span-1">
            <RelativePriceIndicator currentGas={currentGas} />
          </div>

          <div className="lg:col-span-2">
            <BestTimeWidget currentGas={currentGas} />
          </div>

          {/* Gas Price Graph */}
          <div className="lg:col-span-3">
            <GasPriceGraph />
          </div>

          {/* Prediction Cards */}
          <div className="lg:col-span-3">
            <PredictionCards />
          </div>

          {/* NEW: Enterprise ML Features */}
          {/* Model Status Widget - Compact */}
          <div className="lg:col-span-1">
            <ModelStatusWidget />
          </div>

          {/* Network Intelligence Panel */}
          <div className="lg:col-span-2">
            <NetworkIntelligencePanel />
          </div>

          {/* Validation Metrics Dashboard - Full Width */}
          <div className="lg:col-span-3">
            <ValidationMetricsDashboard />
          </div>

          {/* Model Accuracy Dashboard */}
          <div className="lg:col-span-3">
            <ModelAccuracy />
          </div>

          <div className="lg:col-span-1">
            <GasLeaderboard />
            <div className="mt-6">
              {currentGas > 0 && (
                <SavingsCalculator
                  currentGas={currentGas}
                  predictions={predictions}
                  ethPrice={3000}
                />
              )}
            </div>
            {walletAddress && (
              <div className="mt-6">
                <UserTransactionHistory address={walletAddress} />
              </div>
            )}
          </div>
          <div className="lg:col-span-2">
            <GasPriceTable />
            <div className="mt-6">
              <SavingsLeaderboard walletAddress={walletAddress} />
            </div>
          </div>
        </main>

        <footer className="mt-12 py-8 text-center border-t border-gray-800">
          <p className="text-gray-400 text-sm mb-2">
            AI-powered gas price predictions for Base network
          </p>
          <p className="text-gray-500 text-xs">
            Chain ID: 8453 • Built with ML • Powered by Base
          </p>
        </footer>
      </div>
    </div>
  );
};

export default Dashboard;
