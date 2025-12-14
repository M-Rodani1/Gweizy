import React, { useEffect, useState } from 'react';
import OnboardingFlow from '../components/onboarding/OnboardingFlow';
import GasPriceGraph from '../components/GasPriceGraph';
import GasLeaderboard from '../components/GasLeaderboard';
import GasPriceTable from '../components/GasPriceTable';
import PredictionCards from '../components/PredictionCards';
import SavingsCalculator from '../components/SavingsCalculator';
import ModelAccuracy from '../components/ModelAccuracy';
import WalletConnect from '../components/WalletConnect';
import UserTransactionHistory from '../components/UserTransactionHistory';
import GasWasteCalculator from '../components/GasWasteCalculator';
import ShareResults from '../components/ShareResults';
import SavingsLeaderboard from '../components/SavingsLeaderboard';
import { GasIcon } from '../components/icons';
import { checkHealth, fetchCurrentGas, fetchPredictions } from '../src/api/gasApi';
import { getCurrentAccount, onAccountsChanged } from '../src/utils/wallet';

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
        const [currentGasData, predictionsResult] = await Promise.all([
          fetchCurrentGas(),
          fetchPredictions()
        ]);

        setCurrentGas(currentGasData.current_gas);

        // Extract first prediction from each horizon
        const preds: { '1h': number; '4h': number; '24h': number } = {
          '1h': 0,
          '4h': 0,
          '24h': 0
        };

        (['1h', '4h', '24h'] as const).forEach((horizon) => {
          const horizonData = predictionsResult.predictions[horizon];
          if (horizonData && horizonData.length > 0 && horizonData[0].predictedGwei) {
            preds[horizon] = horizonData[0].predictedGwei;
          }
        });

        setPredictions(preds);
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
    <div className="min-h-screen bg-gray-900 text-white font-sans p-4 sm:p-6 lg:p-8">
      <div className="max-w-7xl mx-auto">
        <header className="mb-8">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <GasIcon className="w-8 h-8 text-cyan-400" />
              <div>
                <h1 className="text-3xl font-bold text-gray-100">Base Gas Optimizer</h1>
                <p className="text-gray-400 mt-1">ML-powered gas price predictions for Base network</p>
              </div>
            </div>
            
            {/* API Status Indicator & Wallet */}
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2">
                <div className={`w-2 h-2 rounded-full ${
                  apiStatus === 'online' ? 'bg-green-500' : 
                  apiStatus === 'offline' ? 'bg-red-500' : 
                  'bg-yellow-500'
                }`}></div>
                <span className="text-sm text-gray-400">
                  {apiStatus === 'online' ? 'API Connected' : 
                   apiStatus === 'offline' ? 'API Offline' : 
                   'Checking...'}
                </span>
              </div>
              <WalletConnect />
            </div>
          </div>
        </header>

        <main className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-3">
            <GasPriceGraph />
          </div>
          
          {/* Prediction Cards */}
          <div className="lg:col-span-3">
            <PredictionCards />
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
            <div className="mt-6">
              <GasWasteCalculator walletAddress={walletAddress} />
            </div>
            {walletAddress && (
              <div className="mt-6">
                <ShareResults
                  savings={0.21}
                  savingsPercent={30}
                  period="month"
                  walletAddress={walletAddress}
                />
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

        <footer className="mt-8 text-center text-gray-500 text-sm">
          <p>Powered by Machine Learning on Base • Chain ID: 8453</p>
        </footer>
      </div>
    </div>
  );
};

export default Dashboard;

