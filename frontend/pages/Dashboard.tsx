import React, { useEffect, useState, lazy } from 'react';
import StickyHeader from '../src/components/StickyHeader';
import TransactionPilot from '../src/components/TransactionPilot';
import MultiChainComparison from '../src/components/MultiChainComparison';
import CompactForecast from '../src/components/CompactForecast';
import { LazySection } from '../src/components/LazySection';
import { checkHealth } from '../src/api/gasApi';
import { useChain } from '../src/contexts/ChainContext';
import { useEthPrice } from '../src/hooks/useEthPrice';

// Lazy load secondary components
const ScheduledTransactionsList = lazy(() => import('../src/components/ScheduledTransactionsList'));
const GasAlertSettings = lazy(() => import('../src/components/GasAlertSettings'));
const GasPriceGraph = lazy(() => import('../src/components/GasPriceGraph'));
const GasPatternHeatmap = lazy(() => import('../src/components/GasPatternHeatmap'));

const Dashboard: React.FC = () => {
  const [apiStatus, setApiStatus] = useState<'checking' | 'online' | 'offline'>('checking');
  const { selectedChain, multiChainGas } = useChain();
  const { ethPrice } = useEthPrice(60000);

  const currentGas = multiChainGas[selectedChain.id]?.gasPrice || 0;

  useEffect(() => {
    const checkAPI = async () => {
      const isHealthy = await checkHealth();
      setApiStatus(isHealthy ? 'online' : 'offline');
    };

    checkAPI();
    const interval = setInterval(checkAPI, 60000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-gray-950">
      {/* Header */}
      <StickyHeader apiStatus={apiStatus} currentGas={currentGas} />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {/* Hero: AI Transaction Pilot */}
        <div className="mb-6">
          <TransactionPilot ethPrice={ethPrice} />
        </div>

        {/* Main Grid */}
        <div className="grid grid-cols-12 gap-6">
          {/* Left Column: Chain Comparison + Forecast */}
          <div className="col-span-12 lg:col-span-4 space-y-6">
            <MultiChainComparison txType="swap" ethPrice={ethPrice} />
            <CompactForecast />
          </div>

          {/* Right Column: Transaction Management */}
          <div className="col-span-12 lg:col-span-8 space-y-6">
            {/* Scheduled Transactions */}
            <LazySection>
              <ScheduledTransactionsList />
            </LazySection>

            {/* Gas Alerts */}
            <LazySection>
              <GasAlertSettings currentGas={currentGas} />
            </LazySection>
          </div>

          {/* Full Width: Charts */}
          <div className="col-span-12">
            <LazySection rootMargin="200px">
              <GasPriceGraph />
            </LazySection>
          </div>

          <div className="col-span-12">
            <LazySection rootMargin="300px">
              <GasPatternHeatmap />
            </LazySection>
          </div>
        </div>

        {/* Footer */}
        <footer className="mt-12 py-6 text-center border-t border-gray-800">
          <p className="text-sm text-gray-500">
            Gweizy — AI-Powered Multi-Chain Gas Optimizer
          </p>
          <p className="text-xs text-gray-600 mt-1">
            {selectedChain.name} • Chain ID: {selectedChain.id} • Powered by DQN Neural Networks
          </p>
          <p className="text-xs text-cyan-500 mt-1">
            v2.0 - AI Pilot Edition (Dec 28, 2025)
          </p>
          <div className="mt-4 flex justify-center gap-4 text-xs text-gray-500">
            <a href="/analytics" className="hover:text-cyan-400 transition-colors">Analytics</a>
            <a href="/docs" className="hover:text-cyan-400 transition-colors">Docs</a>
            <a href="/pricing" className="hover:text-cyan-400 transition-colors">API</a>
          </div>
        </footer>
      </div>
    </div>
  );
};

export default Dashboard;
