import React, { useEffect, useState, lazy } from 'react';
import { Bell, Calendar } from 'lucide-react';
import StickyHeader from '../src/components/StickyHeader';
import TransactionPilot from '../src/components/TransactionPilot';
import MultiChainComparison from '../src/components/MultiChainComparison';
import CompactForecast from '../src/components/CompactForecast';
import { LazySection } from '../src/components/LazySection';
import CollapsibleSection from '../src/components/ui/CollapsibleSection';
import ApiStatusPanel from '../src/components/ApiStatusPanel';
import AccuracyMetricsCard from '../src/components/AccuracyMetricsCard';
import AccuracyMetricsDashboard from '../src/components/AccuracyMetricsDashboard';
import FeatureImportanceChart from '../src/components/FeatureImportanceChart';
import DriftAlertBanner from '../src/components/DriftAlertBanner';
import PersonalizationPanel from '../src/components/PersonalizationPanel';
import PersonalizedRecommendations from '../src/components/PersonalizedRecommendations';
import { checkHealth } from '../src/api/gasApi';
import { useChain } from '../src/contexts/ChainContext';
import { useEthPrice } from '../src/hooks/useEthPrice';
import { useWalletAddress } from '../src/hooks/useWalletAddress';

// Lazy load secondary components
const ScheduledTransactionsList = lazy(() => import('../src/components/ScheduledTransactionsList'));
const GasAlertSettings = lazy(() => import('../src/components/GasAlertSettings'));
const GasPriceGraph = lazy(() => import('../src/components/GasPriceGraph'));
const GasPatternHeatmap = lazy(() => import('../src/components/GasPatternHeatmap'));

const Dashboard: React.FC = () => {
  const [apiStatus, setApiStatus] = useState<'checking' | 'online' | 'offline'>('checking');
  const { selectedChain, multiChainGas } = useChain();
  const { ethPrice } = useEthPrice(60000);
  const walletAddress = useWalletAddress();

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
    <div className="min-h-screen app-shell">
      {/* Header */}
      <StickyHeader apiStatus={apiStatus} currentGas={currentGas} />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Drift Alert Banner */}
        <DriftAlertBanner />

        {/* Hero: AI Transaction Pilot - Sticky on scroll */}
        <div className="mb-8 lg:sticky lg:top-[88px] lg:z-40 lg:bg-[#05070f] lg:pb-4 lg:pt-2 lg:-mx-4 lg:mx-0 lg:px-4 lg:shadow-lg lg:border-b lg:border-gray-800/50 lg:backdrop-blur-sm">
          <TransactionPilot ethPrice={ethPrice} />
        </div>

        {/* Main Grid */}
        <div className="grid grid-cols-12 gap-8 relative">
          {/* Left Column: Chain Comparison + Forecast */}
          <div className="col-span-12 lg:col-span-4 space-y-8">
            <div className="space-y-4">
              <div className="text-[11px] tracking-[0.2em] text-gray-500 uppercase">Profile</div>
              <PersonalizationPanel />
              {walletAddress && (
                <PersonalizedRecommendations walletAddress={walletAddress} />
              )}
            </div>
            <div className="space-y-4">
              <div className="text-[11px] tracking-[0.2em] text-gray-500 uppercase">Network Insights</div>
              <MultiChainComparison txType="swap" ethPrice={ethPrice} />
              <CompactForecast />
            </div>
            <div className="space-y-4">
              <div className="text-[11px] tracking-[0.2em] text-gray-500 uppercase">System</div>
              <ApiStatusPanel />
              <AccuracyMetricsCard />
              <AccuracyMetricsDashboard />
              <FeatureImportanceChart />
            </div>
          </div>

          {/* Right Column: Transaction Management - Collapsible */}
          <div className="col-span-12 lg:col-span-8 space-y-6">
            {/* Scheduled Transactions - Collapsible */}
            <CollapsibleSection
              title="Scheduled Transactions"
              icon={<Calendar className="w-4 h-4 text-cyan-300" />}
              defaultExpanded={false}
            >
              <LazySection>
                <ScheduledTransactionsList />
              </LazySection>
            </CollapsibleSection>

            {/* Gas Alerts - Collapsible */}
            <CollapsibleSection
              title="Gas Price Alerts"
              icon={<Bell className="w-4 h-4 text-cyan-300" />}
              defaultExpanded={false}
            >
              <LazySection>
                <GasAlertSettings currentGas={currentGas} />
              </LazySection>
            </CollapsibleSection>
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
