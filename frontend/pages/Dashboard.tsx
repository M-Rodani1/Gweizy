import React, { useEffect, useState, lazy } from 'react';
import { Bell, Calendar, BarChart3, Settings2, LayoutDashboard, ChevronRight } from 'lucide-react';
import StickyHeader from '../src/components/StickyHeader';
import TransactionPilot from '../src/components/TransactionPilot';
import MultiChainComparison from '../src/components/MultiChainComparison';
import CompactForecast from '../src/components/CompactForecast';
import { LazySection } from '../src/components/LazySection';
import CollapsibleSection from '../src/components/ui/CollapsibleSection';
import { SectionErrorBoundary } from '../src/components/SectionErrorBoundary';
import ApiStatusPanel from '../src/components/ApiStatusPanel';
import AccuracyMetricsCard from '../src/components/AccuracyMetricsCard';
import AccuracyMetricsDashboard from '../src/components/AccuracyMetricsDashboard';
import PatternMatchingCard from '../src/components/PatternMatchingCard';
import MempoolStatusCard from '../src/components/MempoolStatusCard';
import FeatureImportanceChart from '../src/components/FeatureImportanceChart';
import DriftAlertBanner from '../src/components/DriftAlertBanner';
import PersonalizationPanel from '../src/components/PersonalizationPanel';
import PersonalizedRecommendations from '../src/components/PersonalizedRecommendations';
import AdvancedAnalyticsPanel from '../src/components/AdvancedAnalyticsPanel';
import ModelTrainingPanel from '../src/components/ModelTrainingPanel';
import { checkHealth } from '../src/api/gasApi';
import { useChain } from '../src/contexts/ChainContext';
import { useEthPrice } from '../src/hooks/useEthPrice';
import { useWalletAddress } from '../src/hooks/useWalletAddress';

// Lazy load secondary components
const ScheduledTransactionsList = lazy(() => import('../src/components/ScheduledTransactionsList'));
const GasAlertSettings = lazy(() => import('../src/components/GasAlertSettings'));
const GasPriceGraph = lazy(() => import('../src/components/GasPriceGraph'));
const GasPatternHeatmap = lazy(() => import('../src/components/GasPatternHeatmap'));

type DashboardTab = 'overview' | 'analytics' | 'system';

const Dashboard: React.FC = () => {
  const [apiStatus, setApiStatus] = useState<'checking' | 'online' | 'offline'>('checking');
  const [activeTab, setActiveTab] = useState<DashboardTab>('overview');
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

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-4 pb-8">
        {/* Drift Alert Banner */}
        <DriftAlertBanner />

        {/* Hero: AI Transaction Pilot */}
        <div className="mb-6">
          <SectionErrorBoundary sectionName="Transaction Pilot">
            <TransactionPilot ethPrice={ethPrice} />
          </SectionErrorBoundary>
        </div>

        {/* Tab Navigation */}
        <div className="flex items-center gap-2 mb-6 border-b border-gray-800 pb-4">
          <button
            onClick={() => setActiveTab('overview')}
            className={`flex items-center gap-2 px-4 py-2.5 rounded-lg font-medium transition-all ${
              activeTab === 'overview'
                ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30'
                : 'text-gray-400 hover:text-white hover:bg-gray-800/50'
            }`}
          >
            <LayoutDashboard className="w-4 h-4" />
            Overview
          </button>
          <button
            onClick={() => setActiveTab('analytics')}
            className={`flex items-center gap-2 px-4 py-2.5 rounded-lg font-medium transition-all ${
              activeTab === 'analytics'
                ? 'bg-purple-500/20 text-purple-400 border border-purple-500/30'
                : 'text-gray-400 hover:text-white hover:bg-gray-800/50'
            }`}
          >
            <BarChart3 className="w-4 h-4" />
            Analytics
          </button>
          <button
            onClick={() => setActiveTab('system')}
            className={`flex items-center gap-2 px-4 py-2.5 rounded-lg font-medium transition-all ${
              activeTab === 'system'
                ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
                : 'text-gray-400 hover:text-white hover:bg-gray-800/50'
            }`}
          >
            <Settings2 className="w-4 h-4" />
            System
          </button>
        </div>

        {/* Tab Content */}
        <div className="space-y-6">
          {/* OVERVIEW TAB */}
          {activeTab === 'overview' && (
            <div className="space-y-6 animate-fadeIn">
              {/* Top row: Forecast + Network */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <SectionErrorBoundary sectionName="Price Forecast">
                  <CompactForecast />
                </SectionErrorBoundary>
                <SectionErrorBoundary sectionName="Network Insights">
                  <MultiChainComparison txType="swap" ethPrice={ethPrice} />
                </SectionErrorBoundary>
              </div>

              {/* Second row: Personalization + Recommendations */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <SectionErrorBoundary sectionName="Profile Settings">
                  <PersonalizationPanel />
                </SectionErrorBoundary>
                <SectionErrorBoundary sectionName="Recommendations">
                  {walletAddress ? (
                    <PersonalizedRecommendations walletAddress={walletAddress} />
                  ) : (
                    <div className="bg-gradient-to-br from-blue-900/20 to-purple-900/20 border border-blue-500/30 rounded-2xl p-6 h-full min-h-[200px] flex flex-col items-center justify-center shadow-xl">
                      <p className="text-gray-400 text-center">Connect your wallet to see personalized recommendations</p>
                    </div>
                  )}
                </SectionErrorBoundary>
              </div>

              {/* Transaction Management - Compact */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <CollapsibleSection
                  title="Scheduled Transactions"
                  icon={<Calendar className="w-4 h-4 text-cyan-300" />}
                  defaultExpanded={false}
                >
                  <LazySection>
                    <ScheduledTransactionsList />
                  </LazySection>
                </CollapsibleSection>

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

              {/* Quick link to other tabs */}
              <div className="flex gap-4 pt-4">
                <button
                  onClick={() => setActiveTab('analytics')}
                  className="flex items-center gap-2 text-sm text-gray-400 hover:text-purple-400 transition-colors"
                >
                  View detailed analytics <ChevronRight className="w-4 h-4" />
                </button>
                <button
                  onClick={() => setActiveTab('system')}
                  className="flex items-center gap-2 text-sm text-gray-400 hover:text-amber-400 transition-colors"
                >
                  View system status <ChevronRight className="w-4 h-4" />
                </button>
              </div>
            </div>
          )}

          {/* ANALYTICS TAB */}
          {activeTab === 'analytics' && (
            <div className="space-y-6 animate-fadeIn">
              {/* Charts */}
              <SectionErrorBoundary sectionName="Gas Price Chart">
                <LazySection rootMargin="200px">
                  <GasPriceGraph />
                </LazySection>
              </SectionErrorBoundary>

              <SectionErrorBoundary sectionName="Gas Pattern Heatmap">
                <LazySection rootMargin="200px">
                  <GasPatternHeatmap />
                </LazySection>
              </SectionErrorBoundary>

              {/* Pattern Matching + Feature Importance */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <SectionErrorBoundary sectionName="Pattern Matching">
                  <PatternMatchingCard />
                </SectionErrorBoundary>
                <SectionErrorBoundary sectionName="Feature Importance">
                  <FeatureImportanceChart />
                </SectionErrorBoundary>
              </div>

              {/* Advanced Analytics */}
              <SectionErrorBoundary sectionName="Advanced Analytics">
                <AdvancedAnalyticsPanel />
              </SectionErrorBoundary>
            </div>
          )}

          {/* SYSTEM TAB */}
          {activeTab === 'system' && (
            <div className="space-y-6 animate-fadeIn">
              {/* Status Cards - 2x2 grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <SectionErrorBoundary sectionName="API Status">
                  <ApiStatusPanel />
                </SectionErrorBoundary>
                <SectionErrorBoundary sectionName="Mempool Status">
                  <MempoolStatusCard />
                </SectionErrorBoundary>
              </div>

              {/* Model Status */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <SectionErrorBoundary sectionName="Model Accuracy">
                  <AccuracyMetricsCard />
                </SectionErrorBoundary>
                <SectionErrorBoundary sectionName="Model Training">
                  <ModelTrainingPanel />
                </SectionErrorBoundary>
              </div>

              {/* Accuracy Dashboard - Full width */}
              <SectionErrorBoundary sectionName="Accuracy Dashboard">
                <AccuracyMetricsDashboard />
              </SectionErrorBoundary>
            </div>
          )}
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
