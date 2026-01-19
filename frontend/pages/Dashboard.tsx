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
import OnboardingTour from '../src/components/OnboardingTour';
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
      {/* Skip Navigation Link */}
      <a href="#main-content" className="skip-nav">
        Skip to main content
      </a>

      {/* Header */}
      <StickyHeader apiStatus={apiStatus} currentGas={currentGas} />

      <main
        id="main-content"
        className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-4 pb-8"
        role="main"
        aria-label="Dashboard content"
      >
        {/* Drift Alert Banner */}
        <DriftAlertBanner />

        {/* Hero: AI Transaction Pilot */}
        <div className="mb-6" data-tour="pilot">
          <SectionErrorBoundary sectionName="Transaction Pilot">
            <TransactionPilot ethPrice={ethPrice} />
          </SectionErrorBoundary>
        </div>

        {/* Tab Navigation */}
        <nav aria-label="Dashboard sections" data-tour="tabs">
          <div
            className="flex items-center gap-2 mb-6 border-b border-gray-800 pb-4"
            role="tablist"
            aria-label="Dashboard tabs"
          >
            <button
              onClick={() => setActiveTab('overview')}
              role="tab"
              aria-selected={activeTab === 'overview'}
              aria-controls="panel-overview"
              id="tab-overview"
              tabIndex={activeTab === 'overview' ? 0 : -1}
              className={`flex items-center gap-2 px-4 py-2.5 rounded-lg font-medium btn-press ripple ${
                activeTab === 'overview'
                  ? 'bg-cyan-500/20 text-cyan-400 border border-cyan-500/30 hover-glow'
                  : 'text-gray-400 hover:text-white hover:bg-gray-800/50'
              }`}
            >
              <LayoutDashboard className="w-4 h-4" aria-hidden="true" />
              Overview
            </button>
            <button
              onClick={() => setActiveTab('analytics')}
              role="tab"
              aria-selected={activeTab === 'analytics'}
              aria-controls="panel-analytics"
              id="tab-analytics"
              tabIndex={activeTab === 'analytics' ? 0 : -1}
              className={`flex items-center gap-2 px-4 py-2.5 rounded-lg font-medium btn-press ripple ${
                activeTab === 'analytics'
                  ? 'bg-purple-500/20 text-purple-400 border border-purple-500/30 hover-glow'
                  : 'text-gray-400 hover:text-white hover:bg-gray-800/50'
              }`}
            >
              <BarChart3 className="w-4 h-4" aria-hidden="true" />
              Analytics
            </button>
            <button
              onClick={() => setActiveTab('system')}
              role="tab"
              aria-selected={activeTab === 'system'}
              aria-controls="panel-system"
              id="tab-system"
              tabIndex={activeTab === 'system' ? 0 : -1}
              className={`flex items-center gap-2 px-4 py-2.5 rounded-lg font-medium btn-press ripple ${
                activeTab === 'system'
                  ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30 hover-glow'
                  : 'text-gray-400 hover:text-white hover:bg-gray-800/50'
              }`}
            >
              <Settings2 className="w-4 h-4" aria-hidden="true" />
              System
            </button>
          </div>
        </nav>

        {/* Tab Content */}
        <div className="space-y-6">
          {/* OVERVIEW TAB */}
          {activeTab === 'overview' && (
            <section
              id="panel-overview"
              role="tabpanel"
              aria-labelledby="tab-overview"
              className="space-y-6 animate-fadeIn"
              tabIndex={0}
            >
              {/* Top row: Forecast + Network */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 lg:min-h-[400px]">
                <div data-tour="forecast" className="h-full">
                  <SectionErrorBoundary sectionName="Price Forecast" className="h-full">
                    <CompactForecast />
                  </SectionErrorBoundary>
                </div>
                <SectionErrorBoundary sectionName="Network Insights" className="h-full">
                  <MultiChainComparison txType="swap" ethPrice={ethPrice} />
                </SectionErrorBoundary>
              </div>

              {/* Second row: Personalization + Recommendations */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 lg:min-h-[320px]">
                <div data-tour="profile" className="h-full">
                  <SectionErrorBoundary sectionName="Profile Settings" className="h-full">
                    <PersonalizationPanel />
                  </SectionErrorBoundary>
                </div>
                <SectionErrorBoundary sectionName="Recommendations" className="h-full">
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
                  aria-label="Switch to analytics tab"
                >
                  View detailed analytics <ChevronRight className="w-4 h-4" aria-hidden="true" />
                </button>
                <button
                  onClick={() => setActiveTab('system')}
                  className="flex items-center gap-2 text-sm text-gray-400 hover:text-amber-400 transition-colors"
                  aria-label="Switch to system tab"
                >
                  View system status <ChevronRight className="w-4 h-4" aria-hidden="true" />
                </button>
              </div>
            </section>
          )}

          {/* ANALYTICS TAB */}
          {activeTab === 'analytics' && (
            <section
              id="panel-analytics"
              role="tabpanel"
              aria-labelledby="tab-analytics"
              className="space-y-6 animate-fadeIn"
              tabIndex={0}
            >
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
            </section>
          )}

          {/* SYSTEM TAB */}
          {activeTab === 'system' && (
            <section
              id="panel-system"
              role="tabpanel"
              aria-labelledby="tab-system"
              className="space-y-6 animate-fadeIn"
              tabIndex={0}
            >
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
            </section>
          )}
        </div>

        {/* Footer */}
        <footer className="mt-12 py-6 text-center border-t border-gray-800" role="contentinfo">
          <p className="text-sm text-gray-500">
            Gweizy — AI-Powered Multi-Chain Gas Optimizer
          </p>
          <p className="text-xs text-gray-600 mt-1">
            <span className="sr-only">Current network: </span>
            {selectedChain.name} • Chain ID: {selectedChain.id} • Powered by DQN Neural Networks
          </p>
          <p className="text-xs text-cyan-500 mt-1">
            v2.0 - AI Pilot Edition (Dec 28, 2025)
          </p>
          <nav className="mt-4 flex justify-center gap-4 text-xs text-gray-500" aria-label="Footer navigation">
            <a href="/analytics" className="hover:text-cyan-400 transition-colors">Analytics</a>
            <a href="/docs" className="hover:text-cyan-400 transition-colors">Documentation</a>
            <a href="/pricing" className="hover:text-cyan-400 transition-colors">API Pricing</a>
          </nav>
        </footer>
      </main>

      {/* Onboarding Tour */}
      <OnboardingTour />
    </div>
  );
};

export default Dashboard;
