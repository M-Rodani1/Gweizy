import React, { useEffect, useState, lazy } from 'react';
import { Bell, Calendar, Sparkles, User } from 'lucide-react';
import TransactionPilot from '../src/components/TransactionPilot';
import MultiChainComparison from '../src/components/MultiChainComparison';
import CompactForecast from '../src/components/CompactForecast';
import { LazySection } from '../src/components/LazySection';
import CollapsibleSection from '../src/components/ui/CollapsibleSection';
import { SectionErrorBoundary } from '../src/components/SectionErrorBoundary';
import OnboardingTour from '../src/components/OnboardingTour';
import DriftAlertBanner from '../src/components/DriftAlertBanner';
import PersonalizationPanel from '../src/components/PersonalizationPanel';
import PersonalizedRecommendations from '../src/components/PersonalizedRecommendations';
import { useChain } from '../src/contexts/ChainContext';
import { useEthPrice } from '../src/hooks/useEthPrice';
import { useWalletAddress } from '../src/hooks/useWalletAddress';
import { useGasWebSocket } from '../src/hooks/useGasWebSocket';
import { API_CONFIG, getApiUrl } from '../src/config/api';
import AppShell from '../src/components/layout/AppShell';

// Lazy load secondary components
const ScheduledTransactionsList = lazy(() => import('../src/components/ScheduledTransactionsList'));
const GasAlertSettings = lazy(() => import('../src/components/GasAlertSettings'));
const Dashboard: React.FC = () => {
  // TODO: wire utilization into a shared gas data hook when available
  const [networkUtilization, setNetworkUtilization] = useState<number>(0.2);
  const { selectedChain, multiChainGas } = useChain();
  const { ethPrice } = useEthPrice(60000);
  const walletAddress = useWalletAddress();
  const { isConnected: isWebSocketConnected } = useGasWebSocket({ enabled: true });

  // Memoize currentGas calculation
  const currentGas = multiChainGas[selectedChain.id]?.gasPrice || 0;

  // Fetch network utilization
  useEffect(() => {
    const fetchNetworkUtilization = async () => {
      try {
        const response = await fetch(getApiUrl(API_CONFIG.ENDPOINTS.ONCHAIN_NETWORK_STATE));
        if (response.ok) {
          const data = await response.json();
          const utilization = data?.network_state?.avg_utilization || 0;
          setNetworkUtilization(utilization);
        }
      } catch (error) {
        console.error('Error fetching network utilization:', error);
      }
    };

    fetchNetworkUtilization();
    const interval = setInterval(fetchNetworkUtilization, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, []);

  return (
    <AppShell activePath="/app">
      <div
        className="max-w-7xl mx-auto pt-4 pb-10 space-y-8"
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

        {/* Overview Content */}
        <section className="space-y-8 animate-fadeIn">
          {/* Top row: Forecast + Network & Gas (combined) */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
            <div data-tour="forecast" className="h-full">
              <SectionErrorBoundary sectionName="Price Forecast" className="h-full">
                <CompactForecast />
              </SectionErrorBoundary>
            </div>
            <SectionErrorBoundary sectionName="Network & Gas" className="h-full">
              <MultiChainComparison
                txType="swap"
                ethPrice={ethPrice}
                networkUtilization={networkUtilization}
                isConnected={isWebSocketConnected}
              />
            </SectionErrorBoundary>
          </div>

          {/* Second row: Personalization + Recommendations (collapsed by default) */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div data-tour="profile">
              <CollapsibleSection
                title="Profile Settings"
                icon={<User className="w-4 h-4 text-cyan-300" />}
                defaultExpanded={false}
              >
                <SectionErrorBoundary sectionName="Profile Settings">
                  <PersonalizationPanel />
                </SectionErrorBoundary>
              </CollapsibleSection>
            </div>
            <CollapsibleSection
              title="Recommendations"
              icon={<Sparkles className="w-4 h-4 text-cyan-300" />}
              defaultExpanded={false}
            >
              <SectionErrorBoundary sectionName="Recommendations">
                <PersonalizedRecommendations walletAddress={walletAddress} />
              </SectionErrorBoundary>
            </CollapsibleSection>
          </div>

          {/* Transaction Management - Compact */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
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
                <GasAlertSettings currentGas={currentGas} walletAddress={walletAddress} />
              </LazySection>
            </CollapsibleSection>
          </div>

        </section>

        {/* Footer */}
        <footer className="mt-10 py-6 text-center border-t border-gray-800" role="contentinfo">
          <p className="text-sm text-gray-500">
            Gweizy — AI-Powered Gas Optimizer
          </p>
          <p className="text-xs text-gray-500 mt-1">
            v2.0 • Powered by DQN Neural Networks
          </p>
        </footer>
      </div>

      {/* Onboarding Tour */}
      <OnboardingTour />
    </AppShell>
  );
};

export default Dashboard;
