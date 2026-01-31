import React from 'react';
import { Link } from 'react-router-dom';
import { ServerCog, ShieldCheck } from 'lucide-react';
import { SectionErrorBoundary } from '../src/components/SectionErrorBoundary';
import ApiStatusPanel from '../src/components/ApiStatusPanel';
import MempoolStatusCard from '../src/components/MempoolStatusCard';
import ModelMetricsPanel from '../src/components/ModelMetricsPanel';
import AppShell from '../src/components/layout/AppShell';
import { useChain } from '../src/contexts/ChainContext';

const SystemStatus: React.FC = () => {
  const { selectedChain } = useChain();

  return (
    <AppShell activePath="/system">
      <div className="max-w-7xl mx-auto py-2 space-y-8" role="main" aria-label="System status">
        <header className="flex flex-col gap-2">
          <div className="flex items-center gap-2 text-sm text-gray-500">
            <Link to="/app" className="hover:text-cyan-400 transition-colors">Dashboard</Link>
            <span>/</span>
            <span className="text-gray-300">System</span>
          </div>
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-amber-500/15 border border-amber-500/30">
              <ServerCog className="w-5 h-5 text-amber-300" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-white">System Status</h1>
              <p className="text-gray-400 text-sm">
                API health, mempool signals, and model performance for {selectedChain.name}
              </p>
            </div>
          </div>
        </header>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <SectionErrorBoundary sectionName="API Status">
            <ApiStatusPanel />
          </SectionErrorBoundary>
          <SectionErrorBoundary sectionName="Mempool Status">
            <MempoolStatusCard />
          </SectionErrorBoundary>
        </div>

        <SectionErrorBoundary sectionName="Model Metrics">
          <ModelMetricsPanel />
        </SectionErrorBoundary>

        <footer className="pt-6 border-t border-gray-800">
          <div className="flex items-center gap-2 text-xs text-gray-500">
            <ShieldCheck className="w-4 h-4" />
            <span>Real-time monitoring across chains • Updated every 60s</span>
          </div>
        </footer>
      </div>
    </AppShell>
  );
};

export default SystemStatus;
