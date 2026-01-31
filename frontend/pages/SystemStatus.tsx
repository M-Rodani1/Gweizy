import React from 'react';
import { Link } from 'react-router-dom';
import { Activity, ServerCog, ShieldCheck } from 'lucide-react';
import { SectionErrorBoundary } from '../src/components/SectionErrorBoundary';
import ApiStatusPanel from '../src/components/ApiStatusPanel';
import MempoolStatusCard from '../src/components/MempoolStatusCard';
import AccuracyMetricsCard from '../src/components/AccuracyMetricsCard';
import ModelHealthCard from '../src/components/ModelHealthCard';
import AccuracyMetricsDashboard from '../src/components/AccuracyMetricsDashboard';
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

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <SectionErrorBoundary sectionName="Model Accuracy">
            <AccuracyMetricsCard />
          </SectionErrorBoundary>
          <SectionErrorBoundary sectionName="Model Health">
            <ModelHealthCard />
          </SectionErrorBoundary>
        </div>

        <SectionErrorBoundary sectionName="Accuracy Dashboard">
          <div
            className="rounded-2xl border border-gray-800 bg-gray-900/60 p-4 focus-card"
            role="article"
            aria-label="Model accuracy dashboard"
            tabIndex={0}
          >
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-2 text-white font-semibold">
                <Activity className="w-4 h-4 text-cyan-400" />
                Accuracy Dashboard
              </div>
              <a href="/docs#model-accuracy" className="text-sm text-cyan-300 hover:text-cyan-100">
                Learn more
              </a>
            </div>
            <AccuracyMetricsDashboard />
          </div>
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
