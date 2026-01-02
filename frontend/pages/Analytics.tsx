import React, { lazy, Suspense } from 'react';
import { Brain, Calendar, ClipboardList, Sparkles, Target, TrendingUp, Activity } from 'lucide-react';
import { Link } from 'react-router-dom';
import StickyHeader from '../src/components/StickyHeader';
import AccuracyMetricsCard from '../src/components/AccuracyMetricsCard';
import FeatureImportanceChart from '../src/components/FeatureImportanceChart';
import AccuracyHistoryChart from '../src/components/AccuracyHistoryChart';
import DriftAlertBanner from '../src/components/DriftAlertBanner';
import ApiStatusPanel from '../src/components/ApiStatusPanel';
import { useChain } from '../src/contexts/ChainContext';
import { useEthPrice } from '../src/hooks/useEthPrice';

// Lazy load analytics components
const GasPriceGraph = lazy(() => import('../src/components/GasPriceGraph'));
const GasPatternHeatmap = lazy(() => import('../src/components/GasPatternHeatmap'));
const PredictionCards = lazy(() => import('../src/components/PredictionCards'));
const ModelAccuracy = lazy(() => import('../src/components/ModelAccuracy'));
const ValidationMetricsDashboard = lazy(() => import('../src/components/ValidationMetricsDashboard'));
const NetworkIntelligencePanel = lazy(() => import('../src/components/NetworkIntelligencePanel'));
const BestTimeWidget = lazy(() => import('../src/components/BestTimeWidget'));
const GasPriceTable = lazy(() => import('../src/components/GasPriceTable'));

const LoadingSpinner = () => (
  <div className="flex items-center justify-center p-8">
    <div className="w-8 h-8 border-2 border-gray-600 border-t-cyan-400 rounded-full animate-spin" />
  </div>
);

const Analytics: React.FC = () => {
  const { selectedChain, multiChainGas } = useChain();
  const { ethPrice } = useEthPrice(60000);
  const currentGas = multiChainGas[selectedChain.id]?.gasPrice || 0;

  return (
    <div className="min-h-screen app-shell">
      <StickyHeader apiStatus="online" currentGas={currentGas} />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
        {/* Page Header */}
        <div className="mb-8">
          <div className="flex items-center gap-2 text-sm text-gray-500 mb-2">
            <Link to="/" className="hover:text-cyan-400 transition-colors">Dashboard</Link>
            <span>/</span>
            <span className="text-gray-300">Analytics</span>
          </div>
          <h1 className="text-2xl font-bold text-white">Gas Analytics</h1>
          <p className="text-gray-400 mt-1">
            Detailed gas price analysis, predictions, and network intelligence for {selectedChain.name}
          </p>
        </div>

        {/* Drift Alert Banner */}
        <DriftAlertBanner />

        {/* Model Health Overview */}
        <section className="mb-8">
          <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <Activity className="w-4 h-4 text-cyan-400" />
            Model Health
          </h2>
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2">
              <AccuracyMetricsCard />
            </div>
            <div>
              <ApiStatusPanel />
            </div>
          </div>
        </section>

        {/* Accuracy Trends */}
        <section className="mb-8">
          <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-cyan-400" />
            Accuracy Trends
          </h2>
          <AccuracyHistoryChart />
        </section>

        {/* Feature Analysis */}
        <section className="mb-8">
          <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <Brain className="w-4 h-4 text-cyan-400" />
            Feature Analysis
          </h2>
          <FeatureImportanceChart />
        </section>

        {/* Predictions Section */}
        <section className="mb-8">
          <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <Sparkles className="w-4 h-4 text-cyan-400" />
            ML Predictions
          </h2>
          <Suspense fallback={<LoadingSpinner />}>
            <PredictionCards />
          </Suspense>
        </section>

        {/* Charts Section */}
        <section className="mb-8">
          <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <TrendingUp className="w-4 h-4 text-cyan-400" />
            Price Charts
          </h2>
          <div className="grid grid-cols-1 gap-6">
            <Suspense fallback={<LoadingSpinner />}>
              <GasPriceGraph />
            </Suspense>
          </div>
        </section>

        {/* Pattern Analysis */}
        <section className="mb-8">
          <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <Calendar className="w-4 h-4 text-cyan-400" />
            Pattern Analysis
          </h2>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Suspense fallback={<LoadingSpinner />}>
              <GasPatternHeatmap />
            </Suspense>
            <Suspense fallback={<LoadingSpinner />}>
              <BestTimeWidget currentGas={currentGas} />
            </Suspense>
          </div>
        </section>

        {/* Model Performance */}
        <section className="mb-8">
          <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <Target className="w-4 h-4 text-cyan-400" />
            Model Performance
          </h2>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Suspense fallback={<LoadingSpinner />}>
              <ModelAccuracy />
            </Suspense>
            <Suspense fallback={<LoadingSpinner />}>
              <ValidationMetricsDashboard />
            </Suspense>
          </div>
        </section>

        {/* Network Intelligence */}
        <section className="mb-8">
          <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <Brain className="w-4 h-4 text-cyan-400" />
            Network Intelligence
          </h2>
          <Suspense fallback={<LoadingSpinner />}>
            <NetworkIntelligencePanel />
          </Suspense>
        </section>

        {/* Historical Data */}
        <section className="mb-8">
          <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
            <ClipboardList className="w-4 h-4 text-cyan-400" />
            Historical Data
          </h2>
          <Suspense fallback={<LoadingSpinner />}>
            <GasPriceTable />
          </Suspense>
        </section>

        {/* Footer */}
        <footer className="mt-12 py-6 text-center border-t border-gray-800">
          <p className="text-sm text-gray-500">
            Gweizy Analytics — Deep dive into gas price data
          </p>
          <div className="mt-4 flex justify-center gap-4 text-xs text-gray-500">
            <Link to="/" className="hover:text-cyan-400 transition-colors">Dashboard</Link>
            <Link to="/docs" className="hover:text-cyan-400 transition-colors">Docs</Link>
            <Link to="/pricing" className="hover:text-cyan-400 transition-colors">API</Link>
          </div>
        </footer>
      </div>
    </div>
  );
};

export default Analytics;
