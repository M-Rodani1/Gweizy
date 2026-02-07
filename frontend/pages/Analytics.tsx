import React, { lazy, Suspense, useState, useEffect } from 'react';
import { Brain, Calendar, ClipboardList, Sparkles, TrendingUp, Activity, Grid3X3 } from 'lucide-react';
import { Link } from 'react-router-dom';
import FeatureImportanceChart from '../src/components/FeatureImportanceChart';
import AccuracyHistoryChart from '../src/components/AccuracyHistoryChart';
import DriftAlertBanner from '../src/components/DriftAlertBanner';
import ApiStatusPanel from '../src/components/ApiStatusPanel';
import HourlyHeatmap from '../src/components/HourlyHeatmap';
import ModelMetricsPanel from '../src/components/ModelMetricsPanel';
import { CardSkeleton, ChartSkeleton, HeatmapSkeleton } from '../src/components/SkeletonLoader';
import { useChain } from '../src/contexts/ChainContext';
import { useEthPrice } from '../src/hooks/useEthPrice';
import { fetchHybridPrediction } from '../src/api/gasApi';
import { HybridPrediction } from '../types';
import AppShell from '../src/components/layout/AppShell';
import { SectionErrorBoundary } from '../src/components/SectionErrorBoundary';

// Lazy load analytics components
const GasPriceGraph = lazy(() => import('../src/components/GasPriceGraph'));
const GasPatternHeatmap = lazy(() => import('../src/components/GasPatternHeatmap'));
const PredictionCards = lazy(() => import('../src/components/PredictionCards'));
const NetworkIntelligencePanel = lazy(() => import('../src/components/NetworkIntelligencePanel'));
const BestTimeWidget = lazy(() => import('../src/components/BestTimeWidget'));
const GasPriceTable = lazy(() => import('../src/components/GasPriceTable'));

const Analytics: React.FC = () => {
  const { selectedChain, multiChainGas } = useChain();
  useEthPrice(60000); // Keep price updated in context
  const currentGas = multiChainGas[selectedChain.id]?.gasPrice || 0;
  const [hybridData, setHybridData] = useState<HybridPrediction | undefined>(undefined);

  // Fetch hybrid prediction data
  useEffect(() => {
    const loadHybridData = async () => {
      try {
        const data = await fetchHybridPrediction();
        setHybridData(data);
      } catch (error) {
        console.error('Failed to fetch hybrid prediction:', error);
      }
    };

    loadHybridData();
    const interval = setInterval(loadHybridData, 60000); // Refresh every 60 seconds
    return () => clearInterval(interval);
  }, []);

  return (
    <AppShell activePath="/analytics">
      <div className="max-w-7xl mx-auto py-2">
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
        <SectionErrorBoundary sectionName="Model Health">
          <section className="mb-8">
            <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
              <Activity className="w-4 h-4 text-cyan-400" />
              Model Health & Metrics
            </h2>
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="lg:col-span-2">
                <ModelMetricsPanel />
              </div>
              <div>
                <ApiStatusPanel />
              </div>
            </div>
          </section>
        </SectionErrorBoundary>

        {/* Accuracy Trends */}
        <SectionErrorBoundary sectionName="Accuracy Trends">
          <section className="mb-8">
            <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-cyan-400" />
              Accuracy Trends
            </h2>
            <AccuracyHistoryChart />
          </section>
        </SectionErrorBoundary>

        {/* Feature Analysis */}
        <SectionErrorBoundary sectionName="Feature Analysis">
          <section className="mb-8">
            <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
              <Brain className="w-4 h-4 text-cyan-400" />
              Feature Analysis
            </h2>
            <FeatureImportanceChart />
          </section>
        </SectionErrorBoundary>

        {/* Predictions Section */}
        <SectionErrorBoundary sectionName="ML Predictions">
          <section className="mb-8">
            <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-cyan-400" />
              ML Predictions
            </h2>
            <Suspense fallback={<CardSkeleton rows={4} />}>
              <PredictionCards hybridData={hybridData} />
            </Suspense>
          </section>
        </SectionErrorBoundary>

        {/* Charts Section */}
        <SectionErrorBoundary sectionName="Price Charts">
          <section className="mb-8">
            <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
              <TrendingUp className="w-4 h-4 text-cyan-400" />
              Price Charts
            </h2>
            <div className="grid grid-cols-1 gap-6">
              <Suspense fallback={<ChartSkeleton height={250} />}>
                <GasPriceGraph />
              </Suspense>
            </div>
          </section>
        </SectionErrorBoundary>

        {/* Weekly Heatmap */}
        <SectionErrorBoundary sectionName="Weekly Patterns">
          <section className="mb-8">
            <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
              <Grid3X3 className="w-4 h-4 text-cyan-400" />
              Weekly Patterns
            </h2>
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <HourlyHeatmap />
              <Suspense fallback={<CardSkeleton rows={3} />}>
                <BestTimeWidget currentGas={currentGas} />
              </Suspense>
            </div>
          </section>
        </SectionErrorBoundary>

        {/* Pattern Analysis */}
        <SectionErrorBoundary sectionName="Pattern Analysis">
          <section className="mb-8">
            <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
              <Calendar className="w-4 h-4 text-cyan-400" />
              Pattern Analysis
            </h2>
            <Suspense fallback={<HeatmapSkeleton />}>
              <GasPatternHeatmap />
            </Suspense>
          </section>
        </SectionErrorBoundary>

        {/* Network Intelligence */}
        <SectionErrorBoundary sectionName="Network Intelligence">
          <section className="mb-8">
            <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
              <Brain className="w-4 h-4 text-cyan-400" />
              Network Intelligence
            </h2>
            <Suspense fallback={<CardSkeleton rows={5} />}>
              <NetworkIntelligencePanel />
            </Suspense>
          </section>
        </SectionErrorBoundary>

        {/* Historical Data */}
        <SectionErrorBoundary sectionName="Historical Data">
          <section className="mb-8">
            <h2 className="text-lg font-semibold text-white mb-4 flex items-center gap-2">
              <ClipboardList className="w-4 h-4 text-cyan-400" />
              Historical Data
            </h2>
            <Suspense fallback={<ChartSkeleton height={300} />}>
              <GasPriceTable />
            </Suspense>
          </section>
        </SectionErrorBoundary>

        {/* Footer */}
        <footer className="mt-12 py-6 text-center border-t border-gray-800">
          <p className="text-sm text-gray-500">
            Gweizy Analytics — Deep dive into gas price data
          </p>
          <div className="mt-4 flex justify-center gap-4 text-xs text-gray-500">
            <Link to="/" className="hover:text-cyan-400 transition-colors">Dashboard</Link>
            <Link to="/pricing" className="hover:text-cyan-400 transition-colors">API</Link>
          </div>
        </footer>
      </div>
    </AppShell>
  );
};

export default Analytics;
