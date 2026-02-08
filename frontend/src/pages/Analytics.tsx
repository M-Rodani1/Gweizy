import React from 'react';
import AccuracyDashboard from '../components/AccuracyDashboard';
import { useChain } from '../contexts/ChainContext';

const Analytics: React.FC = () => {
  const { selectedChainId } = useChain();

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-900 to-gray-800">
      <div className="container mx-auto px-4 py-8">
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-white mb-2">Analytics</h1>
          <p className="text-gray-400">
            Track model performance, accuracy trends, and prediction success rates over time
          </p>
        </div>

        <AccuracyDashboard selectedChain={String(selectedChainId)} />
      </div>
    </div>
  );
};

export default Analytics;
