import React, { useState, useEffect } from 'react';
import { fetchGlobalStats } from '../api/gasApi';

interface Stats {
  total_saved_k: number;
  accuracy_percent: number;
  predictions_k: number;
}

const SocialProof: React.FC = () => {
  const [stats, setStats] = useState<Stats | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadStats = async () => {
      try {
        const data = await fetchGlobalStats();
        setStats({
          total_saved_k: data.total_saved_k || 0,
          accuracy_percent: data.accuracy_percent || 0,
          predictions_k: data.predictions_k || 0
        });
      } catch (err) {
        console.error('Failed to fetch global stats:', err);
        // No fallback - show loading state or zeros
      } finally {
        setLoading(false);
      }
    };

    loadStats();
    const interval = setInterval(loadStats, 300000); // Refresh every 5 minutes
    return () => clearInterval(interval);
  }, []);

  const formatSaved = (value: number) => {
    if (value >= 1000) return `$${(value / 1000).toFixed(0)}M+`;
    if (value >= 1) return `$${value.toFixed(0)}K+`;
    return '$0';
  };

  const formatPredictions = (value: number) => {
    if (value >= 1000) return `${(value / 1000).toFixed(0)}M+`;
    if (value >= 1) return `${value.toFixed(0)}K+`;
    return '0';
  };

  return (
    <div className="bg-gradient-to-r from-cyan-500/10 to-emerald-500/10 border border-cyan-500/30 rounded-lg p-4 md:p-6 mb-6">
      <div className="flex flex-col md:flex-row items-center justify-between gap-4">
        {/* Hackathon Winner Badge */}
        <div className="flex items-center gap-3">
          <div className="text-4xl md:text-5xl">🏆</div>
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="text-lg md:text-xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-emerald-400">
                Coinbase 2025 Hackathon Winner
              </span>
            </div>
            <p className="text-sm text-gray-400">
              Built at Coinbase Hackathon
            </p>
          </div>
        </div>

        {/* Stats - Live from API */}
        <div className="flex gap-6 md:gap-8">
          <div className="text-center">
            <div className="text-2xl md:text-3xl font-bold text-white">
              {loading ? (
                <span className="animate-pulse">---</span>
              ) : (
                formatSaved(stats?.total_saved_k || 0)
              )}
            </div>
            <div className="text-xs md:text-sm text-gray-400">
              Saved in Fees
            </div>
          </div>
          <div className="text-center">
            <div className="text-2xl md:text-3xl font-bold text-white">
              {loading ? (
                <span className="animate-pulse">--</span>
              ) : (
                `${stats?.accuracy_percent || 0}%`
              )}
            </div>
            <div className="text-xs md:text-sm text-gray-400">
              Accuracy
            </div>
          </div>
          <div className="text-center">
            <div className="text-2xl md:text-3xl font-bold text-white">
              {loading ? (
                <span className="animate-pulse">---</span>
              ) : (
                formatPredictions(stats?.predictions_k || 0)
              )}
            </div>
            <div className="text-xs md:text-sm text-gray-400">
              Predictions
            </div>
          </div>
        </div>
      </div>

      {/* Trust Indicators */}
      <div className="mt-4 pt-4 border-t border-cyan-500/20">
        <div className="flex flex-wrap items-center justify-center gap-4 md:gap-6 text-xs md:text-sm text-gray-400">
          <div className="flex items-center gap-1.5">
            <span className="text-green-400">✓</span>
            <span>AI-Powered Predictions</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="text-green-400">✓</span>
            <span>Real-time Data</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="text-green-400">✓</span>
            <span>Base Network Optimized</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="text-green-400">✓</span>
            <span>Free to Use</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SocialProof;
