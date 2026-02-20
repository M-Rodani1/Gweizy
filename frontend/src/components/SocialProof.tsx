import React, { useState, useEffect } from 'react';
import { fetchGlobalStats } from '../api/gasApi';
import { REFRESH_INTERVALS } from '../constants';
import type { GlobalStatsResponse } from '../../types';

const SocialProof: React.FC = () => {
  const [stats, setStats] = useState<GlobalStatsResponse | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadStats = async () => {
      try {
        const data = await fetchGlobalStats();
        setStats(data);
      } catch (err) {
        console.error('Failed to fetch global stats:', err);
        // No fallback - show loading state or zeros
      } finally {
        setLoading(false);
      }
    };

    void loadStats();
    const refreshIfVisible = () => {
      if (typeof document === 'undefined' || document.visibilityState === 'visible') {
        void loadStats();
      }
    };
    const interval = setInterval(refreshIfVisible, REFRESH_INTERVALS.HISTORICAL); // Refresh every 5 minutes
    const handleVisibilityChange = () => {
      if (document.visibilityState === 'visible') {
        void loadStats();
      }
    };
    document.addEventListener('visibilitychange', handleVisibilityChange);

    return () => {
      clearInterval(interval);
      document.removeEventListener('visibilitychange', handleVisibilityChange);
    };
  }, []);

  const formatSaved = (value: number) => {
    if (value >= 1_000_000) return `$${(value / 1_000_000).toFixed(1)}M+`;
    if (value >= 1_000) return `$${(value / 1_000).toFixed(0)}K+`;
    return '$0';
  };

  const formatPredictions = (value: number) => {
    if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(1)}M+`;
    if (value >= 1_000) return `${(value / 1_000).toFixed(0)}K+`;
    return '0';
  };

  const getAccuracyPercent = (value: number) => {
    return value <= 1 ? value * 100 : value;
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
                formatSaved(stats?.total_savings_usd || 0)
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
                `${getAccuracyPercent(stats?.average_accuracy || 0).toFixed(0)}%`
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
                formatPredictions(stats?.predictions_made || 0)
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
            <span>Base Network Optimised</span>
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
