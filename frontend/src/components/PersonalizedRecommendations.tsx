/**
 * Personalized Recommendations Component
 * Shows personalized gas optimisation recommendations based on user transaction history
 */
import React, { useState, useEffect } from 'react';
import { useChain } from '../contexts/ChainContext';
import { getApiUrl } from '../config/api';
import { Clock, TrendingUp, Target, DollarSign } from 'lucide-react';
import LoadingSpinner from './LoadingSpinner';

interface PersonalizedRecommendationsProps {
  walletAddress: string | null;
}

const PersonalizedRecommendations: React.FC<PersonalizedRecommendationsProps> = ({ walletAddress }) => {
  const { selectedChainId } = useChain();
  const [recommendations, setRecommendations] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<string>('');

  useEffect(() => {
    if (!walletAddress) {
      setLoading(false);
      return;
    }

    const loadRecommendations = async () => {
      try {
        setLoading(true);
        setError(null);

        const response = await fetch(
          getApiUrl(`/personalization/recommendations/${walletAddress}`, { chain_id: selectedChainId })
        );
        const data = await response.json();

        if (data.success) {
          setRecommendations(data);
          const now = new Date();
          setLastUpdated(now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }));
        } else {
          setError(data.error || 'Failed to load recommendations');
        }
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load recommendations');
      } finally {
        setLoading(false);
      }
    };

    loadRecommendations();
  }, [walletAddress, selectedChainId]);

  if (!walletAddress) {
    return (
      <div
        className="bg-gradient-to-br from-blue-900/30 to-gray-900/30 border border-blue-500/30 rounded-2xl p-6 h-full flex flex-col justify-between shadow-xl focus-card"
        role="article"
        aria-label="Personalized recommendations onboarding"
        tabIndex={0}
      >
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <Target className="w-5 h-5 text-blue-300" />
            <h2 className="text-lg font-semibold text-white">Personalized Recommendations</h2>
          </div>
          <p className="text-sm text-gray-300">
            Connect your wallet to unlock timing tips based on your on-chain history.
          </p>
          <ul className="text-xs text-gray-400 space-y-1">
            <li>• Best hour to transact today</li>
            <li>• Typical gas you pay vs. market</li>
            <li>• Savings estimate for your wallet</li>
          </ul>
        </div>
        <div className="pt-4 text-xs text-gray-400">
          <span>Use the header wallet button to connect</span>
        </div>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="bg-gray-800 rounded-lg p-6">
        <LoadingSpinner message="Analysing your transaction patterns..." />
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-gray-800 rounded-lg p-6">
        <p className="text-red-400 text-center">{error}</p>
      </div>
    );
  }

  if (!recommendations) {
    return null;
  }

  const { recommended_time, hours_until_best, confidence, reason, potential_savings, patterns } = recommendations;

  return (
    <div
      className="bg-gradient-to-br from-blue-900/20 to-gray-900/20 border border-blue-500/30 rounded-2xl p-6 h-full flex flex-col shadow-xl widget-glow w-full max-w-full overflow-y-auto focus-card"
      role="article"
      aria-label="Personalized recommendations"
      tabIndex={0}
    >
      <div className="flex items-center justify-between mb-5 min-w-0">
        <div className="flex items-center gap-3 min-w-0 flex-1">
          <Target className="w-6 h-6 text-blue-400 flex-shrink-0" />
          <h2 className="text-xl font-bold text-white break-words min-w-0">Personalized Recommendations</h2>
        </div>
        {lastUpdated && (
          <div className="last-updated flex-shrink-0 ml-2">
            {lastUpdated}
          </div>
        )}
      </div>

      <div className="space-y-5">
        {/* Best Time Recommendation */}
        <div className="bg-gray-800/50 rounded-lg p-5 border border-gray-700">
          <div className="flex items-center gap-2 mb-3 min-w-0">
            <Clock className="w-5 h-5 text-blue-400 flex-shrink-0" />
            <h3 className="font-semibold text-white break-words min-w-0">Best Time to Transact</h3>
          </div>
          <div className="text-2xl font-bold text-blue-400 mb-2">{recommended_time}</div>
          <p className="text-sm text-gray-400 mb-2">{reason}</p>
          {hours_until_best > 0 && (
            <p className="text-xs text-gray-500 mb-2">
              {hours_until_best} hour{hours_until_best !== 1 ? 's' : ''} until optimal time
            </p>
          )}
          <div className="mt-3">
            <span className={`badge ${
              confidence === 'high' ? 'badge-success' :
              confidence === 'medium' ? 'badge-warning' :
              'badge-info'
            }`}>
              {confidence} confidence
            </span>
          </div>
        </div>

        {/* Savings Potential */}
        {potential_savings && (
          <div className="bg-gray-800/50 rounded-lg p-5 border border-gray-700">
            <div className="flex items-center gap-2 mb-3 min-w-0">
              <DollarSign className="w-5 h-5 text-green-400 flex-shrink-0" />
              <h3 className="font-semibold text-white break-words min-w-0">Potential Savings</h3>
            </div>
            <div className="text-2xl font-bold text-green-400 mb-2">
              {potential_savings}% savings
            </div>
            <p className="text-sm text-gray-400">
              By following recommendations, you could save this much on gas fees
            </p>
          </div>
        )}

        {/* Transaction Patterns */}
        {patterns && patterns.total_transactions > 0 && (
          <div className="bg-gray-800/50 rounded-lg p-5 border border-gray-700">
            <div className="flex items-center gap-2 mb-4 min-w-0">
              <TrendingUp className="w-5 h-5 text-cyan-400 flex-shrink-0" />
              <h3 className="font-semibold text-white break-words min-w-0">Your Patterns</h3>
            </div>
            <div className="grid grid-cols-2 gap-3 text-sm">
              <div>
                <p className="text-gray-400">Total Transactions</p>
                <p className="text-white font-semibold">{patterns.total_transactions}</p>
              </div>
              <div>
                <p className="text-gray-400">Usual Time</p>
                <p className="text-white font-semibold">{patterns.recommendations?.usual_time || 'Varies'}</p>
              </div>
              {patterns.statistics && (
                <>
                  <div>
                    <p className="text-gray-400">Avg Gas Price</p>
                    <p className="text-white font-semibold">{patterns.statistics.avg_gas_price_gwei.toFixed(4)} gwei</p>
                  </div>
                  <div>
                    <p className="text-gray-400">Total Gas Paid</p>
                    <p className="text-white font-semibold">{patterns.statistics.total_gas_paid_eth.toFixed(6)} ETH</p>
                  </div>
                </>
              )}
            </div>
          </div>
        )}

        {/* Suggestion */}
        {patterns && patterns.recommendations && patterns.recommendations.suggestion && (
          <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-5">
            <p className="text-sm text-blue-300">{patterns.recommendations.suggestion}</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default PersonalizedRecommendations;
