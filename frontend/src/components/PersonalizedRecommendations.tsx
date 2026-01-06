/**
 * Personalized Recommendations Component
 * Shows personalized gas optimization recommendations based on user transaction history
 */
import React, { useState, useEffect } from 'react';
import { useChain } from '../contexts/ChainContext';
import { getApiUrl } from '../config/api';
import { Clock, TrendingDown, TrendingUp, Target, DollarSign } from 'lucide-react';
import LoadingSpinner from './LoadingSpinner';

interface PersonalizedRecommendationsProps {
  walletAddress: string | null;
}

const PersonalizedRecommendations: React.FC<PersonalizedRecommendationsProps> = ({ walletAddress }) => {
  const { selectedChainId } = useChain();
  const [recommendations, setRecommendations] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

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
      <div className="bg-gray-800 rounded-lg p-6">
        <p className="text-gray-400 text-center">Connect your wallet to see personalized recommendations</p>
      </div>
    );
  }

  if (loading) {
    return (
      <div className="bg-gray-800 rounded-lg p-6">
        <LoadingSpinner message="Analyzing your transaction patterns..." />
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
    <div className="bg-gradient-to-br from-blue-900/20 to-purple-900/20 border border-blue-500/30 rounded-2xl p-6 h-full flex flex-col shadow-xl w-full max-w-full overflow-hidden">
      <div className="flex items-center gap-3 mb-5">
        <Target className="w-6 h-6 text-blue-400" />
        <h2 className="text-xl font-bold text-white">Personalized Recommendations</h2>
      </div>

      <div className="space-y-5">
        {/* Best Time Recommendation */}
        <div className="bg-gray-800/50 rounded-lg p-5 border border-gray-700">
          <div className="flex items-center gap-2 mb-3">
            <Clock className="w-5 h-5 text-blue-400" />
            <h3 className="font-semibold text-white">Best Time to Transact</h3>
          </div>
          <div className="text-2xl font-bold text-blue-400 mb-2">{recommended_time}</div>
          <p className="text-sm text-gray-400 mb-2">{reason}</p>
          {hours_until_best > 0 && (
            <p className="text-xs text-gray-500 mb-2">
              {hours_until_best} hour{hours_until_best !== 1 ? 's' : ''} until optimal time
            </p>
          )}
          <div className="mt-3">
            <span className={`text-xs px-2 py-1 rounded ${
              confidence === 'high' ? 'bg-green-500/20 text-green-400' :
              confidence === 'medium' ? 'bg-yellow-500/20 text-yellow-400' :
              'bg-gray-500/20 text-gray-400'
            }`}>
              {confidence} confidence
            </span>
          </div>
        </div>

        {/* Savings Potential */}
        {potential_savings && (
          <div className="bg-gray-800/50 rounded-lg p-5 border border-gray-700">
            <div className="flex items-center gap-2 mb-3">
              <DollarSign className="w-5 h-5 text-green-400" />
              <h3 className="font-semibold text-white">Potential Savings</h3>
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
            <div className="flex items-center gap-2 mb-4">
              <TrendingUp className="w-5 h-5 text-purple-400" />
              <h3 className="font-semibold text-white">Your Patterns</h3>
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

