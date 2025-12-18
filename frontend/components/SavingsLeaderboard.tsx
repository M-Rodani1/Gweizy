import React, { useState, useEffect } from 'react';
import { fetchLeaderboard } from '../src/api/gasApi';

interface LeaderboardEntry {
  address: string;
  savings: number;
  rank: number;
  badge?: string;
  streak?: number;
}

interface SavingsLeaderboardProps {
  walletAddress?: string | null;
}

const SavingsLeaderboard: React.FC<SavingsLeaderboardProps> = ({ walletAddress }) => {
  const [leaderboard, setLeaderboard] = useState<LeaderboardEntry[]>([]);
  const [userRank, setUserRank] = useState<number | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadLeaderboard = async () => {
      try {
        setLoading(true);
        setError(null);
        const data = await fetchLeaderboard();
        setLeaderboard(data.leaderboard || []);
        setUserRank(data.user_rank || null);
      } catch (err) {
        console.error('Error loading leaderboard:', err);
        setError(err instanceof Error ? err.message : 'Failed to load leaderboard');
        // Use mock data on error
        setLeaderboard([
          { address: '0x1234...5678', savings: 12.45, rank: 1, badge: '🥇', streak: 7 },
          { address: '0x8765...4321', savings: 8.90, rank: 2, badge: '🥈', streak: 5 },
          { address: '0xabcd...efgh', savings: 6.78, rank: 3, badge: '🥉', streak: 3 },
          { address: '0x9876...5432', savings: 5.23, rank: 4, streak: 2 },
          { address: '0xfedc...ba98', savings: 4.56, rank: 5, streak: 1 }
        ]);
      } finally {
        setLoading(false);
      }
    };

    loadLeaderboard();
    const interval = setInterval(loadLeaderboard, 60000); // Refresh every minute
    return () => clearInterval(interval);
  }, [walletAddress]);

  const formatAddress = (address: string) => {
    if (!address) return '';
    return `${address.slice(0, 6)}...${address.slice(-4)}`;
  };

  const getBadge = (rank: number) => {
    if (rank === 1) return '🥇';
    if (rank === 2) return '🥈';
    if (rank === 3) return '🥉';
    return '';
  };

  const getUserEntry = () => {
    if (!walletAddress || userRank === null) return null;
    return leaderboard.find(entry => entry.address.toLowerCase() === walletAddress.toLowerCase());
  };

  const userEntry = getUserEntry();

  if (loading && leaderboard.length === 0) {
    return (
      <div className="bg-gray-800 p-6 rounded-lg shadow-lg border border-gray-700">
        <h3 className="text-lg font-semibold text-gray-200 mb-4">🏆 Top Savers This Week</h3>
        <div className="text-gray-400 text-sm">Loading leaderboard...</div>
      </div>
    );
  }

  if (error && leaderboard.length === 0) {
    return (
      <div className="bg-gray-800 p-6 rounded-lg shadow-lg border border-gray-700">
        <h3 className="text-lg font-semibold text-gray-200 mb-4">🏆 Top Savers This Week</h3>
        <p className="text-red-400 text-sm mb-2">⚠️ {error}</p>
      </div>
    );
  }

  return (
    <div className="bg-gray-800 p-6 rounded-lg shadow-lg border border-gray-700">
      <h3 className="text-lg font-semibold text-gray-200 mb-4">🏆 Top Savers This Week</h3>

      {/* Leaderboard */}
      <div className="space-y-2 mb-4">
        {leaderboard.slice(0, 10).map((entry) => {
          const isUser = walletAddress && entry.address.toLowerCase() === walletAddress.toLowerCase();
          return (
            <div
              key={entry.rank}
              className={`flex items-center justify-between p-3 rounded-md ${
                isUser
                  ? 'bg-cyan-500/20 border-2 border-cyan-500/50'
                  : 'bg-gray-700/50 border border-gray-600'
              }`}
            >
              <div className="flex items-center space-x-3">
                <span className="text-xl">{getBadge(entry.rank)}</span>
                <div>
                  <div className="text-sm font-medium text-gray-200">
                    {isUser ? 'You' : formatAddress(entry.address)}
                  </div>
                  {entry.badge && (
                    <div className="text-xs text-yellow-400">{entry.badge}</div>
                  )}
                  {entry.streak && entry.streak > 0 && (
                    <div className="text-xs text-gray-400">
                      🔥 {entry.streak} day streak
                    </div>
                  )}
                </div>
              </div>
              <div className="text-right">
                <div className="text-sm font-bold text-green-400">
                  ${entry.savings !== undefined && entry.savings !== null ? entry.savings.toFixed(2) : 'N/A'}
                </div>
                <div className="text-xs text-gray-400">Saved</div>
              </div>
            </div>
          );
        })}
      </div>

      {/* User Rank (if not in top 10) */}
      {userEntry && userEntry.rank > 10 && (
        <div className="mt-4 pt-4 border-t border-gray-700">
          <div className="flex items-center justify-between p-3 bg-cyan-500/20 border-2 border-cyan-500/50 rounded-md">
            <div>
              <div className="text-sm font-medium text-gray-200">You</div>
              <div className="text-xs text-gray-400">Rank #{userEntry.rank}</div>
            </div>
            <div className="text-right">
              <div className="text-sm font-bold text-green-400">
                ${userEntry.savings !== undefined && userEntry.savings !== null ? userEntry.savings.toFixed(2) : 'N/A'}
              </div>
              <div className="text-xs text-gray-400">Saved</div>
            </div>
          </div>
        </div>
      )}

      {/* Gamification Badges */}
      {walletAddress && (
        <div className="mt-4 pt-4 border-t border-gray-700">
          <div className="text-sm font-semibold text-gray-300 mb-2">Your Badges:</div>
          <div className="flex flex-wrap gap-2">
            {userEntry && userEntry.savings > 10 && (
              <span className="px-2 py-1 bg-yellow-500/20 text-yellow-400 rounded text-xs">
                ⭐ Gas Ninja
              </span>
            )}
            {userEntry && userEntry.streak && userEntry.streak >= 7 && (
              <span className="px-2 py-1 bg-purple-500/20 text-purple-400 rounded text-xs">
                🔥 Optimizer Pro
              </span>
            )}
            {userEntry && userEntry.savings > 5 && (
              <span className="px-2 py-1 bg-green-500/20 text-green-400 rounded text-xs">
                💰 Savings Master
              </span>
            )}
            {(!userEntry || (userEntry.savings === 0 && (!userEntry.streak || userEntry.streak === 0))) && (
              <span className="text-xs text-gray-500 italic">Start saving to earn badges!</span>
            )}
          </div>
        </div>
      )}

      {/* Challenges */}
      <div className="mt-4 pt-4 border-t border-gray-700">
        <div className="text-sm font-semibold text-gray-300 mb-2">Active Challenges:</div>
        <div className="space-y-2 text-xs">
          <div className="flex items-center justify-between p-2 bg-gray-700/30 rounded">
            <span className="text-gray-300">Save $10 this month</span>
            <span className="text-cyan-400">0%</span>
          </div>
          <div className="flex items-center justify-between p-2 bg-gray-700/30 rounded">
            <span className="text-gray-300">7 days of optimal transactions</span>
            <span className="text-cyan-400">0%</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SavingsLeaderboard;

