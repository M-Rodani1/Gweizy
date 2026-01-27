import React from 'react';
import { useFarcaster } from '../contexts/FarcasterContext';

const FarcasterWidget: React.FC = () => {
  const { isSDKLoaded, user, error, openUrl } = useFarcaster();

  // Don't show widget if not in Farcaster or if there's an error
  if (!isSDKLoaded || error) {
    return null;
  }

  const handleSharePrediction = () => {
    const shareText = "Just checked gas prices on Base with Gas Optimiser! 📊 Save up to 40% on transaction fees with AI-powered predictions.";
    const shareUrl = "https://basegasfeesml.netlify.app";

    // Farcaster share URL
    const farcasterShareUrl = `https://warpcast.com/~/compose?text=${encodeURIComponent(shareText)}&embeds[]=${encodeURIComponent(shareUrl)}`;

    openUrl(farcasterShareUrl);
  };

  return (
    <div className="bg-gradient-to-r from-purple-500/10 to-pink-500/10 border border-purple-500/20 rounded-xl p-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-purple-500/20 rounded-full flex items-center justify-center">
            <svg className="w-5 h-5 text-purple-400" fill="currentColor" viewBox="0 0 24 24">
              <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8z"/>
              <path d="M12 6c-3.31 0-6 2.69-6 6s2.69 6 6 6 6-2.69 6-6-2.69-6-6-6zm0 10c-2.21 0-4-1.79-4-4s1.79-4 4-4 4 1.79 4 4-1.79 4-4 4z"/>
            </svg>
          </div>

          <div>
            <p className="text-sm font-medium text-white">
              {user?.displayName || user?.username || 'Farcaster User'}
            </p>
            <p className="text-xs text-gray-400">
              Connected via Farcaster
            </p>
          </div>
        </div>

        <button
          onClick={handleSharePrediction}
          className="px-4 py-2 bg-purple-500 hover:bg-purple-600 text-white text-sm font-medium rounded-lg transition-colors"
        >
          Share on Warpcast
        </button>
      </div>

      {user?.fid && (
        <div className="mt-3 pt-3 border-t border-purple-500/20">
          <p className="text-xs text-gray-400">
            FID: {user.fid}
          </p>
        </div>
      )}
    </div>
  );
};

export default FarcasterWidget;
