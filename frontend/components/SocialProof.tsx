import React from 'react';

const SocialProof = () => {
  return (
    <div className="bg-gradient-to-r from-cyan-500/10 to-emerald-500/10 border border-cyan-500/30 rounded-lg p-4 md:p-6 mb-6">
      <div className="flex flex-col md:flex-row items-center justify-between gap-4">
        {/* Hackathon Winner Badge */}
        <div className="flex items-center gap-3">
          <div className="text-4xl md:text-5xl">🏆</div>
          <div>
            <div className="flex items-center gap-2 mb-1">
              <span className="text-lg md:text-xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-emerald-400">
                Hackathon Winner
              </span>
              <span className="px-2 py-0.5 bg-cyan-500/20 text-cyan-300 text-xs font-semibold rounded">
                2024
              </span>
            </div>
            <p className="text-sm text-gray-400">
              Backed by Coinbase
            </p>
          </div>
        </div>

        {/* Stats */}
        <div className="flex gap-6 md:gap-8">
          <div className="text-center">
            <div className="text-2xl md:text-3xl font-bold text-white">
              $50K+
            </div>
            <div className="text-xs md:text-sm text-gray-400">
              Saved in Fees
            </div>
          </div>
          <div className="text-center">
            <div className="text-2xl md:text-3xl font-bold text-white">
              95%
            </div>
            <div className="text-xs md:text-sm text-gray-400">
              Accuracy
            </div>
          </div>
          <div className="text-center">
            <div className="text-2xl md:text-3xl font-bold text-white">
              5K+
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
