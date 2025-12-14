import React, { useState } from 'react';

interface ShareResultsProps {
  savings: number;
  savingsPercent: number;
  period: string;
  walletAddress?: string | null;
}

const ShareResults: React.FC<ShareResultsProps> = ({ savings, savingsPercent, period, walletAddress }) => {
  const [copied, setCopied] = useState(false);

  const shareUrl = walletAddress 
    ? `https://basegasoptimizer.com/?ref=${walletAddress.slice(2, 10)}`
    : 'https://basegasoptimizer.com';

  const shareText = {
    twitter: `I saved ${savingsPercent.toFixed(0)}% on @base gas fees this ${period}! Saved $${savings.toFixed(2)} using Base Gas Optimizer 💰\n\n${shareUrl}`,
    linkedin: `I've been optimizing my Base network gas fees using Base Gas Optimizer. Saved ${savingsPercent.toFixed(0)}% ($${savings.toFixed(2)}) this ${period} by timing my transactions better.\n\n${shareUrl}`,
    generic: `I saved ${savingsPercent.toFixed(0)}% on Base gas fees! Check out Base Gas Optimizer: ${shareUrl}`
  };

  const handleCopyLink = async () => {
    try {
      await navigator.clipboard.writeText(shareUrl);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy:', err);
    }
  };

  const handleShare = (platform: 'twitter' | 'linkedin') => {
    const text = shareText[platform];
    const url = platform === 'twitter' 
      ? `https://twitter.com/intent/tweet?text=${encodeURIComponent(text)}`
      : `https://www.linkedin.com/sharing/share-offsite/?url=${encodeURIComponent(shareUrl)}`;
    
    window.open(url, '_blank', 'width=600,height=400');
  };

  const handleDownloadImage = () => {
    // Create a canvas with the share card
    const canvas = document.createElement('canvas');
    canvas.width = 1200;
    canvas.height = 630;
    const ctx = canvas.getContext('2d');
    
    if (!ctx) return;

    // Background
    ctx.fillStyle = '#1F2937';
    ctx.fillRect(0, 0, 1200, 630);

    // Title
    ctx.fillStyle = '#E5E7EB';
    ctx.font = 'bold 48px Arial';
    ctx.textAlign = 'center';
    ctx.fillText('I saved on Base gas fees!', 600, 150);

    // Savings
    ctx.fillStyle = '#10B981';
    ctx.font = 'bold 72px Arial';
    ctx.fillText(`${savingsPercent.toFixed(0)}%`, 600, 280);
    ctx.fillStyle = '#9CA3AF';
    ctx.font = '36px Arial';
    ctx.fillText(`$${savings.toFixed(2)} saved this ${period}`, 600, 340);

    // Logo/URL
    ctx.fillStyle = '#60A5FA';
    ctx.font = '32px Arial';
    ctx.fillText('basegasoptimizer.com', 600, 500);

    // Download
    canvas.toBlob((blob) => {
      if (blob) {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `gas-savings-${Date.now()}.png`;
        a.click();
        URL.revokeObjectURL(url);
      }
    });
  };

  return (
    <div className="bg-gray-800 p-6 rounded-lg shadow-lg border border-gray-700">
      <h3 className="text-lg font-semibold text-gray-200 mb-4">📱 Share Your Savings</h3>

      <div className="space-y-3">
        {/* Share Buttons */}
        <div className="grid grid-cols-2 gap-3">
          <button
            onClick={() => handleShare('twitter')}
            className="flex items-center justify-center space-x-2 px-4 py-3 bg-blue-500 hover:bg-blue-600 rounded-md transition-colors text-white font-medium"
          >
            <span>🐦</span>
            <span>Twitter</span>
          </button>

          <button
            onClick={() => handleShare('linkedin')}
            className="flex items-center justify-center space-x-2 px-4 py-3 bg-blue-700 hover:bg-blue-800 rounded-md transition-colors text-white font-medium"
          >
            <span>💼</span>
            <span>LinkedIn</span>
          </button>
        </div>

        {/* Copy Link */}
        <button
          onClick={handleCopyLink}
          className="w-full flex items-center justify-center space-x-2 px-4 py-3 bg-gray-700 hover:bg-gray-600 rounded-md transition-colors text-gray-200 font-medium"
        >
          <span>{copied ? '✅' : '📋'}</span>
          <span>{copied ? 'Copied!' : 'Copy Link'}</span>
        </button>

        {/* Download Image */}
        <button
          onClick={handleDownloadImage}
          className="w-full flex items-center justify-center space-x-2 px-4 py-3 bg-cyan-500 hover:bg-cyan-600 rounded-md transition-colors text-white font-medium"
        >
          <span>🖼️</span>
          <span>Download Image</span>
        </button>
      </div>

      {/* Referral Link */}
      {walletAddress && (
        <div className="mt-4 pt-4 border-t border-gray-700">
          <div className="text-sm text-gray-400 mb-2">Your Referral Link:</div>
          <div className="flex items-center space-x-2">
            <input
              type="text"
              value={shareUrl}
              readOnly
              className="flex-1 bg-gray-700 text-gray-200 text-xs px-3 py-2 rounded border border-gray-600"
            />
            <button
              onClick={handleCopyLink}
              className="px-3 py-2 bg-cyan-500 hover:bg-cyan-600 rounded text-sm text-white"
            >
              Copy
            </button>
          </div>
          <div className="text-xs text-gray-500 mt-2">
            Invite friends and track adoption
          </div>
        </div>
      )}
    </div>
  );
};

export default ShareResults;

