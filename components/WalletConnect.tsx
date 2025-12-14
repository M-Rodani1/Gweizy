import React, { useState, useEffect } from 'react';
import {
  isWalletInstalled,
  connectWallet,
  getCurrentAccount,
  getCurrentChainId,
  formatAddress,
  copyToClipboard,
  getBaseScanAddressUrl,
  onAccountsChanged,
  onChainChanged
} from '../src/utils/wallet';

const BASE_CHAIN_ID = 8453;

const WalletConnect: React.FC = () => {
  const [address, setAddress] = useState<string | null>(null);
  const [chainId, setChainId] = useState<number | null>(null);
  const [isConnecting, setIsConnecting] = useState(false);
  const [showDropdown, setShowDropdown] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const checkConnection = async () => {
    if (!isWalletInstalled()) return;

    try {
      const account = await getCurrentAccount();
      const currentChainId = await getCurrentChainId();
      
      setAddress(account);
      setChainId(currentChainId);
    } catch (err) {
      console.error('Error checking connection:', err);
    }
  };

  useEffect(() => {
    checkConnection();

    // Listen for account changes
    const unsubscribeAccounts = onAccountsChanged((accounts) => {
      setAddress(accounts.length > 0 ? accounts[0] : null);
      setShowDropdown(false);
    });

    // Listen for chain changes
    const unsubscribeChain = onChainChanged((newChainId) => {
      setChainId(parseInt(newChainId, 16));
      setShowDropdown(false);
    });

    return () => {
      unsubscribeAccounts();
      unsubscribeChain();
    };
  }, []);

  const handleConnect = async () => {
    setIsConnecting(true);
    setError(null);

    try {
      const account = await connectWallet();
      setAddress(account);
      const currentChainId = await getCurrentChainId();
      setChainId(currentChainId);
    } catch (err: any) {
      setError(err.message || 'Failed to connect wallet');
      console.error('Error connecting wallet:', err);
    } finally {
      setIsConnecting(false);
    }
  };

  const handleDisconnect = () => {
    setAddress(null);
    setChainId(null);
    setShowDropdown(false);
  };

  const handleCopyAddress = async () => {
    if (address) {
      await copyToClipboard(address);
      setShowDropdown(false);
      // You could add a toast notification here
    }
  };

  const handleViewOnBaseScan = () => {
    if (address) {
      window.open(getBaseScanAddressUrl(address), '_blank');
      setShowDropdown(false);
    }
  };

  const isOnBaseNetwork = chainId === BASE_CHAIN_ID;

  if (!isWalletInstalled()) {
    return (
      <div className="flex items-center space-x-2">
        <a
          href="https://metamask.io/download/"
          target="_blank"
          rel="noopener noreferrer"
          className="px-4 py-2 bg-cyan-500 hover:bg-cyan-600 rounded-md transition-colors text-sm font-medium"
        >
          Install Wallet
        </a>
      </div>
    );
  }

  if (!address) {
    return (
      <div className="flex items-center space-x-2">
        <button
          onClick={handleConnect}
          disabled={isConnecting}
          className="px-4 py-2 bg-cyan-500 hover:bg-cyan-600 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-md transition-colors text-sm font-medium"
        >
          {isConnecting ? 'Connecting...' : 'Connect Wallet'}
        </button>
        {error && (
          <span className="text-xs text-red-400 max-w-xs truncate" title={error}>
            {error}
          </span>
        )}
      </div>
    );
  }

  return (
    <div className="relative">
      <button
        onClick={() => setShowDropdown(!showDropdown)}
        className="flex items-center space-x-2 px-4 py-2 bg-gray-700 hover:bg-gray-600 rounded-md transition-colors text-sm font-medium"
      >
        <div className={`w-2 h-2 rounded-full ${isOnBaseNetwork ? 'bg-green-500' : 'bg-yellow-500'}`}></div>
        <span className="text-gray-200">{formatAddress(address)}</span>
        <span className="text-gray-400">▼</span>
      </button>

      {showDropdown && (
        <>
          <div
            className="fixed inset-0 z-10"
            onClick={() => setShowDropdown(false)}
          ></div>
          <div className="absolute right-0 mt-2 w-56 bg-gray-800 rounded-lg shadow-lg border border-gray-700 z-20">
            <div className="p-2">
              <div className="px-3 py-2 text-xs text-gray-400 border-b border-gray-700 mb-2">
                {formatAddress(address)}
              </div>

              {!isOnBaseNetwork && (
                <div className="px-3 py-2 text-xs text-yellow-400 mb-2 bg-yellow-500/10 rounded">
                  ⚠️ Switch to Base network
                </div>
              )}

              <button
                onClick={handleCopyAddress}
                className="w-full text-left px-3 py-2 text-sm text-gray-200 hover:bg-gray-700 rounded transition-colors"
              >
                📋 Copy Address
              </button>

              <button
                onClick={handleViewOnBaseScan}
                className="w-full text-left px-3 py-2 text-sm text-gray-200 hover:bg-gray-700 rounded transition-colors"
              >
                🔍 View on BaseScan
              </button>

              <div className="border-t border-gray-700 my-2"></div>

              <button
                onClick={handleDisconnect}
                className="w-full text-left px-3 py-2 text-sm text-red-400 hover:bg-gray-700 rounded transition-colors"
              >
                Disconnect
              </button>
            </div>
          </div>
        </>
      )}
    </div>
  );
};

export default WalletConnect;

