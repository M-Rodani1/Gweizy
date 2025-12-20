import React, { useEffect, useState } from 'react';
import { GasIcon } from './icons';
import WalletConnect from './WalletConnect';

interface StickyHeaderProps {
  apiStatus: 'checking' | 'online' | 'offline';
  currentGas: number;
}

const StickyHeader: React.FC<StickyHeaderProps> = ({ apiStatus, currentGas }) => {
  const [isScrolled, setIsScrolled] = useState(false);

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 20);
    };

    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  return (
    <header
      className={`sticky top-0 z-50 transition-all duration-300 ${
        isScrolled
          ? 'bg-gray-900/80 backdrop-blur-xl shadow-lg shadow-cyan-500/5 border-b border-cyan-500/10'
          : 'bg-transparent'
      }`}
    >
      <div className="w-full mx-auto px-4 sm:px-6 lg:px-8 py-4">
        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
          <div className="flex items-center space-x-3">
            <div className="relative">
              <GasIcon className={`w-7 h-7 sm:w-8 sm:h-8 text-cyan-400 flex-shrink-0 transition-transform duration-300 ${
                isScrolled ? 'scale-90' : 'scale-100'
              }`} />
              {isScrolled && currentGas > 0 && (
                <div className="absolute -top-1 -right-1 w-3 h-3 bg-green-400 rounded-full animate-pulse" />
              )}
            </div>
            <div>
              <h1 className={`font-bold text-gray-100 transition-all duration-300 ${
                isScrolled ? 'text-lg sm:text-xl' : 'text-xl sm:text-2xl md:text-3xl'
              }`}>
                Base Gas Optimizer
              </h1>
              {!isScrolled && (
                <p className="text-xs sm:text-sm text-gray-400 mt-1 hidden sm:block animate-fade-in">
                  Know the best times to transact on Base network
                </p>
              )}
              {isScrolled && currentGas > 0 && (
                <p className="text-xs text-cyan-400 font-semibold animate-fade-in">
                  {currentGas.toFixed(2)} Gwei
                </p>
              )}
            </div>
          </div>

          <div className="flex items-center justify-between sm:justify-end gap-3 sm:gap-4">
            <div className="flex items-center space-x-2 px-3 py-1.5 rounded-full bg-gray-800/50 backdrop-blur-sm border border-gray-700/50">
              <div className={`w-2 h-2 rounded-full transition-all duration-300 ${
                apiStatus === 'online' ? 'bg-green-500 shadow-lg shadow-green-500/50' :
                apiStatus === 'offline' ? 'bg-red-500 shadow-lg shadow-red-500/50' :
                'bg-yellow-500 shadow-lg shadow-yellow-500/50'
              }`}></div>
              <span className="text-xs sm:text-sm text-gray-300 font-medium">
                {apiStatus === 'online' ? 'Live' :
                 apiStatus === 'offline' ? 'Offline' :
                 'Checking...'}
              </span>
            </div>
            <WalletConnect />
          </div>
        </div>
      </div>
    </header>
  );
};

export default StickyHeader;
