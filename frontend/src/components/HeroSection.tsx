import React, { useEffect, useState } from 'react';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';

interface HeroSectionProps {
  currentGas: number;
  predictions?: {
    '1h': number;
    '4h': number;
    '24h': number;
  };
  ethPrice?: number;
}

// Calculate USD cost for a typical swap (150k gas)
const calculateSwapCost = (gasGwei: number, ethPrice: number): string => {
  const gasUnits = 150000; // Typical Uniswap swap
  const ethCost = (gasGwei * gasUnits) / 1e9;
  const usdCost = ethCost * ethPrice;
  return usdCost < 0.01 ? '<$0.01' : `$${usdCost.toFixed(2)}`;
};

const HeroSection: React.FC<HeroSectionProps> = ({ currentGas, predictions, ethPrice = 3500 }) => {
  const [previousGas, setPreviousGas] = useState(currentGas);
  const [trend, setTrend] = useState<'up' | 'down' | 'stable'>('stable');
  const [animateValue, setAnimateValue] = useState(false);

  useEffect(() => {
    if (currentGas !== previousGas && currentGas > 0) {
      setAnimateValue(true);
      if (currentGas > previousGas) {
        setTrend('up');
      } else if (currentGas < previousGas) {
        setTrend('down');
      } else {
        setTrend('stable');
      }
      setPreviousGas(currentGas);

      const timer = setTimeout(() => setAnimateValue(false), 600);
      return () => clearTimeout(timer);
    }
  }, [currentGas, previousGas]);

  const next1h = predictions?.['1h'] || 0;
  const changePercent = currentGas > 0 ? ((next1h - currentGas) / currentGas) * 100 : 0;

  const getPriceStatus = () => {
    if (currentGas < 0.05) return { label: 'Extremely Low', color: 'text-green-400', bg: 'bg-green-500/10', border: 'border-green-500/30' };
    if (currentGas < 0.1) return { label: 'Very Low', color: 'text-green-300', bg: 'bg-green-500/10', border: 'border-green-500/30' };
    if (currentGas < 0.2) return { label: 'Low', color: 'text-cyan-400', bg: 'bg-cyan-500/10', border: 'border-cyan-500/30' };
    if (currentGas < 0.5) return { label: 'Moderate', color: 'text-yellow-400', bg: 'bg-yellow-500/10', border: 'border-yellow-500/30' };
    if (currentGas < 1.0) return { label: 'High', color: 'text-orange-400', bg: 'bg-orange-500/10', border: 'border-orange-500/30' };
    return { label: 'Very High', color: 'text-red-400', bg: 'bg-red-500/10', border: 'border-red-500/30' };
  };

  const status = getPriceStatus();

  return (
    <div className="relative overflow-hidden rounded-3xl bg-gradient-to-br from-gray-800 via-gray-900 to-black border border-cyan-500/20 shadow-2xl shadow-cyan-500/10 mb-8">
      {/* Animated background gradient */}
      <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/5 via-cyan-500/5 to-cyan-500/5 animate-gradient-x" />

      {/* Floating orbs */}
      <div className="absolute top-0 left-1/4 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl animate-pulse" />
      <div className="absolute bottom-0 right-1/4 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl animate-pulse delay-1000" />

      <div className="relative z-10 px-6 py-8 sm:px-8 sm:py-12 lg:px-12 lg:py-16">
        <div className="max-w-6xl mx-auto">
          <div className="text-center">
            {/* Status Badge */}
            <div className="inline-flex items-center justify-center mb-6">
              <div className={`px-4 py-2 rounded-full ${status.bg} ${status.border} border backdrop-blur-sm`}>
                <span className={`text-sm font-semibold ${status.color}`}>
                  {status.label} Gas Price
                </span>
              </div>
            </div>

            {/* Main Gas Price Display */}
            <div className="mb-8">
              <div className="flex items-center justify-center gap-3 mb-2">
                <span className="text-base sm:text-lg text-gray-400 font-medium">Current Gas Price</span>
                {trend !== 'stable' && (
                  <div className={`transition-all duration-300 ${animateValue ? 'scale-125' : 'scale-100'}`}>
                    {trend === 'up' && <TrendingUp className="w-5 h-5 text-red-400" />}
                    {trend === 'down' && <TrendingDown className="w-5 h-5 text-green-400" />}
                  </div>
                )}
              </div>

              <div className={`transition-all duration-500 ${animateValue ? 'scale-105' : 'scale-100'}`}>
                <div className="flex items-baseline justify-center gap-2">
                  <span className={`text-6xl sm:text-7xl md:text-8xl font-bold ${status.color} tracking-tight`}>
                    {currentGas > 0 ? currentGas.toFixed(4) : '---'}
                  </span>
                  <span className="text-2xl sm:text-3xl md:text-4xl text-gray-500 font-semibold">
                    Gwei
                  </span>
                </div>
              </div>

              {/* Current Gas USD Cost */}
              {currentGas > 0 && (
                <div className="mt-2 text-sm text-gray-400">
                  Swap cost: <span className="text-cyan-400 font-semibold">{calculateSwapCost(currentGas, ethPrice)}</span>
                  <span className="text-gray-500 ml-1">(150k gas)</span>
                </div>
              )}

              {/* 1h Prediction Preview */}
              {next1h > 0 && currentGas > 0 && (
                <div className="mt-4 flex items-center justify-center gap-2 text-sm sm:text-base">
                  <span className="text-gray-400">In 1 hour:</span>
                  <span className={`font-semibold ${changePercent > 0 ? 'text-red-400' : changePercent < 0 ? 'text-green-400' : 'text-gray-300'}`}>
                    {next1h.toFixed(4)} Gwei
                  </span>
                  <span className="text-gray-500">({calculateSwapCost(next1h, ethPrice)})</span>
                  <span className={`flex items-center gap-1 ${changePercent > 0 ? 'text-red-400' : changePercent < 0 ? 'text-green-400' : 'text-gray-400'}`}>
                    {changePercent > 0 && <TrendingUp className="w-4 h-4" />}
                    {changePercent < 0 && <TrendingDown className="w-4 h-4" />}
                    {changePercent === 0 && <Minus className="w-4 h-4" />}
                    {Math.abs(changePercent).toFixed(1)}%
                  </span>
                </div>
              )}
            </div>

            {/* Quick Stats */}
            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 max-w-3xl mx-auto">
              <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-4 border border-gray-700/50 hover:border-cyan-500/30 transition-all duration-300 hover:shadow-lg hover:shadow-cyan-500/5">
                <div className="text-gray-400 text-xs sm:text-sm mb-1">1 Hour</div>
                <div className="text-cyan-400 text-lg sm:text-xl font-bold">
                  {predictions?.['1h'] ? predictions['1h'].toFixed(4) : '---'} <span className="text-sm text-gray-500">Gwei</span>
                </div>
                <div className="text-xs text-gray-500 mt-1">
                  {predictions?.['1h'] ? calculateSwapCost(predictions['1h'], ethPrice) : '---'}
                </div>
              </div>

              <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-4 border border-gray-700/50 hover:border-cyan-500/30 transition-all duration-300 hover:shadow-lg hover:shadow-cyan-500/5">
                <div className="text-gray-400 text-xs sm:text-sm mb-1">4 Hours</div>
                <div className="text-cyan-400 text-lg sm:text-xl font-bold">
                  {predictions?.['4h'] ? predictions['4h'].toFixed(4) : '---'} <span className="text-sm text-gray-500">Gwei</span>
                </div>
                <div className="text-xs text-gray-500 mt-1">
                  {predictions?.['4h'] ? calculateSwapCost(predictions['4h'], ethPrice) : '---'}
                </div>
              </div>

              <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-4 border border-gray-700/50 hover:border-cyan-500/30 transition-all duration-300 hover:shadow-lg hover:shadow-cyan-500/5">
                <div className="text-gray-400 text-xs sm:text-sm mb-1">24 Hours</div>
                <div className="text-cyan-400 text-lg sm:text-xl font-bold">
                  {predictions?.['24h'] ? predictions['24h'].toFixed(4) : '---'} <span className="text-sm text-gray-500">Gwei</span>
                </div>
                <div className="text-xs text-gray-500 mt-1">
                  {predictions?.['24h'] ? calculateSwapCost(predictions['24h'], ethPrice) : '---'}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HeroSection;
