import React, { useEffect, useState, memo } from 'react';
import { ArrowRight, Sparkles, TrendingDown, TrendingUp } from 'lucide-react';
import { useChain } from '../contexts/ChainContext';
import { fetchPredictions } from '../api/gasApi';
import { SkeletonForecast } from './ui/Skeleton';
import AnimatedNumber from './ui/AnimatedNumber';
import { CorrectionBadge } from './ui/CorrectionBadge';
import { BiasCorrection } from '../../types';

interface Prediction {
  horizon: '1h' | '4h' | '24h';
  predicted: number;
  confidence: number;
  direction: 'up' | 'down' | 'stable';
  changePercent: number;
  biasCorrection?: BiasCorrection;
}

const CompactForecast: React.FC = () => {
  const { selectedChain, multiChainGas } = useChain();
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(true);
  const [lastUpdated, setLastUpdated] = useState<string>('');

  const currentGas = multiChainGas[selectedChain.id]?.gasPrice || 0;

  useEffect(() => {
    const loadPredictions = async () => {
      try {
        setLoading(true);
        const result = await fetchPredictions(selectedChain.id);

        if (result?.predictions) {
          const preds: Prediction[] = [];

          (['1h', '4h', '24h'] as const).forEach((horizon) => {
            const data = result.predictions[horizon];
            if (Array.isArray(data) && data.length > 0) {
              const predicted = data[0].predictedGwei || 0;
              const confidence = data[0].confidence || 0.5;
              const changePercent = currentGas > 0 ? ((predicted - currentGas) / currentGas) * 100 : 0;
              const biasCorrection = data[0].bias_correction;

              preds.push({
                horizon,
                predicted,
                confidence,
                direction: changePercent > 5 ? 'up' : changePercent < -5 ? 'down' : 'stable',
                changePercent,
                biasCorrection
              });
            }
          });

          setPredictions(preds);
          const now = new Date();
          setLastUpdated(now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }));
        }
      } catch (err) {
        console.error('Failed to load predictions:', err);
      } finally {
        setLoading(false);
      }
    };

    loadPredictions();
    const interval = setInterval(loadPredictions, 60000);
    return () => clearInterval(interval);
  }, [selectedChain.id, currentGas]);

  const getDirectionIcon = (direction: string) => {
    switch (direction) {
      case 'up':
        return <TrendingUp className="w-4 h-4 text-red-400 icon-bounce" />;
      case 'down':
        return <TrendingDown className="w-4 h-4 text-green-400 icon-bounce" />;
      default:
        return <ArrowRight className="w-4 h-4 text-yellow-400 icon-bounce" />;
    }
  };

  const getDirectionColor = (direction: string) => {
    switch (direction) {
      case 'up': return 'text-red-400';
      case 'down': return 'text-green-400';
      default: return 'text-yellow-400';
    }
  };

  // Show skeleton while loading
  if (loading) {
    return <SkeletonForecast className="shadow-xl widget-glow" />;
  }

  return (
    <div
      className="bg-gray-800/50 border border-gray-700 rounded-2xl overflow-hidden h-full flex flex-col shadow-xl widget-glow bg-pattern-dots"
      role="region"
      aria-label="Gas price forecast"
    >
      {/* Header */}
      <div className="px-6 py-4 border-b border-gray-700/50 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Sparkles className="w-4 h-4 text-cyan-400" aria-hidden="true" />
          <h3 className="font-semibold text-white" id="forecast-heading">Price Forecast</h3>
        </div>
        <div className="flex items-center gap-2">
          {lastUpdated && (
            <div className="last-updated text-xs">
              {lastUpdated}
            </div>
          )}
          <div className="text-xs text-gray-500">{selectedChain.name}</div>
        </div>
      </div>

      {/* Current Price */}
      <div className="px-6 py-4 border-b border-gray-700/30 bg-gray-800/30">
        <div className="flex items-center justify-between">
          <span className="text-sm text-gray-400">Current</span>
          <span className="font-mono font-bold text-cyan-400 text-lg">
            <AnimatedNumber value={currentGas} decimals={4} suffix=" gwei" />
          </span>
        </div>
      </div>

      {/* Predictions */}
      <div className="divide-y divide-gray-700/30 flex-1" role="list" aria-label="Gas price predictions">
          {predictions.map((pred) => (
            <div key={pred.horizon} className="px-6 py-4 flex items-center justify-between" role="listitem">
              <div className="flex items-center gap-3">
                <div className="w-12 text-sm font-medium text-gray-400">
                  {pred.horizon === '1h' ? '1 Hour' : pred.horizon === '4h' ? '4 Hours' : '24 Hours'}
                </div>
                <span aria-label={`Trend: ${pred.direction}`}>{getDirectionIcon(pred.direction)}</span>
              </div>

              <div className="flex items-center gap-4">
                <div className="text-right">
                  <div className="font-mono font-bold text-white flex items-center justify-end">
                    <AnimatedNumber value={pred.predicted} decimals={4} />
                    <CorrectionBadge biasCorrection={pred.biasCorrection} />
                  </div>
                  <div className={`text-xs ${getDirectionColor(pred.direction)}`}>
                    {pred.changePercent > 0 ? '+' : ''}
                    <AnimatedNumber value={pred.changePercent} decimals={1} suffix="%" />
                  </div>
                </div>

                {/* Confidence bar */}
                <div className="w-16" role="meter" aria-label="Prediction confidence" aria-valuenow={Math.round(pred.confidence * 100)} aria-valuemin={0} aria-valuemax={100}>
                  <div className="h-1.5 bg-gray-700 rounded-full overflow-hidden" aria-hidden="true">
                    <div
                      className={`h-full transition-all ${
                        pred.confidence > 0.7 ? 'bg-green-500' :
                        pred.confidence > 0.4 ? 'bg-yellow-500' : 'bg-red-500'
                      }`}
                      style={{ width: `${pred.confidence * 100}%` }}
                    />
                  </div>
                  <div className="text-xs text-gray-500 text-center mt-1">
                    {Math.round(pred.confidence * 100)}%
                  </div>
                </div>
              </div>
            </div>
          ))}
      </div>

      {/* Best time indicator */}
      {predictions.length > 0 && (
        <div className="px-6 py-4 bg-gray-800/50 border-t border-gray-700/30">
          {(() => {
            const bestPred = predictions.reduce((best, current) =>
              current.predicted < best.predicted ? current : best
            );

            if (bestPred.direction === 'down') {
              return (
                <div className="flex items-center gap-2 text-sm">
                  <span className="text-green-400">Best time to transact:</span>
                  <span className="font-medium text-white">
                    Wait {bestPred.horizon === '1h' ? '~1 hour' : bestPred.horizon === '4h' ? '~4 hours' : '~24 hours'}
                  </span>
                </div>
              );
            }

            return (
              <div className="flex items-center gap-2 text-sm">
                <span className="text-cyan-400">Gas is optimal now</span>
                <span className="text-gray-400">— prices may rise</span>
              </div>
            );
          })()}
        </div>
      )}
    </div>
  );
};

export default memo(CompactForecast);
