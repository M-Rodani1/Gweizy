import React, { useEffect, useState } from 'react';
import { useChain } from '../contexts/ChainContext';
import { fetchPredictions } from '../api/gasApi';

interface Prediction {
  horizon: '1h' | '4h' | '24h';
  predicted: number;
  confidence: number;
  direction: 'up' | 'down' | 'stable';
  changePercent: number;
}

const CompactForecast: React.FC = () => {
  const { selectedChain, multiChainGas } = useChain();
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(true);

  const currentGas = multiChainGas[selectedChain.id]?.gasPrice || 0;

  useEffect(() => {
    const loadPredictions = async () => {
      try {
        setLoading(true);
        const result = await fetchPredictions();

        if (result?.predictions) {
          const preds: Prediction[] = [];

          (['1h', '4h', '24h'] as const).forEach((horizon) => {
            const data = result.predictions[horizon];
            if (Array.isArray(data) && data.length > 0) {
              const predicted = data[0].predictedGwei || 0;
              const confidence = data[0].confidence || 0.5;
              const changePercent = currentGas > 0 ? ((predicted - currentGas) / currentGas) * 100 : 0;

              preds.push({
                horizon,
                predicted,
                confidence,
                direction: changePercent > 5 ? 'up' : changePercent < -5 ? 'down' : 'stable',
                changePercent
              });
            }
          });

          setPredictions(preds);
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
  }, [currentGas]);

  const getDirectionIcon = (direction: string) => {
    switch (direction) {
      case 'up': return '📈';
      case 'down': return '📉';
      default: return '➡️';
    }
  };

  const getDirectionColor = (direction: string) => {
    switch (direction) {
      case 'up': return 'text-red-400';
      case 'down': return 'text-green-400';
      default: return 'text-yellow-400';
    }
  };

  return (
    <div className="bg-gray-800/50 border border-gray-700 rounded-xl overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 border-b border-gray-700/50 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-lg">🔮</span>
          <h3 className="font-semibold text-white">Price Forecast</h3>
        </div>
        <div className="text-xs text-gray-500">{selectedChain.name}</div>
      </div>

      {/* Current Price */}
      <div className="px-4 py-3 border-b border-gray-700/30 bg-gray-800/30">
        <div className="flex items-center justify-between">
          <span className="text-sm text-gray-400">Current</span>
          <span className="font-mono font-bold text-cyan-400 text-lg">
            {currentGas.toFixed(4)} gwei
          </span>
        </div>
      </div>

      {/* Predictions */}
      {loading ? (
        <div className="p-4 flex items-center justify-center">
          <div className="w-5 h-5 border-2 border-gray-600 border-t-cyan-400 rounded-full animate-spin" />
        </div>
      ) : (
        <div className="divide-y divide-gray-700/30">
          {predictions.map((pred) => (
            <div key={pred.horizon} className="px-4 py-3 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-12 text-sm font-medium text-gray-400">
                  {pred.horizon === '1h' ? '1 Hour' : pred.horizon === '4h' ? '4 Hours' : '24 Hours'}
                </div>
                <span className="text-lg">{getDirectionIcon(pred.direction)}</span>
              </div>

              <div className="flex items-center gap-4">
                <div className="text-right">
                  <div className="font-mono font-bold text-white">
                    {pred.predicted.toFixed(4)}
                  </div>
                  <div className={`text-xs ${getDirectionColor(pred.direction)}`}>
                    {pred.changePercent > 0 ? '+' : ''}{pred.changePercent.toFixed(1)}%
                  </div>
                </div>

                {/* Confidence bar */}
                <div className="w-16">
                  <div className="h-1.5 bg-gray-700 rounded-full overflow-hidden">
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
      )}

      {/* Best time indicator */}
      {predictions.length > 0 && (
        <div className="px-4 py-3 bg-gray-800/50 border-t border-gray-700/30">
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

export default CompactForecast;
