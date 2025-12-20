import React, { useState, useEffect } from 'react';
import { fetchPredictions, fetchCurrentGas } from '../src/api/gasApi';
import LoadingSpinner from './LoadingSpinner';
import PredictionExplanation from './PredictionExplanation';

interface PredictionCard {
  horizon: '1h' | '4h' | '24h';
  current: number;
  predicted: number;
  lowerBound?: number;
  upperBound?: number;
  confidence?: number;
  confidenceLevel?: 'high' | 'medium' | 'low';
  confidenceEmoji?: string;
  confidenceColor?: string;
  changePercent: number;
  recommendation: string;
  color: 'red' | 'green' | 'yellow';
  icon: string;
  isBest?: boolean;
}

const PredictionCards: React.FC = () => {
  const [cards, setCards] = useState<PredictionCard[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showExplanation, setShowExplanation] = useState<string | null>(null);

  const loadData = async () => {
    try {
      setLoading(true);
      setError(null);

      const [predictionsResult, currentGasData] = await Promise.all([
        fetchPredictions(),
        fetchCurrentGas()
      ]);

      const current = currentGasData?.current_gas || 0;
      const newCards: PredictionCard[] = [];

      // First pass: collect all predictions to find the best one
      const horizonPredictions: { horizon: string; predicted: number }[] = [];
      (['1h', '4h', '24h'] as const).forEach((horizon) => {
        const predictions = predictionsResult?.predictions?.[horizon];
        if (Array.isArray(predictions) && predictions.length > 0) {
          const firstPrediction = predictions[0];
          const predicted = firstPrediction.predictedGwei || 0;
          if (predicted > 0) {
            horizonPredictions.push({ horizon, predicted });
          }
        }
      });

      // Find the best (lowest) predicted gas
      const bestPrediction = horizonPredictions.length > 0
        ? horizonPredictions.reduce((best, current) => 
            current.predicted < best.predicted ? current : best
          )
        : null;

      // Process each prediction horizon
      (['1h', '4h', '24h'] as const).forEach((horizon) => {
        const predictions = predictionsResult?.predictions?.[horizon];
        if (Array.isArray(predictions) && predictions.length > 0) {
          const firstPrediction = predictions[0];
          const predicted = firstPrediction?.predictedGwei || 0;
          const lowerBound = firstPrediction?.lowerBound;
          const upperBound = firstPrediction?.upperBound;
          const confidence = firstPrediction?.confidence;
          const confidenceLevel = firstPrediction?.confidenceLevel || 'medium';
          const confidenceEmoji = firstPrediction?.confidenceEmoji || '';
          const confidenceColor = firstPrediction?.confidenceColor || 'yellow';

          // Prevent division by zero
          const changePercent = current === 0 ? 0 : ((predicted - current) / current) * 100;
          
          // Check if this is the best prediction (considering confidence)
          const isBest = bestPrediction && bestPrediction.horizon === horizon && confidenceLevel === 'high';

          let color: 'red' | 'green' | 'yellow';
          let icon: string;
          let recommendation: string;

          // Update recommendation based on confidence
          if (predicted < current * 0.9) {
            color = 'red';
            icon = '';
            if (confidenceLevel === 'high') {
              recommendation = `Gas expected to drop significantly. Wait ${horizon} before transacting.`;
            } else if (confidenceLevel === 'medium') {
              recommendation = `Gas likely to drop, but prediction has moderate uncertainty. Consider waiting.`;
            } else {
              recommendation = `Gas may drop, but prediction is uncertain. Risky to wait.`;
            }
          } else if (predicted > current * 1.1) {
            color = 'green';
            icon = '';
            recommendation = `Gas expected to rise. Consider transacting now.`;
          } else {
            color = 'yellow';
            icon = '';
            recommendation = `Gas expected to remain stable. No urgent action needed.`;
          }

          newCards.push({
            horizon,
            current,
            predicted,
            lowerBound,
            upperBound,
            confidence,
            confidenceLevel,
            confidenceEmoji,
            confidenceColor,
            changePercent,
            recommendation,
            color,
            icon,
            isBest: isBest || false
          });
        }
      });

      setCards(newCards);
    } catch (err) {
      console.error('Error loading prediction cards:', err);
      setError(err instanceof Error ? err.message : 'Failed to load predictions');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, []);

  if (loading && cards.length === 0) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <div className="bg-gray-800 p-4 md:p-6 rounded-lg shadow-lg">
          <LoadingSpinner message="Loading predictions..." />
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <div className="bg-gray-800 p-4 md:p-6 rounded-lg shadow-lg">
          <p className="text-red-400 mb-4">⚠️ {error}</p>
          <button
            onClick={loadData}
            className="px-6 py-3 bg-cyan-500 hover:bg-cyan-600 rounded-md transition-colors text-sm min-h-[44px]"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  const getColorClasses = (color: string) => {
    switch (color) {
      case 'red':
        return 'border-red-500 bg-red-500/10';
      case 'green':
        return 'border-green-500 bg-green-500/10';
      case 'yellow':
        return 'border-yellow-500 bg-yellow-500/10';
      default:
        return 'border-gray-600 bg-gray-800';
    }
  };

  const getTextColor = (color: string) => {
    switch (color) {
      case 'red':
        return 'text-red-400';
      case 'green':
        return 'text-green-400';
      case 'yellow':
        return 'text-yellow-400';
      default:
        return 'text-gray-400';
    }
  };

  return (
    <>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        {cards && cards.length > 0 ? cards.map((card) => (
          <div
            key={card.horizon}
            className={`bg-gray-800 p-4 md:p-6 rounded-lg shadow-lg border-2 ${getColorClasses(card.color)} relative`}
          >
          {/* BEST TIME Badge */}
          {card.isBest && (
            <div className="absolute -top-3 left-4 bg-gradient-to-r from-yellow-400 to-yellow-500 text-gray-900 px-3 py-1 rounded-full text-xs font-bold shadow-lg">
              BEST TIME
            </div>
          )}
          
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-200">
              {card.horizon.toUpperCase()} PREDICTION
            </h3>
          </div>

          <div className="mb-4">
            <div className={`text-xl font-bold mb-2 ${getTextColor(card.color)}`}>
              {card.changePercent < -10
                ? `WAIT - Gas Dropping ${Math.abs(card.changePercent).toFixed(0)}%`
                : card.changePercent > 10
                ? `TRANSACT NOW - Gas Rising ${card.changePercent.toFixed(0)}%`
                : 'NEUTRAL - Gas Stable'}
            </div>
          </div>

          {/* Prediction Range */}
          {card.lowerBound !== undefined && card.upperBound !== undefined && (
            <div className="bg-gray-900/50 rounded-lg p-4 mb-4">
              <div className="text-xs text-gray-400 mb-3">Predicted Range:</div>
              
              {/* Visual range slider */}
              <div className="relative h-8 mb-2">
                <div className="absolute inset-0 flex items-center">
                  <div className="w-full h-1 bg-gray-700 rounded"></div>
                </div>
                
                {/* Range bar */}
                <div 
                  className="absolute inset-0 flex items-center"
                  style={{
                    left: `${Math.max(0, (card.lowerBound / (card.upperBound * 1.2)) * 100)}%`,
                    right: `${Math.max(0, 100 - (card.upperBound / (card.upperBound * 1.2)) * 100)}%`
                  }}
                >
                  <div className="w-full h-2 bg-gradient-to-r from-green-500 via-yellow-500 to-red-500 rounded"></div>
                </div>
                
                {/* Prediction marker */}
                <div 
                  className="absolute top-0 flex items-center justify-center"
                  style={{
                    left: `${Math.min(100, Math.max(0, (card.predicted / (card.upperBound * 1.2)) * 100))}%`,
                    transform: 'translateX(-50%)'
                  }}
                >
                  <div className="w-3 h-3 bg-white rounded-full border-2 border-cyan-400 shadow-lg"></div>
                </div>
              </div>

              {/* Range labels */}
              <div className="flex justify-between text-xs">
                <div className="text-green-400">
                  <div className="font-mono">{card.lowerBound !== undefined && card.lowerBound !== null ? card.lowerBound.toFixed(4) : 'N/A'}</div>
                  <div className="text-gray-500">Best Case</div>
                </div>
                <div className="text-cyan-400 text-center">
                  <div className="font-mono font-bold">{card.predicted !== undefined && card.predicted !== null ? card.predicted.toFixed(4) : 'N/A'}</div>
                  <div className="text-gray-500">Most Likely</div>
                </div>
                <div className="text-red-400 text-right">
                  <div className="font-mono">{card.upperBound !== undefined && card.upperBound !== null ? card.upperBound.toFixed(4) : 'N/A'}</div>
                  <div className="text-gray-500">Worst Case</div>
                </div>
              </div>
            </div>
          )}

          <div className="space-y-2 mb-4">
            <div className="flex justify-between text-sm">
              <span className="text-gray-400">Current:</span>
              <span className="text-gray-200 font-medium">{card.current !== undefined && card.current !== null ? card.current.toFixed(4) : 'N/A'} gwei</span>
            </div>
            <div className="flex justify-between text-sm">
              <span className="text-gray-400">Predicted:</span>
              <span className="text-gray-200 font-medium">{card.predicted !== undefined && card.predicted !== null ? card.predicted.toFixed(4) : 'N/A'} gwei</span>
            </div>
            <div className="flex justify-between text-sm pt-2 border-t border-gray-700">
              <span className="text-gray-400">Potential Savings:</span>
              <span className={`font-bold ${getTextColor(card.color)}`}>
                {card.changePercent !== undefined && card.changePercent !== null ? Math.abs(card.changePercent).toFixed(0) : '0'}%
              </span>
            </div>
          </div>

          {/* Confidence indicator */}
          {card.confidence !== undefined && (() => {
            const confColors = {
              green: {bg: 'bg-green-500/10', border: 'border-green-500/30', text: 'text-green-400', textLight: 'text-green-300'},
              cyan: {bg: 'bg-cyan-500/10', border: 'border-cyan-500/30', text: 'text-cyan-400', textLight: 'text-cyan-300'},
              yellow: {bg: 'bg-yellow-500/10', border: 'border-yellow-500/30', text: 'text-yellow-400', textLight: 'text-yellow-300'},
              orange: {bg: 'bg-orange-500/10', border: 'border-orange-500/30', text: 'text-orange-400', textLight: 'text-orange-300'},
              red: {bg: 'bg-red-500/10', border: 'border-red-500/30', text: 'text-red-400', textLight: 'text-red-300'}
            };
            const confClass = confColors[card.confidenceColor as keyof typeof confColors] || confColors.yellow;
            return (
              <div className={`${confClass.bg} border ${confClass.border} rounded-md p-3 mb-4`}>
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className={`${confClass.text} font-medium text-sm`}>
                      {card.confidenceLevel?.toUpperCase()} CONFIDENCE
                    </span>
                  </div>
                  <span className={`${confClass.text} font-bold`}>
                    {card.confidence !== undefined && card.confidence !== null ? (card.confidence * 100).toFixed(0) : '0'}%
                  </span>
                </div>
                <p className={`text-xs ${confClass.textLight} mt-2`}>
                  {card.confidenceLevel === 'high' && "Models strongly agree on this prediction"}
                  {card.confidenceLevel === 'medium' && "Moderate agreement between models"}
                  {card.confidenceLevel === 'low' && "High uncertainty - models disagree"}
                </p>
              </div>
            );
          })()}

          <div className="mt-4 pt-4 border-t border-gray-700">
            <p className="text-sm text-gray-300 leading-relaxed mb-3">{card.recommendation}</p>
            <button
              onClick={() => setShowExplanation(card.horizon)}
              className="w-full py-3 bg-gray-700 hover:bg-gray-600 text-cyan-400 rounded-md text-sm transition min-h-[44px]"
            >
              Why this prediction?
            </button>
          </div>
        </div>
        )) : (
          <div className="col-span-3 text-gray-400 text-center p-8">Loading predictions...</div>
        )}
      </div>

      {/* Explanation Modal */}
      {showExplanation && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50 p-4">
          <div className="bg-gray-900 rounded-xl max-w-2xl w-full max-h-[90vh] overflow-y-auto">
            <div className="sticky top-0 bg-gray-900 border-b border-gray-700 p-4 flex justify-between items-center">
              <h2 className="text-xl font-bold text-white">{showExplanation.toUpperCase()} Prediction Explained</h2>
              <button
                onClick={() => setShowExplanation(null)}
                className="text-gray-400 hover:text-white text-2xl"
              >
                ×
              </button>
            </div>
            <div className="p-6">
              <PredictionExplanation horizon={showExplanation as '1h' | '4h' | '24h'} />
            </div>
          </div>
        </div>
      )}
    </>
  );
};

export default PredictionCards;

