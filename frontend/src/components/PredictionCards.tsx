import React, { useState, useEffect, useCallback, memo } from 'react';
import { fetchPredictions, fetchCurrentGas } from '../api/gasApi';
import { useChain } from '../contexts/ChainContext';
import LoadingSpinner from './LoadingSpinner';
import ConfidenceBar from './ui/ConfidenceBar';
import { HybridPrediction } from '../../types';

interface PredictionCardsProps {
  hybridData?: HybridPrediction;
}

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
  trendSignal4h?: number; // The "Macro" signal (-1.0 to 1.0)
  probabilities?: {
    wait: number;
    normal: number;
    urgent: number;
  };
}

const PredictionCards: React.FC<PredictionCardsProps> = ({ hybridData }) => {
  const { selectedChainId } = useChain();
  const [cards, setCards] = useState<PredictionCard[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [showExplanation, setShowExplanation] = useState<string | null>(null);

  // All hooks must be called before any early returns
  const getColorClasses = useCallback((color: string) => {
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
  }, []);

  const getTextColor = useCallback((color: string) => {
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
  }, []);

  // Determine classification state from probabilities
  const getClassificationState = useCallback((probs: { wait: number; normal: number; urgent: number } | undefined) => {
    if (!probs) return null;
    const max = Math.max(probs.wait, probs.normal, probs.urgent);
    if (probs.urgent === max && probs.urgent > 0.3) return 'spike';
    if (probs.wait === max && probs.wait > 0.3) return 'elevated';
    return 'normal';
  }, []);

  const loadData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      const [predictionsResult, currentGasData] = await Promise.all([
        fetchPredictions(selectedChainId),
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
          
          // Extract hybrid prediction data (trend_signal_4h and probabilities)
          const classification = firstPrediction?.classification;
          const trendSignal4h = firstPrediction?.trend_signal_4h ?? classification?.trend_signal_4h;
          const probabilities = classification?.probabilities || firstPrediction?.probabilities;
          
          // Normalize probabilities if they use different keys (normal/elevated/spike -> wait/normal/urgent)
          let normalizedProbs: { wait: number; normal: number; urgent: number } | undefined;
          if (probabilities) {
            if ('wait' in probabilities && 'normal' in probabilities && 'urgent' in probabilities) {
              normalizedProbs = probabilities as { wait: number; normal: number; urgent: number };
            } else if ('normal' in probabilities && 'elevated' in probabilities && 'spike' in probabilities) {
              // Map: normal -> normal, elevated -> wait, spike -> urgent
              const probs = probabilities as { normal?: number; elevated?: number; spike?: number };
              normalizedProbs = {
                wait: probs.elevated ?? 0,
                normal: probs.normal ?? 0,
                urgent: probs.spike ?? 0,
              };
            }
          }

          // Prevent division by zero
          const changePercent = current === 0 ? 0 : ((predicted - current) / current) * 100;
          
          // Check if this is the best prediction (considering confidence)
          const isBest = bestPrediction && bestPrediction.horizon === horizon && confidenceLevel === 'high';

          let color: 'red' | 'green' | 'yellow';
          let icon: string;
          let recommendation: string;
          const classifState = getClassificationState(normalizedProbs);

          // Update recommendation based on confidence and classification
          if (predicted < current * 0.9) {
            color = 'red';
            icon = '';
            if (classifState === 'elevated') {
              recommendation = `Gas prices are elevated (50-85th percentile). Consider delaying if possible. Expected to drop ${Math.abs(changePercent).toFixed(0)}%.`;
            } else if (confidenceLevel === 'high') {
              recommendation = `Gas expected to drop significantly. Wait ${horizon} before transacting.`;
            } else if (confidenceLevel === 'medium') {
              recommendation = `Gas likely to drop, but prediction has moderate uncertainty. Consider waiting.`;
            } else {
              recommendation = `Gas may drop, but prediction is uncertain. Risky to wait.`;
            }
          } else if (predicted > current * 1.1) {
            color = 'green';
            icon = '';
            if (classifState === 'spike') {
              recommendation = `Spike detected! Gas prices spiking above normal levels. Transact immediately if possible.`;
            } else {
              recommendation = `Gas expected to rise. Consider transacting now.`;
            }
          } else {
            color = 'yellow';
            icon = '';
            if (classifState === 'elevated') {
              recommendation = `Gas prices are elevated and stable. No urgent action, but consider waiting for lower prices.`;
            } else {
              recommendation = `Gas expected to remain stable. No urgent action needed.`;
            }
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
            isBest: isBest || false,
            trendSignal4h,
            probabilities: normalizedProbs
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
  }, [selectedChainId]);

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, [loadData, selectedChainId]);

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

          {/* Classification Badge - Shows Normal/Elevated/Spike */}
          {(() => {
            const classifState = getClassificationState(card.probabilities);
            if (!classifState) return null;

            const badges = {
              normal: { bg: 'bg-green-500/20', border: 'border-green-500/30', text: 'text-green-400', label: 'Normal' },
              elevated: { bg: 'bg-amber-500/20', border: 'border-amber-500/30', text: 'text-amber-400', label: 'Elevated' },
              spike: { bg: 'bg-red-500/20', border: 'border-red-500/30', text: 'text-red-400', label: 'Spike' }
            };

            const badge = badges[classifState as keyof typeof badges];
            return (
              <div className={`absolute -top-3 right-4 ${badge.bg} border ${badge.border} ${badge.text} px-3 py-1 rounded-full text-xs font-bold shadow-lg`}>
                {badge.label}
              </div>
            );
          })()}
          
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <h3 className="text-lg font-semibold text-gray-200">
                {card.horizon.toUpperCase()} PREDICTION
              </h3>
              {card.horizon === '24h' && (
                <span className="px-2 py-0.5 rounded-full text-xs font-semibold bg-cyan-500/10 text-cyan-300 border border-cyan-500/30">
                  Most Reliable
                </span>
              )}
              {card.horizon === '4h' && (() => {
                const trendSignal = card.trendSignal4h ?? hybridData?.trend_signal_4h;
                if (trendSignal === undefined || trendSignal === null) {
                  return null;
                }
                if (trendSignal > 0.2) {
                  return (
                    <span className="px-2 py-0.5 rounded-full text-xs font-semibold bg-red-500/10 text-red-300 border border-red-500/30">
                      ↗ Rising
                    </span>
                  );
                }
                if (trendSignal < -0.2) {
                  return (
                    <span className="px-2 py-0.5 rounded-full text-xs font-semibold bg-green-500/10 text-green-300 border border-green-500/30">
                      ↘ Falling
                    </span>
                  );
                }
                return null;
              })()}
            </div>
          </div>

          <div className="mb-4">
            <div className={`text-xl font-bold mb-2 ${getTextColor(card.color)}`}>
              {/* Binary action recommendation based on classifier probabilities */}
              {card.probabilities && (card.probabilities.urgent + card.probabilities.wait) > 0.5
                ? `WAIT - Consider Delaying`
                : card.changePercent < -10
                ? `WAIT - Gas Dropping ${Math.abs(card.changePercent).toFixed(0)}%`
                : card.changePercent > 10
                ? `TRANSACT NOW - Gas Rising ${card.changePercent.toFixed(0)}%`
                : 'TRANSACT - Good Time'}
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

          {/* Probability Breakdown - Show for all cards */}
          {(hybridData?.probabilities || card.probabilities) && (
            <div className="mb-4">
              <div className="text-xs text-gray-400 mb-2 font-medium">Classification Probabilities</div>
              {hybridData?.probabilities ? (
                <ConfidenceBar probs={hybridData.probabilities} showLabels="classification" />
              ) : card.probabilities ? (
                <ConfidenceBar probs={card.probabilities} showLabels="classification" />
              ) : (
                <div className="h-4 bg-gray-800/50 rounded-full animate-pulse mt-4" />
              )}
            </div>
          )}

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
                  {card.confidenceLevel === 'high' && "Model is confident in this prediction"}
                  {card.confidenceLevel === 'medium' && "Moderate prediction confidence"}
                  {card.confidenceLevel === 'low' && "High uncertainty - prediction less reliable"}
                </p>
                {(card.horizon === '1h' || card.horizon === '4h') && (
                  <p className="text-xs text-gray-500 italic mt-2">
                    Short-term predictions have higher uncertainty
                  </p>
                )}
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
              <div className="space-y-4">
                <div className="bg-gray-800 rounded-lg p-4">
                  <h4 className="text-cyan-400 font-semibold mb-2">How We Predict Gas Prices</h4>
                  <p className="text-gray-300 text-sm">
                    Our ML models analyze historical gas patterns, network congestion, and time-based trends
                    to forecast future gas prices for the {showExplanation} horizon.
                  </p>
                </div>
                <div className="bg-gray-800 rounded-lg p-4">
                  <h4 className="text-cyan-400 font-semibold mb-2">Confidence Levels</h4>
                  <ul className="text-gray-300 text-sm space-y-2">
                    <li><span className="text-green-400">High confidence:</span> Model is confident in this prediction</li>
                    <li><span className="text-yellow-400">Medium confidence:</span> Moderate prediction certainty</li>
                    <li><span className="text-red-400">Low confidence:</span> High uncertainty - prediction less reliable</li>
                  </ul>
                </div>
                <div className="bg-gray-800 rounded-lg p-4">
                  <h4 className="text-cyan-400 font-semibold mb-2">Classification States</h4>
                  <ul className="text-gray-300 text-sm space-y-2">
                    <li><span className="text-green-400">Normal:</span> Gas prices in typical range (0-50th percentile)</li>
                    <li><span className="text-amber-400">Elevated:</span> Gas prices elevated (50-85th percentile) - consider waiting if possible</li>
                    <li><span className="text-red-400">Spike:</span> Gas prices spiking (85th+ percentile) - transact immediately if needed</li>
                  </ul>
                </div>
                <div className="bg-gray-800 rounded-lg p-4">
                  <h4 className="text-cyan-400 font-semibold mb-2">Prediction Range</h4>
                  <p className="text-gray-300 text-sm">
                    The range shows best-case and worst-case scenarios based on historical volatility patterns.
                    Wider ranges indicate higher uncertainty.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </>
  );
};

// Memoize the component to prevent unnecessary re-renders
export default memo(PredictionCards);
