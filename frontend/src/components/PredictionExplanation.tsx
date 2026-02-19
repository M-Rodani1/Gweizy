import React, { useState, useEffect } from 'react';
import { Lightbulb, ChevronDown, ChevronUp, TrendingUp, TrendingDown, RefreshCw, Sparkles } from 'lucide-react';
import { API_CONFIG, getApiUrl } from '../config/api';
import { withTimeout } from '../utils/withTimeout';

interface Factor {
  name: string;
  description: string;
  weight: number;
  value: number;
}

interface ExplanationData {
  llm_explanation: string;
  technical_explanation: string;
  technical_details: {
    feature_importance: Record<string, { value: number; importance: number; impact: string }>;
    increasing_factors: Factor[];
    decreasing_factors: Factor[];
    similar_cases: Array<{ timestamp: string; gas_price: number; similarity: number }>;
    classification?: string;
    classification_confidence?: number;
  };
  prediction: number;
  current_gas: number;
}

interface PredictionExplanationProps {
  prediction?: number;
  currentGas?: number;
  horizon?: '1h' | '4h' | '24h';
  compact?: boolean;
}

const PredictionExplanation: React.FC<PredictionExplanationProps> = ({
  prediction,
  currentGas,
  horizon = '1h',
  compact = false
}) => {
  const [explanation, setExplanation] = useState<ExplanationData | null>(null);
  const [loading, setLoading] = useState(false);
  const [showDetails, setShowDetails] = useState(false);

  const fetchExplanation = async () => {
    if (!prediction || !currentGas) return;

    try {
      setLoading(true);

      const response = await withTimeout(
        fetch(getApiUrl(`${API_CONFIG.ENDPOINTS.EXPLAIN}/${horizon}`)),
        API_CONFIG.TIMEOUT,
        `Request timed out: explanation ${horizon}`
      );

      if (!response.ok) {
        // Generate fallback explanation
        setExplanation(generateFallbackExplanation(prediction, currentGas));
        return;
      }

      const data = await response.json();
      if (data.success) {
        setExplanation(data);
      } else {
        setExplanation(generateFallbackExplanation(prediction, currentGas));
      }
    } catch {
      setExplanation(generateFallbackExplanation(prediction, currentGas));
    } finally {
      setLoading(false);
    }
  };

  const generateFallbackExplanation = (pred: number, current: number): ExplanationData => {
    const changePercent = ((pred - current) / current) * 100;
    const direction = pred < current ? 'drop' : pred > current ? 'rise' : 'stay stable';
    const now = new Date();
    const hour = now.getHours();
    const isWeekend = now.getDay() === 0 || now.getDay() === 6;

    let explanation = '';
    let classification = 'Normal';
    const increasing: Factor[] = [];
    const decreasing: Factor[] = [];

    // Determine classification based on prediction
    if (pred > current * 1.15) {
      classification = 'Spike';
    } else if (pred > current * 1.05) {
      classification = 'Elevated';
    } else {
      classification = 'Normal';
    }

    if (direction === 'drop') {
      if (isWeekend) {
        explanation = `Gas is expected to drop about ${Math.abs(changePercent).toFixed(0)}% because it's the weekend when Base network activity is typically lower.`;
        decreasing.push({
          name: 'is_weekend',
          description: 'Weekend: Lower network activity',
          weight: 45,
          value: 1
        });
      } else if (hour >= 0 && hour <= 6) {
        explanation = `Gas is expected to drop about ${Math.abs(changePercent).toFixed(0)}% because it's late night when fewer transactions are happening.`;
        decreasing.push({
          name: 'hour',
          description: 'Late night: Minimal activity',
          weight: 50,
          value: hour
        });
      } else {
        explanation = `Gas is expected to drop about ${Math.abs(changePercent).toFixed(0)}% based on recent network trends and historical patterns.`;
        decreasing.push({
          name: 'trend_1h',
          description: 'Downward trend detected',
          weight: 35,
          value: -0.1
        });
      }
    } else if (direction === 'rise') {
      if (hour >= 10 && hour <= 16) {
        explanation = `Gas is expected to rise about ${Math.abs(changePercent).toFixed(0)}% because it's peak hours when network activity is highest.`;
        increasing.push({
          name: 'hour',
          description: 'Peak hours: High activity',
          weight: 55,
          value: hour
        });
      } else {
        explanation = `Gas is expected to rise about ${Math.abs(changePercent).toFixed(0)}% based on increasing network demand.`;
        increasing.push({
          name: 'trend_1h',
          description: 'Upward trend detected',
          weight: 40,
          value: 0.15
        });
      }
    } else {
      explanation = `Gas is expected to stay relatively stable around ${pred.toFixed(4)} gwei with minimal change expected.`;
    }

    // Add classification context
    if (classification === 'Spike') {
      explanation += ` Market conditions suggest ${classification.toLowerCase()} activity with prices expected in high percentile ranges.`;
    } else if (classification === 'Elevated') {
      explanation += ` Gas prices are predicted to be elevated (50-85th percentile). Consider waiting if your transaction is not urgent.`;
    }

    return {
      llm_explanation: explanation,
      technical_explanation: explanation,
      technical_details: {
        feature_importance: {},
        increasing_factors: increasing,
        decreasing_factors: decreasing,
        similar_cases: [],
        classification,
        classification_confidence: 0.7
      },
      prediction: pred,
      current_gas: current
    };
  };

  useEffect(() => {
    if (prediction && currentGas) {
      fetchExplanation();
    }
  }, [prediction, currentGas, horizon]);

  const changePercent = prediction && currentGas
    ? ((prediction - currentGas) / currentGas) * 100
    : 0;
  const isDropping = changePercent < -1;
  const isRising = changePercent > 1;

  if (!prediction || !currentGas) {
    return null;
  }

  if (compact) {
    return (
      <div className="flex items-start gap-2 p-3 bg-gray-800/40 rounded-xl">
        <Lightbulb className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
        <div className="text-sm text-gray-300">
          {loading ? (
            <span className="text-gray-500">Analysing prediction...</span>
          ) : explanation?.llm_explanation || (
            <span className="text-gray-500">Explanation unavailable</span>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gray-900/50 border border-gray-800 rounded-2xl p-4 shadow-xl">
      {/* Header */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Sparkles className="w-4 h-4 text-amber-400" />
          <h3 className="font-semibold text-white text-sm">Why This Prediction?</h3>
        </div>
        <button
          onClick={fetchExplanation}
          className="p-1 text-gray-500 hover:text-gray-300 transition-colors"
          disabled={loading}
        >
          <RefreshCw className={`w-3 h-3 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {/* Main explanation */}
      <div className="p-3 bg-gray-800/40 rounded-xl mb-3">
        {loading ? (
          <div className="flex items-center gap-2 text-gray-400">
            <RefreshCw className="w-4 h-4 animate-spin" />
            <span className="text-sm">Analysing prediction factors...</span>
          </div>
        ) : (
          <p className="text-sm text-gray-200 leading-relaxed">
            {explanation?.llm_explanation || 'Unable to generate explanation.'}
          </p>
        )}
      </div>

      {/* Change summary */}
      <div className="flex items-center gap-3 mb-3">
        <div className={`
          flex items-center gap-1.5 px-2 py-1 rounded-lg text-sm font-medium
          ${isDropping ? 'bg-green-500/20 text-green-400' : isRising ? 'bg-red-500/20 text-red-400' : 'bg-gray-700/50 text-gray-400'}
        `}>
          {isDropping ? (
            <TrendingDown className="w-4 h-4" />
          ) : isRising ? (
            <TrendingUp className="w-4 h-4" />
          ) : null}
          <span>{isDropping ? '-' : isRising ? '+' : ''}{Math.abs(changePercent).toFixed(1)}%</span>
        </div>
        <div className="text-xs text-gray-500">
          {currentGas.toFixed(4)} → {prediction.toFixed(4)} gwei
        </div>
      </div>

      {/* Expand/Collapse details */}
      {explanation?.technical_details && (
        <>
          <button
            onClick={() => setShowDetails(!showDetails)}
            className="flex items-center gap-1 text-xs text-gray-500 hover:text-gray-300 transition-colors"
          >
            {showDetails ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
            {showDetails ? 'Hide' : 'Show'} technical details
          </button>

          {showDetails && (
            <div className="mt-3 space-y-3 animate-in fade-in slide-in-from-top-2 duration-200">
              {/* Classification section */}
              {explanation.technical_details.classification && (
                <div>
                  <div className={`text-xs mb-2 flex items-center gap-1 font-semibold ${
                    explanation.technical_details.classification === 'Spike' ? 'text-red-400' :
                    explanation.technical_details.classification === 'Elevated' ? 'text-amber-400' :
                    'text-green-400'
                  }`}>
                    {explanation.technical_details.classification === 'Spike' && '🔴'}
                    {explanation.technical_details.classification === 'Elevated' && '🟡'}
                    {explanation.technical_details.classification === 'Normal' && '🟢'}
                    Classification: {explanation.technical_details.classification}
                  </div>
                  <p className="text-xs text-gray-400">
                    {explanation.technical_details.classification === 'Spike'
                      ? 'Prices in very high percentile (85%+). Market volatility is significant.'
                      : explanation.technical_details.classification === 'Elevated'
                      ? 'Prices in elevated range (50-85 percentile). Consider waiting if possible.'
                      : 'Prices in normal range (0-50 percentile). Good time to transact.'}
                  </p>
                </div>
              )}

              {/* Factors pushing down */}
              {explanation.technical_details.decreasing_factors.length > 0 && (
                <div>
                  <div className="text-xs text-green-400 mb-2 flex items-center gap-1">
                    <TrendingDown className="w-3 h-3" />
                    Factors pushing gas DOWN
                  </div>
                  <div className="space-y-1">
                    {explanation.technical_details.decreasing_factors.slice(0, 3).map((factor, i) => (
                      <div key={i} className="flex items-center justify-between text-xs">
                        <span className="text-gray-400">{factor.description}</span>
                        <span className="text-green-400 font-mono">{factor.weight}%</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Factors pushing up */}
              {explanation.technical_details.increasing_factors.length > 0 && (
                <div>
                  <div className="text-xs text-red-400 mb-2 flex items-center gap-1">
                    <TrendingUp className="w-3 h-3" />
                    Factors pushing gas UP
                  </div>
                  <div className="space-y-1">
                    {explanation.technical_details.increasing_factors.slice(0, 3).map((factor, i) => (
                      <div key={i} className="flex items-center justify-between text-xs">
                        <span className="text-gray-400">{factor.description}</span>
                        <span className="text-red-400 font-mono">{factor.weight}%</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Similar historical cases */}
              {explanation.technical_details.similar_cases.length > 0 && (
                <div>
                  <div className="text-xs text-gray-500 mb-2">Similar historical situations</div>
                  <div className="text-xs text-gray-400 font-mono">
                    Avg: {(explanation.technical_details.similar_cases.reduce((a, b) => a + b.gas_price, 0) / explanation.technical_details.similar_cases.length).toFixed(4)} gwei
                    <span className="text-gray-500 ml-2">
                      ({explanation.technical_details.similar_cases.length} cases)
                    </span>
                  </div>
                </div>
              )}
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default PredictionExplanation;
