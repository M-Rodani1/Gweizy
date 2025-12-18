import React, { useState, useEffect } from 'react';

interface TechnicalDetails {
  feature_importance: any;
  increasing_factors: Array<{
    name: string;
    description: string;
    weight: number;
    value: number;
  }>;
  decreasing_factors: Array<{
    name: string;
    description: string;
    weight: number;
    value: number;
  }>;
  similar_cases: Array<{
    timestamp: string;
    gas_price: number;
    similarity: number;
  }>;
}

interface ExplanationData {
  llm_explanation: string;
  technical_explanation?: string;  // Legacy fallback
  technical_details: TechnicalDetails;
  prediction: number;
  current_gas: number;
  horizon: string;
}

interface Props {
  horizon: '1h' | '4h' | '24h';
}

const PredictionExplanation: React.FC<Props> = ({ horizon }) => {
  const [data, setData] = useState<ExplanationData | null>(null);
  const [loading, setLoading] = useState(true);
  const [showTechnicalDetails, setShowTechnicalDetails] = useState(false);

  useEffect(() => {
    loadExplanation();
  }, [horizon]);

  const loadExplanation = async () => {
    try {
      setLoading(true);
      const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5001/api';
      const response = await fetch(`${API_BASE_URL}/explain/${horizon}`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' },
        mode: 'cors',
      });
      if (!response.ok) {
        const errorText = await response.text();
        console.error(`Explanation API error: ${response.status} - ${errorText}`);
        throw new Error(`Failed to fetch explanation: ${response.statusText}`);
      }
      const result = await response.json();
      
      // Handle both new format (with llm_explanation) and legacy format
      if (result.llm_explanation) {
        setData(result);
      } else if (result.explanation) {
        // Legacy format - convert to new format
        setData({
          llm_explanation: result.explanation,
          technical_explanation: result.explanation,
          technical_details: {
            feature_importance: result.feature_importance || {},
            increasing_factors: result.increasing_factors || [],
            decreasing_factors: result.decreasing_factors || [],
            similar_cases: result.similar_cases || []
          },
          prediction: result.prediction || 0,
          current_gas: result.current_gas || 0,
          horizon: result.horizon || horizon
        });
      } else {
        throw new Error('Invalid response format');
      }
    } catch (error) {
      console.error('Error loading explanation:', error);
      // Set error state so user sees something
      setData({
        llm_explanation: `Unable to load explanation. Error: ${error instanceof Error ? error.message : 'Unknown error'}. The backend may need to be restarted or the explain endpoint may not be fully configured.`,
        technical_explanation: '',
        technical_details: {
          feature_importance: {},
          increasing_factors: [],
          decreasing_factors: [],
          similar_cases: []
        },
        prediction: 0,
        current_gas: 0,
        horizon: horizon
      });
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="bg-gray-800 p-6 rounded-lg">
        <div className="flex items-center justify-center gap-3">
          <div className="w-5 h-5 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin"></div>
          <span className="text-gray-400">Claude is analyzing the prediction...</span>
        </div>
      </div>
    );
  }

  if (!data) return null;

  const { llm_explanation, technical_details } = data;

  const totalWeight = 
    technical_details.increasing_factors.reduce((sum, f) => sum + f.weight, 0) +
    technical_details.decreasing_factors.reduce((sum, f) => sum + f.weight, 0);

  const increasingPercent = totalWeight > 0 
    ? (technical_details.increasing_factors.reduce((sum, f) => sum + f.weight, 0) / totalWeight) * 100
    : 0;
  
  const decreasingPercent = 100 - increasingPercent;

  return (
    <div className="bg-gray-800 p-6 rounded-lg shadow-lg">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-2">
          <h3 className="text-xl font-bold text-white">
            Why We Predict {data.prediction ? data.prediction.toFixed(4) : '0.0000'} Gwei
          </h3>
        </div>
        <button
          onClick={() => setShowTechnicalDetails(!showTechnicalDetails)}
          className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-cyan-400 rounded-md text-sm transition"
        >
          {showTechnicalDetails ? 'Hide Details' : 'Show Details'}
        </button>
      </div>

      {/* LLM Explanation (Always visible) */}
      <div className="bg-gradient-to-r from-cyan-500/10 to-emerald-500/10 border border-cyan-500/30 rounded-lg p-5 mb-6">
        <p className="text-cyan-50 text-lg leading-relaxed">
          {llm_explanation}
        </p>
      </div>

      {/* Factors Overview */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-3">
          <span className="text-sm text-gray-400">Factors influencing this prediction:</span>
        </div>

        {/* Visual bar */}
        <div className="flex h-8 rounded-lg overflow-hidden mb-2">
          <div 
            className="bg-gradient-to-r from-red-500 to-red-600 flex items-center justify-center text-white text-xs font-bold"
            style={{ width: `${increasingPercent}%` }}
          >
            {increasingPercent > 15 && `↑ ${increasingPercent.toFixed(0)}%`}
          </div>
          <div 
            className="bg-gradient-to-r from-green-500 to-green-600 flex items-center justify-center text-white text-xs font-bold"
            style={{ width: `${decreasingPercent}%` }}
          >
            {decreasingPercent > 15 && `↓ ${decreasingPercent.toFixed(0)}%`}
          </div>
        </div>

        <div className="flex justify-between text-xs text-gray-400">
          <span>Increasing Gas ({increasingPercent.toFixed(0)}%)</span>
          <span>Decreasing Gas ({decreasingPercent.toFixed(0)}%)</span>
        </div>
      </div>

      {/* Technical Details (Collapsible) */}
      {showTechnicalDetails && (
        <div className="space-y-6 animate-fadeIn">
          {/* Visual Factor Breakdown */}
          <div>
            <div className="flex items-center justify-between mb-3">
              <span className="text-sm text-gray-400">Factors influencing this prediction:</span>
            </div>

            {/* Visual bar */}
            <div className="flex h-8 rounded-lg overflow-hidden mb-2 shadow-lg">
              <div 
                className="bg-gradient-to-r from-red-500 to-red-600 flex items-center justify-center text-white text-xs font-bold transition-all"
                style={{ width: `${increasingPercent}%` }}
              >
                {increasingPercent > 15 && `↑ ${increasingPercent.toFixed(0)}%`}
              </div>
              <div 
                className="bg-gradient-to-r from-green-500 to-green-600 flex items-center justify-center text-white text-xs font-bold transition-all"
                style={{ width: `${decreasingPercent}%` }}
              >
                {decreasingPercent > 15 && `↓ ${decreasingPercent.toFixed(0)}%`}
              </div>
            </div>

            <div className="flex justify-between text-xs text-gray-400">
              <span>Increasing Gas ({increasingPercent.toFixed(0)}%)</span>
              <span>Decreasing Gas ({decreasingPercent.toFixed(0)}%)</span>
            </div>
          </div>

          {/* Decreasing factors */}
          {technical_details.decreasing_factors.length > 0 && (
            <div>
              <h4 className="text-green-400 font-semibold mb-3">
                DECREASING GAS ({decreasingPercent.toFixed(0)}% weight)
              </h4>
              <div className="space-y-2">
                {technical_details.decreasing_factors.map((factor, i) => (
                  <div key={i} className="bg-green-500/10 border border-green-500/30 rounded-md p-3 hover:bg-green-500/20 transition-colors">
                    <div className="flex justify-between items-start mb-1">
                      <span className="text-gray-300 text-sm">{factor.description}</span>
                      <span className="text-green-400 text-sm font-mono font-bold">
                        {factor.weight}%
                      </span>
                    </div>
                    <div className="text-xs text-gray-500 font-mono mt-1">
                      Value: {factor.value !== undefined && factor.value !== null ? factor.value.toFixed(4) : 'N/A'}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Increasing factors */}
          {technical_details.increasing_factors.length > 0 && (
            <div>
              <h4 className="text-red-400 font-semibold mb-3">
                INCREASING GAS ({increasingPercent.toFixed(0)}% weight)
              </h4>
              <div className="space-y-2">
                {technical_details.increasing_factors.map((factor, i) => (
                  <div key={i} className="bg-red-500/10 border border-red-500/30 rounded-md p-3 hover:bg-red-500/20 transition-colors">
                    <div className="flex justify-between items-start mb-1">
                      <span className="text-gray-300 text-sm">{factor.description}</span>
                      <span className="text-red-400 text-sm font-mono font-bold">
                        {factor.weight}%
                      </span>
                    </div>
                    <div className="text-xs text-gray-500 font-mono mt-1">
                      Value: {factor.value !== undefined && factor.value !== null ? factor.value.toFixed(4) : 'N/A'}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Similar historical cases */}
          {technical_details.similar_cases && technical_details.similar_cases.length > 0 && (
            <div>
              <h4 className="text-cyan-400 font-semibold mb-3">
                Similar Historical Situations
              </h4>
              <div className="space-y-2">
                {technical_details.similar_cases.map((case_, i) => (
                  <div key={i} className="bg-cyan-500/10 border border-cyan-500/30 rounded-md p-3 hover:bg-cyan-500/20 transition-colors">
                    <div className="flex justify-between items-center">
                      <span className="text-gray-300 text-sm">{case_.timestamp}</span>
                      <span className="text-cyan-400 font-mono text-sm font-bold">
                        {case_.gas_price !== undefined && case_.gas_price !== null ? case_.gas_price.toFixed(4) : 'N/A'} gwei
                      </span>
                    </div>
                  </div>
                ))}
                <div className="bg-gray-700/50 rounded-md p-3 mt-3">
                  <div className="flex justify-between items-center">
                    <span className="text-gray-400 text-sm">Historical Average:</span>
                    <span className="text-white font-mono text-sm font-bold">
                      {technical_details.similar_cases.length > 0
                        ? (technical_details.similar_cases.reduce((sum, c) => sum + (c.gas_price || 0), 0) /
                           technical_details.similar_cases.length).toFixed(4)
                        : 'N/A'} gwei
                    </span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Feature Importance Table */}
          {technical_details.feature_importance && Object.keys(technical_details.feature_importance).length > 0 && (
            <div>
              <h4 className="text-amber-400 font-semibold mb-3">
                Raw Model Data
              </h4>
              <div className="bg-gray-900/50 rounded-lg p-4 overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-gray-700">
                      <th className="text-left text-gray-400 pb-2">Feature</th>
                      <th className="text-right text-gray-400 pb-2">Value</th>
                      <th className="text-right text-gray-400 pb-2">Importance</th>
                    </tr>
                  </thead>
                  <tbody className="font-mono">
                    {Object.entries(technical_details.feature_importance).map(([feature, data]: [string, any], i) => (
                      <tr key={i} className="border-b border-gray-800 hover:bg-gray-800/50">
                        <td className="py-2 text-gray-300">{feature}</td>
                        <td className="py-2 text-right text-cyan-400">
                          {data.value !== undefined && data.value !== null ? data.value.toFixed(4) : 'N/A'}
                        </td>
                        <td className="py-2 text-right text-amber-400">
                          {data.importance !== undefined && data.importance !== null ? data.importance.toFixed(6) : 'N/A'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Disclaimer */}
      <div className="mt-6 pt-4 border-t border-gray-700">
        <p className="text-xs text-gray-500 text-center">
          {showTechnicalDetails 
            ? "Technical data shows the mathematical reasoning behind our prediction. While we strive for accuracy, gas prices can be unpredictable."
            : "Our AI analyzes multiple factors to generate predictions. Click 'Show Details' to see the technical breakdown."
          }
        </p>
      </div>
    </div>
  );
};

export default PredictionExplanation;

