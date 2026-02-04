import React, { useEffect, useState, useCallback } from 'react';
import { Bot, CheckCircle, Clock, Coins, Zap } from 'lucide-react';
import { API_CONFIG, getApiUrl } from '../config/api';

interface AgentResponse {
  success: boolean;
  recommendation: {
    action: string;
    confidence: number;
    recommended_gas: number;
    expected_savings: number;
    reasoning: string;
    urgency_factor: number;
    q_values?: Record<string, number>;
  };
  context: {
    current_gas: number;
    predictions: Record<string, number>;
    timestamp: string;
  };
}

interface AgentRecommendationProps {
  currentGas?: number;
}

const AgentRecommendation: React.FC<AgentRecommendationProps> = ({ currentGas: _currentGas = 0 }) => {
  const [recommendation, setRecommendation] = useState<AgentResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [urgency, setUrgency] = useState(0.5);

  const fetchRecommendation = useCallback(async () => {
    try {
      const response = await fetch(
        getApiUrl(API_CONFIG.ENDPOINTS.AGENT_RECOMMEND, { urgency })
      );
      const data = await response.json();

      if (data.success) {
        setRecommendation(data);
        setError(null);
      } else {
        setError(data.error || 'Failed to get recommendation');
      }
    } catch (err) {
      setError('Unable to connect to agent');
      console.error('Agent fetch error:', err);
    } finally {
      setLoading(false);
    }
  }, [urgency]);

  useEffect(() => {
    fetchRecommendation();
    const interval = setInterval(fetchRecommendation, 30000); // Refresh every 30s
    return () => clearInterval(interval);
  }, [fetchRecommendation]);

  const getActionConfig = (action: string) => {
    switch (action) {
      case 'WAIT':
        return {
          icon: <Clock className="w-6 h-6" />,
          color: 'yellow',
          bgClass: 'bg-yellow-500/10 border-yellow-500/50',
          textClass: 'text-yellow-400',
          label: 'Wait',
          description: 'Hold off - better prices expected soon'
        };
      case 'SUBMIT_NOW':
        return {
          icon: <CheckCircle className="w-6 h-6" />,
          color: 'green',
          bgClass: 'bg-green-500/10 border-green-500/50',
          textClass: 'text-green-400',
          label: 'Submit Now',
          description: 'Good time to submit your transaction'
        };
      case 'SUBMIT_LOW':
        return {
          icon: <Coins className="w-6 h-6" />,
          color: 'cyan',
          bgClass: 'bg-cyan-500/10 border-cyan-500/50',
          textClass: 'text-cyan-400',
          label: 'Submit Low',
          description: 'Try 10% below current price (~15% fail risk)'
        };
      case 'SUBMIT_HIGH':
        return {
          icon: <Zap className="w-6 h-6" />,
          color: 'purple',
          bgClass: 'bg-purple-500/10 border-purple-500/50',
          textClass: 'text-purple-400',
          label: 'Submit High',
          description: 'Priority submission for faster confirmation'
        };
      default:
        return {
          icon: <Bot className="w-6 h-6" />,
          color: 'gray',
          bgClass: 'bg-gray-500/10 border-gray-500/50',
          textClass: 'text-gray-400',
          label: action,
          description: 'Agent recommendation'
        };
    }
  };

  if (loading) {
    return (
      <div className="bg-gradient-to-br from-gray-800 to-gray-900 p-4 sm:p-6 rounded-2xl shadow-2xl border border-gray-700/50 animate-pulse">
        <div className="h-6 bg-gray-700 rounded w-48 mb-4"></div>
        <div className="h-32 bg-gray-700 rounded"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-gradient-to-br from-gray-800 to-gray-900 p-4 sm:p-6 rounded-2xl shadow-2xl border border-red-500/30">
        <div className="flex items-center gap-2 mb-4">
          <Bot className="w-5 h-5 text-gray-400" />
          <h3 className="text-lg sm:text-xl font-bold text-gray-300">AI Agent</h3>
        </div>
        <div className="text-red-400 text-sm">{error}</div>
      </div>
    );
  }

  if (!recommendation) return null;

  const { recommendation: rec, context } = recommendation;
  const actionConfig = getActionConfig(rec.action);
  const confidencePercent = Math.round(rec.confidence * 100);

  return (
    <div className="bg-gradient-to-br from-gray-800 to-gray-900 p-4 sm:p-6 rounded-2xl shadow-2xl border border-gray-700/50 card-hover">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Bot className="w-5 h-5 text-purple-400" />
          <h3 className="text-lg sm:text-xl font-bold bg-gradient-to-r from-purple-400 to-pink-500 bg-clip-text text-transparent">
            AI Transaction Agent
          </h3>
        </div>
        <div className="text-xs text-gray-500">
          DQN Neural Network
        </div>
      </div>

      {/* Main Recommendation */}
      <div className={`${actionConfig.bgClass} border-2 rounded-xl p-4 mb-4`}>
        <div className="flex items-center gap-3 mb-2">
          <span className={actionConfig.textClass}>{actionConfig.icon}</span>
          <div>
            <div className={`text-xl font-bold ${actionConfig.textClass}`}>
              {actionConfig.label}
            </div>
            <div className="text-sm text-gray-400">{actionConfig.description}</div>
          </div>
        </div>

        {/* Reasoning */}
        <div className="mt-3 p-3 bg-gray-800/50 rounded-lg">
          <div className="text-xs text-gray-500 mb-1">Agent Reasoning:</div>
          <div className="text-sm text-gray-300">{rec.reasoning}</div>
        </div>
      </div>

      {/* Confidence Meter */}
      <div className="mb-4">
        <div className="flex justify-between text-xs text-gray-400 mb-1">
          <span>Confidence</span>
          <span>{confidencePercent}%</span>
        </div>
        <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
          <div
            className={`h-full transition-all duration-500 ${
              confidencePercent > 70 ? 'bg-green-500' :
              confidencePercent > 40 ? 'bg-yellow-500' : 'bg-red-500'
            }`}
            style={{ width: `${confidencePercent}%` }}
          />
        </div>
      </div>

      {/* Urgency Slider */}
      <div className="mb-4">
        <div className="flex justify-between text-xs text-gray-400 mb-2">
          <span>Transaction Urgency</span>
          <span>{Math.round(urgency * 100)}%</span>
        </div>
        <input
          type="range"
          min="0"
          max="1"
          step="0.1"
          value={urgency}
          onChange={(e) => setUrgency(parseFloat(e.target.value))}
          className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-purple-500"
        />
        <div className="flex justify-between text-xs text-gray-500 mt-1">
          <span>No Rush</span>
          <span>Urgent</span>
        </div>
      </div>

      {/* Context Info */}
      <div className="grid grid-cols-2 gap-3 text-sm">
        <div className="bg-gray-800/50 rounded-lg p-3">
          <div className="text-xs text-gray-500">Current Gas</div>
          <div className="font-mono font-bold text-gray-200">
            {context.current_gas.toFixed(4)} gwei
          </div>
        </div>
        <div className="bg-gray-800/50 rounded-lg p-3">
          <div className="text-xs text-gray-500">Recommended</div>
          <div className="font-mono font-bold text-gray-200">
            {rec.recommended_gas.toFixed(4)} gwei
          </div>
        </div>
      </div>

      {/* Predictions */}
      <div className="mt-3 p-3 bg-gray-800/30 rounded-lg">
        <div className="text-xs text-gray-500 mb-2">Price Predictions</div>
        <div className="grid grid-cols-3 gap-2 text-xs">
          {Object.entries(context.predictions).map(([horizon, price]) => (
            <div key={horizon} className="text-center">
              <div className="text-gray-500">{horizon}</div>
              <div className={`font-mono ${
                price < context.current_gas ? 'text-green-400' : 'text-red-400'
              }`}>
                {typeof price === 'number' ? price.toFixed(4) : 'N/A'}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Q-Values (for advanced users) */}
      {rec.q_values && (
        <details className="mt-3">
          <summary className="text-xs text-gray-500 cursor-pointer hover:text-gray-400">
            Show Q-Values (Advanced)
          </summary>
          <div className="mt-2 grid grid-cols-2 gap-2 text-xs">
            {Object.entries(rec.q_values).map(([action, value]) => (
              <div key={action} className="flex justify-between bg-gray-800/30 p-2 rounded">
                <span className="text-gray-400">{action}</span>
                <span className="font-mono text-gray-300">{value.toFixed(3)}</span>
              </div>
            ))}
          </div>
        </details>
      )}

      {/* Refresh Button */}
      <button
        onClick={fetchRecommendation}
        className="mt-4 w-full py-2 px-4 bg-purple-600/20 hover:bg-purple-600/30 border border-purple-500/50 rounded-lg text-purple-400 text-sm font-medium transition-colors"
      >
        Refresh Recommendation
      </button>
    </div>
  );
};

export default AgentRecommendation;
