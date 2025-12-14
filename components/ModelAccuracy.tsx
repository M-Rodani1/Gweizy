import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import { fetchAccuracy } from '../src/api/gasApi';
import LoadingSpinner from './LoadingSpinner';

interface AccuracyData {
  mae: number;
  rmse: number;
  r2: number;
  directional_accuracy: number;
  recent_predictions?: Array<{
    timestamp: string;
    predicted: number;
    actual: number;
    error: number;
  }>;
  last_updated?: string;
}

const ModelAccuracy: React.FC = () => {
  const [data, setData] = useState<AccuracyData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [isExpanded, setIsExpanded] = useState(false);

  const loadData = async () => {
    try {
      setLoading(true);
      setError(null);
      const accuracyData = await fetchAccuracy();
      setData(accuracyData);
    } catch (err) {
      console.error('Error loading accuracy data:', err);
      setError(err instanceof Error ? err.message : 'Failed to load accuracy data');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 300000); // Refresh every 5 minutes
    return () => clearInterval(interval);
  }, []);

  const getConfidenceLevel = (accuracy: number) => {
    // accuracy is a decimal (0.0-1.0)
    if (accuracy >= 0.7) return { level: 'High', color: 'green', emoji: '🟢' };
    if (accuracy >= 0.6) return { level: 'Medium', color: 'yellow', emoji: '🟡' };
    return { level: 'Low', color: 'red', emoji: '🔴' };
  };

  const getMetricColor = (value: number, type: 'mae' | 'rmse' | 'r2' | 'directional') => {
    if (type === 'r2') {
      // R² is returned as decimal (0.0-1.0), but we display as percentage
      const percent = value * 100;
      if (percent >= 70) return 'text-green-400';
      if (percent >= 50) return 'text-yellow-400';
      return 'text-red-400';
    }
    if (type === 'directional') {
      // Directional accuracy is returned as decimal (0.0-1.0), but we display as percentage
      const percent = value * 100;
      if (percent >= 70) return 'text-green-400';
      if (percent >= 60) return 'text-yellow-400';
      return 'text-red-400';
    }
    // For MAE and RMSE, lower is better
    if (value < 0.001) return 'text-green-400';
    if (value < 0.01) return 'text-yellow-400';
    return 'text-red-400';
  };

  const prepareChartData = () => {
    if (!data?.recent_predictions) return [];
    
    return data.recent_predictions.map((pred) => ({
      time: new Date(pred.timestamp).toLocaleTimeString('en-US', { 
        hour: '2-digit', 
        minute: '2-digit' 
      }),
      predicted: pred.predicted,
      actual: pred.actual,
      error: pred.error,
      isClose: pred.error < 0.001 // Green if error < 0.001 gwei
    }));
  };

  const chartData = prepareChartData();
  const confidence = data ? getConfidenceLevel(data.directional_accuracy) : null;

  if (loading && !data) {
    return (
      <div className="bg-gray-800 p-6 rounded-lg shadow-lg border border-gray-700">
        <LoadingSpinner message="Loading model accuracy..." />
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-gray-800 p-6 rounded-lg shadow-lg border border-gray-700">
        <p className="text-red-400 mb-4">⚠️ {error}</p>
        <button
          onClick={loadData}
          className="px-4 py-2 bg-cyan-500 hover:bg-cyan-600 rounded-md transition-colors text-sm"
        >
          Retry
        </button>
      </div>
    );
  }

  if (!data) return null;

  return (
    <div className="bg-gray-800 p-6 rounded-lg shadow-lg border border-gray-700">
      {/* Header with Collapse Toggle */}
      <div 
        className="flex items-center justify-between mb-6 cursor-pointer"
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <div className="flex items-center space-x-3">
          <span className="text-2xl">📊</span>
          <h2 className="text-xl font-bold text-gray-200">Model Performance</h2>
        </div>
        <button className="text-gray-400 hover:text-gray-200 transition-colors">
          {isExpanded ? '▼' : '▶'}
        </button>
      </div>

      {isExpanded && (
        <>
          {/* Accuracy Metrics Grid */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div className="bg-gray-700/50 rounded-lg p-4 border border-gray-600">
              <div className="text-sm text-gray-400 mb-1">MAE</div>
              <div className={`text-2xl font-bold ${getMetricColor(data.mae, 'mae')}`}>
                {data.mae.toFixed(6)}
              </div>
              <div className="text-xs text-gray-500 mt-1">Mean Absolute Error</div>
            </div>

            <div className="bg-gray-700/50 rounded-lg p-4 border border-gray-600">
              <div className="text-sm text-gray-400 mb-1">RMSE</div>
              <div className={`text-2xl font-bold ${getMetricColor(data.rmse, 'rmse')}`}>
                {data.rmse.toFixed(6)}
              </div>
              <div className="text-xs text-gray-500 mt-1">Root Mean Squared Error</div>
            </div>

            <div className="bg-gray-700/50 rounded-lg p-4 border border-gray-600">
              <div className="text-sm text-gray-400 mb-1">R² Score</div>
              <div className={`text-2xl font-bold ${getMetricColor(data.r2, 'r2')}`}>
                {(data.r2 * 100).toFixed(1)}%
              </div>
              <div className="text-xs text-gray-500 mt-1">Variance Explained</div>
            </div>

            <div className="bg-gray-700/50 rounded-lg p-4 border border-gray-600">
              <div className="text-sm text-gray-400 mb-1">Directional Accuracy</div>
              <div className={`text-2xl font-bold ${getMetricColor(data.directional_accuracy, 'directional')}`}>
                {(data.directional_accuracy * 100).toFixed(1)}%
              </div>
              <div className="text-xs text-gray-500 mt-1">Up/Down Prediction</div>
            </div>
          </div>

          {/* Model Confidence Indicator */}
          {confidence && (
            <div className="mb-6 p-4 bg-gray-700/30 rounded-lg border border-gray-600">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <span className="text-2xl">{confidence.emoji}</span>
                  <div>
                    <div className="text-sm text-gray-400">Model Confidence</div>
                    <div className={`text-xl font-bold text-${confidence.color}-400`}>
                      {confidence.level} Confidence
                    </div>
                  </div>
                </div>
                <div className="text-right">
                  <div className="text-sm text-gray-400">Overall Accuracy</div>
                  <div className={`text-2xl font-bold text-${confidence.color}-400`}>
                    {(data.directional_accuracy * 100).toFixed(0)}%
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Prediction vs Actual Graph */}
          {chartData.length > 0 && (
            <div className="mb-6">
              <h3 className="text-lg font-semibold text-gray-200 mb-4">
                Prediction vs Actual (Last 24 Hours)
              </h3>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#4A5568" />
                    <XAxis 
                      dataKey="time" 
                      stroke="#A0AEC0"
                      tick={{ fill: '#A0AEC0', fontSize: 12 }}
                    />
                    <YAxis 
                      stroke="#A0AEC0"
                      tick={{ fill: '#A0AEC0', fontSize: 12 }}
                    />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: '#1A202C',
                        borderColor: '#4A5568',
                        color: '#E2E8F0'
                      }}
                      labelStyle={{ color: '#E2E8F0' }}
                    />
                    <Line
                      type="monotone"
                      dataKey="actual"
                      name="Actual Gas Price"
                      stroke="#4FD1C5"
                      strokeWidth={2}
                      dot={{ r: 3 }}
                    />
                    <Line
                      type="monotone"
                      dataKey="predicted"
                      name="Predicted Gas Price"
                      stroke="#F6E05E"
                      strokeWidth={2}
                      strokeDasharray="5 5"
                      dot={{ r: 3 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </div>
          )}

          {/* Performance Over Time */}
          <div className="mb-4 p-4 bg-gray-700/30 rounded-lg border border-gray-600">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-sm text-gray-400 mb-1">Performance Over Time</div>
                <div className="text-lg font-semibold text-gray-200">
                  Our model has been <span className="text-green-400 font-bold">
                    {(data.directional_accuracy * 100).toFixed(0)}%
                  </span> accurate over the past week
                </div>
              </div>
              {confidence && (
                <div className="text-4xl">{confidence.emoji}</div>
              )}
            </div>
          </div>

          {/* Last Updated */}
          {data.last_updated && (
            <div className="text-xs text-gray-500 text-center mt-4">
              Last updated: {new Date(data.last_updated).toLocaleString()}
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default ModelAccuracy;

