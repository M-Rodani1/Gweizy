import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { GraphDataPoint } from '../types';
import { fetchPredictions, fetchCurrentGas } from '../api/gasApi';
import LoadingSpinner from './LoadingSpinner';

type TimeScale = '1h' | '4h' | '24h' | 'historical';

const GasPriceGraph: React.FC = () => {
  const [timeScale, setTimeScale] = useState<TimeScale>('24h');
  const [data, setData] = useState<GraphDataPoint[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [currentGas, setCurrentGas] = useState<number | null>(null);

  const loadData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);

      // Fetch both predictions and current gas
      const [predictionsResult, currentGasData] = await Promise.all([
        fetchPredictions(),
        fetchCurrentGas()
      ]);

      setCurrentGas(currentGasData?.current_gas || null);

      // Get data for selected timeframe - with safe access
      let selectedData = predictionsResult?.predictions?.[timeScale] || [];

      // For prediction timeframes (1h, 4h, 24h), add current gas as first point
      if (timeScale !== 'historical' && currentGasData?.current_gas) {
        const currentPoint: GraphDataPoint = {
          time: 'now',
          gwei: currentGasData.current_gas,
          predictedGwei: null
        };

        // Get the first predicted point for this timeframe
        if (Array.isArray(selectedData) && selectedData.length > 0 && selectedData[0]?.predictedGwei !== undefined) {
          const firstPredicted: GraphDataPoint = {
            time: selectedData[0].time,
            gwei: null,
            predictedGwei: selectedData[0].predictedGwei
          };
          // Combine current point + predicted point to show connection
          selectedData = [currentPoint, firstPredicted, ...selectedData.slice(1)];
        } else {
          // If no predictions yet, just show current
          selectedData = [currentPoint, ...selectedData];
        }
      }

      setData(selectedData);

    } catch (err) {
      console.error('Error loading graph data:', err);
      setError(err instanceof Error ? err.message : 'Failed to load data');
    } finally {
      setLoading(false);
    }
  }, [timeScale]);

  useEffect(() => {
    loadData();

    // Auto-refresh every 30 seconds
    const interval = setInterval(loadData, 30000);
    return () => clearInterval(interval);
  }, [loadData]);

  if (loading && data.length === 0) {
    return (
      <div className="bg-gray-800 p-4 sm:p-6 rounded-lg shadow-lg h-64 md:h-80 lg:h-96">
        <LoadingSpinner message="Loading gas price data..." />
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-gray-800 p-4 sm:p-6 rounded-lg shadow-lg h-64 md:h-80 lg:h-96">
        <div className="flex flex-col items-center justify-center h-full">
          <p className="text-red-400 mb-4">⚠️ {error}</p>
          <button
            onClick={loadData}
            className="px-4 py-2 bg-cyan-500 hover:bg-cyan-600 rounded-md transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gradient-to-br from-gray-800 to-gray-900 p-4 sm:p-6 rounded-2xl shadow-2xl border border-gray-700/50 card-hover h-64 md:h-80 lg:h-96">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-4 gap-3">
        <h2 className="text-lg sm:text-xl font-bold bg-gradient-to-r from-cyan-400 to-blue-500 bg-clip-text text-transparent">
          Gas Price Predictions - {timeScale.toUpperCase()}
        </h2>
        <div className="flex flex-wrap gap-1 bg-gray-700/50 p-1 rounded-md">
          {(['1h', '4h', '24h', 'historical'] as TimeScale[]).map((scale) => (
            <button
              key={scale}
              onClick={() => setTimeScale(scale)}
              className={`px-3 py-2 text-sm font-medium rounded-md transition-colors min-w-[60px] ${
                timeScale === scale
                  ? 'bg-cyan-500 text-white'
                  : 'text-gray-300 hover:bg-gray-600'
              }`}
            >
              {scale === 'historical' ? 'History' : scale.toUpperCase()}
            </button>
          ))}
        </div>
      </div>

      <ResponsiveContainer width="100%" height="85%">
        <LineChart data={data} margin={{ top: 5, right: 10, left: 0, bottom: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#4A5568" />
          <XAxis dataKey="time" stroke="#A0AEC0" />
          <YAxis stroke="#A0AEC0" />
          <Tooltip
            contentStyle={{
              backgroundColor: '#1A202C',
              borderColor: '#4A5568'
            }}
            labelStyle={{ color: '#E2E8F0' }}
            formatter={(value: any, name: string) => {
              if (value === null || value === undefined) return 'N/A';
              return `${Number(value).toFixed(4)} gwei`;
            }}
            labelFormatter={(label: string) => {
              if (label === 'now') return 'Current (Now)';
              return `Time: ${label}`;
            }}
          />
          <Legend wrapperStyle={{ color: '#E2E8F0', paddingTop: '20px' }} />
          <Line
            type="monotone"
            dataKey="gwei"
            name="Current/Actual Price"
            stroke="#4FD1C5"
            strokeWidth={3}
            dot={{ r: 6, fill: '#4FD1C5' }}
            activeDot={{ r: 8, stroke: '#81E6D9', strokeWidth: 2 }}
            connectNulls={false}
          />
          <Line
            type="monotone"
            dataKey="predictedGwei"
            name="Predicted Price"
            stroke="#F6E05E"
            strokeWidth={2}
            strokeDasharray="5 5"
            dot={{ r: 5, fill: '#F6E05E' }}
            activeDot={{ r: 8, stroke: '#FAF089', strokeWidth: 2 }}
            connectNulls={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default GasPriceGraph;
