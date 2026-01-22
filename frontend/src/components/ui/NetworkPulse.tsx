import React from 'react';
import { motion } from 'framer-motion';
import { Activity } from 'lucide-react';

interface NetworkPulseProps {
  utilization: number; // 0.0 to 1.0 (e.g. 0.85 for 85%)
  isConnected: boolean;
}

/**
 * NetworkPulse Component
 * 
 * Visualizes the live network load (Block Headers) with a breathing
 * progress bar that pulses based on utilization level.
 */
const NetworkPulse: React.FC<NetworkPulseProps> = ({ utilization, isConnected }) => {
  // Clamp utilization between 0 and 1
  const clampedUtil = Math.max(0, Math.min(1, utilization));
  const percentage = (clampedUtil * 100).toFixed(1);

  // Color logic: <50% (Green), 50-80% (Yellow), >80% (Red)
  const getColorClasses = () => {
    if (clampedUtil < 0.5) {
      return {
        bg: 'bg-green-500',
        gradient: 'from-green-500 to-green-600',
        text: 'text-green-400',
        glow: 'shadow-glow-cyan',
      };
    } else if (clampedUtil < 0.8) {
      return {
        bg: 'bg-yellow-500',
        gradient: 'from-yellow-500 to-yellow-600',
        text: 'text-yellow-400',
        glow: 'shadow-glow-cyan',
      };
    } else {
      return {
        bg: 'bg-red-500',
        gradient: 'from-red-500 to-red-600',
        text: 'text-red-400',
        glow: 'shadow-glow-cyan',
      };
    }
  };

  const colors = getColorClasses();

  // Breathing animation opacity based on utilization
  // Higher utilization = faster pulse
  const pulseSpeed = 1 + clampedUtil * 2; // 1s to 3s

  return (
    <div className="bg-gray-800 rounded-lg shadow-lg border border-gray-700 p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Activity 
            className={`w-5 h-5 ${colors.text} ${isConnected ? 'animate-pulse' : 'opacity-50'}`}
            aria-hidden="true"
          />
          <h3 className="text-lg font-semibold text-gray-200">
            Base Network Load
          </h3>
        </div>
        {isConnected && (
          <span className="text-xs text-green-400 flex items-center gap-1">
            <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
            Live
          </span>
        )}
      </div>

      {/* Large percentage display */}
      <div className="mb-4">
        <div className={`text-4xl font-bold ${colors.text} mb-1`}>
          {percentage}%
        </div>
        <p className="text-sm text-gray-400">
          Network Utilization
        </p>
      </div>

      {/* Breathing progress bar */}
      <div className="relative w-full h-4 bg-gray-900 rounded-full overflow-hidden border border-gray-700">
        {/* Background fill */}
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${clampedUtil * 100}%` }}
          transition={{ duration: 0.8, ease: 'easeOut' }}
          className={`absolute left-0 top-0 h-full bg-gradient-to-r ${colors.gradient}`}
        />

        {/* Breathing pulse overlay */}
        {isConnected && (
          <motion.div
            animate={{
              opacity: [0.3, 0.7, 0.3],
            }}
            transition={{
              duration: pulseSpeed,
              repeat: Infinity,
              ease: 'easeInOut',
            }}
            className={`absolute left-0 top-0 h-full w-full bg-gradient-to-r ${colors.gradient}`}
            style={{ width: `${clampedUtil * 100}%` }}
          />
        )}
      </div>

      {/* Status indicator */}
      <div className="mt-4 flex items-center justify-between text-xs">
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${colors.bg}`} />
          <span className="text-gray-400">
            {clampedUtil < 0.5 && 'Low Load'}
            {clampedUtil >= 0.5 && clampedUtil < 0.8 && 'Moderate Load'}
            {clampedUtil >= 0.8 && 'High Load'}
          </span>
        </div>
        {!isConnected && (
          <span className="text-yellow-400">Using cached data</span>
        )}
      </div>
    </div>
  );
};

export default NetworkPulse;
