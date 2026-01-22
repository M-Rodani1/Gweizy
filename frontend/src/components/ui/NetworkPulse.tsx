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
 * Visualizes the live network load with a breathing progress bar that
 * pulses based on utilization level.
 */
const NetworkPulse: React.FC<NetworkPulseProps> = ({ utilization, isConnected }) => {
  // Clamp utilization between 0 and 1
  const clampedUtil = Math.max(0, Math.min(1, utilization));
  const percentage = Math.round(clampedUtil * 100);

  // Color logic: <50% (Green), 50-80% (Yellow), >80% (Red)
  const getColorClasses = () => {
    if (clampedUtil < 0.5) {
      return {
        gradient: 'from-green-500 to-green-600',
        text: 'text-green-400',
      };
    }
    if (clampedUtil < 0.8) {
      return {
        gradient: 'from-yellow-500 to-yellow-600',
        text: 'text-yellow-400',
      };
    }
    return {
      gradient: 'from-red-500 to-red-600',
      text: 'text-red-400',
    };
  };

  const colors = getColorClasses();

  // Breathing animation opacity based on utilization
  // Higher utilization = faster pulse
  const pulseSpeed = 1 + clampedUtil * 2; // 1s to 3s

  return (
    <div className="bg-gray-800/60 rounded-xl shadow-lg border border-gray-700 p-4 flex flex-col gap-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Activity
            className={`w-4 h-4 ${colors.text} ${isConnected ? 'animate-pulse' : 'opacity-60'}`}
            aria-hidden="true"
          />
          <h3 className="text-sm font-semibold text-gray-200">Network Load</h3>
        </div>
        <span className={`text-xs ${isConnected ? 'text-green-400' : 'text-yellow-400'}`}>
          {isConnected ? 'Live' : 'Cached'}
        </span>
      </div>

      {/* Center value */}
      <div className="flex flex-col items-center justify-center flex-1">
        <div className={`text-3xl font-bold ${colors.text}`}>{percentage}%</div>
        <p className="text-xs text-gray-400">Utilization</p>
      </div>

      {/* Breathing progress bar */}
      <div className="relative w-full h-2 bg-gray-900 rounded-full overflow-hidden border border-gray-700">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${clampedUtil * 100}%` }}
          transition={{ duration: 0.8, ease: 'easeOut' }}
          className={`absolute left-0 top-0 h-full bg-gradient-to-r ${colors.gradient}`}
        />
        {isConnected && (
          <motion.div
            animate={{ opacity: [0.3, 0.7, 0.3] }}
            transition={{
              duration: pulseSpeed,
              repeat: Infinity,
              ease: 'easeInOut',
            }}
            className={`absolute left-0 top-0 h-full bg-gradient-to-r ${colors.gradient}`}
            style={{ width: `${clampedUtil * 100}%` }}
          />
        )}
      </div>
    </div>
  );
};

export default NetworkPulse;
