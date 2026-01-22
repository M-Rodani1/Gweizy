import React from 'react';
import { motion } from 'framer-motion';

interface ConfidenceBarProps {
  probs: {
    wait: number;
    normal: number;
    urgent: number;
  };
}

/**
 * ConfidenceBar Component
 * 
 * Visualizes the probability split between Wait/Normal/Urgent actions
 * as a rounded progress bar with three colored sections.
 */
const ConfidenceBar: React.FC<ConfidenceBarProps> = ({ probs }) => {
  const { wait, normal, urgent } = probs;
  
  // Ensure probabilities sum to 1.0 and are valid
  const total = wait + normal + urgent;
  const normalizedWait = total > 0 ? wait / total : 0;
  const normalizedNormal = total > 0 ? normal / total : 0;
  const normalizedUrgent = total > 0 ? urgent / total : 0;

  // Calculate widths as percentages
  const waitWidth = normalizedWait * 100;
  const normalWidth = normalizedNormal * 100;
  const urgentWidth = normalizedUrgent * 100;

  return (
    <div className="w-full">
      {/* Labels above the bar */}
      <div className="flex justify-between items-center mb-2">
        <div className="flex flex-col items-start">
          <span className="text-xs font-semibold text-green-400">
            Wait: {(normalizedWait * 100).toFixed(1)}%
          </span>
        </div>
        <div className="flex flex-col items-center">
          <span className="text-xs font-semibold text-cyan-400">
            Normal: {(normalizedNormal * 100).toFixed(1)}%
          </span>
        </div>
        <div className="flex flex-col items-end">
          <span className="text-xs font-semibold text-red-400">
            Urgent: {(normalizedUrgent * 100).toFixed(1)}%
          </span>
        </div>
      </div>

      {/* Progress bar container */}
      <div className="relative w-full h-6 bg-gray-800 rounded-full overflow-hidden border border-gray-700">
        {/* Wait section (Green) */}
        {waitWidth > 0 && (
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${waitWidth}%` }}
            transition={{ duration: 0.6, ease: 'easeOut' }}
            className="absolute left-0 top-0 h-full bg-gradient-to-r from-green-500 to-green-600"
            style={{ zIndex: 1 }}
          />
        )}

        {/* Normal section (Blue/Cyan) */}
        {normalWidth > 0 && (
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${normalWidth}%` }}
            transition={{ duration: 0.6, ease: 'easeOut', delay: 0.1 }}
            className="absolute h-full bg-gradient-to-r from-cyan-500 to-cyan-600"
            style={{
              left: `${waitWidth}%`,
              zIndex: 2,
            }}
          />
        )}

        {/* Urgent section (Red) */}
        {urgentWidth > 0 && (
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${urgentWidth}%` }}
            transition={{ duration: 0.6, ease: 'easeOut', delay: 0.2 }}
            className="absolute right-0 top-0 h-full bg-gradient-to-r from-red-500 to-red-600"
            style={{ zIndex: 3 }}
          />
        )}

        {/* Divider lines between sections (optional visual enhancement) */}
        {waitWidth > 0 && normalWidth > 0 && (
          <div
            className="absolute top-0 bottom-0 w-px bg-gray-900/50"
            style={{ left: `${waitWidth}%`, zIndex: 10 }}
          />
        )}
        {normalWidth > 0 && urgentWidth > 0 && (
          <div
            className="absolute top-0 bottom-0 w-px bg-gray-900/50"
            style={{ left: `${waitWidth + normalWidth}%`, zIndex: 10 }}
          />
        )}
      </div>

      {/* Legend below the bar */}
      <div className="flex justify-center gap-4 mt-2 text-xs text-gray-400">
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded-full bg-green-500" />
          <span>Wait</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded-full bg-cyan-500" />
          <span>Normal</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded-full bg-red-500" />
          <span>Urgent</span>
        </div>
      </div>
    </div>
  );
};

export default ConfidenceBar;
