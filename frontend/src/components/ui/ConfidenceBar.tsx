import React from 'react';
import { motion } from 'framer-motion';

interface ConfidenceBarProps {
  probs: {
    wait: number;
    normal: number;
    urgent: number;
  };
  showLabels?: 'action' | 'classification';
}

/**
 * ConfidenceBar Component
 * 
 * Visualizes the probability split between Wait/Normal/Urgent actions
 * as a rounded progress bar with three colored sections.
 */
const ConfidenceBar: React.FC<ConfidenceBarProps> = ({ probs, showLabels = 'action' }) => {
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

  // Label configurations
  const labels = showLabels === 'classification'
    ? { wait: 'Elevated', normal: 'Normal', urgent: 'Spike', waitColor: 'text-amber-400', normalColor: 'text-green-400', urgentColor: 'text-red-400' }
    : { wait: 'Wait', normal: 'Normal', urgent: 'Urgent', waitColor: 'text-green-400', normalColor: 'text-blue-400', urgentColor: 'text-red-400' };

  return (
    <div className="w-full">
      {/* Labels above the bar - flex container displaying % text */}
      <div className="flex justify-between items-center mb-3">
        <span className={`text-xs font-semibold ${labels.waitColor}`}>
          {labels.wait}: {(normalizedWait * 100).toFixed(1)}%
        </span>
        <span className={`text-xs font-semibold ${labels.normalColor}`}>
          {labels.normal}: {(normalizedNormal * 100).toFixed(1)}%
        </span>
        <span className={`text-xs font-semibold ${labels.urgentColor}`}>
          {labels.urgent}: {(normalizedUrgent * 100).toFixed(1)}%
        </span>
      </div>

      {/* Progress bar container */}
      <div className="relative w-full h-6 bg-gray-800 rounded-full overflow-hidden border border-gray-700">
        {/* Wait/Elevated section */}
        {waitWidth > 0 && (
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${waitWidth}%` }}
            transition={{ duration: 0.6, ease: 'easeOut' }}
            className={`absolute left-0 top-0 h-full ${
              showLabels === 'classification'
                ? 'bg-gradient-to-r from-amber-500 to-amber-600'
                : 'bg-gradient-to-r from-green-500 to-green-600'
            }`}
            style={{ zIndex: 1 }}
          />
        )}

        {/* Normal section */}
        {normalWidth > 0 && (
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${normalWidth}%` }}
            transition={{ duration: 0.6, ease: 'easeOut', delay: 0.1 }}
            className={`absolute h-full ${
              showLabels === 'classification'
                ? 'bg-gradient-to-r from-green-500 to-green-600'
                : 'bg-gradient-to-r from-blue-500 to-blue-600'
            }`}
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
          <div className={`w-3 h-3 rounded-full ${showLabels === 'classification' ? 'bg-amber-500' : 'bg-green-500'}`} />
          <span>{labels.wait}</span>
        </div>
        <div className="flex items-center gap-1">
          <div className={`w-3 h-3 rounded-full ${showLabels === 'classification' ? 'bg-green-500' : 'bg-blue-500'}`} />
          <span>{labels.normal}</span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-3 h-3 rounded-full bg-red-500" />
          <span>{labels.urgent}</span>
        </div>
      </div>
    </div>
  );
};

export default ConfidenceBar;
