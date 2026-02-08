/**
 * Utility functions for prediction-related components
 */

export type PredictionColor = 'red' | 'green' | 'yellow';

export interface Probabilities {
  wait: number;
  normal: number;
  urgent: number;
}

/**
 * Get Tailwind border/background classes for a prediction color
 */
export const getColorClasses = (color: string): string => {
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
};

/**
 * Get Tailwind text color class for a prediction color
 */
export const getTextColor = (color: string): string => {
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
};

/**
 * Determine classification state from probabilities
 * Returns 'spike', 'elevated', or 'normal'
 */
export const getClassificationState = (probs: Probabilities | undefined): string | null => {
  if (!probs) return null;
  const max = Math.max(probs.wait, probs.normal, probs.urgent);
  if (probs.urgent === max && probs.urgent > 0.3) return 'spike';
  if (probs.wait === max && probs.wait > 0.3) return 'elevated';
  return 'normal';
};
