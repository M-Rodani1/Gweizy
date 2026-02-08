import { BiasCorrection } from '../../../types';

interface CorrectionBadgeProps {
  biasCorrection?: BiasCorrection;
  showDetails?: boolean;
}

/**
 * Small badge indicating when a prediction has been auto-corrected
 * for bias based on recent prediction performance.
 */
export function CorrectionBadge({ biasCorrection, showDetails = false }: CorrectionBadgeProps) {
  if (!biasCorrection?.applied) return null;

  const periodLabel = biasCorrection.period
    ? `${biasCorrection.period} period`
    : 'overall';

  const tooltipText = `Auto-adjusted for ${periodLabel} bias`;

  return (
    <span
      className="inline-flex items-center gap-1 text-xs text-cyan-400 ml-1"
      title={tooltipText}
    >
      <svg
        className="w-3 h-3"
        fill="none"
        stroke="currentColor"
        viewBox="0 0 24 24"
      >
        <path
          strokeLinecap="round"
          strokeLinejoin="round"
          strokeWidth={2}
          d="M5 13l4 4L19 7"
        />
      </svg>
      {showDetails ? (
        <span>adjusted ({periodLabel})</span>
      ) : (
        <span>adjusted</span>
      )}
    </span>
  );
}

export default CorrectionBadge;
