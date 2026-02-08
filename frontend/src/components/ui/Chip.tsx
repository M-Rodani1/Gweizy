import React from 'react';

interface ChipProps {
  label: string;
  onRemove?: () => void;
  className?: string;
}

export const Chip: React.FC<ChipProps> = ({ label, onRemove, className = '' }) => (
  <span className={`inline-flex items-center gap-2 px-3 py-1 rounded-full bg-[var(--surface-2)] border border-[var(--border)] text-sm text-[var(--text)] ${className}`}>
    {label}
    {onRemove && (
      <button
        type="button"
        onClick={onRemove}
        className="text-[var(--text-muted)] hover:text-[var(--text)] focus:outline-none focus-visible:ring-2 focus-visible:ring-cyan-500 rounded"
        aria-label={`Remove ${label}`}
      >
        ×
      </button>
    )}
  </span>
);

export default Chip;
