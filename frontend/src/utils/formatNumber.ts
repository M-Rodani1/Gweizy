/**
 * Format a number to a specified number of significant decimals
 * Removes trailing zeros for cleaner display
 */
export function formatGwei(value: number, maxDecimals: number = 4): string {
  if (value === 0) return '0';

  // For very small numbers, use scientific notation threshold
  if (value < 0.0001 && value > 0) {
    return value.toExponential(2);
  }

  // For larger numbers, just show 2 decimals
  if (value >= 1000) {
    return value.toFixed(2);
  }

  // For typical gas values, show appropriate decimals
  const fixed = value.toFixed(maxDecimals);
  // Remove trailing zeros after decimal point
  return parseFloat(fixed).toString();
}

/**
 * Format USD values - always show 2-4 decimals depending on size
 */
export function formatUsd(value: number): string {
  if (value === 0) return '$0.00';

  if (value < 0.0001) {
    return '<$0.0001';
  }

  if (value < 0.01) {
    return `$${value.toFixed(4)}`;
  }

  if (value < 1) {
    return `$${value.toFixed(4)}`;
  }

  if (value < 100) {
    return `$${value.toFixed(2)}`;
  }

  // Large values - use locale string with 2 decimals
  return `$${value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

/**
 * Format percentage values
 */
export function formatPercent(value: number, decimals: number = 1): string {
  return `${value >= 0 ? '+' : ''}${value.toFixed(decimals)}%`;
}

/**
 * Format large numbers with K/M/B suffixes
 */
export function formatCompact(value: number): string {
  if (value >= 1_000_000_000) {
    return `${(value / 1_000_000_000).toFixed(1)}B`;
  }
  if (value >= 1_000_000) {
    return `${(value / 1_000_000).toFixed(1)}M`;
  }
  if (value >= 1_000) {
    return `${(value / 1_000).toFixed(1)}K`;
  }
  return value.toLocaleString();
}
