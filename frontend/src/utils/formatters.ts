/**
 * Reusable formatting utility functions
 * Centralized formatting logic to avoid duplication
 */

/**
 * Format gas price from gwei to display string
 * @param gwei - Gas price in gwei
 * @param decimals - Number of decimal places (default: 3)
 * @returns Formatted string with "gwei" suffix
 */
export function formatGasPrice(gwei: number | null | undefined, decimals: number = 3): string {
  if (gwei === null || gwei === undefined || isNaN(gwei)) {
    return 'N/A';
  }
  return `${(gwei * 1000).toFixed(decimals)} gwei`;
}

/**
 * Format USD amount
 * @param amount - Amount in USD
 * @param decimals - Number of decimal places (default: 2)
 * @returns Formatted string with "$" prefix
 */
export function formatUSD(amount: number | null | undefined, decimals: number = 2): string {
  if (amount === null || amount === undefined || isNaN(amount)) {
    return '$0.00';
  }
  return `$${amount.toFixed(decimals)}`;
}

/**
 * Format percentage
 * @param value - Percentage value (0-100)
 * @param decimals - Number of decimal places (default: 1)
 * @returns Formatted string with "%" suffix
 */
export function formatPercent(value: number | null | undefined, decimals: number = 1): string {
  if (value === null || value === undefined || isNaN(value)) {
    return '0%';
  }
  return `${value.toFixed(decimals)}%`;
}

/**
 * Format timestamp to readable date/time string
 * @param timestamp - Unix timestamp in seconds
 * @returns Formatted date/time string
 */
export function formatTimestamp(timestamp: number): string {
  if (!timestamp) return 'N/A';
  const date = new Date(timestamp * 1000);
  return date.toLocaleString();
}

/**
 * Format relative time (e.g., "2 minutes ago")
 * @param timestamp - Unix timestamp in seconds
 * @returns Relative time string
 */
export function formatRelativeTime(timestamp: number): string {
  if (!timestamp) return 'Unknown';
  
  const now = Math.floor(Date.now() / 1000);
  const diff = now - timestamp;
  
  if (diff < 60) return 'just now';
  if (diff < 3600) return `${Math.floor(diff / 60)} minutes ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)} hours ago`;
  return `${Math.floor(diff / 86400)} days ago`;
}

/**
 * Format hour with leading zero
 * @param hour - Hour (0-23)
 * @returns Formatted hour string (00-23)
 */
export function formatHour(hour: number): string {
  return hour.toString().padStart(2, '0');
}

/**
 * Format wallet address (truncate middle)
 * @param address - Full wallet address
 * @param startChars - Number of characters at start (default: 6)
 * @param endChars - Number of characters at end (default: 4)
 * @returns Truncated address string
 */
export function formatAddress(address: string, startChars: number = 6, endChars: number = 4): string {
  if (!address || address.length < startChars + endChars) {
    return address;
  }
  return `${address.slice(0, startChars)}...${address.slice(-endChars)}`;
}

/**
 * Format large numbers with K/M/B suffixes
 * @param num - Number to format
 * @returns Formatted string
 */
export function formatLargeNumber(num: number): string {
  if (num >= 1e9) return `${(num / 1e9).toFixed(2)}B`;
  if (num >= 1e6) return `${(num / 1e6).toFixed(2)}M`;
  if (num >= 1e3) return `${(num / 1e3).toFixed(2)}K`;
  return num.toFixed(0);
}
