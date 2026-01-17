/**
 * Input validation and sanitization utilities
 * Ensures user inputs are safe and valid
 */

/**
 * Validate Ethereum address format
 * @param address - Address to validate
 * @returns true if valid Ethereum address
 */
export function isValidEthereumAddress(address: string): boolean {
  if (!address || typeof address !== 'string') {
    return false;
  }
  
  // Ethereum address format: 0x followed by 40 hex characters
  const ethAddressRegex = /^0x[a-fA-F0-9]{40}$/;
  return ethAddressRegex.test(address);
}

/**
 * Sanitize Ethereum address (lowercase and validate)
 * @param address - Address to sanitize
 * @returns Sanitized address or null if invalid
 */
export function sanitizeAddress(address: string): string | null {
  if (!address) return null;
  
  const trimmed = address.trim();
  if (!isValidEthereumAddress(trimmed)) {
    return null;
  }
  
  return trimmed.toLowerCase();
}

/**
 * Validate number is positive and within range
 * @param value - Value to validate
 * @param min - Minimum value (default: 0)
 * @param max - Maximum value (default: Infinity)
 * @returns true if valid
 */
export function isValidNumber(value: number, min: number = 0, max: number = Infinity): boolean {
  return typeof value === 'number' && 
         !isNaN(value) && 
         isFinite(value) && 
         value >= min && 
         value <= max;
}

/**
 * Sanitize number (clamp to valid range)
 * @param value - Value to sanitize
 * @param min - Minimum value
 * @param max - Maximum value
 * @param defaultValue - Default if invalid
 * @returns Sanitized number
 */
export function sanitizeNumber(
  value: number | string | null | undefined,
  min: number = 0,
  max: number = Infinity,
  defaultValue: number = 0
): number {
  if (value === null || value === undefined) {
    return defaultValue;
  }

  const num = typeof value === 'string' ? parseFloat(value) : value;

  // Check if it's a valid number (not NaN, is finite)
  if (typeof num !== 'number' || isNaN(num) || !isFinite(num)) {
    return defaultValue;
  }

  // Clamp to range
  return Math.max(min, Math.min(max, num));
}

/**
 * Validate URL format
 * @param url - URL to validate
 * @returns true if valid URL
 */
export function isValidUrl(url: string): boolean {
  try {
    new URL(url);
    return true;
  } catch {
    return false;
  }
}

/**
 * Sanitize string input (remove dangerous characters)
 * @param input - String to sanitize
 * @param maxLength - Maximum length
 * @returns Sanitized string
 */
export function sanitizeString(input: string, maxLength: number = 1000): string {
  if (!input || typeof input !== 'string') {
    return '';
  }
  
  // Remove null bytes and trim
  let sanitized = input.replace(/\0/g, '').trim();
  
  // Limit length
  if (sanitized.length > maxLength) {
    sanitized = sanitized.slice(0, maxLength);
  }
  
  return sanitized;
}
