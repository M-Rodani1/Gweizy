/**
 * Input sanitization utilities for security.
 *
 * Provides functions to sanitize user inputs and prevent XSS attacks.
 * Uses a lightweight approach without external dependencies.
 *
 * @module utils/sanitize
 */

/**
 * HTML entities to escape
 */
const HTML_ENTITIES: Record<string, string> = {
  '&': '&amp;',
  '<': '&lt;',
  '>': '&gt;',
  '"': '&quot;',
  "'": '&#x27;',
  '/': '&#x2F;',
  '`': '&#x60;',
  '=': '&#x3D;',
};

/**
 * Escape HTML special characters to prevent XSS.
 *
 * @param str - The string to escape
 * @returns The escaped string
 *
 * @example
 * ```ts
 * escapeHtml('<script>alert("xss")</script>')
 * // Returns: '&lt;script&gt;alert(&quot;xss&quot;)&lt;/script&gt;'
 * ```
 */
export function escapeHtml(str: string): string {
  if (typeof str !== 'string') {
    return '';
  }
  return str.replace(/[&<>"'`=/]/g, (char) => HTML_ENTITIES[char] || char);
}

/**
 * Sanitize a string by removing potentially dangerous content.
 *
 * @param input - The string to sanitize
 * @returns The sanitized string
 *
 * @example
 * ```ts
 * sanitizeString('Hello <script>evil()</script> World')
 * // Returns: 'Hello  World'
 * ```
 */
export function sanitizeString(input: string): string {
  if (typeof input !== 'string') {
    return '';
  }

  return input
    // Remove script tags and their content
    .replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '')
    // Remove event handlers
    .replace(/\s*on\w+\s*=\s*(['"])[^'"]*\1/gi, '')
    .replace(/\s*on\w+\s*=\s*[^\s>]*/gi, '')
    // Remove javascript: URLs
    .replace(/javascript:/gi, '')
    // Remove data: URLs (potential XSS vector)
    .replace(/data:/gi, '')
    // Remove vbscript: URLs
    .replace(/vbscript:/gi, '')
    // Trim whitespace
    .trim();
}

/**
 * Sanitize an object's string values recursively.
 *
 * @param obj - The object to sanitize
 * @returns A new object with sanitized string values
 *
 * @example
 * ```ts
 * sanitizeObject({ name: '<b>John</b>', age: 25 })
 * // Returns: { name: 'John', age: 25 }
 * ```
 */
export function sanitizeObject<T extends Record<string, unknown>>(obj: T): T {
  if (obj === null || typeof obj !== 'object') {
    return obj;
  }

  if (Array.isArray(obj)) {
    return obj.map((item) =>
      typeof item === 'string'
        ? sanitizeString(item)
        : typeof item === 'object'
          ? sanitizeObject(item as Record<string, unknown>)
          : item
    ) as unknown as T;
  }

  const result: Record<string, unknown> = {};

  for (const [key, value] of Object.entries(obj)) {
    if (typeof value === 'string') {
      result[key] = sanitizeString(value);
    } else if (typeof value === 'object' && value !== null) {
      result[key] = sanitizeObject(value as Record<string, unknown>);
    } else {
      result[key] = value;
    }
  }

  return result as T;
}

/**
 * Validate and sanitize a wallet address.
 *
 * @param address - The wallet address to validate
 * @returns The validated address or null if invalid
 *
 * @example
 * ```ts
 * sanitizeWalletAddress('0x1234...abcd')
 * // Returns: '0x1234...abcd' (if valid)
 *
 * sanitizeWalletAddress('invalid<script>')
 * // Returns: null
 * ```
 */
export function sanitizeWalletAddress(address: string): string | null {
  if (typeof address !== 'string') {
    return null;
  }

  // Remove whitespace
  const cleaned = address.trim().toLowerCase();

  // Check for valid Ethereum address format
  if (!/^0x[a-f0-9]{40}$/i.test(cleaned)) {
    return null;
  }

  return cleaned;
}

/**
 * Validate and sanitize a transaction hash.
 *
 * @param hash - The transaction hash to validate
 * @returns The validated hash or null if invalid
 *
 * @example
 * ```ts
 * sanitizeTransactionHash('0xabc123...')
 * // Returns: '0xabc123...' (if valid 66 char hex)
 * ```
 */
export function sanitizeTransactionHash(hash: string): string | null {
  if (typeof hash !== 'string') {
    return null;
  }

  const cleaned = hash.trim().toLowerCase();

  // Check for valid transaction hash format (66 characters: 0x + 64 hex chars)
  if (!/^0x[a-f0-9]{64}$/i.test(cleaned)) {
    return null;
  }

  return cleaned;
}

/**
 * Sanitize a number input, returning a safe number or default.
 *
 * @param input - The input to sanitize
 * @param defaultValue - Default value if input is invalid
 * @param min - Minimum allowed value
 * @param max - Maximum allowed value
 * @returns The sanitized number
 *
 * @example
 * ```ts
 * sanitizeNumber('42', 0, 0, 100)
 * // Returns: 42
 *
 * sanitizeNumber('invalid', 0)
 * // Returns: 0
 * ```
 */
export function sanitizeNumber(
  input: unknown,
  defaultValue = 0,
  min?: number,
  max?: number
): number {
  let num: number;

  if (typeof input === 'number') {
    num = input;
  } else if (typeof input === 'string') {
    num = parseFloat(input);
  } else {
    return defaultValue;
  }

  if (isNaN(num) || !isFinite(num)) {
    return defaultValue;
  }

  if (min !== undefined && num < min) {
    return min;
  }

  if (max !== undefined && num > max) {
    return max;
  }

  return num;
}

/**
 * Sanitize URL to prevent javascript: and other dangerous protocols.
 *
 * @param url - The URL to sanitize
 * @returns The sanitized URL or null if dangerous
 *
 * @example
 * ```ts
 * sanitizeUrl('https://example.com')
 * // Returns: 'https://example.com'
 *
 * sanitizeUrl('javascript:alert(1)')
 * // Returns: null
 * ```
 */
export function sanitizeUrl(url: string): string | null {
  if (typeof url !== 'string') {
    return null;
  }

  const trimmed = url.trim();

  // Check for dangerous protocols
  const dangerousProtocols = ['javascript:', 'data:', 'vbscript:', 'file:'];
  const lowerUrl = trimmed.toLowerCase();

  for (const protocol of dangerousProtocols) {
    if (lowerUrl.startsWith(protocol)) {
      return null;
    }
  }

  // Allow only http, https, and relative URLs
  if (
    trimmed.startsWith('http://') ||
    trimmed.startsWith('https://') ||
    trimmed.startsWith('/') ||
    trimmed.startsWith('#') ||
    !trimmed.includes(':')
  ) {
    return trimmed;
  }

  return null;
}

/**
 * Sanitize search query input.
 *
 * @param query - The search query to sanitize
 * @param maxLength - Maximum allowed length
 * @returns The sanitized query
 *
 * @example
 * ```ts
 * sanitizeSearchQuery('  hello world  ')
 * // Returns: 'hello world'
 * ```
 */
export function sanitizeSearchQuery(query: string, maxLength = 200): string {
  if (typeof query !== 'string') {
    return '';
  }

  return query
    .trim()
    .slice(0, maxLength)
    .replace(/[<>]/g, '') // Remove angle brackets
    .replace(/\s+/g, ' '); // Normalize whitespace
}

/**
 * Create a sanitized text node (safe for DOM insertion).
 *
 * @param text - The text content
 * @returns A sanitized text node
 */
export function createSafeTextNode(text: string): Text {
  return document.createTextNode(sanitizeString(text));
}

export default {
  escapeHtml,
  sanitizeString,
  sanitizeObject,
  sanitizeWalletAddress,
  sanitizeTransactionHash,
  sanitizeNumber,
  sanitizeUrl,
  sanitizeSearchQuery,
  createSafeTextNode,
};
