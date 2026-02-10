/**
 * Fuzz Testing for Input Validation
 *
 * Tests input validation with random, edge case, and malicious inputs
 * to ensure robust handling of unexpected data.
 */

import { describe, it, expect } from 'vitest';

// ============================================================================
// Fuzz Test Data Generators
// ============================================================================

/**
 * Generate random string of specified length
 */
function randomString(length: number, charset = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'): string {
  let result = '';
  for (let i = 0; i < length; i++) {
    result += charset.charAt(Math.floor(Math.random() * charset.length));
  }
  return result;
}

/**
 * Generate random number in range
 */
function randomNumber(min: number, max: number): number {
  return Math.random() * (max - min) + min;
}

/**
 * Generate random integer in range
 */
function randomInt(min: number, max: number): number {
  return Math.floor(randomNumber(min, max + 1));
}

/**
 * Get random item from array
 */
function randomChoice<T>(items: T[]): T {
  return items[randomInt(0, items.length - 1)];
}

/**
 * Generate multiple random values
 */
function generateMany<T>(generator: () => T, count: number): T[] {
  return Array.from({ length: count }, generator);
}

// Edge case strings for fuzzing
const EDGE_CASE_STRINGS = [
  '',                           // Empty string
  ' ',                          // Single space
  '   ',                        // Multiple spaces
  '\t',                         // Tab
  '\n',                         // Newline
  '\r\n',                       // CRLF
  '\0',                         // Null byte
  'null',                       // String "null"
  'undefined',                  // String "undefined"
  'NaN',                        // String "NaN"
  'Infinity',                   // String "Infinity"
  '-Infinity',                  // String "-Infinity"
  'true',                       // String "true"
  'false',                      // String "false"
  '0',                          // String "0"
  '-1',                         // String "-1"
  '1e308',                      // Large exponent
  '1e-308',                     // Small exponent
  'a'.repeat(10000),            // Very long string
  '🔥'.repeat(100),             // Unicode emojis
  '你好世界',                   // Chinese characters
  'مرحبا',                      // Arabic (RTL)
  '    leading spaces',
  'trailing spaces    ',
  '<script>alert(1)</script>',  // XSS attempt
  '"><script>alert(1)</script>', // XSS with quote escape
  "'; DROP TABLE users; --",    // SQL injection
  '${process.env.SECRET}',      // Template injection
  '{{constructor.constructor("return this")()}}', // Prototype pollution
  '../../../etc/passwd',        // Path traversal
  'file:///etc/passwd',         // File URL
  'javascript:alert(1)',        // JavaScript URL
  'data:text/html,<script>alert(1)</script>', // Data URL
];

// Edge case numbers for fuzzing
const EDGE_CASE_NUMBERS = [
  0,
  -0,
  1,
  -1,
  0.1,
  -0.1,
  Number.MAX_VALUE,
  Number.MIN_VALUE,
  Number.MAX_SAFE_INTEGER,
  Number.MIN_SAFE_INTEGER,
  Number.POSITIVE_INFINITY,
  Number.NEGATIVE_INFINITY,
  NaN,
  1e308,
  1e-308,
  Math.PI,
  Math.E,
];

// ============================================================================
// Validation Functions to Test
// ============================================================================

/**
 * Validate wallet address format
 */
function isValidWalletAddress(address: unknown): boolean {
  if (typeof address !== 'string') return false;
  if (!address.startsWith('0x')) return false;
  if (address.length !== 42) return false;
  return /^0x[a-fA-F0-9]{40}$/.test(address);
}

/**
 * Validate gas price value
 */
function isValidGasPrice(price: unknown): boolean {
  if (typeof price !== 'number') return false;
  if (Number.isNaN(price)) return false;
  if (!Number.isFinite(price)) return false;
  if (price < 0) return false;
  if (price > 1000000) return false; // Reasonable upper bound in gwei
  return true;
}

/**
 * Validate chain ID
 */
function isValidChainId(chainId: unknown): boolean {
  if (typeof chainId !== 'number') return false;
  if (!Number.isInteger(chainId)) return false;
  if (chainId < 1) return false;
  if (chainId > Number.MAX_SAFE_INTEGER) return false;
  return true;
}

/**
 * Validate percentage value (0-100)
 */
function isValidPercentage(value: unknown): boolean {
  if (typeof value !== 'number') return false;
  if (Number.isNaN(value)) return false;
  if (value < 0) return false;
  if (value > 100) return false;
  return true;
}

/**
 * Validate email format
 */
function isValidEmail(email: unknown): boolean {
  if (typeof email !== 'string') return false;
  if (email.length === 0) return false;
  if (email.length > 254) return false;
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email);
}

/**
 * Validate URL format
 */
function isValidUrl(url: unknown): boolean {
  if (typeof url !== 'string') return false;
  try {
    const parsed = new URL(url);
    return ['http:', 'https:'].includes(parsed.protocol);
  } catch {
    return false;
  }
}

/**
 * Sanitize user input
 */
function sanitizeInput(input: unknown): string {
  if (typeof input !== 'string') return '';
  return input
    .replace(/[<>]/g, '') // Remove angle brackets
    .replace(/[\x00-\x1F\x7F]/g, '') // Remove control characters
    .trim()
    .slice(0, 1000); // Limit length
}

/**
 * Parse JSON safely
 */
function safeJsonParse<T>(json: unknown, defaultValue: T): T {
  if (typeof json !== 'string') return defaultValue;
  try {
    return JSON.parse(json) as T;
  } catch {
    return defaultValue;
  }
}

// ============================================================================
// Fuzz Tests
// ============================================================================

describe('Fuzz Testing - Input Validation', () => {
  describe('Wallet Address Validation', () => {
    it('should accept valid wallet addresses', () => {
      const validAddresses = [
        '0x1234567890123456789012345678901234567890',
        '0xABCDEF1234567890ABCDEF1234567890ABCDEF12',
        '0xabcdef1234567890abcdef1234567890abcdef12',
      ];

      validAddresses.forEach((addr) => {
        expect(isValidWalletAddress(addr)).toBe(true);
      });
    });

    it('should reject invalid wallet addresses', () => {
      const invalidAddresses = [
        '0x123',                                     // Too short
        '0x12345678901234567890123456789012345678901234', // Too long
        '1234567890123456789012345678901234567890',   // Missing 0x prefix
        '0xGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG', // Invalid hex
        '',                                           // Empty
        null,                                         // Null
        undefined,                                    // Undefined
        123,                                          // Number
        {},                                           // Object
      ];

      invalidAddresses.forEach((addr) => {
        expect(isValidWalletAddress(addr)).toBe(false);
      });
    });

    it('should handle fuzz inputs', () => {
      EDGE_CASE_STRINGS.forEach((input) => {
        const result = isValidWalletAddress(input);
        expect(typeof result).toBe('boolean');
      });

      // Random hex strings
      generateMany(() => randomString(42, '0123456789abcdefABCDEF'), 100).forEach((str) => {
        const result = isValidWalletAddress('0x' + str.slice(2));
        expect(typeof result).toBe('boolean');
      });
    });
  });

  describe('Gas Price Validation', () => {
    it('should accept valid gas prices', () => {
      const validPrices = [0, 0.001, 1, 10, 100.5, 999999];
      validPrices.forEach((price) => {
        expect(isValidGasPrice(price)).toBe(true);
      });
    });

    it('should reject invalid gas prices', () => {
      const invalidPrices = [
        -1,                            // Negative
        NaN,                           // NaN
        Infinity,                      // Infinity
        -Infinity,                     // -Infinity
        1000001,                       // Too high
        '10',                          // String
        null,                          // Null
        undefined,                     // Undefined
      ];

      invalidPrices.forEach((price) => {
        expect(isValidGasPrice(price)).toBe(false);
      });
    });

    it('should handle fuzz number inputs', () => {
      EDGE_CASE_NUMBERS.forEach((num) => {
        const result = isValidGasPrice(num);
        expect(typeof result).toBe('boolean');
      });

      // Random numbers
      generateMany(() => randomNumber(-1000, 2000000), 100).forEach((num) => {
        const result = isValidGasPrice(num);
        expect(typeof result).toBe('boolean');
      });
    });
  });

  describe('Chain ID Validation', () => {
    it('should accept valid chain IDs', () => {
      const validChainIds = [1, 8453, 137, 42161, 10, 56];
      validChainIds.forEach((id) => {
        expect(isValidChainId(id)).toBe(true);
      });
    });

    it('should reject invalid chain IDs', () => {
      const invalidIds = [0, -1, 1.5, NaN, Infinity, '1', null, undefined];
      invalidIds.forEach((id) => {
        expect(isValidChainId(id)).toBe(false);
      });
    });

    it('should handle fuzz integer inputs', () => {
      generateMany(() => randomInt(-1000, 1000000), 100).forEach((num) => {
        const result = isValidChainId(num);
        expect(typeof result).toBe('boolean');
      });
    });
  });

  describe('Percentage Validation', () => {
    it('should accept valid percentages', () => {
      const validPercentages = [0, 50, 100, 33.33, 0.01, 99.99];
      validPercentages.forEach((pct) => {
        expect(isValidPercentage(pct)).toBe(true);
      });
    });

    it('should reject invalid percentages', () => {
      const invalidPercentages = [-1, 101, NaN, Infinity, '50', null];
      invalidPercentages.forEach((pct) => {
        expect(isValidPercentage(pct)).toBe(false);
      });
    });

    it('should handle fuzz percentage inputs', () => {
      generateMany(() => randomNumber(-50, 150), 100).forEach((num) => {
        const result = isValidPercentage(num);
        expect(typeof result).toBe('boolean');
      });
    });
  });

  describe('Email Validation', () => {
    it('should accept valid emails', () => {
      const validEmails = [
        'test@example.com',
        'user.name@domain.org',
        'user+tag@example.co.uk',
      ];
      validEmails.forEach((email) => {
        expect(isValidEmail(email)).toBe(true);
      });
    });

    it('should reject invalid emails', () => {
      const invalidEmails = [
        '',
        'notanemail',
        '@nodomain',
        'noat.domain.com',
        'a'.repeat(255) + '@example.com', // Too long
      ];
      invalidEmails.forEach((email) => {
        expect(isValidEmail(email)).toBe(false);
      });
    });

    it('should handle fuzz email inputs', () => {
      EDGE_CASE_STRINGS.forEach((input) => {
        const result = isValidEmail(input);
        expect(typeof result).toBe('boolean');
      });

      // Random strings with @ symbol
      generateMany(() => randomString(5) + '@' + randomString(5) + '.' + randomString(3), 50).forEach((str) => {
        const result = isValidEmail(str);
        expect(typeof result).toBe('boolean');
      });
    });
  });

  describe('URL Validation', () => {
    it('should accept valid URLs', () => {
      const validUrls = [
        'https://example.com',
        'http://localhost:3000',
        'https://api.example.com/v1/endpoint?query=1',
      ];
      validUrls.forEach((url) => {
        expect(isValidUrl(url)).toBe(true);
      });
    });

    it('should reject invalid or dangerous URLs', () => {
      const invalidUrls = [
        '',
        'notaurl',
        'ftp://example.com',           // Wrong protocol
        'javascript:alert(1)',          // JavaScript URL
        'file:///etc/passwd',           // File URL
        'data:text/html,<script>',      // Data URL
      ];
      invalidUrls.forEach((url) => {
        expect(isValidUrl(url)).toBe(false);
      });
    });

    it('should handle fuzz URL inputs', () => {
      EDGE_CASE_STRINGS.forEach((input) => {
        const result = isValidUrl(input);
        expect(typeof result).toBe('boolean');
      });
    });
  });

  describe('Input Sanitization', () => {
    it('should sanitize dangerous input', () => {
      expect(sanitizeInput('<script>alert(1)</script>')).toBe('scriptalert(1)/script');
      expect(sanitizeInput('normal text')).toBe('normal text');
      expect(sanitizeInput('  trimmed  ')).toBe('trimmed');
    });

    it('should handle non-string input', () => {
      expect(sanitizeInput(null)).toBe('');
      expect(sanitizeInput(undefined)).toBe('');
      expect(sanitizeInput(123)).toBe('');
      expect(sanitizeInput({})).toBe('');
      expect(sanitizeInput([])).toBe('');
    });

    it('should limit output length', () => {
      const longInput = 'a'.repeat(10000);
      expect(sanitizeInput(longInput).length).toBeLessThanOrEqual(1000);
    });

    it('should remove control characters', () => {
      expect(sanitizeInput('hello\x00world')).toBe('helloworld');
      expect(sanitizeInput('test\x1Fvalue')).toBe('testvalue');
    });

    it('should handle all edge cases without crashing', () => {
      EDGE_CASE_STRINGS.forEach((input) => {
        const result = sanitizeInput(input);
        expect(typeof result).toBe('string');
        expect(result.length).toBeLessThanOrEqual(1000);
      });
    });
  });

  describe('Safe JSON Parsing', () => {
    it('should parse valid JSON', () => {
      expect(safeJsonParse('{"key": "value"}', {})).toEqual({ key: 'value' });
      expect(safeJsonParse('[1, 2, 3]', [])).toEqual([1, 2, 3]);
      expect(safeJsonParse('"string"', '')).toBe('string');
      expect(safeJsonParse('123', 0)).toBe(123);
    });

    it('should return default for invalid JSON', () => {
      expect(safeJsonParse('not json', 'default')).toBe('default');
      expect(safeJsonParse('{invalid}', {})).toEqual({});
      expect(safeJsonParse('', [])).toEqual([]);
    });

    it('should return default for non-string input', () => {
      expect(safeJsonParse(null, 'default')).toBe('default');
      expect(safeJsonParse(undefined, 'default')).toBe('default');
      expect(safeJsonParse(123, 'default')).toBe('default');
    });

    it('should handle edge case JSON strings', () => {
      EDGE_CASE_STRINGS.forEach((input) => {
        const result = safeJsonParse(input, null);
        // Should not throw
        expect(result === null || result !== undefined).toBe(true);
      });
    });
  });

  describe('Random Fuzz Tests', () => {
    it('should handle random strings in all validators', () => {
      const randomInputs = generateMany(() => randomString(randomInt(0, 100)), 100);

      randomInputs.forEach((input) => {
        // Each validator should return boolean and not throw
        expect(typeof isValidWalletAddress(input)).toBe('boolean');
        expect(typeof isValidGasPrice(input)).toBe('boolean');
        expect(typeof isValidChainId(input)).toBe('boolean');
        expect(typeof isValidPercentage(input)).toBe('boolean');
        expect(typeof isValidEmail(input)).toBe('boolean');
        expect(typeof isValidUrl(input)).toBe('boolean');
        expect(typeof sanitizeInput(input)).toBe('string');
      });
    });

    it('should handle mixed type fuzz inputs', () => {
      const mixedInputs = [
        ...EDGE_CASE_STRINGS,
        ...EDGE_CASE_NUMBERS,
        null,
        undefined,
        {},
        [],
        () => {},
        Symbol('test'),
        new Date(),
        /regex/,
      ];

      mixedInputs.forEach((input) => {
        // All validators should not throw
        expect(() => isValidWalletAddress(input)).not.toThrow();
        expect(() => isValidGasPrice(input)).not.toThrow();
        expect(() => isValidChainId(input)).not.toThrow();
        expect(() => isValidPercentage(input)).not.toThrow();
        expect(() => isValidEmail(input)).not.toThrow();
        expect(() => isValidUrl(input)).not.toThrow();
        expect(() => sanitizeInput(input)).not.toThrow();
        expect(() => safeJsonParse(input, null)).not.toThrow();
      });
    });
  });

  describe('Security-Focused Fuzz Tests', () => {
    const securityPayloads = [
      // XSS payloads
      '<script>alert(1)</script>',
      '<img src=x onerror=alert(1)>',
      '<svg/onload=alert(1)>',
      '"><script>alert(1)</script>',
      "'-alert(1)-'",
      '{{constructor.constructor("return this")()}}',

      // SQL injection payloads
      "'; DROP TABLE users; --",
      "' OR '1'='1",
      "1; SELECT * FROM users",
      "UNION SELECT * FROM passwords",

      // Command injection payloads
      '; ls -la',
      '| cat /etc/passwd',
      '`whoami`',
      '$(id)',

      // Path traversal payloads
      '../../../etc/passwd',
      '..\\..\\..\\windows\\system32',
      '/etc/shadow',
      'file:///etc/passwd',

      // Template injection payloads
      '${7*7}',
      '#{7*7}',
      '{{7*7}}',
      '${process.env.SECRET}',

      // Unicode shenanigans
      'test\u0000value',
      'test\u202Evalue', // RTL override
      '\uFEFFtest', // BOM
    ];

    it('should safely handle all security payloads in validators', () => {
      securityPayloads.forEach((payload) => {
        // Validators should return false for all malicious payloads
        expect(isValidWalletAddress(payload)).toBe(false);
        expect(isValidGasPrice(payload)).toBe(false);
        expect(isValidChainId(payload)).toBe(false);
        expect(isValidEmail(payload)).toBe(false);
        expect(isValidUrl(payload)).toBe(false);
      });
    });

    it('should sanitize all security payloads', () => {
      securityPayloads.forEach((payload) => {
        const sanitized = sanitizeInput(payload);
        // Sanitized output should not contain dangerous characters
        expect(sanitized).not.toContain('<');
        expect(sanitized).not.toContain('>');
        expect(sanitized).not.toContain('\x00');
      });
    });
  });
});
