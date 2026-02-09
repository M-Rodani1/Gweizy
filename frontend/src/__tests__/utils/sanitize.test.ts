/**
 * Tests for sanitization utilities
 */

import { describe, it, expect } from 'vitest';
import {
  escapeHtml,
  sanitizeString,
  sanitizeObject,
  sanitizeWalletAddress,
  sanitizeTransactionHash,
  sanitizeNumber,
  sanitizeUrl,
  sanitizeSearchQuery,
} from '../../utils/sanitize';

describe('escapeHtml', () => {
  it('should escape HTML special characters', () => {
    expect(escapeHtml('<script>')).toBe('&lt;script&gt;');
    expect(escapeHtml('"quotes"')).toBe('&quot;quotes&quot;');
    expect(escapeHtml("'apostrophe'")).toBe('&#x27;apostrophe&#x27;');
    expect(escapeHtml('a & b')).toBe('a &amp; b');
  });

  it('should handle empty strings', () => {
    expect(escapeHtml('')).toBe('');
  });

  it('should return empty string for non-strings', () => {
    expect(escapeHtml(null as unknown as string)).toBe('');
    expect(escapeHtml(undefined as unknown as string)).toBe('');
    expect(escapeHtml(123 as unknown as string)).toBe('');
  });

  it('should handle strings with no special characters', () => {
    expect(escapeHtml('Hello World')).toBe('Hello World');
  });
});

describe('sanitizeString', () => {
  it('should remove script tags', () => {
    expect(sanitizeString('<script>alert("xss")</script>')).toBe('');
    expect(sanitizeString('Hello<script>evil()</script>World')).toBe('HelloWorld');
  });

  it('should remove event handlers', () => {
    expect(sanitizeString('<div onclick="evil()">Click</div>')).toBe('<div>Click</div>');
    expect(sanitizeString('<img onerror="evil()" src="x">')).toBe('<img src="x">');
  });

  it('should remove javascript: URLs', () => {
    expect(sanitizeString('javascript:alert(1)')).toBe('alert(1)');
    expect(sanitizeString('JAVASCRIPT:evil()')).toBe('evil()');
  });

  it('should remove data: URLs', () => {
    // data: is removed, then script tags are also removed
    expect(sanitizeString('data:text/html,<script>evil()</script>')).toBe('text/html,');
  });

  it('should handle empty strings', () => {
    expect(sanitizeString('')).toBe('');
  });

  it('should return empty string for non-strings', () => {
    expect(sanitizeString(null as unknown as string)).toBe('');
  });

  it('should trim whitespace', () => {
    expect(sanitizeString('  hello  ')).toBe('hello');
  });
});

describe('sanitizeObject', () => {
  it('should sanitize string values', () => {
    const input = { name: '<script>evil()</script>John' };
    const result = sanitizeObject(input);
    expect(result.name).toBe('John');
  });

  it('should preserve non-string values', () => {
    const input = { age: 25, active: true, score: null };
    const result = sanitizeObject(input);
    expect(result).toEqual({ age: 25, active: true, score: null });
  });

  it('should handle nested objects', () => {
    const input = {
      user: {
        name: '<b>John</b>',
        email: 'john@example.com',
      },
    };
    const result = sanitizeObject(input);
    expect(result.user.name).toBe('<b>John</b>');
    expect(result.user.email).toBe('john@example.com');
  });

  it('should handle arrays', () => {
    const input = { tags: ['<script>evil</script>', 'safe'] };
    const result = sanitizeObject(input);
    expect(result.tags).toEqual(['', 'safe']);
  });

  it('should return non-objects as-is', () => {
    expect(sanitizeObject(null as unknown as Record<string, unknown>)).toBeNull();
  });
});

describe('sanitizeWalletAddress', () => {
  it('should accept valid Ethereum addresses', () => {
    const validAddress = '0x1234567890123456789012345678901234567890';
    expect(sanitizeWalletAddress(validAddress)).toBe(validAddress.toLowerCase());
  });

  it('should normalize to lowercase', () => {
    const address = '0xABCDEF1234567890123456789012345678901234';
    expect(sanitizeWalletAddress(address)).toBe(address.toLowerCase());
  });

  it('should reject invalid addresses', () => {
    expect(sanitizeWalletAddress('invalid')).toBeNull();
    expect(sanitizeWalletAddress('0x123')).toBeNull();
    expect(sanitizeWalletAddress('0xGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG')).toBeNull();
    expect(sanitizeWalletAddress('<script>alert(1)</script>')).toBeNull();
  });

  it('should trim whitespace', () => {
    const address = '  0x1234567890123456789012345678901234567890  ';
    expect(sanitizeWalletAddress(address)).toBe('0x1234567890123456789012345678901234567890');
  });

  it('should return null for non-strings', () => {
    expect(sanitizeWalletAddress(null as unknown as string)).toBeNull();
    expect(sanitizeWalletAddress(123 as unknown as string)).toBeNull();
  });
});

describe('sanitizeTransactionHash', () => {
  it('should accept valid transaction hashes', () => {
    const validHash = '0x' + 'a'.repeat(64);
    expect(sanitizeTransactionHash(validHash)).toBe(validHash);
  });

  it('should normalize to lowercase', () => {
    const hash = '0x' + 'A'.repeat(64);
    expect(sanitizeTransactionHash(hash)).toBe(hash.toLowerCase());
  });

  it('should reject invalid hashes', () => {
    expect(sanitizeTransactionHash('invalid')).toBeNull();
    expect(sanitizeTransactionHash('0x123')).toBeNull();
    expect(sanitizeTransactionHash('0x' + 'g'.repeat(64))).toBeNull();
  });

  it('should return null for non-strings', () => {
    expect(sanitizeTransactionHash(null as unknown as string)).toBeNull();
  });
});

describe('sanitizeNumber', () => {
  it('should parse valid numbers', () => {
    expect(sanitizeNumber('42')).toBe(42);
    expect(sanitizeNumber('3.14')).toBe(3.14);
    expect(sanitizeNumber(100)).toBe(100);
  });

  it('should return default for invalid input', () => {
    expect(sanitizeNumber('invalid', 0)).toBe(0);
    expect(sanitizeNumber(NaN, 5)).toBe(5);
    expect(sanitizeNumber(Infinity, 0)).toBe(0);
    expect(sanitizeNumber(null, 10)).toBe(10);
  });

  it('should enforce min/max bounds', () => {
    expect(sanitizeNumber(50, 0, 0, 100)).toBe(50);
    expect(sanitizeNumber(-10, 0, 0, 100)).toBe(0);
    expect(sanitizeNumber(150, 0, 0, 100)).toBe(100);
  });

  it('should handle edge cases', () => {
    expect(sanitizeNumber('0', 1)).toBe(0);
    expect(sanitizeNumber('-5', 0, -10)).toBe(-5);
  });
});

describe('sanitizeUrl', () => {
  it('should allow safe URLs', () => {
    expect(sanitizeUrl('https://example.com')).toBe('https://example.com');
    expect(sanitizeUrl('http://example.com')).toBe('http://example.com');
    expect(sanitizeUrl('/path/to/page')).toBe('/path/to/page');
    expect(sanitizeUrl('#anchor')).toBe('#anchor');
  });

  it('should reject dangerous protocols', () => {
    expect(sanitizeUrl('javascript:alert(1)')).toBeNull();
    expect(sanitizeUrl('JAVASCRIPT:evil()')).toBeNull();
    expect(sanitizeUrl('data:text/html,<script>')).toBeNull();
    expect(sanitizeUrl('vbscript:evil')).toBeNull();
    expect(sanitizeUrl('file:///etc/passwd')).toBeNull();
  });

  it('should handle edge cases', () => {
    expect(sanitizeUrl('  https://example.com  ')).toBe('https://example.com');
    expect(sanitizeUrl('')).toBe('');
    expect(sanitizeUrl(null as unknown as string)).toBeNull();
  });

  it('should allow relative URLs without protocol', () => {
    expect(sanitizeUrl('path/to/resource')).toBe('path/to/resource');
  });
});

describe('sanitizeSearchQuery', () => {
  it('should trim and normalize whitespace', () => {
    expect(sanitizeSearchQuery('  hello   world  ')).toBe('hello world');
  });

  it('should remove angle brackets', () => {
    expect(sanitizeSearchQuery('search <query>')).toBe('search query');
  });

  it('should enforce max length', () => {
    const longQuery = 'a'.repeat(300);
    expect(sanitizeSearchQuery(longQuery, 100).length).toBe(100);
  });

  it('should handle empty input', () => {
    expect(sanitizeSearchQuery('')).toBe('');
    expect(sanitizeSearchQuery(null as unknown as string)).toBe('');
  });

  it('should preserve valid search terms', () => {
    expect(sanitizeSearchQuery('0x1234abcd')).toBe('0x1234abcd');
    expect(sanitizeSearchQuery('gas price ethereum')).toBe('gas price ethereum');
  });
});
