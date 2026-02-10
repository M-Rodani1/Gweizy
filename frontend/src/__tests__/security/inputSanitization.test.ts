import { describe, it, expect } from 'vitest';
import {
  sanitizeNumber,
  sanitizeWalletAddress,
  sanitizeTransactionHash,
  sanitizeString,
} from '../../utils/sanitize';

describe('input sanitization utilities', () => {
  it('sanitizes numbers with bounds and defaults', () => {
    expect(sanitizeNumber('42', 0, 0, 100)).toBe(42);
    expect(sanitizeNumber('invalid', 5)).toBe(5);
    expect(sanitizeNumber(-10, 0, 0, 100)).toBe(0);
    expect(sanitizeNumber(250, 0, 0, 100)).toBe(100);
  });

  it('validates wallet addresses', () => {
    expect(sanitizeWalletAddress('0x0000000000000000000000000000000000000000')).toBe(
      '0x0000000000000000000000000000000000000000'
    );
    expect(sanitizeWalletAddress('0xINVALID')).toBeNull();
  });

  it('validates transaction hashes', () => {
    const hash = '0x' + 'a'.repeat(64);
    expect(sanitizeTransactionHash(hash)).toBe(hash);
    expect(sanitizeTransactionHash('0x1234')).toBeNull();
  });

  it('removes scripts and javascript URLs from input', () => {
    const input = 'Hello <script>alert(1)</script> javascript:evil()';
    expect(sanitizeString(input)).toBe('Hello  evil()');
  });
});
