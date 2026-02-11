import { describe, it, expect } from 'vitest';
import { isValidSignatureFormat, parseSignature, normalizeSignatureV } from '../../utils/signatures';

describe('Transaction signature utilities', () => {
  it('parses valid 65-byte signatures', () => {
    const sig = `0x${'a'.repeat(64)}${'b'.repeat(64)}1b`;
    const parsed = parseSignature(sig);

    expect(parsed).toBeTruthy();
    expect(parsed?.r).toBe(`0x${'a'.repeat(64)}`);
    expect(parsed?.s).toBe(`0x${'b'.repeat(64)}`);
    expect(parsed?.v).toBe(27);
  });

  it('normalizes recovery id values', () => {
    expect(normalizeSignatureV(0)).toBe(27);
    expect(normalizeSignatureV(1)).toBe(28);
    expect(normalizeSignatureV(27)).toBe(27);
  });

  it('validates signature format and recovery id', () => {
    const valid = `0x${'a'.repeat(64)}${'b'.repeat(64)}1c`;
    const invalid = `0x${'a'.repeat(64)}${'b'.repeat(64)}02`;

    expect(isValidSignatureFormat(valid)).toBe(true);
    expect(isValidSignatureFormat(invalid)).toBe(false);
  });
});
