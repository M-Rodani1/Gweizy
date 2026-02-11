import { describe, it, expect, vi } from 'vitest';
import { getSecureRandomBytes, getSecureRandomHex } from '../../utils/secureRandom';

describe('Secure randomness', () => {
  it('uses crypto.getRandomValues for bytes', () => {
    const getRandomValues = vi.fn((array: Uint8Array) => {
      array.set([1, 2, 3]);
      return array;
    });

    const originalCrypto = globalThis.crypto;
    vi.stubGlobal('crypto', { getRandomValues } as Crypto);

    const bytes = getSecureRandomBytes(3);

    expect(getRandomValues).toHaveBeenCalled();
    expect(bytes).toEqual(new Uint8Array([1, 2, 3]));

    vi.stubGlobal('crypto', originalCrypto);
  });

  it('generates hex strings of expected length', () => {
    const getRandomValues = vi.fn((array: Uint8Array) => {
      array.set([255, 0, 16, 32]);
      return array;
    });

    const originalCrypto = globalThis.crypto;
    vi.stubGlobal('crypto', { getRandomValues } as Crypto);

    const hex = getSecureRandomHex(4);
    expect(hex).toBe('ff001020');

    vi.stubGlobal('crypto', originalCrypto);
  });

  it('throws when crypto is unavailable', () => {
    const originalCrypto = globalThis.crypto;
    vi.stubGlobal('crypto', undefined as unknown as Crypto);

    expect(() => getSecureRandomBytes(4)).toThrow('Secure random generator unavailable');

    vi.stubGlobal('crypto', originalCrypto);
  });
});
