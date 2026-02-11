import { describe, it, expect, vi } from 'vitest';
import { createCspNonce } from '../../utils/cspNonce';

describe('CSP nonce generation', () => {
  it('generates a base64 nonce with minimum length', () => {
    const nonce = createCspNonce(12);

    expect(nonce).toMatch(/^[A-Za-z0-9+/]+$/);
    expect(nonce.length).toBeGreaterThan(0);
  });

  it('falls back to Math.random when crypto is unavailable', () => {
    const originalCrypto = globalThis.crypto;
    vi.stubGlobal('crypto', undefined as unknown as Crypto);

    const nonce = createCspNonce(12);
    expect(nonce).toMatch(/^[A-Za-z0-9+/]+$/);

    vi.stubGlobal('crypto', originalCrypto);
  });
});
