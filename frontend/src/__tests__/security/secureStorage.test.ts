import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { secureStorage } from '../../utils/secureStorage';

describe('secureStorage', () => {
  beforeEach(() => {
    sessionStorage.clear();
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    sessionStorage.clear();
    secureStorage.clear();
  });

  it('stores and retrieves values with crypto', async () => {
    const subtle = {
      generateKey: vi.fn().mockResolvedValue('key'),
      encrypt: vi.fn().mockResolvedValue(new Uint8Array([1, 2, 3, 4]).buffer),
      decrypt: vi.fn().mockResolvedValue(new TextEncoder().encode('secret').buffer),
    };

    vi.stubGlobal('crypto', {
      subtle,
      getRandomValues: (array: Uint8Array) => {
        array.fill(1);
        return array;
      },
    } as Crypto);

    await secureStorage.setItem('token', 'secret');
    const value = await secureStorage.getItem('token');

    expect(subtle.generateKey).toHaveBeenCalled();
    expect(subtle.encrypt).toHaveBeenCalled();
    expect(subtle.decrypt).toHaveBeenCalled();
    expect(value).toBe('secret');
  });

  it('falls back to memory storage when crypto is unavailable', async () => {
    vi.stubGlobal('crypto', undefined);

    await secureStorage.setItem('token', 'memory');
    const value = await secureStorage.getItem('token');

    expect(value).toBe('memory');
  });
});
