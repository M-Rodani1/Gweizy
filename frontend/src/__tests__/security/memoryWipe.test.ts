import { describe, it, expect } from 'vitest';
import { wipeUint8Array, wipeString } from '../../utils/memoryWipe';

describe('Private key memory wiping', () => {
  it('zeroes out byte buffers', () => {
    const buffer = new Uint8Array([1, 2, 3, 4]);
    wipeUint8Array(buffer);

    expect(Array.from(buffer)).toEqual([0, 0, 0, 0]);
  });

  it('returns an empty string for sensitive string cleanup', () => {
    expect(wipeString()).toBe('');
  });
});
