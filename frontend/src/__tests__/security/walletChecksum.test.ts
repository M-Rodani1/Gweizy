import { describe, it, expect } from 'vitest';
import { isChecksumAddress, isValidAddress, toChecksumAddress } from '../../utils/walletAddress';

describe('Wallet address checksum validation', () => {
  it('validates checksum addresses per EIP-55', () => {
    const address = '0x52908400098527886E0F7030069857D2E4169EE7';

    expect(isValidAddress(address)).toBe(true);
    expect(isChecksumAddress(address)).toBe(true);
  });

  it('computes checksum casing from lowercase input', () => {
    const lower = '0x52908400098527886e0f7030069857d2e4169ee7';
    const checksummed = toChecksumAddress(lower);

    expect(checksummed).toBe('0x52908400098527886E0F7030069857D2E4169EE7');
  });

  it('rejects invalid checksums while still validating format', () => {
    const lower = '0x8617e340b3d01fa5f11f306f4090fd50e238070d';
    const checksummed = '0x8617E340B3D01FA5F11F306F4090FD50E238070D';

    expect(isValidAddress(lower)).toBe(true);
    expect(isChecksumAddress(lower)).toBe(false);
    expect(isChecksumAddress(checksummed)).toBe(true);
  });
});
