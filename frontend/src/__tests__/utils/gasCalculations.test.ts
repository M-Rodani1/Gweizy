/**
 * Unit tests for gas calculation utilities
 */

import { describe, it, expect } from 'vitest';
import { calculateGasCost, formatGasCost } from '../../utils/gasCalculations';

describe('gasCalculations', () => {
  describe('calculateGasCost', () => {
    it('should calculate gas cost for a swap transaction', () => {
      const result = calculateGasCost('swap', 30, 3000);

      expect(result.gasUnits).toBe(150000); // TX_GAS_ESTIMATES.swap
      expect(result.costEth).toBeCloseTo(0.0045); // (30 * 150000) / 1e9
      expect(result.costUsd).toBeCloseTo(13.5); // 0.0045 * 3000
    });

    it('should calculate gas cost for a transfer transaction', () => {
      const result = calculateGasCost('transfer', 20, 2500);

      expect(result.gasUnits).toBe(21000); // TX_GAS_ESTIMATES.transfer
      expect(result.costEth).toBeCloseTo(0.00042); // (20 * 21000) / 1e9
      expect(result.costUsd).toBeCloseTo(1.05); // 0.00042 * 2500
    });

    it('should calculate gas cost for an NFT mint', () => {
      const result = calculateGasCost('nftMint', 50, 3500);

      expect(result.gasUnits).toBe(100000); // TX_GAS_ESTIMATES.nftMint
      expect(result.costEth).toBeCloseTo(0.005); // (50 * 100000) / 1e9
      expect(result.costUsd).toBeCloseTo(17.5); // 0.005 * 3500
    });

    it('should handle zero gas price', () => {
      const result = calculateGasCost('swap', 0, 3000);

      expect(result.costEth).toBe(0);
      expect(result.costUsd).toBe(0);
    });

    it('should handle zero ETH price', () => {
      const result = calculateGasCost('swap', 30, 0);

      expect(result.costEth).toBeCloseTo(0.0045);
      expect(result.costUsd).toBe(0);
    });
  });

  describe('formatGasCost', () => {
    it('should format costs less than $0.01', () => {
      expect(formatGasCost(0.001)).toBe('<$0.01');
      expect(formatGasCost(0.009)).toBe('<$0.01');
    });

    it('should format costs between $0.01 and $1', () => {
      expect(formatGasCost(0.01)).toBe('$0.01');
      expect(formatGasCost(0.5)).toBe('$0.50');
      expect(formatGasCost(0.99)).toBe('$0.99');
    });

    it('should format costs $1 and above', () => {
      expect(formatGasCost(1)).toBe('$1.00');
      expect(formatGasCost(10.5)).toBe('$10.50');
      expect(formatGasCost(100.123)).toBe('$100.12');
    });

    it('should handle zero', () => {
      expect(formatGasCost(0)).toBe('<$0.01');
    });
  });
});
