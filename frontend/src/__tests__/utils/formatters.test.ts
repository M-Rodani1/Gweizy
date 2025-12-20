/**
 * Unit tests for formatter utilities
 */

import { describe, it, expect } from 'vitest';
import {
  formatGasPrice,
  formatUSD,
  formatPercent,
  formatTimestamp,
  formatRelativeTime,
  formatHour,
  formatAddress,
  formatLargeNumber
} from '../../utils/formatters';

describe('formatters', () => {
  describe('formatGasPrice', () => {
    it('should format gas price correctly', () => {
      expect(formatGasPrice(0.001)).toBe('1.000 gwei');
      expect(formatGasPrice(0.0025)).toBe('2.500 gwei');
    });

    it('should handle null/undefined', () => {
      expect(formatGasPrice(null)).toBe('N/A');
      expect(formatGasPrice(undefined)).toBe('N/A');
    });

    it('should handle custom decimals', () => {
      expect(formatGasPrice(0.001, 2)).toBe('1.00 gwei');
    });
  });

  describe('formatUSD', () => {
    it('should format USD correctly', () => {
      expect(formatUSD(10.5)).toBe('$10.50');
      expect(formatUSD(100)).toBe('$100.00');
    });

    it('should handle null/undefined', () => {
      expect(formatUSD(null)).toBe('$0.00');
      expect(formatUSD(undefined)).toBe('$0.00');
    });
  });

  describe('formatPercent', () => {
    it('should format percentage correctly', () => {
      expect(formatPercent(50)).toBe('50.0%');
      expect(formatPercent(25.5)).toBe('25.5%');
    });

    it('should handle null/undefined', () => {
      expect(formatPercent(null)).toBe('0%');
      expect(formatPercent(undefined)).toBe('0%');
    });
  });

  describe('formatAddress', () => {
    it('should truncate address correctly', () => {
      const address = '0x1234567890123456789012345678901234567890';
      expect(formatAddress(address)).toBe('0x1234...7890');
    });

    it('should handle short addresses', () => {
      expect(formatAddress('0x123')).toBe('0x123');
    });
  });

  describe('formatLargeNumber', () => {
    it('should format large numbers with suffixes', () => {
      expect(formatLargeNumber(1000)).toBe('1.00K');
      expect(formatLargeNumber(1000000)).toBe('1.00M');
      expect(formatLargeNumber(1000000000)).toBe('1.00B');
    });
  });
});
