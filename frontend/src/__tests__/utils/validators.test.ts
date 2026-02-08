/**
 * Unit tests for validator utilities
 */

import { describe, it, expect } from 'vitest';
import {
  isValidEthereumAddress,
  sanitizeAddress,
  isValidNumber,
  sanitizeNumber,
  isValidUrl
} from '../../utils/validators';

describe('validators', () => {
  describe('isValidEthereumAddress', () => {
    it('should validate correct addresses', () => {
      expect(isValidEthereumAddress('0x1234567890123456789012345678901234567890')).toBe(true);
    });

    it('should reject invalid addresses', () => {
      expect(isValidEthereumAddress('0x123')).toBe(false);
      expect(isValidEthereumAddress('invalid')).toBe(false);
      expect(isValidEthereumAddress('')).toBe(false);
    });
  });

  describe('sanitizeAddress', () => {
    it('should sanitize valid addresses', () => {
      const address = '0x1234567890123456789012345678901234567890';
      expect(sanitizeAddress(address)).toBe(address.toLowerCase());
    });

    it('should return null for invalid addresses', () => {
      expect(sanitizeAddress('invalid')).toBeNull();
      expect(sanitizeAddress('')).toBeNull();
    });
  });

  describe('isValidNumber', () => {
    it('should validate numbers in range', () => {
      expect(isValidNumber(5, 0, 10)).toBe(true);
      expect(isValidNumber(0, 0, 10)).toBe(true);
      expect(isValidNumber(10, 0, 10)).toBe(true);
    });

    it('should reject numbers out of range', () => {
      expect(isValidNumber(-1, 0, 10)).toBe(false);
      expect(isValidNumber(11, 0, 10)).toBe(false);
    });
  });

  describe('sanitizeNumber', () => {
    it('should sanitize valid numbers', () => {
      expect(sanitizeNumber(5, 0, 10, 0)).toBe(5);
      expect(sanitizeNumber('5', 0, 10, 0)).toBe(5);
    });

    it('should clamp numbers to range', () => {
      expect(sanitizeNumber(15, 0, 10, 0)).toBe(10);
      expect(sanitizeNumber(-5, 0, 10, 0)).toBe(0);
    });

    it('should return default for invalid input', () => {
      expect(sanitizeNumber(null, 0, 10, 5)).toBe(5);
      expect(sanitizeNumber('invalid', 0, 10, 5)).toBe(5);
    });
  });

  describe('isValidUrl', () => {
    it('should validate correct URLs', () => {
      expect(isValidUrl('https://example.com')).toBe(true);
      expect(isValidUrl('http://example.com')).toBe(true);
    });

    it('should reject invalid URLs', () => {
      expect(isValidUrl('not-a-url')).toBe(false);
      expect(isValidUrl('')).toBe(false);
    });
  });
});
