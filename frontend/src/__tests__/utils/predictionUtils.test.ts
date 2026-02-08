/**
 * Unit tests for prediction utility functions
 */

import { describe, it, expect } from 'vitest';
import {
  getColorClasses,
  getTextColor,
  getClassificationState,
  Probabilities
} from '../../utils/predictionUtils';

describe('predictionUtils', () => {
  describe('getColorClasses', () => {
    it('should return red classes for red color', () => {
      expect(getColorClasses('red')).toBe('border-red-500 bg-red-500/10');
    });

    it('should return green classes for green color', () => {
      expect(getColorClasses('green')).toBe('border-green-500 bg-green-500/10');
    });

    it('should return yellow classes for yellow color', () => {
      expect(getColorClasses('yellow')).toBe('border-yellow-500 bg-yellow-500/10');
    });

    it('should return default gray classes for unknown color', () => {
      expect(getColorClasses('blue')).toBe('border-gray-600 bg-gray-800');
      expect(getColorClasses('')).toBe('border-gray-600 bg-gray-800');
    });
  });

  describe('getTextColor', () => {
    it('should return correct text color for red', () => {
      expect(getTextColor('red')).toBe('text-red-400');
    });

    it('should return correct text color for green', () => {
      expect(getTextColor('green')).toBe('text-green-400');
    });

    it('should return correct text color for yellow', () => {
      expect(getTextColor('yellow')).toBe('text-yellow-400');
    });

    it('should return default gray text for unknown color', () => {
      expect(getTextColor('purple')).toBe('text-gray-400');
      expect(getTextColor('')).toBe('text-gray-400');
    });
  });

  describe('getClassificationState', () => {
    it('should return null for undefined probabilities', () => {
      expect(getClassificationState(undefined)).toBeNull();
    });

    it('should return spike when urgent is highest and > 0.3', () => {
      const probs: Probabilities = { wait: 0.2, normal: 0.3, urgent: 0.5 };
      expect(getClassificationState(probs)).toBe('spike');
    });

    it('should return elevated when wait is highest and > 0.3', () => {
      const probs: Probabilities = { wait: 0.5, normal: 0.3, urgent: 0.2 };
      expect(getClassificationState(probs)).toBe('elevated');
    });

    it('should return normal when normal is highest', () => {
      const probs: Probabilities = { wait: 0.2, normal: 0.6, urgent: 0.2 };
      expect(getClassificationState(probs)).toBe('normal');
    });

    it('should return normal when wait is highest but <= 0.3', () => {
      const probs: Probabilities = { wait: 0.3, normal: 0.2, urgent: 0.2 };
      expect(getClassificationState(probs)).toBe('normal');
    });

    it('should return normal when urgent is highest but <= 0.3', () => {
      const probs: Probabilities = { wait: 0.2, normal: 0.2, urgent: 0.3 };
      expect(getClassificationState(probs)).toBe('normal');
    });

    it('should handle edge case with equal probabilities', () => {
      const probs: Probabilities = { wait: 0.33, normal: 0.33, urgent: 0.34 };
      expect(getClassificationState(probs)).toBe('spike');
    });
  });
});
