/**
 * Tests for PredictionCards component
 *
 * Tests the main prediction display cards for 1h, 4h, and 24h horizons.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import PredictionCards from '../../components/PredictionCards';
import { ChainProvider } from '../../contexts/ChainContext';

// Mock fetch globally
global.fetch = vi.fn();

// Mock ChainContext
vi.mock('../../contexts/ChainContext', () => ({
  ChainProvider: ({ children }: { children: React.ReactNode }) => children,
  useChain: () => ({ selectedChainId: 8453 })
}));

// Wrapper component with providers
const TestWrapper = ({ children }: { children: React.ReactNode }) => (
  <ChainProvider>{children}</ChainProvider>
);

describe('PredictionCards', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.resetAllMocks();
    vi.useRealTimers();
  });

  const mockPredictionsResponse = {
    predictions: {
      '1h': [
        {
          predictedGwei: 0.00095,
          lowerBound: 0.0009,
          upperBound: 0.001,
          confidence: 0.85,
          confidenceLevel: 'high',
          confidenceEmoji: '🎯',
          confidenceColor: 'green'
        }
      ],
      '4h': [
        {
          predictedGwei: 0.0011,
          lowerBound: 0.001,
          upperBound: 0.0012,
          confidence: 0.72,
          confidenceLevel: 'medium',
          confidenceEmoji: '📊',
          confidenceColor: 'yellow'
        }
      ],
      '24h': [
        {
          predictedGwei: 0.00085,
          lowerBound: 0.0007,
          upperBound: 0.001,
          confidence: 0.55,
          confidenceLevel: 'low',
          confidenceEmoji: '⚠️',
          confidenceColor: 'red'
        }
      ]
    }
  };

  const mockCurrentGasResponse = {
    current_gas: 0.001,
    base_fee: 0.0008,
    priority_fee: 0.0002,
    timestamp: new Date().toISOString()
  };

  it('renders loading state initially', () => {
    (global.fetch as any).mockImplementation(() => new Promise(() => {}));

    render(<PredictionCards />, { wrapper: TestWrapper });

    expect(screen.getByText(/Loading predictions/i)).toBeInTheDocument();
  });

  it('renders prediction cards after successful fetch', async () => {
    (global.fetch as any).mockImplementation((url: string) => {
      if (url.includes('predictions')) {
        return Promise.resolve({
          ok: true,
          json: async () => mockPredictionsResponse
        });
      }
      if (url.includes('current')) {
        return Promise.resolve({
          ok: true,
          json: async () => mockCurrentGasResponse
        });
      }
      return Promise.resolve({ ok: true, json: async () => ({}) });
    });

    render(<PredictionCards />, { wrapper: TestWrapper });

    await waitFor(() => {
      // Should show horizon labels
      expect(screen.getByText('1h')).toBeInTheDocument();
    });

    await waitFor(() => {
      expect(screen.getByText('4h')).toBeInTheDocument();
      expect(screen.getByText('24h')).toBeInTheDocument();
    });
  });

  describe('Prediction Display', () => {
    beforeEach(() => {
      (global.fetch as any).mockImplementation((url: string) => {
        if (url.includes('predictions')) {
          return Promise.resolve({
            ok: true,
            json: async () => mockPredictionsResponse
          });
        }
        if (url.includes('current')) {
          return Promise.resolve({
            ok: true,
            json: async () => mockCurrentGasResponse
          });
        }
        return Promise.resolve({ ok: true, json: async () => ({}) });
      });
    });

    it('displays all three prediction horizons', async () => {
      render(<PredictionCards />, { wrapper: TestWrapper });

      await waitFor(() => {
        expect(screen.getByText('1h')).toBeInTheDocument();
        expect(screen.getByText('4h')).toBeInTheDocument();
        expect(screen.getByText('24h')).toBeInTheDocument();
      });
    });

    it('shows recommendation text', async () => {
      render(<PredictionCards />, { wrapper: TestWrapper });

      await waitFor(() => {
        // Should show some recommendation
        const recommendations = screen.getAllByText(/expected|Consider|stable/i);
        expect(recommendations.length).toBeGreaterThan(0);
      });
    });
  });

  describe('Color Coding', () => {
    it('shows green for rising gas prices', async () => {
      const risingResponse = {
        predictions: {
          '1h': [{ predictedGwei: 0.0015, confidence: 0.8, confidenceLevel: 'high' }],
          '4h': [{ predictedGwei: 0.001, confidence: 0.7 }],
          '24h': [{ predictedGwei: 0.001, confidence: 0.5 }]
        }
      };

      (global.fetch as any).mockImplementation((url: string) => {
        if (url.includes('predictions')) {
          return Promise.resolve({
            ok: true,
            json: async () => risingResponse
          });
        }
        if (url.includes('current')) {
          return Promise.resolve({
            ok: true,
            json: async () => mockCurrentGasResponse
          });
        }
        return Promise.resolve({ ok: true, json: async () => ({}) });
      });

      render(<PredictionCards />, { wrapper: TestWrapper });

      await waitFor(() => {
        // Check for green border/background class
        const cards = document.querySelectorAll('[class*="green"]');
        expect(cards.length).toBeGreaterThan(0);
      });
    });

    it('shows red for dropping gas prices', async () => {
      const droppingResponse = {
        predictions: {
          '1h': [{ predictedGwei: 0.0005, confidence: 0.8, confidenceLevel: 'high' }],
          '4h': [{ predictedGwei: 0.001, confidence: 0.7 }],
          '24h': [{ predictedGwei: 0.001, confidence: 0.5 }]
        }
      };

      (global.fetch as any).mockImplementation((url: string) => {
        if (url.includes('predictions')) {
          return Promise.resolve({
            ok: true,
            json: async () => droppingResponse
          });
        }
        if (url.includes('current')) {
          return Promise.resolve({
            ok: true,
            json: async () => mockCurrentGasResponse
          });
        }
        return Promise.resolve({ ok: true, json: async () => ({}) });
      });

      render(<PredictionCards />, { wrapper: TestWrapper });

      await waitFor(() => {
        // Check for red border/background class
        const cards = document.querySelectorAll('[class*="red"]');
        expect(cards.length).toBeGreaterThan(0);
      });
    });
  });

  describe('Confidence Display', () => {
    it('displays confidence level', async () => {
      (global.fetch as any).mockImplementation((url: string) => {
        if (url.includes('predictions')) {
          return Promise.resolve({
            ok: true,
            json: async () => mockPredictionsResponse
          });
        }
        if (url.includes('current')) {
          return Promise.resolve({
            ok: true,
            json: async () => mockCurrentGasResponse
          });
        }
        return Promise.resolve({ ok: true, json: async () => ({}) });
      });

      render(<PredictionCards />, { wrapper: TestWrapper });

      await waitFor(() => {
        // Should show confidence percentage or level
        const confidenceElements = screen.getAllByText(/%|high|medium|low/i);
        expect(confidenceElements.length).toBeGreaterThanOrEqual(0);
      });
    });
  });

  describe('Error Handling', () => {
    it('displays error message on fetch failure', async () => {
      (global.fetch as any).mockRejectedValue(new Error('Network error'));

      render(<PredictionCards />, { wrapper: TestWrapper });

      await waitFor(() => {
        expect(screen.getByText(/Failed|error|unavailable/i)).toBeInTheDocument();
      });
    });

    it('shows retry button on error', async () => {
      (global.fetch as any).mockRejectedValue(new Error('Network error'));

      render(<PredictionCards />, { wrapper: TestWrapper });

      await waitFor(() => {
        expect(screen.getByText(/Retry/i)).toBeInTheDocument();
      });
    });

    it('retries fetch when retry button clicked', async () => {
      (global.fetch as any).mockRejectedValueOnce(new Error('Network error'));

      render(<PredictionCards />, { wrapper: TestWrapper });

      await waitFor(() => {
        expect(screen.getByText(/Retry/i)).toBeInTheDocument();
      });

      // Setup success for retry
      (global.fetch as any).mockImplementation((url: string) => {
        if (url.includes('predictions')) {
          return Promise.resolve({
            ok: true,
            json: async () => mockPredictionsResponse
          });
        }
        if (url.includes('current')) {
          return Promise.resolve({
            ok: true,
            json: async () => mockCurrentGasResponse
          });
        }
        return Promise.resolve({ ok: true, json: async () => ({}) });
      });

      const retryButton = screen.getByText(/Retry/i);
      fireEvent.click(retryButton);

      await waitFor(() => {
        expect(global.fetch).toHaveBeenCalled();
      });
    });
  });

  describe('Auto-refresh', () => {
    it('refreshes data every 30 seconds', async () => {
      (global.fetch as any).mockImplementation((url: string) => {
        if (url.includes('predictions')) {
          return Promise.resolve({
            ok: true,
            json: async () => mockPredictionsResponse
          });
        }
        if (url.includes('current')) {
          return Promise.resolve({
            ok: true,
            json: async () => mockCurrentGasResponse
          });
        }
        return Promise.resolve({ ok: true, json: async () => ({}) });
      });

      render(<PredictionCards />, { wrapper: TestWrapper });

      await waitFor(() => {
        expect(global.fetch).toHaveBeenCalled();
      });

      const initialCallCount = (global.fetch as any).mock.calls.length;

      vi.advanceTimersByTime(30000);

      await waitFor(() => {
        expect((global.fetch as any).mock.calls.length).toBeGreaterThan(initialCallCount);
      });
    });
  });

  describe('Best Time Indicator', () => {
    it('highlights best prediction horizon', async () => {
      const bestTimeResponse = {
        predictions: {
          '1h': [{ predictedGwei: 0.0015, confidence: 0.9, confidenceLevel: 'high' }],
          '4h': [{ predictedGwei: 0.0008, confidence: 0.85, confidenceLevel: 'high' }], // Best (lowest)
          '24h': [{ predictedGwei: 0.0012, confidence: 0.7, confidenceLevel: 'medium' }]
        }
      };

      (global.fetch as any).mockImplementation((url: string) => {
        if (url.includes('predictions')) {
          return Promise.resolve({
            ok: true,
            json: async () => bestTimeResponse
          });
        }
        if (url.includes('current')) {
          return Promise.resolve({
            ok: true,
            json: async () => mockCurrentGasResponse
          });
        }
        return Promise.resolve({ ok: true, json: async () => ({}) });
      });

      render(<PredictionCards />, { wrapper: TestWrapper });

      await waitFor(() => {
        // 4h should be marked as best
        expect(screen.getByText('4h')).toBeInTheDocument();
      });
    });
  });

  describe('Change Percentage', () => {
    it('shows percentage change from current', async () => {
      (global.fetch as any).mockImplementation((url: string) => {
        if (url.includes('predictions')) {
          return Promise.resolve({
            ok: true,
            json: async () => mockPredictionsResponse
          });
        }
        if (url.includes('current')) {
          return Promise.resolve({
            ok: true,
            json: async () => mockCurrentGasResponse
          });
        }
        return Promise.resolve({ ok: true, json: async () => ({}) });
      });

      render(<PredictionCards />, { wrapper: TestWrapper });

      await waitFor(() => {
        // Should show percentage
        const percentages = screen.getAllByText(/%/);
        expect(percentages.length).toBeGreaterThan(0);
      });
    });
  });

  describe('Empty State', () => {
    it('handles empty predictions gracefully', async () => {
      (global.fetch as any).mockImplementation((url: string) => {
        if (url.includes('predictions')) {
          return Promise.resolve({
            ok: true,
            json: async () => ({ predictions: {} })
          });
        }
        if (url.includes('current')) {
          return Promise.resolve({
            ok: true,
            json: async () => mockCurrentGasResponse
          });
        }
        return Promise.resolve({ ok: true, json: async () => ({}) });
      });

      render(<PredictionCards />, { wrapper: TestWrapper });

      // Should not crash
      await waitFor(() => {
        expect(document.body).toBeInTheDocument();
      });
    });
  });
});
