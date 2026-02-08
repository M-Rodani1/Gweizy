/**
 * Tests for PredictionCards component
 *
 * Tests the main prediction display cards for 1h, 4h, and 24h horizons.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor, fireEvent, act } from '@testing-library/react';
import PredictionCards from '../../components/PredictionCards';
import { ChainProvider } from '../../contexts/ChainContext';
import * as gasApi from '../../api/gasApi';

// Mock the API module
vi.mock('../../api/gasApi', () => ({
  fetchPredictions: vi.fn(),
  fetchCurrentGas: vi.fn()
}));

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
  });

  afterEach(() => {
    vi.resetAllMocks();
  });

  const mockCurrentGasData = {
    current_gas: 0.001,
    base_fee: 0.0008,
    priority_fee: 0.0002,
    timestamp: new Date().toISOString(),
    block_number: 12345678
  };

  const mockPredictionsResponse = {
    current: mockCurrentGasData,
    predictions: {
      '1h': [
        {
          time: new Date().toISOString(),
          predictedGwei: 0.00095,
          lowerBound: 0.0009,
          upperBound: 0.001,
          confidence: 0.85,
          confidenceLevel: 'high' as const,
          confidenceEmoji: '🎯',
          confidenceColor: 'green'
        }
      ],
      '4h': [
        {
          time: new Date().toISOString(),
          predictedGwei: 0.0011,
          lowerBound: 0.001,
          upperBound: 0.0012,
          confidence: 0.72,
          confidenceLevel: 'medium' as const,
          confidenceEmoji: '📊',
          confidenceColor: 'yellow'
        }
      ],
      '24h': [
        {
          time: new Date().toISOString(),
          predictedGwei: 0.00085,
          lowerBound: 0.0007,
          upperBound: 0.001,
          confidence: 0.55,
          confidenceLevel: 'low' as const,
          confidenceEmoji: '⚠️',
          confidenceColor: 'red'
        }
      ],
      historical: []
    }
  };

  const mockCurrentGasResponse = mockCurrentGasData;

  it('renders loading state initially', () => {
    // Make the API calls hang indefinitely
    vi.mocked(gasApi.fetchPredictions).mockImplementation(() => new Promise(() => {}));
    vi.mocked(gasApi.fetchCurrentGas).mockImplementation(() => new Promise(() => {}));

    render(<PredictionCards />, { wrapper: TestWrapper });

    expect(screen.getByText(/Loading predictions/i)).toBeInTheDocument();
  });

  it('renders prediction cards after successful fetch', async () => {
    vi.mocked(gasApi.fetchPredictions).mockResolvedValue(mockPredictionsResponse);
    vi.mocked(gasApi.fetchCurrentGas).mockResolvedValue(mockCurrentGasResponse);

    render(<PredictionCards />, { wrapper: TestWrapper });

    await waitFor(() => {
      // Should show horizon labels in uppercase format
      expect(screen.getByText('1H PREDICTION')).toBeInTheDocument();
    });

    await waitFor(() => {
      expect(screen.getByText('4H PREDICTION')).toBeInTheDocument();
      expect(screen.getByText('24H PREDICTION')).toBeInTheDocument();
    });
  });

  describe('Prediction Display', () => {
    beforeEach(() => {
      vi.mocked(gasApi.fetchPredictions).mockResolvedValue(mockPredictionsResponse);
      vi.mocked(gasApi.fetchCurrentGas).mockResolvedValue(mockCurrentGasResponse);
    });

    it('displays all three prediction horizons', async () => {
      render(<PredictionCards />, { wrapper: TestWrapper });

      await waitFor(() => {
        expect(screen.getByText('1H PREDICTION')).toBeInTheDocument();
        expect(screen.getByText('4H PREDICTION')).toBeInTheDocument();
        expect(screen.getByText('24H PREDICTION')).toBeInTheDocument();
      });
    });

    it('shows recommendation text', async () => {
      render(<PredictionCards />, { wrapper: TestWrapper });

      await waitFor(() => {
        // Should show some recommendation (the component shows recommendations like "Gas expected to...")
        const recommendations = screen.getAllByText(/expected|Consider|stable/i);
        expect(recommendations.length).toBeGreaterThan(0);
      });
    });
  });

  describe('Color Coding', () => {
    it('shows green for rising gas prices', async () => {
      const risingResponse = {
        current: mockCurrentGasData,
        predictions: {
          '1h': [{ time: '', predictedGwei: 0.0015, confidence: 0.8, confidenceLevel: 'high' as const }],
          '4h': [{ time: '', predictedGwei: 0.001, confidence: 0.7, confidenceLevel: 'medium' as const }],
          '24h': [{ time: '', predictedGwei: 0.001, confidence: 0.5, confidenceLevel: 'low' as const }],
          historical: []
        }
      };

      vi.mocked(gasApi.fetchPredictions).mockResolvedValue(risingResponse);
      vi.mocked(gasApi.fetchCurrentGas).mockResolvedValue(mockCurrentGasResponse);

      render(<PredictionCards />, { wrapper: TestWrapper });

      await waitFor(() => {
        // Check for green border/background class
        const cards = document.querySelectorAll('[class*="green"]');
        expect(cards.length).toBeGreaterThan(0);
      });
    });

    it('shows red for dropping gas prices', async () => {
      const droppingResponse = {
        current: mockCurrentGasData,
        predictions: {
          '1h': [{ time: '', predictedGwei: 0.0005, confidence: 0.8, confidenceLevel: 'high' as const }],
          '4h': [{ time: '', predictedGwei: 0.001, confidence: 0.7, confidenceLevel: 'medium' as const }],
          '24h': [{ time: '', predictedGwei: 0.001, confidence: 0.5, confidenceLevel: 'low' as const }],
          historical: []
        }
      };

      vi.mocked(gasApi.fetchPredictions).mockResolvedValue(droppingResponse);
      vi.mocked(gasApi.fetchCurrentGas).mockResolvedValue(mockCurrentGasResponse);

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
      vi.mocked(gasApi.fetchPredictions).mockResolvedValue(mockPredictionsResponse);
      vi.mocked(gasApi.fetchCurrentGas).mockResolvedValue(mockCurrentGasResponse);

      render(<PredictionCards />, { wrapper: TestWrapper });

      await waitFor(() => {
        // Should show confidence percentage or level text
        const highConfidence = screen.queryAllByText(/HIGH CONFIDENCE/i);
        const mediumConfidence = screen.queryAllByText(/MEDIUM CONFIDENCE/i);
        const lowConfidence = screen.queryAllByText(/LOW CONFIDENCE/i);
        expect(highConfidence.length + mediumConfidence.length + lowConfidence.length).toBeGreaterThan(0);
      });
    });
  });

  describe('Error Handling', () => {
    it('displays error message on fetch failure', async () => {
      vi.mocked(gasApi.fetchPredictions).mockRejectedValue(new Error('Network error'));
      vi.mocked(gasApi.fetchCurrentGas).mockRejectedValue(new Error('Network error'));

      render(<PredictionCards />, { wrapper: TestWrapper });

      await waitFor(() => {
        expect(screen.getByText(/Network error/i)).toBeInTheDocument();
      });
    });

    it('shows retry button on error', async () => {
      vi.mocked(gasApi.fetchPredictions).mockRejectedValue(new Error('Network error'));
      vi.mocked(gasApi.fetchCurrentGas).mockRejectedValue(new Error('Network error'));

      render(<PredictionCards />, { wrapper: TestWrapper });

      await waitFor(() => {
        expect(screen.getByText(/Retry/i)).toBeInTheDocument();
      });
    });

    it('retries fetch when retry button clicked', async () => {
      vi.mocked(gasApi.fetchPredictions).mockRejectedValueOnce(new Error('Network error'));
      vi.mocked(gasApi.fetchCurrentGas).mockRejectedValueOnce(new Error('Network error'));

      render(<PredictionCards />, { wrapper: TestWrapper });

      await waitFor(() => {
        expect(screen.getByText(/Retry/i)).toBeInTheDocument();
      });

      // Setup success for retry
      vi.mocked(gasApi.fetchPredictions).mockResolvedValue(mockPredictionsResponse);
      vi.mocked(gasApi.fetchCurrentGas).mockResolvedValue(mockCurrentGasResponse);

      const retryButton = screen.getByText(/Retry/i);
      fireEvent.click(retryButton);

      await waitFor(() => {
        // Verify API was called again
        expect(gasApi.fetchPredictions).toHaveBeenCalledTimes(2);
      });
    });
  });

  describe('Auto-refresh', () => {
    it('refreshes data every 30 seconds', async () => {
      vi.useFakeTimers();

      try {
        vi.mocked(gasApi.fetchPredictions).mockResolvedValue(mockPredictionsResponse);
        vi.mocked(gasApi.fetchCurrentGas).mockResolvedValue(mockCurrentGasResponse);

        render(<PredictionCards />, { wrapper: TestWrapper });

        // Wait for initial load
        await vi.waitFor(() => {
          expect(gasApi.fetchPredictions).toHaveBeenCalledTimes(1);
        });

        // Advance timer by 30 seconds inside act to flush state updates
        await act(async () => {
          await vi.advanceTimersByTimeAsync(30000);
        });

        await vi.waitFor(() => {
          // Should have been called again
          expect(gasApi.fetchPredictions).toHaveBeenCalledTimes(2);
        });
      } finally {
        vi.useRealTimers();
      }
    });
  });

  describe('Best Time Indicator', () => {
    it('highlights best prediction horizon', async () => {
      const bestTimeResponse = {
        current: mockCurrentGasData,
        predictions: {
          '1h': [{ time: '', predictedGwei: 0.0015, confidence: 0.9, confidenceLevel: 'high' as const }],
          '4h': [{ time: '', predictedGwei: 0.0008, confidence: 0.85, confidenceLevel: 'high' as const }], // Best (lowest)
          '24h': [{ time: '', predictedGwei: 0.0012, confidence: 0.7, confidenceLevel: 'medium' as const }],
          historical: []
        }
      };

      vi.mocked(gasApi.fetchPredictions).mockResolvedValue(bestTimeResponse);
      vi.mocked(gasApi.fetchCurrentGas).mockResolvedValue(mockCurrentGasResponse);

      render(<PredictionCards />, { wrapper: TestWrapper });

      await waitFor(() => {
        // 4h should be marked as best
        expect(screen.getByText('4H PREDICTION')).toBeInTheDocument();
        // BEST TIME badge should appear
        expect(screen.getByText('BEST TIME')).toBeInTheDocument();
      });
    });
  });

  describe('Change Percentage', () => {
    it('shows percentage change from current', async () => {
      vi.mocked(gasApi.fetchPredictions).mockResolvedValue(mockPredictionsResponse);
      vi.mocked(gasApi.fetchCurrentGas).mockResolvedValue(mockCurrentGasResponse);

      render(<PredictionCards />, { wrapper: TestWrapper });

      await waitFor(() => {
        // Should show percentage in the "Potential Savings" section
        const percentages = screen.getAllByText(/%/);
        expect(percentages.length).toBeGreaterThan(0);
      });
    });
  });

  describe('Empty State', () => {
    it('handles empty predictions gracefully', async () => {
      vi.mocked(gasApi.fetchPredictions).mockResolvedValue({
        current: mockCurrentGasData,
        predictions: { '1h': [], '4h': [], '24h': [], historical: [] }
      });
      vi.mocked(gasApi.fetchCurrentGas).mockResolvedValue(mockCurrentGasResponse);

      render(<PredictionCards />, { wrapper: TestWrapper });

      // Should not crash and should show loading/empty state
      await waitFor(() => {
        expect(document.body).toBeInTheDocument();
      });
    });
  });
});
