/**
 * Tests for PatternMatchingCard component
 *
 * Tests pattern matching display, predictions, and historical pattern analysis.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import PatternMatchingCard from '../../components/PatternMatchingCard';

// Mock fetch globally
global.fetch = vi.fn();

describe('PatternMatchingCard', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.resetAllMocks();
    vi.useRealTimers();
  });

  const mockPatternResponse = {
    available: true,
    current_price: 0.001,
    data_points: 150,
    search_period_hours: 72,
    match_count: 5,
    predictions: {
      available: true,
      match_count: 5,
      avg_correlation: 0.85,
      confidence: 0.78,
      '1h': {
        predicted_change: -0.02,
        predicted_price: 0.00098,
        std_dev: 0.0001
      },
      '4h': {
        predicted_change: -0.03,
        predicted_price: 0.00097,
        std_dev: 0.00015
      },
      '24h': {
        predicted_change: 0.01,
        predicted_price: 0.00101,
        std_dev: 0.0002
      },
      pattern_insight: 'Similar patterns suggest stable prices ahead'
    },
    top_matches: [
      {
        timestamp: '2024-01-15T10:30:00Z',
        correlation: 0.92,
        time_similarity: 0.85,
        combined_score: 0.88,
        outcome: {
          '1h_change': -0.015,
          '4h_change': -0.025
        }
      },
      {
        timestamp: '2024-01-10T14:20:00Z',
        correlation: 0.87,
        time_similarity: 0.82,
        combined_score: 0.84,
        outcome: {
          '1h_change': -0.02,
          '4h_change': -0.01
        }
      }
    ],
    timestamp: new Date().toISOString()
  };

  const mockInsufficientDataResponse = {
    available: false,
    reason: 'Insufficient data',
    data_points: 30,
    minimum_required: 50
  };

  it('renders loading state initially', () => {
    (global.fetch as any).mockImplementation(() => new Promise(() => {}));

    render(<PatternMatchingCard />);

    expect(screen.getByText('Pattern Analysis')).toBeInTheDocument();
    expect(document.querySelector('.animate-spin')).toBeTruthy();
  });

  it('renders pattern data after successful fetch', async () => {
    (global.fetch as any).mockResolvedValueOnce({
      ok: true,
      json: async () => mockPatternResponse
    });

    render(<PatternMatchingCard />);

    await waitFor(() => {
      expect(screen.getByText('Pattern Analysis')).toBeInTheDocument();
    });

    await waitFor(() => {
      expect(screen.getByText('5 matches')).toBeInTheDocument();
    });
  });

  describe('Predictions Display', () => {
    it('displays prediction horizons', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockPatternResponse
      });

      render(<PatternMatchingCard />);

      await waitFor(() => {
        expect(screen.getByText('1h')).toBeInTheDocument();
        expect(screen.getByText('4h')).toBeInTheDocument();
        expect(screen.getByText('24h')).toBeInTheDocument();
      });
    });

    it('displays confidence level', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockPatternResponse
      });

      render(<PatternMatchingCard />);

      await waitFor(() => {
        expect(screen.getByText(/78%/)).toBeInTheDocument();
      });
    });

    it('displays pattern insight', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockPatternResponse
      });

      render(<PatternMatchingCard />);

      await waitFor(() => {
        expect(screen.getByText(/Similar patterns suggest/)).toBeInTheDocument();
      });
    });
  });

  describe('Match Information', () => {
    it('displays match count', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockPatternResponse
      });

      render(<PatternMatchingCard />);

      await waitFor(() => {
        expect(screen.getByText(/5 matches/)).toBeInTheDocument();
      });
    });

    it('displays average correlation', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockPatternResponse
      });

      render(<PatternMatchingCard />);

      await waitFor(() => {
        expect(screen.getByText(/85%/)).toBeInTheDocument();
      });
    });

    it('displays data points analyzed', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockPatternResponse
      });

      render(<PatternMatchingCard />);

      await waitFor(() => {
        expect(screen.getByText(/150/)).toBeInTheDocument();
      });
    });
  });

  describe('Change Formatting', () => {
    it('formats positive changes with plus sign', async () => {
      const positiveResponse = {
        ...mockPatternResponse,
        predictions: {
          ...mockPatternResponse.predictions,
          '1h': {
            predicted_change: 0.05,
            predicted_price: 0.00105,
            std_dev: 0.0001
          }
        }
      };

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => positiveResponse
      });

      render(<PatternMatchingCard />);

      await waitFor(() => {
        expect(screen.getByText(/\+5\.0%/)).toBeInTheDocument();
      });
    });

    it('formats negative changes correctly', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockPatternResponse
      });

      render(<PatternMatchingCard />);

      await waitFor(() => {
        expect(screen.getByText(/-2\.0%/)).toBeInTheDocument();
      });
    });
  });

  describe('Insufficient Data State', () => {
    it('displays message when data unavailable', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockInsufficientDataResponse
      });

      render(<PatternMatchingCard />);

      await waitFor(() => {
        expect(screen.getByText(/Insufficient data/i)).toBeInTheDocument();
      });
    });
  });

  describe('Refresh Functionality', () => {
    it('renders refresh button', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockPatternResponse
      });

      render(<PatternMatchingCard />);

      await waitFor(() => {
        const refreshButton = document.querySelector('button[title="Refresh patterns"]');
        expect(refreshButton).toBeTruthy();
      });
    });

    it('calls fetch on refresh click', async () => {
      (global.fetch as any).mockResolvedValue({
        ok: true,
        json: async () => mockPatternResponse
      });

      render(<PatternMatchingCard />);

      await waitFor(() => {
        expect(screen.getByText('Pattern Analysis')).toBeInTheDocument();
      });

      vi.clearAllMocks();
      (global.fetch as any).mockResolvedValue({
        ok: true,
        json: async () => mockPatternResponse
      });

      const refreshButton = document.querySelector('button[title="Refresh patterns"]');
      if (refreshButton) {
        fireEvent.click(refreshButton);

        await waitFor(() => {
          expect(global.fetch).toHaveBeenCalled();
        });
      }
    });
  });

  describe('Error Handling', () => {
    it('displays error message on fetch failure', async () => {
      (global.fetch as any).mockRejectedValueOnce(new Error('Network error'));

      render(<PatternMatchingCard />);

      await waitFor(() => {
        expect(screen.getByText('Pattern analysis unavailable')).toBeInTheDocument();
      });
    });

    it('displays timeout message on abort', async () => {
      const abortError = new Error('Aborted');
      abortError.name = 'AbortError';
      (global.fetch as any).mockRejectedValueOnce(abortError);

      render(<PatternMatchingCard />);

      await waitFor(() => {
        expect(screen.getByText('Request timed out')).toBeInTheDocument();
      });
    });
  });

  describe('Auto-refresh', () => {
    it('refreshes data every 60 seconds', async () => {
      (global.fetch as any).mockResolvedValue({
        ok: true,
        json: async () => mockPatternResponse
      });

      render(<PatternMatchingCard />);

      await waitFor(() => {
        expect(global.fetch).toHaveBeenCalledTimes(1);
      });

      vi.clearAllMocks();
      (global.fetch as any).mockResolvedValue({
        ok: true,
        json: async () => mockPatternResponse
      });

      vi.advanceTimersByTime(60000);

      await waitFor(() => {
        expect(global.fetch).toHaveBeenCalled();
      });
    });
  });

  describe('Trend Icons', () => {
    it('shows upward trend icon for positive change > 2%', async () => {
      const risingResponse = {
        ...mockPatternResponse,
        predictions: {
          ...mockPatternResponse.predictions,
          '1h': {
            predicted_change: 0.05,
            predicted_price: 0.00105,
            std_dev: 0.0001
          }
        }
      };

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => risingResponse
      });

      render(<PatternMatchingCard />);

      await waitFor(() => {
        // Check for red color class (indicates rising prices)
        const changeElement = screen.getByText(/\+5\.0%/);
        expect(changeElement.className).toContain('text-red');
      });
    });

    it('shows downward trend for negative change < -2%', async () => {
      const fallingResponse = {
        ...mockPatternResponse,
        predictions: {
          ...mockPatternResponse.predictions,
          '1h': {
            predicted_change: -0.05,
            predicted_price: 0.00095,
            std_dev: 0.0001
          }
        }
      };

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => fallingResponse
      });

      render(<PatternMatchingCard />);

      await waitFor(() => {
        // Check for green color class (indicates falling prices - good for users)
        const changeElement = screen.getByText(/-5\.0%/);
        expect(changeElement.className).toContain('text-green');
      });
    });
  });
});
