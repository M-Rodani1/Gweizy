/**
 * Tests for AccuracyMetricsCard component
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import AccuracyMetricsCard from '../../components/AccuracyMetricsCard';

// Mock fetch globally
global.fetch = vi.fn();

describe('AccuracyMetricsCard', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.resetAllMocks();
  });

  const mockMetricsResponse = {
    success: true,
    metrics: {
      '1h': { mae: 0.000275, rmse: 0.000442, r2: 0.71, directional_accuracy: 0.65, n: 100 },
      '4h': { mae: 0.000312, rmse: 0.000521, r2: 0.55, directional_accuracy: 0.58, n: 50 },
      '24h': { mae: 0.000498, rmse: 0.000712, r2: 0.35, directional_accuracy: 0.52, n: 20 }
    }
  };

  it('renders loading state initially', () => {
    (global.fetch as any).mockImplementation(() => new Promise(() => {}));

    render(<AccuracyMetricsCard />);

    // Should show loading spinner (RefreshCw with animate-spin)
    expect(document.querySelector('.animate-spin')).toBeTruthy();
  });

  it('renders metrics after successful fetch', async () => {
    (global.fetch as any).mockResolvedValueOnce({
      ok: true,
      json: async () => mockMetricsResponse,
    });

    render(<AccuracyMetricsCard />);

    await waitFor(() => {
      expect(screen.getByText('Model Accuracy')).toBeInTheDocument();
    });

    // Check that R² score is displayed
    await waitFor(() => {
      expect(screen.getByText('R² Score')).toBeInTheDocument();
    });
  });

  it('renders horizon tabs', async () => {
    (global.fetch as any).mockResolvedValueOnce({
      ok: true,
      json: async () => mockMetricsResponse,
    });

    render(<AccuracyMetricsCard />);

    await waitFor(() => {
      expect(screen.getByText('1h')).toBeInTheDocument();
      expect(screen.getByText('4h')).toBeInTheDocument();
      expect(screen.getByText('24h')).toBeInTheDocument();
    });
  });

  it('switches horizons when tab clicked', async () => {
    (global.fetch as any).mockResolvedValueOnce({
      ok: true,
      json: async () => mockMetricsResponse,
    });

    render(<AccuracyMetricsCard />);

    await waitFor(() => {
      expect(screen.getByText('1h')).toBeInTheDocument();
    });

    // Click on 4h tab
    fireEvent.click(screen.getByText('4h'));

    // The 4h tab should now be selected (has different styling)
    const tab4h = screen.getByText('4h');
    expect(tab4h.className).toContain('bg-purple-500');
  });

  it('renders refresh button', async () => {
    (global.fetch as any).mockResolvedValueOnce({
      ok: true,
      json: async () => mockMetricsResponse,
    });

    render(<AccuracyMetricsCard />);

    await waitFor(() => {
      expect(screen.getByText('Refresh')).toBeInTheDocument();
    });
  });

  it('calls fetch on refresh button click', async () => {
    (global.fetch as any).mockResolvedValue({
      ok: true,
      json: async () => mockMetricsResponse,
    });

    render(<AccuracyMetricsCard />);

    await waitFor(() => {
      expect(screen.getByText('Refresh')).toBeInTheDocument();
    });

    // Clear previous calls
    vi.clearAllMocks();

    // Click refresh
    fireEvent.click(screen.getByText('Refresh'));

    // Fetch should be called again
    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalled();
    });
  });

  it('shows error state when fetch fails', async () => {
    (global.fetch as any).mockRejectedValueOnce(new Error('Network error'));

    render(<AccuracyMetricsCard />);

    await waitFor(() => {
      // Should show fallback mock data and error message
      expect(screen.getByText(/Could not load/i)).toBeInTheDocument();
    });
  });

  it('shows no metrics message when data is empty', async () => {
    (global.fetch as any).mockResolvedValueOnce({
      ok: true,
      json: async () => ({
        success: true,
        metrics: {
          '1h': { mae: null, rmse: null, r2: null, directional_accuracy: null, n: 0 },
          '4h': { mae: null, rmse: null, r2: null, directional_accuracy: null, n: 0 },
          '24h': { mae: null, rmse: null, r2: null, directional_accuracy: null, n: 0 }
        }
      }),
    });

    render(<AccuracyMetricsCard />);

    await waitFor(() => {
      // Use getAllByText since the message appears in multiple places
      const elements = screen.getAllByText(/No metrics available/i);
      expect(elements.length).toBeGreaterThan(0);
    });
  });

  it('displays correct metric labels', async () => {
    (global.fetch as any).mockResolvedValueOnce({
      ok: true,
      json: async () => mockMetricsResponse,
    });

    render(<AccuracyMetricsCard />);

    await waitFor(() => {
      expect(screen.getByText('R² Score')).toBeInTheDocument();
      expect(screen.getByText('Direction')).toBeInTheDocument();
      expect(screen.getByText('MAE')).toBeInTheDocument();
      expect(screen.getByText('Samples')).toBeInTheDocument();
    });
  });

  it('displays metric descriptions', async () => {
    (global.fetch as any).mockResolvedValueOnce({
      ok: true,
      json: async () => mockMetricsResponse,
    });

    render(<AccuracyMetricsCard />);

    await waitFor(() => {
      expect(screen.getByText('Variance explained')).toBeInTheDocument();
      expect(screen.getByText('Trend prediction')).toBeInTheDocument();
      expect(screen.getByText('Mean abs. error')).toBeInTheDocument();
      expect(screen.getByText('Predictions tracked')).toBeInTheDocument();
    });
  });
});
