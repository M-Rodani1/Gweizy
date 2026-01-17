/**
 * Tests for MempoolStatusCard component
 *
 * Tests mempool status display, congestion indicators, and momentum formatting.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import MempoolStatusCard from '../../components/MempoolStatusCard';

// Mock fetch globally
global.fetch = vi.fn();

describe('MempoolStatusCard', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.resetAllMocks();
  });

  const mockMempoolResponse = {
    status: 'active',
    metrics: {
      pending_count: 45,
      avg_gas_price: 0.00115,
      median_gas_price: 0.001,
      p90_gas_price: 0.0015,
      gas_price_spread: 0.0005,
      large_tx_ratio: 0.1,
      arrival_rate: 12.5
    },
    signals: {
      is_congested: false,
      congestion_level: 'low',
      count_momentum: 0.02,
      gas_momentum: -0.03
    },
    interpretation: {
      trend: 'stable',
      trend_description: 'Gas prices are stable',
      recommendation: 'Good time to transact'
    },
    timestamp: new Date().toISOString()
  };

  const mockCongestedResponse = {
    ...mockMempoolResponse,
    signals: {
      is_congested: true,
      congestion_level: 'high',
      count_momentum: 0.15,
      gas_momentum: 0.12
    },
    interpretation: {
      trend: 'rising',
      trend_description: 'Gas prices are rising',
      recommendation: 'Consider waiting for lower gas'
    }
  };

  it('renders loading state initially', () => {
    (global.fetch as any).mockImplementation(() => new Promise(() => {}));

    render(<MempoolStatusCard />);

    expect(screen.getByText('Mempool Status')).toBeInTheDocument();
    expect(document.querySelector('.animate-spin')).toBeTruthy();
  });

  it('renders mempool data after successful fetch', async () => {
    (global.fetch as any).mockResolvedValueOnce({
      ok: true,
      json: async () => mockMempoolResponse
    });

    render(<MempoolStatusCard />);

    await waitFor(() => {
      expect(screen.getByText('Mempool Status')).toBeInTheDocument();
    });

    await waitFor(() => {
      expect(screen.getByText('45')).toBeInTheDocument();
    });
  });

  describe('Congestion Status', () => {
    it('displays clear status when not congested', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockMempoolResponse
      });

      render(<MempoolStatusCard />);

      await waitFor(() => {
        expect(screen.getByText('Clear')).toBeInTheDocument();
      });
    });

    it('displays congested status when congested', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockCongestedResponse
      });

      render(<MempoolStatusCard />);

      await waitFor(() => {
        expect(screen.getByText('Congested')).toBeInTheDocument();
      });
    });

    it('displays moderate status correctly', async () => {
      const moderateResponse = {
        ...mockMempoolResponse,
        signals: {
          ...mockMempoolResponse.signals,
          congestion_level: 'moderate'
        }
      };

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => moderateResponse
      });

      render(<MempoolStatusCard />);

      await waitFor(() => {
        expect(screen.getByText('Moderate')).toBeInTheDocument();
      });
    });
  });

  describe('Metrics Display', () => {
    it('displays pending transaction count', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockMempoolResponse
      });

      render(<MempoolStatusCard />);

      await waitFor(() => {
        expect(screen.getByText('Pending Txs')).toBeInTheDocument();
        expect(screen.getByText('45')).toBeInTheDocument();
      });
    });

    it('displays average gas price', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockMempoolResponse
      });

      render(<MempoolStatusCard />);

      await waitFor(() => {
        expect(screen.getByText('Avg Gas Price')).toBeInTheDocument();
        // avg_gas_price: 0.00115 with toFixed(4) = "0.0011"
        expect(screen.getByText('0.0011')).toBeInTheDocument();
      });
    });

    it('displays arrival rate', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockMempoolResponse
      });

      render(<MempoolStatusCard />);

      await waitFor(() => {
        expect(screen.getByText('Arrival Rate')).toBeInTheDocument();
        expect(screen.getByText('12.5')).toBeInTheDocument();
      });
    });
  });

  describe('Momentum Indicators', () => {
    it('shows positive momentum with plus sign', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockMempoolResponse
      });

      render(<MempoolStatusCard />);

      await waitFor(() => {
        expect(screen.getByText('+2.0%')).toBeInTheDocument();
      });
    });

    it('shows negative momentum correctly', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockMempoolResponse
      });

      render(<MempoolStatusCard />);

      await waitFor(() => {
        expect(screen.getByText('-3.0%')).toBeInTheDocument();
      });
    });

    it('applies correct color for high momentum', async () => {
      const highMomentumResponse = {
        ...mockMempoolResponse,
        signals: {
          ...mockMempoolResponse.signals,
          gas_momentum: 0.12
        }
      };

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => highMomentumResponse
      });

      render(<MempoolStatusCard />);

      await waitFor(() => {
        // Red color for high positive momentum
        const momentumText = screen.getByText('+12.0%');
        expect(momentumText.className).toContain('text-red-400');
      });
    });
  });

  describe('Recommendation', () => {
    it('displays recommendation text', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockMempoolResponse
      });

      render(<MempoolStatusCard />);

      await waitFor(() => {
        expect(screen.getByText('Good time to transact')).toBeInTheDocument();
      });
    });

    it('displays wait recommendation when congested', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockCongestedResponse
      });

      render(<MempoolStatusCard />);

      await waitFor(() => {
        expect(screen.getByText('Consider waiting for lower gas')).toBeInTheDocument();
      });
    });
  });

  describe('Status Indicator', () => {
    it('shows live indicator when active', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockMempoolResponse
      });

      render(<MempoolStatusCard />);

      await waitFor(() => {
        expect(screen.getByText('Live')).toBeInTheDocument();
      });
    });

    it('shows inactive when status is inactive', async () => {
      const inactiveResponse = {
        ...mockMempoolResponse,
        status: 'inactive'
      };

      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => inactiveResponse
      });

      render(<MempoolStatusCard />);

      await waitFor(() => {
        expect(screen.getByText('Inactive')).toBeInTheDocument();
      });
    });
  });

  describe('Refresh Functionality', () => {
    it('renders refresh button', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => mockMempoolResponse
      });

      render(<MempoolStatusCard />);

      await waitFor(() => {
        const refreshButton = document.querySelector('button[title="Refresh mempool"]');
        expect(refreshButton).toBeTruthy();
      });
    });

    it('calls fetch on refresh click', async () => {
      (global.fetch as any).mockResolvedValue({
        ok: true,
        json: async () => mockMempoolResponse
      });

      render(<MempoolStatusCard />);

      await waitFor(() => {
        expect(screen.getByText('Mempool Status')).toBeInTheDocument();
      });

      vi.clearAllMocks();
      (global.fetch as any).mockResolvedValue({
        ok: true,
        json: async () => mockMempoolResponse
      });

      const refreshButton = document.querySelector('button[title="Refresh mempool"]');
      fireEvent.click(refreshButton!);

      await waitFor(() => {
        expect(global.fetch).toHaveBeenCalled();
      });
    });
  });

  describe('Error Handling', () => {
    it('displays error message on fetch failure', async () => {
      (global.fetch as any).mockRejectedValueOnce(new Error('Network error'));

      render(<MempoolStatusCard />);

      await waitFor(() => {
        expect(screen.getByText('Mempool data unavailable')).toBeInTheDocument();
      });
    });

    it('displays timeout message on abort', async () => {
      const abortError = new Error('Aborted');
      abortError.name = 'AbortError';
      (global.fetch as any).mockRejectedValueOnce(abortError);

      render(<MempoolStatusCard />);

      await waitFor(() => {
        expect(screen.getByText('Request timed out')).toBeInTheDocument();
      });
    });

    it('handles missing metrics gracefully', async () => {
      (global.fetch as any).mockResolvedValueOnce({
        ok: true,
        json: async () => ({ status: 'active', signals: {}, interpretation: {} })
      });

      render(<MempoolStatusCard />);

      await waitFor(() => {
        expect(screen.getByText('Mempool data unavailable')).toBeInTheDocument();
      });
    });
  });

  describe('Auto-refresh', () => {
    it('refreshes data every 30 seconds', async () => {
      vi.useFakeTimers();

      (global.fetch as any).mockResolvedValue({
        ok: true,
        json: async () => mockMempoolResponse
      });

      render(<MempoolStatusCard />);

      // Wait for initial load using vi.waitFor which works with fake timers
      await vi.waitFor(() => {
        expect(global.fetch).toHaveBeenCalled();
      });

      const initialCallCount = (global.fetch as any).mock.calls.length;

      // Advance timer by 30 seconds
      await vi.advanceTimersByTimeAsync(30000);

      // Should have been called again
      expect((global.fetch as any).mock.calls.length).toBeGreaterThan(initialCallCount);

      vi.useRealTimers();
    });
  });
});
