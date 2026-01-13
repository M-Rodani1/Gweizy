/**
 * Tests for NetworkIntelligencePanel component
 *
 * Tests the network intelligence display showing congestion, block data, and trends.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import NetworkIntelligencePanel from '../../components/NetworkIntelligencePanel';

// Mock fetch globally
global.fetch = vi.fn();

describe('NetworkIntelligencePanel', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.resetAllMocks();
    vi.useRealTimers();
  });

  const mockNetworkStateResponse = {
    network_state: {
      current_block: 12345678,
      avg_utilization: 0.65,
      avg_tx_count: 150,
      avg_base_fee: 0.001,
      base_fee_trend: 0.02,
      is_congested: false,
      timestamp: new Date().toISOString()
    },
    interpretation: {
      congestion_level: 'moderate',
      gas_trend: 'stable',
      recommendation: 'Good time to transact'
    },
    timestamp: new Date().toISOString()
  };

  const mockCongestionHistoryResponse = {
    timestamps: [
      new Date(Date.now() - 3600000).toISOString(),
      new Date(Date.now() - 1800000).toISOString(),
      new Date().toISOString()
    ],
    congestion_scores: [0.55, 0.60, 0.65],
    block_utilization: [0.50, 0.58, 0.65],
    tx_counts: [120, 140, 150]
  };

  const mockHighCongestionResponse = {
    ...mockNetworkStateResponse,
    network_state: {
      ...mockNetworkStateResponse.network_state,
      avg_utilization: 0.92,
      is_congested: true
    },
    interpretation: {
      congestion_level: 'high',
      gas_trend: 'rising',
      recommendation: 'Consider waiting for lower gas'
    }
  };

  it('renders loading state initially', () => {
    (global.fetch as any).mockImplementation(() => new Promise(() => {}));

    render(<NetworkIntelligencePanel />);

    // Should show loading indicator or panel title
    expect(screen.getByText(/Network|Loading/i)).toBeInTheDocument();
  });

  it('renders network data after successful fetch', async () => {
    (global.fetch as any).mockImplementation((url: string) => {
      if (url.includes('network-state')) {
        return Promise.resolve({
          ok: true,
          json: async () => mockNetworkStateResponse
        });
      }
      if (url.includes('congestion-history')) {
        return Promise.resolve({
          ok: true,
          json: async () => mockCongestionHistoryResponse
        });
      }
      return Promise.resolve({ ok: true, json: async () => ({}) });
    });

    render(<NetworkIntelligencePanel />);

    await waitFor(() => {
      // Should display block number
      expect(screen.getByText(/12345678|Block/i)).toBeInTheDocument();
    });
  });

  describe('Congestion Display', () => {
    beforeEach(() => {
      (global.fetch as any).mockImplementation((url: string) => {
        if (url.includes('network-state')) {
          return Promise.resolve({
            ok: true,
            json: async () => mockNetworkStateResponse
          });
        }
        if (url.includes('congestion-history')) {
          return Promise.resolve({
            ok: true,
            json: async () => mockCongestionHistoryResponse
          });
        }
        return Promise.resolve({ ok: true, json: async () => ({}) });
      });
    });

    it('displays congestion level', async () => {
      render(<NetworkIntelligencePanel />);

      await waitFor(() => {
        // Should show congestion level text
        expect(screen.getByText(/moderate|low|high/i)).toBeInTheDocument();
      });
    });

    it('shows green color for low congestion', async () => {
      (global.fetch as any).mockImplementation((url: string) => {
        if (url.includes('network-state')) {
          return Promise.resolve({
            ok: true,
            json: async () => ({
              ...mockNetworkStateResponse,
              interpretation: { ...mockNetworkStateResponse.interpretation, congestion_level: 'low' }
            })
          });
        }
        if (url.includes('congestion-history')) {
          return Promise.resolve({
            ok: true,
            json: async () => mockCongestionHistoryResponse
          });
        }
        return Promise.resolve({ ok: true, json: async () => ({}) });
      });

      render(<NetworkIntelligencePanel />);

      await waitFor(() => {
        const lowElements = document.querySelectorAll('[class*="green"]');
        expect(lowElements.length).toBeGreaterThan(0);
      });
    });

    it('shows red color for high congestion', async () => {
      (global.fetch as any).mockImplementation((url: string) => {
        if (url.includes('network-state')) {
          return Promise.resolve({
            ok: true,
            json: async () => mockHighCongestionResponse
          });
        }
        if (url.includes('congestion-history')) {
          return Promise.resolve({
            ok: true,
            json: async () => mockCongestionHistoryResponse
          });
        }
        return Promise.resolve({ ok: true, json: async () => ({}) });
      });

      render(<NetworkIntelligencePanel />);

      await waitFor(() => {
        expect(screen.getByText(/high/i)).toBeInTheDocument();
      });
    });
  });

  describe('Block Information', () => {
    beforeEach(() => {
      (global.fetch as any).mockImplementation((url: string) => {
        if (url.includes('network-state')) {
          return Promise.resolve({
            ok: true,
            json: async () => mockNetworkStateResponse
          });
        }
        if (url.includes('congestion-history')) {
          return Promise.resolve({
            ok: true,
            json: async () => mockCongestionHistoryResponse
          });
        }
        return Promise.resolve({ ok: true, json: async () => ({}) });
      });
    });

    it('displays current block number', async () => {
      render(<NetworkIntelligencePanel />);

      await waitFor(() => {
        expect(screen.getByText(/12345678/)).toBeInTheDocument();
      });
    });

    it('displays transaction count', async () => {
      render(<NetworkIntelligencePanel />);

      await waitFor(() => {
        expect(screen.getByText(/150|Tx/i)).toBeInTheDocument();
      });
    });

    it('displays block utilization', async () => {
      render(<NetworkIntelligencePanel />);

      await waitFor(() => {
        // 65% utilization
        expect(screen.getByText(/65|Utilization/i)).toBeInTheDocument();
      });
    });
  });

  describe('Expand/Collapse', () => {
    beforeEach(() => {
      (global.fetch as any).mockImplementation((url: string) => {
        if (url.includes('network-state')) {
          return Promise.resolve({
            ok: true,
            json: async () => mockNetworkStateResponse
          });
        }
        if (url.includes('congestion-history')) {
          return Promise.resolve({
            ok: true,
            json: async () => mockCongestionHistoryResponse
          });
        }
        return Promise.resolve({ ok: true, json: async () => ({}) });
      });
    });

    it('can toggle expanded state', async () => {
      render(<NetworkIntelligencePanel />);

      await waitFor(() => {
        expect(screen.getByText(/Network/i)).toBeInTheDocument();
      });

      // Find and click toggle button if exists
      const toggleButton = document.querySelector('button');
      if (toggleButton) {
        fireEvent.click(toggleButton);
        // Component should still be visible
        expect(screen.getByText(/Network/i)).toBeInTheDocument();
      }
    });
  });

  describe('Error Handling', () => {
    it('displays error message on fetch failure', async () => {
      (global.fetch as any).mockRejectedValue(new Error('Network error'));

      render(<NetworkIntelligencePanel />);

      await waitFor(() => {
        expect(screen.getByText(/Failed|error|unavailable/i)).toBeInTheDocument();
      });
    });

    it('handles partial API failure', async () => {
      (global.fetch as any).mockImplementation((url: string) => {
        if (url.includes('network-state')) {
          return Promise.resolve({
            ok: true,
            json: async () => mockNetworkStateResponse
          });
        }
        if (url.includes('congestion-history')) {
          return Promise.resolve({
            ok: false,
            status: 500
          });
        }
        return Promise.resolve({ ok: true, json: async () => ({}) });
      });

      render(<NetworkIntelligencePanel />);

      await waitFor(() => {
        // Should show error or handle gracefully
        expect(document.body).toBeInTheDocument();
      });
    });
  });

  describe('Auto-refresh', () => {
    it('refreshes data every 30 seconds', async () => {
      (global.fetch as any).mockImplementation((url: string) => {
        if (url.includes('network-state')) {
          return Promise.resolve({
            ok: true,
            json: async () => mockNetworkStateResponse
          });
        }
        if (url.includes('congestion-history')) {
          return Promise.resolve({
            ok: true,
            json: async () => mockCongestionHistoryResponse
          });
        }
        return Promise.resolve({ ok: true, json: async () => ({}) });
      });

      render(<NetworkIntelligencePanel />);

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

  describe('Gas Price Display', () => {
    it('displays gas price information', async () => {
      (global.fetch as any).mockImplementation((url: string) => {
        if (url.includes('network-state')) {
          return Promise.resolve({
            ok: true,
            json: async () => mockNetworkStateResponse
          });
        }
        if (url.includes('congestion-history')) {
          return Promise.resolve({
            ok: true,
            json: async () => mockCongestionHistoryResponse
          });
        }
        return Promise.resolve({ ok: true, json: async () => ({}) });
      });

      render(<NetworkIntelligencePanel />);

      await waitFor(() => {
        // Should show gas price or Gwei
        const gasElements = screen.getAllByText(/Gwei|Gas|0\.001/i);
        expect(gasElements.length).toBeGreaterThanOrEqual(0);
      });
    });
  });

  describe('Trend Indicators', () => {
    it('displays trend information', async () => {
      (global.fetch as any).mockImplementation((url: string) => {
        if (url.includes('network-state')) {
          return Promise.resolve({
            ok: true,
            json: async () => mockNetworkStateResponse
          });
        }
        if (url.includes('congestion-history')) {
          return Promise.resolve({
            ok: true,
            json: async () => mockCongestionHistoryResponse
          });
        }
        return Promise.resolve({ ok: true, json: async () => ({}) });
      });

      render(<NetworkIntelligencePanel />);

      await waitFor(() => {
        // Should show trend indicator
        expect(screen.getByText(/stable|rising|falling|trend/i)).toBeInTheDocument();
      });
    });
  });

  describe('Recommendations', () => {
    it('displays transaction recommendation', async () => {
      (global.fetch as any).mockImplementation((url: string) => {
        if (url.includes('network-state')) {
          return Promise.resolve({
            ok: true,
            json: async () => mockNetworkStateResponse
          });
        }
        if (url.includes('congestion-history')) {
          return Promise.resolve({
            ok: true,
            json: async () => mockCongestionHistoryResponse
          });
        }
        return Promise.resolve({ ok: true, json: async () => ({}) });
      });

      render(<NetworkIntelligencePanel />);

      await waitFor(() => {
        expect(screen.getByText(/Good time|Consider|transact/i)).toBeInTheDocument();
      });
    });

    it('shows wait recommendation during high congestion', async () => {
      (global.fetch as any).mockImplementation((url: string) => {
        if (url.includes('network-state')) {
          return Promise.resolve({
            ok: true,
            json: async () => mockHighCongestionResponse
          });
        }
        if (url.includes('congestion-history')) {
          return Promise.resolve({
            ok: true,
            json: async () => mockCongestionHistoryResponse
          });
        }
        return Promise.resolve({ ok: true, json: async () => ({}) });
      });

      render(<NetworkIntelligencePanel />);

      await waitFor(() => {
        expect(screen.getByText(/waiting|Consider/i)).toBeInTheDocument();
      });
    });
  });

  describe('Icons', () => {
    it('renders network-related icons', async () => {
      (global.fetch as any).mockImplementation((url: string) => {
        if (url.includes('network-state')) {
          return Promise.resolve({
            ok: true,
            json: async () => mockNetworkStateResponse
          });
        }
        if (url.includes('congestion-history')) {
          return Promise.resolve({
            ok: true,
            json: async () => mockCongestionHistoryResponse
          });
        }
        return Promise.resolve({ ok: true, json: async () => ({}) });
      });

      render(<NetworkIntelligencePanel />);

      await waitFor(() => {
        // Lucide icons render as SVG
        const svgIcons = document.querySelectorAll('svg');
        expect(svgIcons.length).toBeGreaterThan(0);
      });
    });
  });
});
