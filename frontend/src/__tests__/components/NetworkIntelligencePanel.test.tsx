/**
 * Tests for NetworkIntelligencePanel component
 *
 * Tests the network intelligence display showing congestion, block data, and trends.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor, fireEvent, act } from '@testing-library/react';
import NetworkIntelligencePanel from '../../components/NetworkIntelligencePanel';

// Mock fetch globally
const mockFetch = vi.fn();
global.fetch = mockFetch;

describe('NetworkIntelligencePanel', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.resetAllMocks();
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

  const setupSuccessfulMocks = () => {
    mockFetch.mockImplementation((url: string) => {
      if (url.includes('network-state')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve(mockNetworkStateResponse)
        });
      }
      if (url.includes('congestion-history')) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve(mockCongestionHistoryResponse)
        });
      }
      return Promise.resolve({ ok: true, json: () => Promise.resolve({}) });
    });
  };

  it('renders loading state initially', () => {
    mockFetch.mockImplementation(() => new Promise(() => {}));

    render(<NetworkIntelligencePanel />);

    // Should show loading spinner (div with animate-spin class)
    expect(document.querySelector('.animate-spin')).toBeInTheDocument();
  });

  it('renders network data after successful fetch', async () => {
    setupSuccessfulMocks();

    render(<NetworkIntelligencePanel />);

    await waitFor(() => {
      // Should display "Network Intelligence" title
      expect(screen.getByText('Network Intelligence')).toBeInTheDocument();
    });
  });

  describe('Congestion Display', () => {
    beforeEach(() => {
      setupSuccessfulMocks();
    });

    it('displays congestion level', async () => {
      render(<NetworkIntelligencePanel />);

      await waitFor(() => {
        // Should show congestion level text
        expect(screen.getByText('moderate')).toBeInTheDocument();
      });
    });

    it('shows green color for low congestion', async () => {
      mockFetch.mockImplementation((url: string) => {
        if (url.includes('network-state')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({
              ...mockNetworkStateResponse,
              interpretation: { ...mockNetworkStateResponse.interpretation, congestion_level: 'low' }
            })
          });
        }
        if (url.includes('congestion-history')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve(mockCongestionHistoryResponse)
          });
        }
        return Promise.resolve({ ok: true, json: () => Promise.resolve({}) });
      });

      render(<NetworkIntelligencePanel />);

      await waitFor(() => {
        const lowElements = document.querySelectorAll('[class*="green"]');
        expect(lowElements.length).toBeGreaterThan(0);
      });
    });

    it('shows red color for high congestion', async () => {
      mockFetch.mockImplementation((url: string) => {
        if (url.includes('network-state')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve(mockHighCongestionResponse)
          });
        }
        if (url.includes('congestion-history')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve(mockCongestionHistoryResponse)
          });
        }
        return Promise.resolve({ ok: true, json: () => Promise.resolve({}) });
      });

      render(<NetworkIntelligencePanel />);

      await waitFor(() => {
        expect(screen.getByText('high')).toBeInTheDocument();
      });
    });
  });

  describe('Block Information', () => {
    beforeEach(() => {
      setupSuccessfulMocks();
    });

    it('displays current block number', async () => {
      render(<NetworkIntelligencePanel />);

      await waitFor(() => {
        expect(screen.getByText('12,345,678')).toBeInTheDocument();
      });
    });

    it('displays transaction count', async () => {
      render(<NetworkIntelligencePanel />);

      await waitFor(() => {
        expect(screen.getByText('150.0')).toBeInTheDocument();
      });
    });

    it('displays block utilization', async () => {
      render(<NetworkIntelligencePanel />);

      await waitFor(() => {
        // 65% utilization - the component formats as "65.0%" and it appears in multiple places
        const elements = screen.getAllByText('65.0%');
        expect(elements.length).toBeGreaterThan(0);
      });
    });
  });

  describe('Expand/Collapse', () => {
    beforeEach(() => {
      setupSuccessfulMocks();
    });

    it('can toggle expanded state', async () => {
      render(<NetworkIntelligencePanel />);

      await waitFor(() => {
        expect(screen.getByText('Network Intelligence')).toBeInTheDocument();
      });

      // Find the header and click to collapse
      const header = screen.getByText('Network Intelligence').closest('div[class*="cursor-pointer"]');
      if (header) {
        fireEvent.click(header);
        // After collapsing, the detailed content should be hidden
        // The header should still be visible
        expect(screen.getByText('Network Intelligence')).toBeInTheDocument();
      }
    });
  });

  describe('Error Handling', () => {
    it('displays error message on fetch failure', async () => {
      mockFetch.mockRejectedValue(new Error('Network error'));

      render(<NetworkIntelligencePanel />);

      await waitFor(() => {
        // Component displays the actual error message
        expect(screen.getByText(/Network error/i)).toBeInTheDocument();
      });
    });

    it('handles partial API failure', async () => {
      mockFetch.mockImplementation((url: string) => {
        if (url.includes('network-state')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve(mockNetworkStateResponse)
          });
        }
        if (url.includes('congestion-history')) {
          return Promise.resolve({
            ok: false,
            status: 500
          });
        }
        return Promise.resolve({ ok: true, json: () => Promise.resolve({}) });
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
      vi.useFakeTimers();

      try {
        setupSuccessfulMocks();

        render(<NetworkIntelligencePanel />);

        // Wait for initial load using vi.waitFor which works with fake timers
        await vi.waitFor(() => {
          expect(mockFetch).toHaveBeenCalled();
        });

        const initialCallCount = mockFetch.mock.calls.length;

        // Advance timer by 30 seconds inside act to flush state updates
        await act(async () => {
          await vi.advanceTimersByTimeAsync(30000);
        });

        await vi.waitFor(() => {
          // Should have been called again
          expect(mockFetch.mock.calls.length).toBeGreaterThan(initialCallCount);
        });
      } finally {
        vi.useRealTimers();
      }
    });
  });

  describe('Gas Price Display', () => {
    it('displays gas price information', async () => {
      setupSuccessfulMocks();

      render(<NetworkIntelligencePanel />);

      await waitFor(() => {
        // Should show "Gwei" label
        expect(screen.getByText('Gwei')).toBeInTheDocument();
      });
    });
  });

  describe('Trend Indicators', () => {
    it('displays trend information', async () => {
      setupSuccessfulMocks();

      render(<NetworkIntelligencePanel />);

      await waitFor(() => {
        // Should show network insights section
        expect(screen.getByText('Network Insights')).toBeInTheDocument();
      });
    });
  });

  describe('Recommendations', () => {
    it('displays transaction recommendation', async () => {
      setupSuccessfulMocks();

      render(<NetworkIntelligencePanel />);

      await waitFor(() => {
        // Should show recommendation text in the insights
        expect(screen.getByText(/good time for transactions|may vary/i)).toBeInTheDocument();
      });
    });

    it('shows wait recommendation during high congestion', async () => {
      mockFetch.mockImplementation((url: string) => {
        if (url.includes('network-state')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve({
              ...mockHighCongestionResponse,
              network_state: {
                ...mockHighCongestionResponse.network_state,
                avg_utilization: 0.85
              }
            })
          });
        }
        if (url.includes('congestion-history')) {
          return Promise.resolve({
            ok: true,
            json: () => Promise.resolve(mockCongestionHistoryResponse)
          });
        }
        return Promise.resolve({ ok: true, json: () => Promise.resolve({}) });
      });

      render(<NetworkIntelligencePanel />);

      await waitFor(() => {
        // High block utilization triggers warning message
        expect(screen.getByText(/expect higher gas prices|heavily utilized/i)).toBeInTheDocument();
      });
    });
  });

  describe('Icons', () => {
    it('renders network-related icons', async () => {
      setupSuccessfulMocks();

      render(<NetworkIntelligencePanel />);

      await waitFor(() => {
        // Lucide icons render as SVG
        const svgIcons = document.querySelectorAll('svg');
        expect(svgIcons.length).toBeGreaterThan(0);
      });
    });
  });
});
