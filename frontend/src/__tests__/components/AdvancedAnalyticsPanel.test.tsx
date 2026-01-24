/**
 * Tests for AdvancedAnalyticsPanel component
 *
 * Tests the display and interaction of:
 * - Gas Volatility Index
 * - Whale Activity Monitor
 * - Anomaly Detection
 * - Model Ensemble Status
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor, fireEvent, act } from '@testing-library/react';
import AdvancedAnalyticsPanel from '../../components/AdvancedAnalyticsPanel';

// Mock fetch globally
global.fetch = vi.fn();

describe('AdvancedAnalyticsPanel', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.resetAllMocks();
  });

  const mockVolatilityResponse = {
    available: true,
    volatility_index: 45,
    level: 'moderate',
    description: 'Normal gas price fluctuations',
    color: 'yellow',
    trend: 'stable',
    metrics: {
      current_price: 0.001,
      avg_price: 0.00095
    }
  };

  const mockWhalesResponse = {
    available: true,
    current: {
      whale_count: 3,
      activity_level: 'moderate',
      description: 'Moderate whale activity',
      estimated_price_impact_pct: 6,
      impact: 'moderate'
    }
  };

  const mockAnomaliesResponse = {
    available: true,
    status: 'normal',
    status_color: 'green',
    anomaly_count: 0,
    anomalies: [],
    current_analysis: {
      price: 0.001,
      z_score: 0.5,
      vs_average_pct: 5
    }
  };

  const mockEnsembleResponse = {
    available: true,
    health: {
      status: 'healthy',
      color: 'green',
      loaded_models: 4,
      total_models: 5,
      health_pct: 80
    },
    primary_model: 'hybrid_predictor',
    prediction_mode: 'ML-powered predictions active',
    models: [
      { name: 'hybrid_predictor', type: 'primary', loaded: true },
      { name: 'spike_detectors', type: 'classifier', loaded: true },
      { name: 'pattern_matcher', type: 'statistical', loaded: true },
      { name: 'fallback', type: 'heuristic', loaded: true }
    ]
  };

  const setupMockFetch = (overrides = {}) => {
    const responses = {
      volatility: mockVolatilityResponse,
      whales: mockWhalesResponse,
      anomalies: mockAnomaliesResponse,
      ensemble: mockEnsembleResponse,
      ...overrides
    };

    (global.fetch as any).mockImplementation((url: string) => {
      if (url.includes('volatility')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(responses.volatility) });
      }
      if (url.includes('whales')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(responses.whales) });
      }
      if (url.includes('anomalies')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(responses.anomalies) });
      }
      if (url.includes('ensemble')) {
        return Promise.resolve({ ok: true, json: () => Promise.resolve(responses.ensemble) });
      }
      return Promise.reject(new Error('Unknown endpoint'));
    });
  };

  it('renders loading state initially', () => {
    (global.fetch as any).mockImplementation(() => new Promise(() => {}));

    render(<AdvancedAnalyticsPanel />);

    expect(screen.getByText('Advanced Analytics')).toBeInTheDocument();
    expect(document.querySelector('.animate-spin')).toBeTruthy();
  });

  it('renders all four analytics cards after loading', async () => {
    setupMockFetch();

    render(<AdvancedAnalyticsPanel />);

    await waitFor(() => {
      expect(screen.getByText('Gas Volatility Index')).toBeInTheDocument();
      expect(screen.getByText('Whale Activity')).toBeInTheDocument();
      expect(screen.getByText('Anomaly Detection')).toBeInTheDocument();
      expect(screen.getByText('Model Ensemble')).toBeInTheDocument();
    });
  });

  describe('Volatility Index Card', () => {
    it('displays volatility index value', async () => {
      setupMockFetch();

      render(<AdvancedAnalyticsPanel />);

      await waitFor(() => {
        expect(screen.getByText('45')).toBeInTheDocument();
      });
    });

    it('displays volatility level badge', async () => {
      setupMockFetch();

      render(<AdvancedAnalyticsPanel />);

      await waitFor(() => {
        // Both volatility and whale activity show "MODERATE", so use getAllByText
        const moderateElements = screen.getAllByText('MODERATE');
        expect(moderateElements.length).toBeGreaterThan(0);
      });
    });

    it('displays volatility description', async () => {
      setupMockFetch();

      render(<AdvancedAnalyticsPanel />);

      await waitFor(() => {
        expect(screen.getByText('Normal gas price fluctuations')).toBeInTheDocument();
      });
    });

    it('shows insufficient data when volatility unavailable', async () => {
      setupMockFetch({ volatility: { available: false } });

      render(<AdvancedAnalyticsPanel />);

      await waitFor(() => {
        expect(screen.getByText('Insufficient data')).toBeInTheDocument();
      });
    });
  });

  describe('Whale Activity Card', () => {
    it('displays whale count', async () => {
      setupMockFetch();

      render(<AdvancedAnalyticsPanel />);

      await waitFor(() => {
        expect(screen.getByText('3')).toBeInTheDocument();
      });
    });

    it('displays activity level', async () => {
      setupMockFetch();

      render(<AdvancedAnalyticsPanel />);

      await waitFor(() => {
        // Both volatility and whale activity show "MODERATE", so use getAllByText
        const moderateElements = screen.getAllByText('MODERATE');
        expect(moderateElements.length).toBeGreaterThan(0);
      });
    });

    it('displays price impact estimate', async () => {
      setupMockFetch();

      render(<AdvancedAnalyticsPanel />);

      await waitFor(() => {
        expect(screen.getByText(/Est. impact: \+6% on gas/)).toBeInTheDocument();
      });
    });

    it('shows monitoring inactive when whales unavailable', async () => {
      setupMockFetch({ whales: { available: false } });

      render(<AdvancedAnalyticsPanel />);

      await waitFor(() => {
        expect(screen.getByText('Monitoring inactive')).toBeInTheDocument();
      });
    });
  });

  describe('Anomaly Detection Card', () => {
    it('displays normal status when no anomalies', async () => {
      setupMockFetch();

      render(<AdvancedAnalyticsPanel />);

      await waitFor(() => {
        expect(screen.getByText('Normal')).toBeInTheDocument();
      });
    });

    it('displays z-score and average comparison', async () => {
      setupMockFetch();

      render(<AdvancedAnalyticsPanel />);

      await waitFor(() => {
        expect(screen.getByText(/Z-score: 0.5/)).toBeInTheDocument();
        expect(screen.getByText(/vs avg: \+5%/)).toBeInTheDocument();
      });
    });

    it('displays alert count when anomalies detected', async () => {
      setupMockFetch({
        anomalies: {
          available: true,
          status: 'warning',
          status_color: 'yellow',
          anomaly_count: 2,
          anomalies: [
            { type: 'spike', severity: 'medium', message: 'Price spike detected' },
            { type: 'volatility', severity: 'low', message: 'High volatility' }
          ],
          current_analysis: { price: 0.001, z_score: 2.5, vs_average_pct: 15 }
        }
      });

      render(<AdvancedAnalyticsPanel />);

      await waitFor(() => {
        expect(screen.getByText('2 Alerts')).toBeInTheDocument();
        expect(screen.getByText('Price spike detected')).toBeInTheDocument();
      });
    });
  });

  describe('Model Ensemble Card', () => {
    it('displays health status', async () => {
      setupMockFetch();

      render(<AdvancedAnalyticsPanel />);

      await waitFor(() => {
        expect(screen.getByText('Healthy')).toBeInTheDocument();
      });
    });

    it('displays model count', async () => {
      setupMockFetch();

      render(<AdvancedAnalyticsPanel />);

      await waitFor(() => {
        expect(screen.getByText('4/5 models')).toBeInTheDocument();
      });
    });

    it('displays prediction mode', async () => {
      setupMockFetch();

      render(<AdvancedAnalyticsPanel />);

      await waitFor(() => {
        expect(screen.getByText('ML-powered predictions active')).toBeInTheDocument();
      });
    });

    it('shows loading when ensemble unavailable', async () => {
      setupMockFetch({ ensemble: { available: false } });

      render(<AdvancedAnalyticsPanel />);

      await waitFor(() => {
        expect(screen.getByText('Loading model info...')).toBeInTheDocument();
      });
    });
  });

  describe('Refresh Functionality', () => {
    it('renders refresh button', async () => {
      setupMockFetch();

      render(<AdvancedAnalyticsPanel />);

      await waitFor(() => {
        const refreshButton = document.querySelector('button[title="Refresh analytics"]');
        expect(refreshButton).toBeTruthy();
      });
    });

    it('calls fetch on refresh click', async () => {
      setupMockFetch();

      render(<AdvancedAnalyticsPanel />);

      await waitFor(() => {
        expect(screen.getByText('Advanced Analytics')).toBeInTheDocument();
      });

      vi.clearAllMocks();
      setupMockFetch();

      const refreshButton = document.querySelector('button[title="Refresh analytics"]');
      fireEvent.click(refreshButton!);

      await waitFor(() => {
        expect(global.fetch).toHaveBeenCalled();
      });
    });
  });

  describe('Error Handling', () => {
    it('displays error message on fetch failure', async () => {
      (global.fetch as any).mockRejectedValue(new Error('Network error'));

      render(<AdvancedAnalyticsPanel />);

      await waitFor(() => {
        // Component uses Promise.allSettled so it handles failures gracefully
        // When all fetches fail, the component shows "Insufficient data" for multiple cards
        const insufficientDataElements = screen.getAllByText('Insufficient data');
        expect(insufficientDataElements.length).toBeGreaterThan(0);
      });
    });

    it('handles partial data gracefully', async () => {
      (global.fetch as any).mockImplementation((url: string) => {
        if (url.includes('volatility')) {
          return Promise.resolve({ ok: true, json: () => Promise.resolve(mockVolatilityResponse) });
        }
        return Promise.resolve({ ok: false });
      });

      render(<AdvancedAnalyticsPanel />);

      await waitFor(() => {
        expect(screen.getByText('Gas Volatility Index')).toBeInTheDocument();
        expect(screen.getByText('45')).toBeInTheDocument();
      });
    });
  });

  describe('Auto-refresh', () => {
    it('refreshes data every 60 seconds', async () => {
      vi.useFakeTimers();
      try {
        setupMockFetch();

        render(<AdvancedAnalyticsPanel />);

        // Wait for initial load using vi.waitFor which works with fake timers
        await vi.waitFor(() => {
          expect(global.fetch).toHaveBeenCalled();
        });

        const initialCallCount = (global.fetch as any).mock.calls.length;

        // Advance timer by 60 seconds inside act to flush state updates
        await act(async () => {
          await vi.advanceTimersByTimeAsync(60000);
        });

        await vi.waitFor(() => {
          // Should have been called again
          expect((global.fetch as any).mock.calls.length).toBeGreaterThan(initialCallCount);
        });
      } finally {
        vi.useRealTimers();
      }
    });
  });
});
