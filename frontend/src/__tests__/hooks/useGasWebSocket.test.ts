/**
 * Tests for useGasWebSocket hook
 *
 * Tests cover:
 * - Connection establishment
 * - Disconnection handling
 * - Automatic reconnection with exponential backoff
 * - Event handling (gas updates, predictions, mempool)
 * - Error handling
 * - Manual disconnect/reconnect
 * - Cleanup on unmount
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act } from '@testing-library/react';
import { useGasWebSocket } from '../../hooks/useGasWebSocket';
import { resetSocketPool } from '../../utils/websocketPool';

// Mock socket.io-client
const mockSocket = {
  on: vi.fn(),
  off: vi.fn(),
  emit: vi.fn(),
  connect: vi.fn(),
  disconnect: vi.fn(),
  connected: false,
};

vi.mock('socket.io-client', () => ({
  io: vi.fn(() => mockSocket),
}));

// Mock config/api
vi.mock('../../config/api', () => ({
  getWebSocketOrigin: vi.fn(() => 'http://localhost:5000'),
}));

describe('useGasWebSocket', () => {
  let eventHandlers: Record<string, (...args: unknown[]) => void> = {};

  beforeEach(() => {
    vi.useFakeTimers();
    eventHandlers = {};
    resetSocketPool();

    // Capture event handlers
    mockSocket.on.mockImplementation((event: string, handler: (...args: unknown[]) => void) => {
      eventHandlers[event] = handler;
      return mockSocket;
    });

    mockSocket.off.mockImplementation(() => mockSocket);
    mockSocket.connect.mockImplementation(() => mockSocket);
    mockSocket.disconnect.mockImplementation(() => mockSocket);
  });

  afterEach(() => {
    vi.useRealTimers();
    vi.clearAllMocks();
    resetSocketPool();
  });

  describe('initial state', () => {
    it('should start with disconnected state', () => {
      const { result } = renderHook(() => useGasWebSocket());

      expect(result.current.isConnected).toBe(false);
      expect(result.current.gasPrice).toBeNull();
      expect(result.current.predictions).toBeNull();
      expect(result.current.mempool).toBeNull();
      expect(result.current.error).toBeNull();
    });

    it('should have socket instance when enabled', () => {
      const { result } = renderHook(() => useGasWebSocket({ enabled: true }));

      expect(result.current.socket).toBeDefined();
    });

    it('should not connect when disabled', () => {
      const { result } = renderHook(() => useGasWebSocket({ enabled: false }));

      expect(result.current.socket).toBeNull();
      expect(mockSocket.on).not.toHaveBeenCalled();
    });
  });

  describe('connection events', () => {
    it('should handle connect event', () => {
      const onConnect = vi.fn();
      renderHook(() => useGasWebSocket({ onConnect }));

      // Simulate connect event
      act(() => {
        eventHandlers['connect']?.();
      });

      expect(onConnect).toHaveBeenCalled();
    });

    it('should set isConnected to true on connect', () => {
      const { result } = renderHook(() => useGasWebSocket());

      act(() => {
        eventHandlers['connect']?.();
      });

      expect(result.current.isConnected).toBe(true);
    });

    it('should clear error on connect', () => {
      const { result } = renderHook(() => useGasWebSocket());

      // First trigger an error
      act(() => {
        eventHandlers['connect_error']?.(new Error('Connection failed'));
      });

      expect(result.current.error).not.toBeNull();

      // Then connect
      act(() => {
        eventHandlers['connect']?.();
      });

      expect(result.current.error).toBeNull();
    });

    it('should handle disconnect event', () => {
      const onDisconnect = vi.fn();
      const { result } = renderHook(() => useGasWebSocket({ onDisconnect }));

      act(() => {
        eventHandlers['connect']?.();
      });

      expect(result.current.isConnected).toBe(true);

      act(() => {
        eventHandlers['disconnect']?.('transport error');
      });

      expect(result.current.isConnected).toBe(false);
      expect(onDisconnect).toHaveBeenCalled();
    });
  });

  describe('reconnection logic', () => {
    it('should attempt reconnection on unintentional disconnect', () => {
      renderHook(() => useGasWebSocket());

      act(() => {
        eventHandlers['connect']?.();
      });

      act(() => {
        eventHandlers['disconnect']?.('transport error');
      });

      // Advance timers for first reconnection attempt
      act(() => {
        vi.advanceTimersByTime(2000);
      });

      expect(mockSocket.connect).toHaveBeenCalled();
    });

    it('should not attempt reconnection on intentional disconnect', () => {
      renderHook(() => useGasWebSocket());

      act(() => {
        eventHandlers['connect']?.();
      });

      const callsBefore = mockSocket.connect.mock.calls.length;

      act(() => {
        eventHandlers['disconnect']?.('io client disconnect');
      });

      // Advance timers
      act(() => {
        vi.advanceTimersByTime(10000);
      });

      // Should not have called connect again
      expect(mockSocket.connect.mock.calls.length).toBe(callsBefore);
    });

    it('should use exponential backoff for reconnection', () => {
      renderHook(() => useGasWebSocket());

      act(() => {
        eventHandlers['connect']?.();
      });

      // First disconnect
      act(() => {
        eventHandlers['disconnect']?.('transport error');
      });

      // First reconnect attempt (2s delay)
      act(() => {
        vi.advanceTimersByTime(2000);
      });

      act(() => {
        eventHandlers['disconnect']?.('transport error');
      });

      // Second reconnect attempt (4s delay)
      act(() => {
        vi.advanceTimersByTime(4000);
      });

      expect(mockSocket.connect.mock.calls.length).toBeGreaterThan(1);
    });

    it('should give up after max reconnection attempts', () => {
      const onError = vi.fn();
      const { result } = renderHook(() => useGasWebSocket({ onError }));

      act(() => {
        eventHandlers['connect']?.();
      });

      // Simulate 5 disconnects (max attempts)
      for (let i = 0; i < 5; i++) {
        act(() => {
          eventHandlers['disconnect']?.('transport error');
          vi.advanceTimersByTime(10000);
        });
      }

      expect(result.current.error?.message).toContain('failed after multiple attempts');
      expect(onError).toHaveBeenCalled();
    });

    it('should reset reconnection attempts on successful connect', () => {
      renderHook(() => useGasWebSocket());

      // Simulate disconnect
      act(() => {
        eventHandlers['connect']?.();
        eventHandlers['disconnect']?.('transport error');
      });

      // Reconnect successfully
      act(() => {
        vi.advanceTimersByTime(2000);
        eventHandlers['connect']?.();
      });

      // Disconnect again - should start from attempt 1
      act(() => {
        eventHandlers['disconnect']?.('transport error');
        vi.advanceTimersByTime(2000);
      });

      expect(mockSocket.connect).toHaveBeenCalled();
    });
  });

  describe('error handling', () => {
    it('should handle connect_error event', () => {
      const onError = vi.fn();
      const { result } = renderHook(() => useGasWebSocket({ onError }));

      const error = new Error('Connection refused');

      act(() => {
        eventHandlers['connect_error']?.(error);
      });

      expect(result.current.error).toEqual(error);
      expect(onError).toHaveBeenCalledWith(error);
    });

    it('should clear error on gas price update', () => {
      const { result } = renderHook(() => useGasWebSocket());

      act(() => {
        eventHandlers['connect_error']?.(new Error('Connection failed'));
      });

      expect(result.current.error).not.toBeNull();

      act(() => {
        eventHandlers['gas_price_update']?.({
          current_gas: 0.00125,
          base_fee: 0.001,
          priority_fee: 0.00025,
          timestamp: new Date().toISOString(),
          collection_count: 1,
        });
      });

      expect(result.current.error).toBeNull();
    });
  });

  describe('data updates', () => {
    it('should handle gas_price_update event', () => {
      const onGasPriceUpdate = vi.fn();
      const { result } = renderHook(() => useGasWebSocket({ onGasPriceUpdate }));

      const gasData = {
        current_gas: 0.00125,
        base_fee: 0.001,
        priority_fee: 0.00025,
        timestamp: new Date().toISOString(),
        collection_count: 10,
      };

      act(() => {
        eventHandlers['gas_price_update']?.(gasData);
      });

      expect(result.current.gasPrice).toEqual(gasData);
      expect(onGasPriceUpdate).toHaveBeenCalledWith(gasData);
    });

    it('should handle prediction_update event', () => {
      const onPredictionUpdate = vi.fn();
      const { result } = renderHook(() => useGasWebSocket({ onPredictionUpdate }));

      const predictionData = {
        current_price: 0.00125,
        predictions: {
          '1h': { price: 0.0013, confidence: 0.85, lower_bound: 0.001, upper_bound: 0.002 },
          '4h': { price: 0.0014, confidence: 0.75, lower_bound: 0.001, upper_bound: 0.003 },
        },
        timestamp: new Date().toISOString(),
      };

      act(() => {
        eventHandlers['prediction_update']?.(predictionData);
      });

      expect(result.current.predictions).toEqual(predictionData);
      expect(onPredictionUpdate).toHaveBeenCalledWith(predictionData);
    });

    it('should handle mempool_update event', () => {
      const onMempoolUpdate = vi.fn();
      const { result } = renderHook(() => useGasWebSocket({ onMempoolUpdate }));

      const mempoolData = {
        pending_count: 150,
        avg_gas_price: 0.0015,
        is_congested: false,
        gas_momentum: 0.1,
        count_momentum: -0.05,
        timestamp: new Date().toISOString(),
      };

      act(() => {
        eventHandlers['mempool_update']?.(mempoolData);
      });

      expect(result.current.mempool).toEqual(mempoolData);
      expect(onMempoolUpdate).toHaveBeenCalledWith(mempoolData);
    });

    it('should handle combined_update event', () => {
      const onGasPriceUpdate = vi.fn();
      const onPredictionUpdate = vi.fn();
      const onMempoolUpdate = vi.fn();

      const { result } = renderHook(() =>
        useGasWebSocket({ onGasPriceUpdate, onPredictionUpdate, onMempoolUpdate })
      );

      const combinedData = {
        gas: {
          current_gas: 0.00125,
          base_fee: 0.001,
          priority_fee: 0.00025,
          timestamp: new Date().toISOString(),
          collection_count: 10,
        },
        predictions: {
          current_price: 0.00125,
          predictions: {
            '1h': { price: 0.0013, confidence: 0.85, lower_bound: 0.001, upper_bound: 0.002 },
          },
          timestamp: new Date().toISOString(),
        },
        mempool: {
          pending_count: 150,
          avg_gas_price: 0.0015,
          is_congested: false,
          gas_momentum: 0.1,
          count_momentum: -0.05,
          timestamp: new Date().toISOString(),
        },
        timestamp: new Date().toISOString(),
      };

      act(() => {
        eventHandlers['combined_update']?.(combinedData);
      });

      expect(result.current.gasPrice).toEqual(combinedData.gas);
      expect(result.current.predictions).toEqual(combinedData.predictions);
      expect(result.current.mempool).toEqual(combinedData.mempool);
      expect(onGasPriceUpdate).toHaveBeenCalledWith(combinedData.gas);
      expect(onPredictionUpdate).toHaveBeenCalledWith(combinedData.predictions);
      expect(onMempoolUpdate).toHaveBeenCalledWith(combinedData.mempool);
    });

    it('should handle partial combined_update', () => {
      const onGasPriceUpdate = vi.fn();
      const onPredictionUpdate = vi.fn();

      const { result } = renderHook(() =>
        useGasWebSocket({ onGasPriceUpdate, onPredictionUpdate })
      );

      const partialUpdate = {
        gas: {
          current_gas: 0.00125,
          base_fee: 0.001,
          priority_fee: 0.00025,
          timestamp: new Date().toISOString(),
          collection_count: 10,
        },
        timestamp: new Date().toISOString(),
      };

      act(() => {
        eventHandlers['combined_update']?.(partialUpdate);
      });

      expect(result.current.gasPrice).toEqual(partialUpdate.gas);
      expect(result.current.predictions).toBeNull();
      expect(onGasPriceUpdate).toHaveBeenCalled();
      expect(onPredictionUpdate).not.toHaveBeenCalled();
    });
  });

  describe('manual controls', () => {
    it('should manually disconnect', () => {
      const { result } = renderHook(() => useGasWebSocket());

      act(() => {
        result.current.disconnect();
      });

      expect(mockSocket.disconnect).toHaveBeenCalled();
    });

    it('should manually reconnect', () => {
      const { result } = renderHook(() => useGasWebSocket());

      const callsBefore = mockSocket.connect.mock.calls.length;

      act(() => {
        result.current.reconnect();
      });

      expect(mockSocket.connect.mock.calls.length).toBeGreaterThan(callsBefore);
    });

    it('should reset reconnection attempts on manual reconnect', () => {
      const onError = vi.fn();
      renderHook(() => useGasWebSocket({ onError }));

      // Simulate multiple disconnects
      act(() => {
        eventHandlers['connect']?.();
      });

      for (let i = 0; i < 3; i++) {
        act(() => {
          eventHandlers['disconnect']?.('transport error');
          vi.advanceTimersByTime(10000);
        });
      }

      // Manually reconnect - should reset attempts
      const { result: newResult } = renderHook(() => useGasWebSocket());

      act(() => {
        newResult.current.reconnect();
        eventHandlers['connect']?.();
        eventHandlers['disconnect']?.('transport error');
      });

      // Should not immediately give up since attempts were reset
      expect(onError).not.toHaveBeenCalledWith(
        expect.objectContaining({ message: expect.stringContaining('failed after') })
      );
    });
  });

  describe('cleanup', () => {
    it('should disconnect on unmount', () => {
      const { unmount } = renderHook(() => useGasWebSocket());

      unmount();

      expect(mockSocket.disconnect).toHaveBeenCalled();
    });

    it('should clear reconnection timeout on unmount', () => {
      const { unmount } = renderHook(() => useGasWebSocket());

      act(() => {
        eventHandlers['connect']?.();
        eventHandlers['disconnect']?.('transport error');
      });

      unmount();

      // Advance timers - should not attempt reconnection
      const callsBefore = mockSocket.connect.mock.calls.length;

      act(() => {
        vi.advanceTimersByTime(10000);
      });

      // Connection calls should not increase after unmount
      expect(mockSocket.connect.mock.calls.length).toBe(callsBefore);
    });
  });

  describe('enabled prop changes', () => {
    it('should disconnect when disabled', () => {
      const { rerender } = renderHook(
        ({ enabled }) => useGasWebSocket({ enabled }),
        { initialProps: { enabled: true } }
      );

      rerender({ enabled: false });

      expect(mockSocket.disconnect).toHaveBeenCalled();
    });

    it('should not create socket when initially disabled', () => {
      const { result } = renderHook(() => useGasWebSocket({ enabled: false }));

      expect(result.current.socket).toBeNull();
    });
  });

  describe('connection_established event', () => {
    it('should handle connection_established event silently', () => {
      renderHook(() => useGasWebSocket());

      // Should not throw
      act(() => {
        eventHandlers['connection_established']?.({ message: 'Connected to gas price service' });
      });
    });
  });
});
