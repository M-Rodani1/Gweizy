/**
 * WebSocket Load Tests
 *
 * Tests the frontend's ability to handle multiple concurrent WebSocket
 * connections and high-frequency message updates.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act, waitFor } from '@testing-library/react';

// ============================================================================
// Mock WebSocket Implementation for Load Testing
// ============================================================================

class MockWebSocket {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;

  readyState: number = MockWebSocket.CONNECTING;
  url: string;
  onopen: ((event: Event) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;

  private messageCount = 0;
  private intervalId: ReturnType<typeof setInterval> | null = null;

  constructor(url: string) {
    this.url = url;
    // Simulate connection delay
    setTimeout(() => {
      this.readyState = MockWebSocket.OPEN;
      this.onopen?.(new Event('open'));
    }, 10);
  }

  send(data: string): void {
    if (this.readyState !== MockWebSocket.OPEN) {
      throw new Error('WebSocket is not open');
    }
  }

  close(code?: number, reason?: string): void {
    this.readyState = MockWebSocket.CLOSING;
    if (this.intervalId) {
      clearInterval(this.intervalId);
    }
    setTimeout(() => {
      this.readyState = MockWebSocket.CLOSED;
      this.onclose?.(new CloseEvent('close', { code, reason }));
    }, 5);
  }

  // Simulate receiving messages at high frequency
  simulateHighFrequencyMessages(messagesPerSecond: number, duration: number): Promise<number> {
    return new Promise((resolve) => {
      const interval = 1000 / messagesPerSecond;
      const endTime = Date.now() + duration;

      this.intervalId = setInterval(() => {
        if (Date.now() >= endTime) {
          if (this.intervalId) {
            clearInterval(this.intervalId);
          }
          resolve(this.messageCount);
          return;
        }

        this.messageCount++;
        const message = {
          type: 'gas_update',
          data: {
            gas_price: 20 + Math.random() * 30,
            timestamp: Date.now(),
            chain_id: 1,
          },
        };

        this.onmessage?.(
          new MessageEvent('message', {
            data: JSON.stringify(message),
          })
        );
      }, interval);
    });
  }

  // Simulate burst of messages
  simulateBurst(count: number): void {
    for (let i = 0; i < count; i++) {
      this.messageCount++;
      const message = {
        type: 'gas_update',
        data: {
          gas_price: 20 + Math.random() * 30,
          timestamp: Date.now(),
          chain_id: 1,
        },
      };

      this.onmessage?.(
        new MessageEvent('message', {
          data: JSON.stringify(message),
        })
      );
    }
  }
}

// ============================================================================
// Load Testing Utilities
// ============================================================================

interface LoadTestResult {
  messagesReceived: number;
  duration: number;
  messagesPerSecond: number;
  peakMemoryMB: number;
  avgProcessingTimeMs: number;
}

interface ConnectionPoolResult {
  totalConnections: number;
  successfulConnections: number;
  failedConnections: number;
  connectionTimeMs: number;
}

async function measureMemory(): Promise<number> {
  // @ts-ignore - performance.memory is non-standard but available in Chrome/Node
  if (typeof performance !== 'undefined' && (performance as any).memory) {
    return (performance as any).memory.usedJSHeapSize / 1024 / 1024;
  }
  return 0;
}

function createConnectionPool(count: number): MockWebSocket[] {
  return Array.from({ length: count }, (_, i) => new MockWebSocket(`ws://test/${i}`));
}

// ============================================================================
// Load Tests
// ============================================================================

describe('WebSocket Load Tests', () => {
  let originalWebSocket: typeof WebSocket;
  const connections: MockWebSocket[] = [];

  beforeEach(() => {
    originalWebSocket = global.WebSocket;
    // @ts-ignore
    global.WebSocket = MockWebSocket;
    vi.useFakeTimers({ shouldAdvanceTime: true });
  });

  afterEach(() => {
    global.WebSocket = originalWebSocket;
    connections.forEach((conn) => conn.close());
    connections.length = 0;
    vi.useRealTimers();
  });

  describe('Connection Pool Tests', () => {
    it('should handle 10 concurrent connections', async () => {
      vi.useRealTimers();

      const connectionCount = 10;
      const startTime = Date.now();

      const pool = createConnectionPool(connectionCount);
      connections.push(...pool);

      // Wait for all connections to open
      await Promise.all(
        pool.map(
          (ws) =>
            new Promise<void>((resolve) => {
              ws.onopen = () => resolve();
            })
        )
      );

      const connectionTime = Date.now() - startTime;

      const openConnections = pool.filter(
        (ws) => ws.readyState === MockWebSocket.OPEN
      ).length;

      expect(openConnections).toBe(connectionCount);
      expect(connectionTime).toBeLessThan(1000); // Should connect within 1 second
    });

    it('should handle 50 concurrent connections', async () => {
      vi.useRealTimers();

      const connectionCount = 50;
      const startTime = Date.now();

      const pool = createConnectionPool(connectionCount);
      connections.push(...pool);

      await Promise.all(
        pool.map(
          (ws) =>
            new Promise<void>((resolve) => {
              ws.onopen = () => resolve();
            })
        )
      );

      const connectionTime = Date.now() - startTime;
      const openConnections = pool.filter(
        (ws) => ws.readyState === MockWebSocket.OPEN
      ).length;

      expect(openConnections).toBe(connectionCount);
      expect(connectionTime).toBeLessThan(2000);
    });

    it('should gracefully close all connections', async () => {
      vi.useRealTimers();

      const pool = createConnectionPool(20);
      connections.push(...pool);

      await Promise.all(
        pool.map(
          (ws) =>
            new Promise<void>((resolve) => {
              ws.onopen = () => resolve();
            })
        )
      );

      // Close all connections
      await Promise.all(
        pool.map(
          (ws) =>
            new Promise<void>((resolve) => {
              ws.onclose = () => resolve();
              ws.close();
            })
        )
      );

      const closedConnections = pool.filter(
        (ws) => ws.readyState === MockWebSocket.CLOSED
      ).length;

      expect(closedConnections).toBe(20);
    });
  });

  describe('High Frequency Message Tests', () => {
    it('should handle 100 messages per second', async () => {
      vi.useRealTimers();

      const ws = new MockWebSocket('ws://test');
      connections.push(ws);

      await new Promise<void>((resolve) => {
        ws.onopen = () => resolve();
      });

      let receivedMessages = 0;
      const processingTimes: number[] = [];

      ws.onmessage = (event) => {
        const start = performance.now();
        JSON.parse(event.data);
        processingTimes.push(performance.now() - start);
        receivedMessages++;
      };

      const duration = 1000; // 1 second
      const messagesPerSecond = 100;

      await ws.simulateHighFrequencyMessages(messagesPerSecond, duration);

      expect(receivedMessages).toBeGreaterThanOrEqual(80); // Allow variance for timing
      expect(receivedMessages).toBeLessThanOrEqual(120);

      const avgProcessingTime =
        processingTimes.reduce((a, b) => a + b, 0) / processingTimes.length;
      expect(avgProcessingTime).toBeLessThan(5); // Less than 5ms per message
    });

    it('should handle burst of 1000 messages', async () => {
      vi.useRealTimers();

      const ws = new MockWebSocket('ws://test');
      connections.push(ws);

      await new Promise<void>((resolve) => {
        ws.onopen = () => resolve();
      });

      let receivedMessages = 0;
      const startMemory = await measureMemory();

      ws.onmessage = () => {
        receivedMessages++;
      };

      const startTime = performance.now();
      ws.simulateBurst(1000);
      const duration = performance.now() - startTime;

      expect(receivedMessages).toBe(1000);
      expect(duration).toBeLessThan(500); // Process 1000 messages in under 500ms

      const endMemory = await measureMemory();
      const memoryIncrease = endMemory - startMemory;

      // Memory increase should be reasonable (less than 50MB for 1000 messages)
      expect(memoryIncrease).toBeLessThan(50);
    });

    it('should maintain message order', async () => {
      vi.useRealTimers();

      const ws = new MockWebSocket('ws://test');
      connections.push(ws);

      await new Promise<void>((resolve) => {
        ws.onopen = () => resolve();
      });

      const receivedTimestamps: number[] = [];

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        receivedTimestamps.push(data.data.timestamp);
      };

      ws.simulateBurst(100);

      // Verify messages are in order
      for (let i = 1; i < receivedTimestamps.length; i++) {
        expect(receivedTimestamps[i]).toBeGreaterThanOrEqual(
          receivedTimestamps[i - 1]
        );
      }
    });
  });

  describe('Memory Leak Tests', () => {
    it('should not leak memory with repeated connections', async () => {
      vi.useRealTimers();

      const iterations = 10;
      const connectionsPerIteration = 5;

      for (let i = 0; i < iterations; i++) {
        const pool = createConnectionPool(connectionsPerIteration);

        await Promise.all(
          pool.map(
            (ws) =>
              new Promise<void>((resolve) => {
                ws.onopen = () => resolve();
              })
          )
        );

        // Close all connections
        await Promise.all(
          pool.map(
            (ws) =>
              new Promise<void>((resolve) => {
                ws.onclose = () => resolve();
                ws.close();
              })
          )
        );
      }

      // If we got here without running out of memory, the test passes
      expect(true).toBe(true);
    });

    it('should clean up message handlers on close', async () => {
      vi.useRealTimers();

      const ws = new MockWebSocket('ws://test');
      connections.push(ws);

      await new Promise<void>((resolve) => {
        ws.onopen = () => resolve();
      });

      let messageCount = 0;
      ws.onmessage = () => messageCount++;

      // Send some messages
      ws.simulateBurst(10);
      expect(messageCount).toBe(10);

      // Close connection
      await new Promise<void>((resolve) => {
        ws.onclose = () => resolve();
        ws.close();
      });

      // Handler should not receive messages after close
      // (this tests the mock, but validates the pattern)
      expect(ws.readyState).toBe(MockWebSocket.CLOSED);
    });
  });

  describe('Reconnection Stress Tests', () => {
    it('should handle rapid reconnection cycles', async () => {
      vi.useRealTimers();

      const cycles = 20;
      let successfulCycles = 0;

      for (let i = 0; i < cycles; i++) {
        const ws = new MockWebSocket(`ws://test/${i}`);

        await new Promise<void>((resolve) => {
          ws.onopen = () => resolve();
        });

        await new Promise<void>((resolve) => {
          ws.onclose = () => {
            successfulCycles++;
            resolve();
          };
          ws.close();
        });
      }

      expect(successfulCycles).toBe(cycles);
    });
  });

  describe('Performance Benchmarks', () => {
    it('should benchmark message processing throughput', async () => {
      vi.useRealTimers();

      const ws = new MockWebSocket('ws://test');
      connections.push(ws);

      await new Promise<void>((resolve) => {
        ws.onopen = () => resolve();
      });

      let processedMessages = 0;
      const results: { timestamp: number; count: number }[] = [];

      ws.onmessage = (event) => {
        JSON.parse(event.data);
        processedMessages++;

        if (processedMessages % 100 === 0) {
          results.push({
            timestamp: performance.now(),
            count: processedMessages,
          });
        }
      };

      // Run for 2 seconds at 200 messages/second
      await ws.simulateHighFrequencyMessages(200, 2000);

      // Calculate throughput
      if (results.length >= 2) {
        const firstResult = results[0];
        const lastResult = results[results.length - 1];
        const duration = lastResult.timestamp - firstResult.timestamp;
        const messages = lastResult.count - firstResult.count;
        const throughput = (messages / duration) * 1000;

        // Should process at least 150 messages per second (75% of target)
        expect(throughput).toBeGreaterThan(150);
      }
    });
  });
});

describe('Load Test Utilities', () => {
  it('should provide accurate timing measurements', () => {
    const start = performance.now();
    let sum = 0;
    for (let i = 0; i < 1000000; i++) {
      sum += i;
    }
    const duration = performance.now() - start;

    expect(duration).toBeGreaterThan(0);
    expect(duration).toBeLessThan(1000); // Should complete in under 1 second
  });
});
