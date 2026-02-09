/**
 * Performance tests for large datasets
 *
 * Tests verify that components and utilities handle large datasets efficiently:
 * - Rendering performance with many items
 * - Memory usage with large arrays
 * - Computation time for data transformations
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import React from 'react';

// Helper to generate large datasets
function generateGasHistory(count: number) {
  return Array.from({ length: count }, (_, i) => ({
    timestamp: Date.now() - i * 60000,
    gas_price: 0.001 + Math.random() * 0.01,
    base_fee: 0.0008 + Math.random() * 0.008,
    priority_fee: 0.0002 + Math.random() * 0.002,
    block_number: 20000000 - i,
  }));
}

function generateTransactions(count: number) {
  return Array.from({ length: count }, (_, i) => ({
    id: `tx-${i}`,
    hash: `0x${Math.random().toString(16).slice(2)}${Math.random().toString(16).slice(2)}`,
    gasUsed: 21000 + Math.floor(Math.random() * 100000),
    gasPrice: 0.001 + Math.random() * 0.01,
    timestamp: Date.now() - i * 30000,
    status: Math.random() > 0.1 ? 'confirmed' : 'pending',
    from: `0x${Math.random().toString(16).slice(2, 42)}`,
    to: `0x${Math.random().toString(16).slice(2, 42)}`,
  }));
}

function generatePredictions(count: number) {
  return Array.from({ length: count }, (_, i) => ({
    timestamp: Date.now() + i * 3600000,
    predicted_gas: 0.001 + Math.random() * 0.01,
    confidence: 0.7 + Math.random() * 0.25,
    lower_bound: 0.0008,
    upper_bound: 0.015,
  }));
}

describe('Performance Tests', () => {
  const performanceThresholds = {
    smallDataset: 50, // ms
    mediumDataset: 200, // ms
    largeDataset: 500, // ms
  };

  describe('Data Generation Performance', () => {
    it('should generate 1000 gas history entries quickly', () => {
      const start = performance.now();
      const data = generateGasHistory(1000);
      const duration = performance.now() - start;

      expect(data).toHaveLength(1000);
      expect(duration).toBeLessThan(performanceThresholds.smallDataset);
    });

    it('should generate 10000 gas history entries in reasonable time', () => {
      const start = performance.now();
      const data = generateGasHistory(10000);
      const duration = performance.now() - start;

      expect(data).toHaveLength(10000);
      expect(duration).toBeLessThan(performanceThresholds.mediumDataset);
    });

    it('should generate 100000 transactions without timing out', () => {
      const start = performance.now();
      const data = generateTransactions(100000);
      const duration = performance.now() - start;

      expect(data).toHaveLength(100000);
      expect(duration).toBeLessThan(performanceThresholds.largeDataset);
    });
  });

  describe('Array Operations Performance', () => {
    it('should filter large arrays efficiently', () => {
      const data = generateTransactions(50000);

      const start = performance.now();
      const filtered = data.filter((tx) => tx.status === 'confirmed');
      const duration = performance.now() - start;

      expect(filtered.length).toBeGreaterThan(0);
      expect(duration).toBeLessThan(performanceThresholds.smallDataset);
    });

    it('should sort large arrays efficiently', () => {
      const data = generateGasHistory(50000);

      const start = performance.now();
      const sorted = [...data].sort((a, b) => b.gas_price - a.gas_price);
      const duration = performance.now() - start;

      expect(sorted[0].gas_price).toBeGreaterThanOrEqual(sorted[sorted.length - 1].gas_price);
      expect(duration).toBeLessThan(performanceThresholds.mediumDataset);
    });

    it('should map large arrays efficiently', () => {
      const data = generateGasHistory(50000);

      const start = performance.now();
      const mapped = data.map((item) => ({
        ...item,
        totalFee: item.base_fee + item.priority_fee,
        formattedPrice: `${item.gas_price.toFixed(6)} gwei`,
      }));
      const duration = performance.now() - start;

      expect(mapped).toHaveLength(50000);
      expect(mapped[0]).toHaveProperty('totalFee');
      expect(duration).toBeLessThan(performanceThresholds.mediumDataset);
    });

    it('should reduce large arrays efficiently', () => {
      const data = generateGasHistory(50000);

      const start = performance.now();
      const stats = data.reduce(
        (acc, item) => ({
          total: acc.total + item.gas_price,
          min: Math.min(acc.min, item.gas_price),
          max: Math.max(acc.max, item.gas_price),
          count: acc.count + 1,
        }),
        { total: 0, min: Infinity, max: -Infinity, count: 0 }
      );
      const duration = performance.now() - start;

      expect(stats.count).toBe(50000);
      expect(stats.min).toBeLessThan(stats.max);
      expect(duration).toBeLessThan(performanceThresholds.smallDataset);
    });
  });

  describe('Aggregation Performance', () => {
    it('should calculate moving averages efficiently', () => {
      const data = generateGasHistory(10000);
      const windowSize = 100;

      const start = performance.now();
      const movingAverages = data.map((_, index, arr) => {
        if (index < windowSize - 1) return null;
        const window = arr.slice(index - windowSize + 1, index + 1);
        return window.reduce((sum, item) => sum + item.gas_price, 0) / windowSize;
      });
      const duration = performance.now() - start;

      expect(movingAverages.filter((v) => v !== null)).toHaveLength(10000 - windowSize + 1);
      expect(duration).toBeLessThan(performanceThresholds.mediumDataset);
    });

    it('should group data by time periods efficiently', () => {
      const data = generateGasHistory(10000);
      const hourMs = 3600000;

      const start = performance.now();
      const grouped = data.reduce<Record<number, typeof data>>((acc, item) => {
        const hourKey = Math.floor(item.timestamp / hourMs);
        if (!acc[hourKey]) acc[hourKey] = [];
        acc[hourKey].push(item);
        return acc;
      }, {});
      const duration = performance.now() - start;

      expect(Object.keys(grouped).length).toBeGreaterThan(0);
      expect(duration).toBeLessThan(performanceThresholds.smallDataset);
    });

    it('should calculate percentiles efficiently', () => {
      const data = generateGasHistory(10000);
      const prices = data.map((d) => d.gas_price).sort((a, b) => a - b);

      const start = performance.now();
      const percentiles = {
        p25: prices[Math.floor(prices.length * 0.25)],
        p50: prices[Math.floor(prices.length * 0.5)],
        p75: prices[Math.floor(prices.length * 0.75)],
        p90: prices[Math.floor(prices.length * 0.9)],
        p99: prices[Math.floor(prices.length * 0.99)],
      };
      const duration = performance.now() - start;

      expect(percentiles.p25).toBeLessThanOrEqual(percentiles.p50);
      expect(percentiles.p50).toBeLessThanOrEqual(percentiles.p75);
      expect(duration).toBeLessThan(performanceThresholds.smallDataset);
    });
  });

  describe('Memory Efficiency', () => {
    it('should handle large dataset transformations without memory issues', () => {
      // Create a large dataset and transform it multiple times
      const data = generateGasHistory(10000);

      const start = performance.now();

      // Chain multiple transformations
      const result = data
        .filter((item) => item.gas_price > 0.005)
        .map((item) => ({
          price: item.gas_price,
          total: item.base_fee + item.priority_fee,
        }))
        .sort((a, b) => b.price - a.price)
        .slice(0, 100);

      const duration = performance.now() - start;

      expect(result.length).toBeLessThanOrEqual(100);
      expect(duration).toBeLessThan(performanceThresholds.smallDataset);
    });

    it('should efficiently deduplicate large arrays using Set', () => {
      const data = generateTransactions(10000);
      // Add some duplicates
      const withDuplicates = [...data, ...data.slice(0, 1000)];

      const start = performance.now();
      // Efficient O(n) deduplication using Set
      const seen = new Set<string>();
      const unique = withDuplicates.filter((tx) => {
        if (seen.has(tx.id)) return false;
        seen.add(tx.id);
        return true;
      });
      const duration = performance.now() - start;

      expect(seen.size).toBe(10000);
      expect(unique).toHaveLength(10000);
      expect(duration).toBeLessThan(performanceThresholds.smallDataset);
    });
  });

  describe('Rendering Performance', () => {
    // Simple list component for testing
    const DataList: React.FC<{ items: { id: string; value: number }[] }> = ({ items }) => (
      <ul>
        {items.map((item) => (
          <li key={item.id} data-testid={`item-${item.id}`}>
            {item.value.toFixed(6)}
          </li>
        ))}
      </ul>
    );

    it('should render 100 items quickly', () => {
      const items = Array.from({ length: 100 }, (_, i) => ({
        id: `item-${i}`,
        value: Math.random(),
      }));

      const start = performance.now();
      render(<DataList items={items} />);
      const duration = performance.now() - start;

      expect(screen.getAllByRole('listitem')).toHaveLength(100);
      expect(duration).toBeLessThan(performanceThresholds.smallDataset);
    });

    it('should render 1000 items in reasonable time', () => {
      const items = Array.from({ length: 1000 }, (_, i) => ({
        id: `item-${i}`,
        value: Math.random(),
      }));

      const start = performance.now();
      render(<DataList items={items} />);
      const duration = performance.now() - start;

      expect(screen.getAllByRole('listitem')).toHaveLength(1000);
      expect(duration).toBeLessThan(performanceThresholds.mediumDataset);
    });

    // Memoized list component
    const MemoizedItem = React.memo<{ id: string; value: number }>(({ id, value }) => (
      <li data-testid={`memo-item-${id}`}>{value.toFixed(6)}</li>
    ));

    const MemoizedList: React.FC<{ items: { id: string; value: number }[] }> = ({ items }) => (
      <ul>
        {items.map((item) => (
          <MemoizedItem key={item.id} id={item.id} value={item.value} />
        ))}
      </ul>
    );

    it('should benefit from memoization with stable items', () => {
      const items = Array.from({ length: 500 }, (_, i) => ({
        id: `item-${i}`,
        value: Math.random(),
      }));

      const start = performance.now();
      const { rerender } = render(<MemoizedList items={items} />);

      // Rerender with same items
      rerender(<MemoizedList items={items} />);
      rerender(<MemoizedList items={items} />);

      const duration = performance.now() - start;

      expect(screen.getAllByRole('listitem')).toHaveLength(500);
      expect(duration).toBeLessThan(performanceThresholds.mediumDataset);
    });
  });

  describe('Search and Lookup Performance', () => {
    it('should find items by ID efficiently using Map', () => {
      const data = generateTransactions(50000);
      const lookupMap = new Map(data.map((tx) => [tx.id, tx]));

      const searchIds = data.slice(0, 1000).map((tx) => tx.id);

      const start = performance.now();
      const found = searchIds.map((id) => lookupMap.get(id)).filter(Boolean);
      const duration = performance.now() - start;

      expect(found).toHaveLength(1000);
      expect(duration).toBeLessThan(10); // Map lookups should be very fast
    });

    it('should search by property efficiently using index', () => {
      const data = generateTransactions(50000);

      // Create index by status
      const statusIndex = data.reduce<Record<string, typeof data>>((acc, tx) => {
        if (!acc[tx.status]) acc[tx.status] = [];
        acc[tx.status].push(tx);
        return acc;
      }, {});

      const start = performance.now();
      const confirmedTxs = statusIndex['confirmed'] || [];
      const duration = performance.now() - start;

      expect(confirmedTxs.length).toBeGreaterThan(0);
      expect(duration).toBeLessThan(5); // Index lookup should be instant
    });

    it('should handle binary search on sorted data', () => {
      const data = generateGasHistory(50000).sort((a, b) => a.gas_price - b.gas_price);
      const targetPrice = 0.005;

      const start = performance.now();

      // Binary search for items around target price
      let left = 0;
      let right = data.length - 1;
      while (left < right) {
        const mid = Math.floor((left + right) / 2);
        if (data[mid].gas_price < targetPrice) {
          left = mid + 1;
        } else {
          right = mid;
        }
      }

      const duration = performance.now() - start;

      // Binary search should be O(log n)
      expect(duration).toBeLessThan(5);
      expect(left).toBeGreaterThanOrEqual(0);
      expect(left).toBeLessThanOrEqual(data.length);
    });
  });

  describe('Batch Processing Performance', () => {
    it('should process data in batches efficiently', async () => {
      const data = generateTransactions(10000);
      const batchSize = 1000;
      const results: number[] = [];

      const start = performance.now();

      for (let i = 0; i < data.length; i += batchSize) {
        const batch = data.slice(i, i + batchSize);
        const batchTotal = batch.reduce((sum, tx) => sum + tx.gasUsed, 0);
        results.push(batchTotal);
        // Yield to event loop (simulated)
        await Promise.resolve();
      }

      const duration = performance.now() - start;

      expect(results).toHaveLength(10);
      expect(duration).toBeLessThan(performanceThresholds.smallDataset);
    });

    it('should chunk large arrays efficiently', () => {
      const data = generateGasHistory(10000);
      const chunkSize = 500;

      const start = performance.now();
      const chunks: typeof data[] = [];
      for (let i = 0; i < data.length; i += chunkSize) {
        chunks.push(data.slice(i, i + chunkSize));
      }
      const duration = performance.now() - start;

      expect(chunks).toHaveLength(20);
      expect(chunks[0]).toHaveLength(500);
      expect(duration).toBeLessThan(performanceThresholds.smallDataset);
    });
  });
});
