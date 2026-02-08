/**
 * Integration tests for the transaction scheduling flow
 */

import React from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act, waitFor } from '@testing-library/react';
import { SchedulerProvider, useScheduler, ScheduledTransaction } from '../../contexts/SchedulerContext';
import { ChainProvider } from '../../contexts/ChainContext';

// Mock localStorage
const localStorageMock = (() => {
  let store: Record<string, string> = {};
  return {
    getItem: vi.fn((key: string) => store[key] || null),
    setItem: vi.fn((key: string, value: string) => { store[key] = value; }),
    removeItem: vi.fn((key: string) => { delete store[key]; }),
    clear: vi.fn(() => { store = {}; })
  };
})();
Object.defineProperty(window, 'localStorage', { value: localStorageMock });

// Mock WebSocket and fetch for ChainContext
vi.mock('../../utils/websocket', () => ({
  createWebSocket: vi.fn(() => ({
    on: vi.fn(),
    off: vi.fn(),
    emit: vi.fn(),
    connect: vi.fn(),
    disconnect: vi.fn(),
    isConnected: vi.fn(() => false)
  }))
}));

global.fetch = vi.fn(() => Promise.resolve({
  ok: true,
  json: () => Promise.resolve({ success: true, gasPrice: 25 })
})) as unknown as typeof fetch;

// Wrapper with both providers
const wrapper = ({ children }: { children: React.ReactNode }) => (
  <ChainProvider>
    <SchedulerProvider>
      {children}
    </SchedulerProvider>
  </ChainProvider>
);

describe('Transaction Scheduling Flow', () => {
  beforeEach(() => {
    localStorageMock.clear();
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('Adding Transactions', () => {
    it('should add a new scheduled transaction', async () => {
      const { result } = renderHook(() => useScheduler(), { wrapper });

      const txId = act(() => {
        return result.current.addTransaction({
          chainId: 1,
          txType: 'swap',
          targetGasPrice: 20,
          maxGasPrice: 30,
          expiresAt: Date.now() + 86400000 // 24 hours
        });
      });

      expect(txId).toBeDefined();
      expect(result.current.transactions).toHaveLength(1);
      expect(result.current.transactions[0].status).toBe('pending');
      expect(result.current.transactions[0].txType).toBe('swap');
    });

    it('should generate unique IDs for transactions', async () => {
      const { result } = renderHook(() => useScheduler(), { wrapper });

      let txId1: string = '';
      let txId2: string = '';

      act(() => {
        txId1 = result.current.addTransaction({
          chainId: 1,
          txType: 'swap',
          targetGasPrice: 20,
          maxGasPrice: 30,
          expiresAt: Date.now() + 86400000
        });
      });

      act(() => {
        txId2 = result.current.addTransaction({
          chainId: 1,
          txType: 'transfer',
          targetGasPrice: 15,
          maxGasPrice: 25,
          expiresAt: Date.now() + 86400000
        });
      });

      expect(txId1).not.toBe(txId2);
      expect(result.current.transactions).toHaveLength(2);
    });

    it('should persist transactions to localStorage', async () => {
      const { result } = renderHook(() => useScheduler(), { wrapper });

      act(() => {
        result.current.addTransaction({
          chainId: 1,
          txType: 'swap',
          targetGasPrice: 20,
          maxGasPrice: 30,
          expiresAt: Date.now() + 86400000
        });
      });

      await waitFor(() => {
        expect(localStorageMock.setItem).toHaveBeenCalled();
      });

      const savedData = localStorageMock.setItem.mock.calls.find(
        call => call[0] === 'gweizy_scheduled_transactions'
      );
      expect(savedData).toBeDefined();
    });
  });

  describe('Removing Transactions', () => {
    it('should remove a transaction by ID', async () => {
      const { result } = renderHook(() => useScheduler(), { wrapper });

      let txId: string = '';
      act(() => {
        txId = result.current.addTransaction({
          chainId: 1,
          txType: 'swap',
          targetGasPrice: 20,
          maxGasPrice: 30,
          expiresAt: Date.now() + 86400000
        });
      });

      expect(result.current.transactions).toHaveLength(1);

      act(() => {
        result.current.removeTransaction(txId);
      });

      expect(result.current.transactions).toHaveLength(0);
    });

    it('should handle removing non-existent transaction gracefully', async () => {
      const { result } = renderHook(() => useScheduler(), { wrapper });

      act(() => {
        result.current.addTransaction({
          chainId: 1,
          txType: 'swap',
          targetGasPrice: 20,
          maxGasPrice: 30,
          expiresAt: Date.now() + 86400000
        });
      });

      act(() => {
        result.current.removeTransaction('non-existent-id');
      });

      // Should still have the original transaction
      expect(result.current.transactions).toHaveLength(1);
    });
  });

  describe('Updating Transactions', () => {
    it('should update transaction properties', async () => {
      const { result } = renderHook(() => useScheduler(), { wrapper });

      let txId: string = '';
      act(() => {
        txId = result.current.addTransaction({
          chainId: 1,
          txType: 'swap',
          targetGasPrice: 20,
          maxGasPrice: 30,
          expiresAt: Date.now() + 86400000
        });
      });

      act(() => {
        result.current.updateTransaction(txId, { targetGasPrice: 15 });
      });

      expect(result.current.transactions[0].targetGasPrice).toBe(15);
    });
  });

  describe('Transaction Status Management', () => {
    it('should mark transaction as executed', async () => {
      const { result } = renderHook(() => useScheduler(), { wrapper });

      let txId: string = '';
      act(() => {
        txId = result.current.addTransaction({
          chainId: 1,
          txType: 'swap',
          targetGasPrice: 20,
          maxGasPrice: 30,
          expiresAt: Date.now() + 86400000
        });
      });

      act(() => {
        result.current.markExecuted(txId);
      });

      expect(result.current.transactions[0].status).toBe('executed');
    });

    it('should mark transaction as cancelled', async () => {
      const { result } = renderHook(() => useScheduler(), { wrapper });

      let txId: string = '';
      act(() => {
        txId = result.current.addTransaction({
          chainId: 1,
          txType: 'swap',
          targetGasPrice: 20,
          maxGasPrice: 30,
          expiresAt: Date.now() + 86400000
        });
      });

      act(() => {
        result.current.markCancelled(txId);
      });

      expect(result.current.transactions[0].status).toBe('cancelled');
    });
  });

  describe('Transaction Counts', () => {
    it('should track pending count correctly', async () => {
      const { result } = renderHook(() => useScheduler(), { wrapper });

      expect(result.current.pendingCount).toBe(0);

      act(() => {
        result.current.addTransaction({
          chainId: 1,
          txType: 'swap',
          targetGasPrice: 20,
          maxGasPrice: 30,
          expiresAt: Date.now() + 86400000
        });
      });

      expect(result.current.pendingCount).toBe(1);

      act(() => {
        result.current.addTransaction({
          chainId: 1,
          txType: 'transfer',
          targetGasPrice: 15,
          maxGasPrice: 25,
          expiresAt: Date.now() + 86400000
        });
      });

      expect(result.current.pendingCount).toBe(2);
    });

    it('should decrease pending count when transaction is cancelled', async () => {
      const { result } = renderHook(() => useScheduler(), { wrapper });

      let txId: string = '';
      act(() => {
        txId = result.current.addTransaction({
          chainId: 1,
          txType: 'swap',
          targetGasPrice: 20,
          maxGasPrice: 30,
          expiresAt: Date.now() + 86400000
        });
      });

      expect(result.current.pendingCount).toBe(1);

      act(() => {
        result.current.markCancelled(txId);
      });

      expect(result.current.pendingCount).toBe(0);
    });
  });

  describe('Transaction with Details', () => {
    it('should store optional transaction details', async () => {
      const { result } = renderHook(() => useScheduler(), { wrapper });

      act(() => {
        result.current.addTransaction({
          chainId: 1,
          txType: 'transfer',
          targetGasPrice: 20,
          maxGasPrice: 30,
          expiresAt: Date.now() + 86400000,
          toAddress: '0x1234567890123456789012345678901234567890',
          amount: '1.5'
        });
      });

      const tx = result.current.transactions[0];
      expect(tx.toAddress).toBe('0x1234567890123456789012345678901234567890');
      expect(tx.amount).toBe('1.5');
    });
  });
});
