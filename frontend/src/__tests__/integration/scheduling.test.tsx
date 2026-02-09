/**
 * Integration tests for the transaction scheduling flow
 */

import React from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, act, waitFor } from '@testing-library/react';
import { SchedulerProvider, useScheduler } from '../../contexts/SchedulerContext';
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

  describe('Ready Transactions', () => {
    it('should filter ready transactions correctly', async () => {
      const { result } = renderHook(() => useScheduler(), { wrapper });

      let txId1: string = '';

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
        result.current.addTransaction({
          chainId: 1,
          txType: 'transfer',
          targetGasPrice: 15,
          maxGasPrice: 25,
          expiresAt: Date.now() + 86400000
        });
      });

      // Initially no ready transactions
      expect(result.current.readyTransactions).toHaveLength(0);
      expect(result.current.readyCount).toBe(0);

      // Manually set one to ready status
      act(() => {
        result.current.updateTransaction(txId1, { status: 'ready' });
      });

      expect(result.current.readyTransactions).toHaveLength(1);
      expect(result.current.readyCount).toBe(1);
      expect(result.current.readyTransactions[0].id).toBe(txId1);
    });

    it('should not include non-ready transactions in readyTransactions', async () => {
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

      // Pending should not appear
      expect(result.current.readyTransactions).toHaveLength(0);

      // Executed should not appear
      act(() => {
        result.current.markExecuted(txId);
      });

      expect(result.current.readyTransactions).toHaveLength(0);
    });
  });

  describe('useScheduler Hook', () => {
    it('should throw when used outside SchedulerProvider', () => {
      const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

      expect(() => {
        renderHook(() => useScheduler());
      }).toThrow('useScheduler must be used within a SchedulerProvider');

      consoleSpy.mockRestore();
    });

    it('should return all required context properties', async () => {
      const { result } = renderHook(() => useScheduler(), { wrapper });

      expect(result.current).toHaveProperty('transactions');
      expect(result.current).toHaveProperty('addTransaction');
      expect(result.current).toHaveProperty('removeTransaction');
      expect(result.current).toHaveProperty('updateTransaction');
      expect(result.current).toHaveProperty('readyTransactions');
      expect(result.current).toHaveProperty('markExecuted');
      expect(result.current).toHaveProperty('markCancelled');
      expect(result.current).toHaveProperty('pendingCount');
      expect(result.current).toHaveProperty('readyCount');
    });
  });

  describe('Transaction Initialization', () => {
    it('should load transactions from localStorage on init', async () => {
      const savedTransactions = [
        {
          id: 'tx_saved_1',
          chainId: 1,
          txType: 'swap',
          targetGasPrice: 20,
          maxGasPrice: 30,
          status: 'pending',
          createdAt: Date.now(),
          expiresAt: Date.now() + 86400000,
          notified: false
        }
      ];

      localStorageMock.getItem.mockReturnValue(JSON.stringify(savedTransactions));

      const { result } = renderHook(() => useScheduler(), { wrapper });

      await waitFor(() => {
        expect(result.current.transactions).toHaveLength(1);
        expect(result.current.transactions[0].id).toBe('tx_saved_1');
      });
    });

    it('should handle invalid localStorage data gracefully', async () => {
      localStorageMock.getItem.mockReturnValue('invalid json');

      const { result } = renderHook(() => useScheduler(), { wrapper });

      // Should start with empty array if JSON parse fails
      expect(result.current.transactions).toHaveLength(0);
    });
  });

  describe('Multiple Transaction Types', () => {
    it('should handle different transaction types', async () => {
      const { result } = renderHook(() => useScheduler(), { wrapper });

      const txTypes = ['transfer', 'swap', 'bridge', 'approve'] as const;

      txTypes.forEach((txType, index) => {
        act(() => {
          result.current.addTransaction({
            chainId: 1,
            txType,
            targetGasPrice: 20 + index,
            maxGasPrice: 30 + index,
            expiresAt: Date.now() + 86400000
          });
        });
      });

      expect(result.current.transactions).toHaveLength(4);
      expect(result.current.transactions.map(tx => tx.txType)).toEqual(txTypes);
    });
  });
});
