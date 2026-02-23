import React, { createContext, useContext, useState, useEffect, useCallback, ReactNode } from 'react';
import { useChain } from './ChainContext';
import { TransactionType } from '../config/chains';
import { safeGetLocalStorageItem, safeSetLocalStorageItem } from '../utils/safeStorage';

export interface ScheduledTransaction {
  id: string;
  chainId: number;
  txType: TransactionType;
  targetGasPrice: number;  // gwei
  maxGasPrice: number;     // max willing to pay
  status: 'pending' | 'ready' | 'executed' | 'expired' | 'cancelled';
  createdAt: number;
  expiresAt: number;       // timestamp
  notified: boolean;
  // Optional tx details
  toAddress?: string;
  amount?: string;
  data?: string;
}

interface SchedulerContextType {
  // Scheduled transactions
  transactions: ScheduledTransaction[];
  addTransaction: (tx: Omit<ScheduledTransaction, 'id' | 'status' | 'createdAt' | 'notified'>) => string;
  removeTransaction: (id: string) => void;
  updateTransaction: (id: string, updates: Partial<ScheduledTransaction>) => void;

  // Ready transactions (gas price met)
  readyTransactions: ScheduledTransaction[];

  // Execute
  markExecuted: (id: string) => void;
  markCancelled: (id: string) => void;

  // Stats
  pendingCount: number;
  readyCount: number;
}

const SchedulerContext = createContext<SchedulerContextType | undefined>(undefined);

const STORAGE_KEY = 'gweizy_scheduled_transactions';

const generateId = () => `tx_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

export const SchedulerProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const { multiChainGas } = useChain();
  const [transactions, setTransactions] = useState<ScheduledTransaction[]>(() => {
    if (typeof window !== 'undefined') {
      const saved = safeGetLocalStorageItem(STORAGE_KEY);
      if (saved) {
        try {
          return JSON.parse(saved);
        } catch {
          return [];
        }
      }
    }
    return [];
  });

  // Persist to localStorage
  useEffect(() => {
    safeSetLocalStorageItem(STORAGE_KEY, JSON.stringify(transactions));
  }, [transactions]);

  // Check for ready transactions and expired ones
  useEffect(() => {
    const now = Date.now();

    setTransactions(prev => prev.map(tx => {
      // Skip non-pending
      if (tx.status !== 'pending') return tx;

      // Check expiry
      if (tx.expiresAt < now) {
        return { ...tx, status: 'expired' as const };
      }

      // Check if gas price target met
      const currentGas = multiChainGas[tx.chainId]?.gasPrice || Infinity;
      if (currentGas <= tx.targetGasPrice) {
        // Notify if not already
        if (!tx.notified && 'Notification' in window && Notification.permission === 'granted') {
          new Notification('Gweizy: Gas Target Reached!', {
            body: `Gas on chain ${tx.chainId} is now ${currentGas.toFixed(4)} gwei (target: ${tx.targetGasPrice.toFixed(4)})`,
            icon: '/favicon.ico'
          });
        }
        return { ...tx, status: 'ready' as const, notified: true };
      }

      return tx;
    }));
  }, [multiChainGas]);

  const addTransaction = useCallback((tx: Omit<ScheduledTransaction, 'id' | 'status' | 'createdAt' | 'notified'>) => {
    const id = generateId();
    const newTx: ScheduledTransaction = {
      ...tx,
      id,
      status: 'pending',
      createdAt: Date.now(),
      notified: false
    };

    setTransactions(prev => [...prev, newTx]);
    return id;
  }, []);

  const removeTransaction = useCallback((id: string) => {
    setTransactions(prev => prev.filter(tx => tx.id !== id));
  }, []);

  const updateTransaction = useCallback((id: string, updates: Partial<ScheduledTransaction>) => {
    setTransactions(prev => prev.map(tx =>
      tx.id === id ? { ...tx, ...updates } : tx
    ));
  }, []);

  const markExecuted = useCallback((id: string) => {
    updateTransaction(id, { status: 'executed' });
  }, [updateTransaction]);

  const markCancelled = useCallback((id: string) => {
    updateTransaction(id, { status: 'cancelled' });
  }, [updateTransaction]);

  const readyTransactions = transactions.filter(tx => tx.status === 'ready');
  const pendingCount = transactions.filter(tx => tx.status === 'pending').length;
  const readyCount = readyTransactions.length;

  const value: SchedulerContextType = {
    transactions,
    addTransaction,
    removeTransaction,
    updateTransaction,
    readyTransactions,
    markExecuted,
    markCancelled,
    pendingCount,
    readyCount
  };

  return (
    <SchedulerContext.Provider value={value}>
      {children}
    </SchedulerContext.Provider>
  );
};

export const useScheduler = (): SchedulerContextType => {
  const context = useContext(SchedulerContext);
  if (!context) {
    throw new Error('useScheduler must be used within a SchedulerProvider');
  }
  return context;
};
