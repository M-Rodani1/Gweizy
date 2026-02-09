/**
 * useTransactionScheduler - Hook for scheduling gas-optimized transactions.
 *
 * Provides a simplified API for scheduling transactions with gas price targets.
 * Includes form state management, validation, and notification handling.
 *
 * @module hooks/useTransactionScheduler
 */

import { useState, useCallback, useMemo, useEffect } from 'react';
import { useScheduler, ScheduledTransaction } from '../contexts/SchedulerContext';
import { useChain } from '../contexts/ChainContext';
import { TransactionType } from '../config/chains';

/**
 * Form state for creating a scheduled transaction.
 */
export interface ScheduleFormState {
  chainId: number;
  txType: TransactionType;
  targetGasPrice: string;
  maxGasPrice: string;
  expiresIn: number; // hours
  toAddress?: string;
  amount?: string;
}

/**
 * Validation errors for the schedule form.
 */
export interface ScheduleFormErrors {
  targetGasPrice?: string;
  maxGasPrice?: string;
  expiresIn?: string;
  toAddress?: string;
  amount?: string;
}

/**
 * Return type for useTransactionScheduler hook.
 */
export interface UseTransactionSchedulerReturn {
  // Form state
  formState: ScheduleFormState;
  setFormState: React.Dispatch<React.SetStateAction<ScheduleFormState>>;
  errors: ScheduleFormErrors;
  isValid: boolean;

  // Form helpers
  updateField: <K extends keyof ScheduleFormState>(field: K, value: ScheduleFormState[K]) => void;
  resetForm: () => void;

  // Actions
  scheduleTransaction: () => string | null;
  cancelTransaction: (id: string) => void;
  retryTransaction: (id: string) => void;

  // Transaction data
  transactions: ScheduledTransaction[];
  pendingTransactions: ScheduledTransaction[];
  readyTransactions: ScheduledTransaction[];
  completedTransactions: ScheduledTransaction[];

  // Stats
  pendingCount: number;
  readyCount: number;
  hasReadyTransactions: boolean;

  // Current gas context
  currentGasPrice: number | null;
  isTargetMet: boolean;
  savingsEstimate: number | null;

  // Notifications
  requestNotificationPermission: () => Promise<boolean>;
  notificationPermission: NotificationPermission | 'unsupported';
}

const DEFAULT_FORM_STATE: ScheduleFormState = {
  chainId: 8453, // Base
  txType: 'transfer' as TransactionType,
  targetGasPrice: '',
  maxGasPrice: '',
  expiresIn: 24, // 24 hours default
};

/**
 * Hook for scheduling gas-optimized transactions.
 *
 * @example
 * ```tsx
 * const {
 *   formState,
 *   updateField,
 *   scheduleTransaction,
 *   errors,
 *   isValid,
 * } = useTransactionScheduler();
 *
 * return (
 *   <form onSubmit={(e) => { e.preventDefault(); scheduleTransaction(); }}>
 *     <input
 *       value={formState.targetGasPrice}
 *       onChange={(e) => updateField('targetGasPrice', e.target.value)}
 *     />
 *     {errors.targetGasPrice && <span>{errors.targetGasPrice}</span>}
 *     <button disabled={!isValid}>Schedule</button>
 *   </form>
 * );
 * ```
 */
export function useTransactionScheduler(
  initialState?: Partial<ScheduleFormState>
): UseTransactionSchedulerReturn {
  const scheduler = useScheduler();
  const { selectedChain, multiChainGas } = useChain();

  const [formState, setFormState] = useState<ScheduleFormState>({
    ...DEFAULT_FORM_STATE,
    chainId: selectedChain?.id ?? DEFAULT_FORM_STATE.chainId,
    ...initialState,
  });

  const [notificationPermission, setNotificationPermission] = useState<
    NotificationPermission | 'unsupported'
  >(() => {
    if (typeof window === 'undefined' || !('Notification' in window)) {
      return 'unsupported';
    }
    return Notification.permission;
  });

  // Update chain when selected chain changes
  useEffect(() => {
    if (selectedChain) {
      setFormState((prev) => ({ ...prev, chainId: selectedChain.id }));
    }
  }, [selectedChain]);

  // Validate form
  const errors = useMemo<ScheduleFormErrors>(() => {
    const errs: ScheduleFormErrors = {};

    const targetPrice = parseFloat(formState.targetGasPrice);
    const maxPrice = parseFloat(formState.maxGasPrice);

    if (!formState.targetGasPrice) {
      errs.targetGasPrice = 'Target gas price is required';
    } else if (isNaN(targetPrice) || targetPrice <= 0) {
      errs.targetGasPrice = 'Must be a positive number';
    }

    if (formState.maxGasPrice) {
      if (isNaN(maxPrice) || maxPrice <= 0) {
        errs.maxGasPrice = 'Must be a positive number';
      } else if (maxPrice < targetPrice) {
        errs.maxGasPrice = 'Max must be >= target price';
      }
    }

    if (formState.expiresIn <= 0) {
      errs.expiresIn = 'Must be greater than 0';
    } else if (formState.expiresIn > 168) {
      errs.expiresIn = 'Max 7 days (168 hours)';
    }

    if (formState.toAddress) {
      const isValidAddress = /^0x[a-fA-F0-9]{40}$/.test(formState.toAddress);
      if (!isValidAddress) {
        errs.toAddress = 'Invalid Ethereum address';
      }
    }

    if (formState.amount) {
      const amount = parseFloat(formState.amount);
      if (isNaN(amount) || amount <= 0) {
        errs.amount = 'Must be a positive number';
      }
    }

    return errs;
  }, [formState]);

  const isValid = Object.keys(errors).length === 0 && formState.targetGasPrice !== '';

  // Current gas price for selected chain
  const currentGasPrice = useMemo(() => {
    const chainGas = multiChainGas[formState.chainId];
    return chainGas?.gasPrice ?? null;
  }, [multiChainGas, formState.chainId]);

  // Check if target is currently met
  const isTargetMet = useMemo(() => {
    if (currentGasPrice === null || !formState.targetGasPrice) return false;
    const target = parseFloat(formState.targetGasPrice);
    return !isNaN(target) && currentGasPrice <= target;
  }, [currentGasPrice, formState.targetGasPrice]);

  // Estimate savings
  const savingsEstimate = useMemo(() => {
    if (currentGasPrice === null || !formState.targetGasPrice) return null;
    const target = parseFloat(formState.targetGasPrice);
    if (isNaN(target)) return null;

    const savings = currentGasPrice - target;
    return savings > 0 ? savings : null;
  }, [currentGasPrice, formState.targetGasPrice]);

  // Update a single field
  const updateField = useCallback(
    <K extends keyof ScheduleFormState>(field: K, value: ScheduleFormState[K]) => {
      setFormState((prev) => ({ ...prev, [field]: value }));
    },
    []
  );

  // Reset form to defaults
  const resetForm = useCallback(() => {
    setFormState({
      ...DEFAULT_FORM_STATE,
      chainId: selectedChain?.id ?? DEFAULT_FORM_STATE.chainId,
    });
  }, [selectedChain]);

  // Schedule a new transaction
  const scheduleTransaction = useCallback((): string | null => {
    if (!isValid) return null;

    const targetPrice = parseFloat(formState.targetGasPrice);
    const maxPrice = formState.maxGasPrice
      ? parseFloat(formState.maxGasPrice)
      : targetPrice * 1.5; // Default max is 150% of target

    const expiresAt = Date.now() + formState.expiresIn * 60 * 60 * 1000;

    const id = scheduler.addTransaction({
      chainId: formState.chainId,
      txType: formState.txType,
      targetGasPrice: targetPrice,
      maxGasPrice: maxPrice,
      expiresAt,
      toAddress: formState.toAddress,
      amount: formState.amount,
    });

    resetForm();
    return id;
  }, [formState, isValid, scheduler, resetForm]);

  // Cancel a transaction
  const cancelTransaction = useCallback(
    (id: string) => {
      scheduler.markCancelled(id);
    },
    [scheduler]
  );

  // Retry a cancelled/expired transaction
  const retryTransaction = useCallback(
    (id: string) => {
      const tx = scheduler.transactions.find((t) => t.id === id);
      if (!tx) return;

      // Create new transaction with same params
      scheduler.addTransaction({
        chainId: tx.chainId,
        txType: tx.txType,
        targetGasPrice: tx.targetGasPrice,
        maxGasPrice: tx.maxGasPrice,
        expiresAt: Date.now() + 24 * 60 * 60 * 1000, // Reset expiry
        toAddress: tx.toAddress,
        amount: tx.amount,
      });

      // Remove old transaction
      scheduler.removeTransaction(id);
    },
    [scheduler]
  );

  // Filter transactions by status
  const pendingTransactions = useMemo(
    () => scheduler.transactions.filter((tx) => tx.status === 'pending'),
    [scheduler.transactions]
  );

  const completedTransactions = useMemo(
    () =>
      scheduler.transactions.filter(
        (tx) => tx.status === 'executed' || tx.status === 'expired' || tx.status === 'cancelled'
      ),
    [scheduler.transactions]
  );

  // Request notification permission
  const requestNotificationPermission = useCallback(async (): Promise<boolean> => {
    if (typeof window === 'undefined' || !('Notification' in window)) {
      return false;
    }

    try {
      const permission = await Notification.requestPermission();
      setNotificationPermission(permission);
      return permission === 'granted';
    } catch {
      return false;
    }
  }, []);

  return {
    // Form state
    formState,
    setFormState,
    errors,
    isValid,

    // Form helpers
    updateField,
    resetForm,

    // Actions
    scheduleTransaction,
    cancelTransaction,
    retryTransaction,

    // Transaction data
    transactions: scheduler.transactions,
    pendingTransactions,
    readyTransactions: scheduler.readyTransactions,
    completedTransactions,

    // Stats
    pendingCount: scheduler.pendingCount,
    readyCount: scheduler.readyCount,
    hasReadyTransactions: scheduler.readyCount > 0,

    // Current gas context
    currentGasPrice,
    isTargetMet,
    savingsEstimate,

    // Notifications
    requestNotificationPermission,
    notificationPermission,
  };
}

export default useTransactionScheduler;
