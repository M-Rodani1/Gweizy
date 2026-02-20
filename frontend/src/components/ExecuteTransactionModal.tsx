import React, { useEffect, useState, useId, useMemo } from 'react';
import { AlertTriangle, CheckCircle, ExternalLink, Send, AlertCircle } from 'lucide-react';
import { SUPPORTED_CHAINS, TransactionType } from '../config/chains';
import { getTxLabel } from '../config/transactions';
import { useChain } from '../contexts/ChainContext';
import { usePreferences } from '../contexts/PreferencesContext';
import {
  connectWallet,
  getCurrentAccount,
  getCurrentChainId,
  sendTransaction,
  switchToChain
} from '../utils/wallet';
import {
  useFormValidation,
  required,
  ethereumAddress,
  positiveNumber
} from '../hooks/useFormValidation';

interface ExecuteTransactionModalProps {
  isOpen: boolean;
  onClose: () => void;
  chainId: number;
  txType?: TransactionType;
  gasGwei?: number | null;
  defaultToAddress?: string;
  defaultAmountEth?: string;
  defaultData?: string;
  onExecuted?: (txHash: string) => void;
}

const ExecuteTransactionModal: React.FC<ExecuteTransactionModalProps> = ({
  isOpen,
  onClose,
  chainId,
  txType = 'transfer',
  gasGwei,
  defaultToAddress,
  defaultAmountEth,
  defaultData,
  onExecuted
}) => {
  const { selectedChainId } = useChain();
  const { preferences } = usePreferences();

  // Generate unique IDs for form accessibility
  const recipientId = useId();
  const amountId = useId();
  const gasPriceId = useId();
  const gasLimitId = useId();
  const dataId = useId();
  const acknowledgeId = useId();
  const modalTitleId = useId();

  const [fromAddress, setFromAddress] = useState<string | null>(null);
  const [amountEth, setAmountEth] = useState(defaultAmountEth || '');
  const [data, setData] = useState(defaultData || '');
  const [gasLimit, setGasLimit] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [txHash, setTxHash] = useState<string | null>(null);
  const [acknowledged, setAcknowledged] = useState(false);

  // Validation schema for required fields
  const validationSchema = useMemo(() => ({
    toAddress: {
      initialValue: defaultToAddress || '',
      rules: [
        required('Recipient address is required'),
        ethereumAddress('Please enter a valid Ethereum address (0x...)')
      ]
    },
    gasPriceGwei: {
      initialValue: gasGwei ? gasGwei.toFixed(6) : '',
      rules: [
        required('Gas price is required'),
        positiveNumber('Gas price must be a positive number')
      ]
    }
  }), [defaultToAddress, gasGwei]);

  const {
    values,
    errors: fieldErrors,
    touched,
    getFieldProps,
    validateAll,
    reset: _resetForm,
    setValues
  } = useFormValidation(validationSchema);

  useEffect(() => {
    if (!isOpen) return;
    setError(null);
    setTxHash(null);
    setAcknowledged(false);
    setValues({
      toAddress: defaultToAddress || '',
      gasPriceGwei: gasGwei ? gasGwei.toFixed(6) : ''
    });
    setAmountEth(defaultAmountEth || '');
    setData(defaultData || '');
    setGasLimit('');

    const loadAccount = async () => {
      const account = await getCurrentAccount();
      setFromAddress(account);
    };
    loadAccount();
  }, [isOpen, defaultToAddress, defaultAmountEth, defaultData, gasGwei, setValues]);

  if (!isOpen) return null;

  const chain = SUPPORTED_CHAINS[chainId] || SUPPORTED_CHAINS[selectedChainId];
  const txLabel = getTxLabel(txType);

  const handleConnect = async () => {
    try {
      setError(null);
      const account = await connectWallet(chainId);
      setFromAddress(account);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to connect wallet');
    }
  };

  const handleExecute = async () => {
    setError(null);
    setTxHash(null);

    if (!fromAddress) {
      setError('Connect a wallet before executing.');
      return;
    }

    // Validate form fields
    if (!validateAll()) {
      return;
    }

    if (!acknowledged) {
      setError('Confirm you understand gas fees before executing.');
      return;
    }

    try {
      setIsSubmitting(true);
      const walletChainId = await getCurrentChainId();
      if (walletChainId !== chainId) {
        await switchToChain(chainId);
      }

      const txData = data.trim();
      const normalizedData = txData ? (txData.startsWith('0x') ? txData : `0x${txData}`) : undefined;
      const tx = await sendTransaction({
        to: values.toAddress,
        valueEth: amountEth || '0',
        gasPriceGwei: values.gasPriceGwei,
        gasLimit: gasLimit || undefined,
        data: normalizedData
      });
      setTxHash(tx);
      onExecuted?.(tx);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Transaction failed');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4" role="presentation">
      {/* Backdrop */}
      <div className="absolute inset-0" onClick={onClose} aria-hidden="true" />

      <div
        className="relative w-full max-w-xl rounded-2xl border border-gray-700 bg-gray-900 shadow-xl"
        role="dialog"
        aria-modal="true"
        aria-labelledby={modalTitleId}
      >
        <div className="flex items-center justify-between border-b border-gray-800 px-6 py-4">
          <div>
            <h2 id={modalTitleId} className="text-lg font-semibold text-white">Execute {txLabel}</h2>
            <p className="text-xs text-gray-400">
              {chain?.name} • Recommended gas {values.gasPriceGwei || 'auto'} gwei
            </p>
          </div>
          <button
            onClick={onClose}
            type="button"
            className="text-gray-400 hover:text-white p-1 rounded-lg focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:ring-offset-2 focus:ring-offset-gray-900"
            aria-label="Close execution modal"
          >
            <span aria-hidden="true">×</span>
          </button>
        </div>

        <div className="space-y-4 px-6 py-5">
          {!fromAddress ? (
            <div className="rounded-lg border border-amber-500/30 bg-amber-500/10 p-4 text-sm text-amber-200">
              Connect your wallet to execute transactions.
            </div>
          ) : (
            <div className="rounded-lg border border-gray-800 bg-gray-950/60 p-3 text-xs text-gray-400">
              From: {fromAddress}
            </div>
          )}

          <div>
            <label htmlFor={recipientId} className="text-xs text-gray-400">
              Recipient <span className="text-red-400" aria-hidden="true">*</span>
            </label>
            <input
              id={recipientId}
              type="text"
              {...getFieldProps('toAddress')}
              onChange={(e) => {
                const trimmed = e.target.value.trim();
                getFieldProps('toAddress').onChange({ target: { value: trimmed } } as React.ChangeEvent<HTMLInputElement>);
              }}
              placeholder="0x..."
              aria-required="true"
              className={`mt-2 w-full rounded-lg border bg-gray-950/60 px-3 py-2 text-sm text-gray-100 focus:outline-none focus:ring-2 transition-colors ${
                touched.toAddress && fieldErrors.toAddress
                  ? 'border-red-500 focus:ring-red-500 focus:border-red-500'
                  : 'border-gray-800 focus:ring-cyan-500 focus:border-cyan-500'
              }`}
            />
            {touched.toAddress && fieldErrors.toAddress && (
              <p id="toAddress-error" className="mt-1 text-xs text-red-400 flex items-center gap-1" role="alert">
                <AlertCircle className="w-3 h-3" aria-hidden="true" />
                {fieldErrors.toAddress}
              </p>
            )}
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <div>
              <label htmlFor={amountId} className="text-xs text-gray-400">Amount (ETH)</label>
              <input
                id={amountId}
                type="number"
                min="0"
                step="0.0001"
                value={amountEth}
                onChange={(e) => setAmountEth(e.target.value)}
                className="mt-2 w-full rounded-lg border border-gray-800 bg-gray-950/60 px-3 py-2 text-sm text-gray-100 focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:border-cyan-500"
                placeholder="0.0"
              />
            </div>
            <div>
              <label htmlFor={gasPriceId} className="text-xs text-gray-400">
                Gas price (gwei) <span className="text-red-400" aria-hidden="true">*</span>
              </label>
              <input
                id={gasPriceId}
                type="number"
                min="0"
                step="0.0001"
                {...getFieldProps('gasPriceGwei')}
                aria-required="true"
                className={`mt-2 w-full rounded-lg border bg-gray-950/60 px-3 py-2 text-sm text-gray-100 focus:outline-none focus:ring-2 transition-colors ${
                  touched.gasPriceGwei && fieldErrors.gasPriceGwei
                    ? 'border-red-500 focus:ring-red-500 focus:border-red-500'
                    : 'border-gray-800 focus:ring-cyan-500 focus:border-cyan-500'
                }`}
                placeholder="0.001"
              />
              {touched.gasPriceGwei && fieldErrors.gasPriceGwei && (
                <p id="gasPriceGwei-error" className="mt-1 text-xs text-red-400 flex items-center gap-1" role="alert">
                  <AlertCircle className="w-3 h-3" aria-hidden="true" />
                  {fieldErrors.gasPriceGwei}
                </p>
              )}
            </div>
          </div>

          {(preferences.showAdvancedFields || data || gasLimit) && (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div>
                <label htmlFor={gasLimitId} className="text-xs text-gray-400">Gas limit (optional)</label>
                <input
                  id={gasLimitId}
                  type="number"
                  min="21000"
                  step="1"
                  value={gasLimit}
                  onChange={(e) => setGasLimit(e.target.value)}
                  className="mt-2 w-full rounded-lg border border-gray-800 bg-gray-950/60 px-3 py-2 text-sm text-gray-100 focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:border-cyan-500"
                  placeholder="21000"
                />
              </div>
              <div>
                <label htmlFor={dataId} className="text-xs text-gray-400">Data (hex, optional)</label>
                <input
                  id={dataId}
                  type="text"
                  value={data}
                  onChange={(e) => setData(e.target.value)}
                  className="mt-2 w-full rounded-lg border border-gray-800 bg-gray-950/60 px-3 py-2 text-sm text-gray-100 focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:border-cyan-500"
                  placeholder="0x"
                />
              </div>
            </div>
          )}

          <div className="flex items-start gap-2">
            <input
              id={acknowledgeId}
              type="checkbox"
              checked={acknowledged}
              onChange={(e) => setAcknowledged(e.target.checked)}
              aria-required="true"
              className="mt-0.5 h-4 w-4 rounded border-gray-700 bg-gray-900 text-cyan-400 focus:ring-2 focus:ring-cyan-500 focus:ring-offset-2 focus:ring-offset-gray-900"
            />
            <label htmlFor={acknowledgeId} className="text-xs text-gray-400 cursor-pointer">
              I understand this sends a live transaction and gas fees apply.
              <span className="text-red-400 ml-1" aria-hidden="true">*</span>
            </label>
          </div>

          {error && (
            <div className="rounded-lg border border-red-500/30 bg-red-500/10 p-3 text-sm text-red-300 flex items-center gap-2" role="alert">
              <AlertTriangle className="w-4 h-4" aria-hidden="true" />
              {error}
            </div>
          )}

          {txHash && (
            <div className="rounded-lg border border-emerald-500/30 bg-emerald-500/10 p-3 text-sm text-emerald-200 flex items-center gap-2" role="status" aria-live="polite">
              <CheckCircle className="w-4 h-4" aria-hidden="true" />
              Transaction submitted.
              <a
                href={`${chain?.blockExplorer}/tx/${txHash}`}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-1 text-emerald-200 underline focus:outline-none focus:ring-2 focus:ring-emerald-400 focus:ring-offset-2 focus:ring-offset-gray-900 rounded"
              >
                View on explorer
                <ExternalLink className="w-3 h-3" aria-hidden="true" />
                <span className="sr-only">(opens in new tab)</span>
              </a>
            </div>
          )}
        </div>

        <div className="flex flex-col sm:flex-row gap-3 border-t border-gray-800 px-6 py-4">
          {!fromAddress ? (
            <button
              onClick={handleConnect}
              type="button"
              className="flex-1 rounded-xl bg-cyan-500 px-4 py-3 text-sm font-semibold text-white hover:bg-cyan-600 focus:outline-none focus:ring-2 focus:ring-cyan-400 focus:ring-offset-2 focus:ring-offset-gray-900"
            >
              Connect Wallet
            </button>
          ) : (
            <button
              onClick={handleExecute}
              disabled={isSubmitting}
              type="button"
              aria-busy={isSubmitting}
              className="flex-1 rounded-xl bg-emerald-500 px-4 py-3 text-sm font-semibold text-white hover:bg-emerald-600 disabled:opacity-60 focus:outline-none focus:ring-2 focus:ring-emerald-400 focus:ring-offset-2 focus:ring-offset-gray-900"
            >
              {isSubmitting ? 'Submitting...' : (
                <span className="inline-flex items-center gap-2">
                  <Send className="w-4 h-4" aria-hidden="true" />
                  Execute Transaction
                </span>
              )}
            </button>
          )}
          <button
            onClick={onClose}
            type="button"
            className="flex-1 rounded-xl border border-gray-700 bg-gray-900 px-4 py-3 text-sm font-semibold text-gray-200 hover:bg-gray-800 focus:outline-none focus:ring-2 focus:ring-gray-500 focus:ring-offset-2 focus:ring-offset-gray-900"
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
};

export default ExecuteTransactionModal;
