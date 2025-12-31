import React, { useEffect, useState } from 'react';
import { AlertTriangle, CheckCircle, ExternalLink, Send } from 'lucide-react';
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
  const [fromAddress, setFromAddress] = useState<string | null>(null);
  const [toAddress, setToAddress] = useState(defaultToAddress || '');
  const [amountEth, setAmountEth] = useState(defaultAmountEth || '');
  const [data, setData] = useState(defaultData || '');
  const [gasPriceGwei, setGasPriceGwei] = useState(
    gasGwei ? gasGwei.toFixed(6) : ''
  );
  const [gasLimit, setGasLimit] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [txHash, setTxHash] = useState<string | null>(null);
  const [acknowledged, setAcknowledged] = useState(false);

  useEffect(() => {
    if (!isOpen) return;
    setError(null);
    setTxHash(null);
    setAcknowledged(false);
    setToAddress(defaultToAddress || '');
    setAmountEth(defaultAmountEth || '');
    setData(defaultData || '');
    setGasPriceGwei(gasGwei ? gasGwei.toFixed(6) : '');
    setGasLimit('');

    const loadAccount = async () => {
      const account = await getCurrentAccount();
      setFromAddress(account);
    };
    loadAccount();
  }, [isOpen, defaultToAddress, defaultAmountEth, defaultData, gasGwei]);

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
    if (!toAddress || !/^0x[a-fA-F0-9]{40}$/.test(toAddress)) {
      setError('Enter a valid recipient address.');
      return;
    }
    if (!gasPriceGwei || Number(gasPriceGwei) <= 0) {
      setError('Enter a valid gas price.');
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
        to: toAddress,
        valueEth: amountEth || '0',
        gasPriceGwei,
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
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
      <div className="w-full max-w-xl rounded-2xl border border-gray-700 bg-gray-900 shadow-xl">
        <div className="flex items-center justify-between border-b border-gray-800 px-6 py-4">
          <div>
            <h2 className="text-lg font-semibold text-white">Execute {txLabel}</h2>
            <p className="text-xs text-gray-400">
              {chain?.name} • Recommended gas {gasPriceGwei || 'auto'} gwei
            </p>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white"
            aria-label="Close execution modal"
          >
            ×
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
            <label className="text-xs text-gray-400">Recipient</label>
            <input
              type="text"
              value={toAddress}
              onChange={(e) => setToAddress(e.target.value.trim())}
              placeholder="0x..."
              className="mt-2 w-full rounded-lg border border-gray-800 bg-gray-950/60 px-3 py-2 text-sm text-gray-100"
            />
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <div>
              <label className="text-xs text-gray-400">Amount (ETH)</label>
              <input
                type="number"
                min="0"
                step="0.0001"
                value={amountEth}
                onChange={(e) => setAmountEth(e.target.value)}
                className="mt-2 w-full rounded-lg border border-gray-800 bg-gray-950/60 px-3 py-2 text-sm text-gray-100"
                placeholder="0.0"
              />
            </div>
            <div>
              <label className="text-xs text-gray-400">Gas price (gwei)</label>
              <input
                type="number"
                min="0"
                step="0.0001"
                value={gasPriceGwei}
                onChange={(e) => setGasPriceGwei(e.target.value)}
                className="mt-2 w-full rounded-lg border border-gray-800 bg-gray-950/60 px-3 py-2 text-sm text-gray-100"
                placeholder="0.001"
              />
            </div>
          </div>

          {(preferences.showAdvancedFields || data || gasLimit) && (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div>
                <label className="text-xs text-gray-400">Gas limit (optional)</label>
                <input
                  type="number"
                  min="21000"
                  step="1"
                  value={gasLimit}
                  onChange={(e) => setGasLimit(e.target.value)}
                  className="mt-2 w-full rounded-lg border border-gray-800 bg-gray-950/60 px-3 py-2 text-sm text-gray-100"
                  placeholder="21000"
                />
              </div>
              <div>
                <label className="text-xs text-gray-400">Data (hex, optional)</label>
                <input
                  type="text"
                  value={data}
                  onChange={(e) => setData(e.target.value)}
                  className="mt-2 w-full rounded-lg border border-gray-800 bg-gray-950/60 px-3 py-2 text-sm text-gray-100"
                  placeholder="0x"
                />
              </div>
            </div>
          )}

          <label className="flex items-start gap-2 text-xs text-gray-400">
            <input
              type="checkbox"
              checked={acknowledged}
              onChange={(e) => setAcknowledged(e.target.checked)}
              className="mt-0.5 h-4 w-4 rounded border-gray-700 bg-gray-900 text-cyan-400"
            />
            I understand this sends a live transaction and gas fees apply.
          </label>

          {error && (
            <div className="rounded-lg border border-red-500/30 bg-red-500/10 p-3 text-sm text-red-300 flex items-center gap-2" role="alert">
              <AlertTriangle className="w-4 h-4" />
              {error}
            </div>
          )}

          {txHash && (
            <div className="rounded-lg border border-emerald-500/30 bg-emerald-500/10 p-3 text-sm text-emerald-200 flex items-center gap-2">
              <CheckCircle className="w-4 h-4" />
              Transaction submitted.
              <a
                href={`${chain?.blockExplorer}/tx/${txHash}`}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-1 text-emerald-200 underline"
              >
                View
                <ExternalLink className="w-3 h-3" />
              </a>
            </div>
          )}
        </div>

        <div className="flex flex-col sm:flex-row gap-3 border-t border-gray-800 px-6 py-4">
          {!fromAddress ? (
            <button
              onClick={handleConnect}
              className="flex-1 rounded-xl bg-cyan-500 px-4 py-3 text-sm font-semibold text-white hover:bg-cyan-600"
            >
              Connect Wallet
            </button>
          ) : (
            <button
              onClick={handleExecute}
              disabled={isSubmitting}
              className="flex-1 rounded-xl bg-emerald-500 px-4 py-3 text-sm font-semibold text-white hover:bg-emerald-600 disabled:opacity-60"
            >
              {isSubmitting ? 'Submitting...' : (
                <span className="inline-flex items-center gap-2">
                  <Send className="w-4 h-4" />
                  Execute Transaction
                </span>
              )}
            </button>
          )}
          <button
            onClick={onClose}
            className="flex-1 rounded-xl border border-gray-700 bg-gray-900 px-4 py-3 text-sm font-semibold text-gray-200 hover:bg-gray-800"
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
};

export default ExecuteTransactionModal;
