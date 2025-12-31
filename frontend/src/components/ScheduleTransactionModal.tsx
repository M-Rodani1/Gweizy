import React, { useState, useEffect } from 'react';
import { useChain } from '../contexts/ChainContext';
import { useScheduler } from '../contexts/SchedulerContext';
import { TransactionType, TX_GAS_ESTIMATES } from '../config/chains';
import ChainBadge from './ChainBadge';
import { getTxLabel } from '../config/transactions';

interface ScheduleTransactionModalProps {
  isOpen: boolean;
  onClose: () => void;
  defaultTxType?: TransactionType;
  suggestedTargetGas?: number;
}

const TX_TYPE_OPTIONS: { type: TransactionType; label: string }[] = [
  { type: 'swap', label: getTxLabel('swap') },
  { type: 'bridge', label: getTxLabel('bridge') },
  { type: 'nftMint', label: getTxLabel('nftMint') },
  { type: 'transfer', label: getTxLabel('transfer') },
  { type: 'erc20Transfer', label: getTxLabel('erc20Transfer') },
  { type: 'approve', label: getTxLabel('approve') },
  { type: 'nftTransfer', label: getTxLabel('nftTransfer') },
  { type: 'contractDeploy', label: getTxLabel('contractDeploy') },
];

const EXPIRY_OPTIONS = [
  { value: 1, label: '1 hour' },
  { value: 4, label: '4 hours' },
  { value: 12, label: '12 hours' },
  { value: 24, label: '24 hours' },
  { value: 72, label: '3 days' },
  { value: 168, label: '1 week' },
];

const ScheduleTransactionModal: React.FC<ScheduleTransactionModalProps> = ({
  isOpen,
  onClose,
  defaultTxType = 'swap',
  suggestedTargetGas
}) => {
  const { selectedChain, multiChainGas, enabledChains } = useChain();
  const { addTransaction } = useScheduler();

  const [txType, setTxType] = useState<TransactionType>(defaultTxType);
  const [chainId, setChainId] = useState(selectedChain.id);
  const [targetGasPrice, setTargetGasPrice] = useState('');
  const [maxGasPrice, setMaxGasPrice] = useState('');
  const [expiryHours, setExpiryHours] = useState(24);
  const [notifyEnabled, setNotifyEnabled] = useState(true);

  const currentGas = multiChainGas[chainId]?.gasPrice || 0;
  const gasUnits = TX_GAS_ESTIMATES[txType];

  // Set suggested target on open
  useEffect(() => {
    if (isOpen) {
      const suggested = suggestedTargetGas || currentGas * 0.85;
      setTargetGasPrice(suggested.toFixed(6));
      setMaxGasPrice((currentGas * 1.1).toFixed(6));
      setChainId(selectedChain.id);
    }
  }, [isOpen, suggestedTargetGas, currentGas, selectedChain.id]);

  // Request notification permission
  useEffect(() => {
    if (notifyEnabled && 'Notification' in window && Notification.permission === 'default') {
      Notification.requestPermission();
    }
  }, [notifyEnabled]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    const target = parseFloat(targetGasPrice);
    const max = parseFloat(maxGasPrice);

    if (isNaN(target) || target <= 0) {
      alert('Please enter a valid target gas price');
      return;
    }

    addTransaction({
      chainId,
      txType,
      targetGasPrice: target,
      maxGasPrice: max || target * 1.5,
      expiresAt: Date.now() + expiryHours * 60 * 60 * 1000
    });

    onClose();
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm flex items-center justify-center z-50 p-4">
      <div className="bg-gray-900 border border-gray-700 rounded-2xl max-w-lg w-full max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="px-6 py-4 border-b border-gray-700 flex items-center justify-between">
          <div>
            <h2 className="text-xl font-bold text-white">Schedule Transaction</h2>
            <p className="text-sm text-gray-400">Execute when gas price drops</p>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white text-2xl leading-none"
          >
            ×
          </button>
        </div>

        <form onSubmit={handleSubmit} className="p-6 space-y-6">
          {/* Chain Selection */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Network
            </label>
            <div className="grid grid-cols-3 gap-2">
              {enabledChains.map(chain => (
                <button
                  key={chain.id}
                  type="button"
                  onClick={() => setChainId(chain.id)}
                  className={`
                    p-3 rounded-lg text-center transition-all
                    ${chainId === chain.id
                      ? 'bg-cyan-500/20 border-2 border-cyan-500'
                      : 'bg-gray-800 border-2 border-transparent hover:border-gray-600'
                    }
                  `}
                >
                  <div className="flex justify-center mb-1">
                    <ChainBadge chain={chain} size="sm" />
                  </div>
                  <div className="text-xs text-gray-300">{chain.shortName}</div>
                  <div className="text-xs text-cyan-400 font-mono">
                    {multiChainGas[chain.id]?.gasPrice?.toFixed(4) || '...'}
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Transaction Type */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Transaction Type
            </label>
            <select
              value={txType}
              onChange={(e) => setTxType(e.target.value as TransactionType)}
              className="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-white focus:ring-2 focus:ring-cyan-500 focus:border-transparent"
            >
              {TX_TYPE_OPTIONS.map(({ type, label }) => (
                <option key={type} value={type}>
                  {label} (~{TX_GAS_ESTIMATES[type].toLocaleString()} gas)
                </option>
              ))}
            </select>
          </div>

          {/* Target Gas Price */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Target Gas Price (gwei)
            </label>
            <div className="relative">
              <input
                type="number"
                step="0.000001"
                value={targetGasPrice}
                onChange={(e) => setTargetGasPrice(e.target.value)}
                placeholder="0.0001"
                className="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-white font-mono focus:ring-2 focus:ring-cyan-500 focus:border-transparent"
              />
              <div className="absolute right-3 top-1/2 -translate-y-1/2 text-sm text-gray-500">
                Current: {currentGas.toFixed(6)}
              </div>
            </div>
            <div className="mt-2 flex gap-2">
              {[0.7, 0.8, 0.9].map(mult => (
                <button
                  key={mult}
                  type="button"
                  onClick={() => setTargetGasPrice((currentGas * mult).toFixed(6))}
                  className="px-3 py-1 text-xs bg-gray-800 text-gray-300 rounded hover:bg-gray-700 transition-colors"
                >
                  {Math.round((1 - mult) * 100)}% below
                </button>
              ))}
            </div>
          </div>

          {/* Max Gas Price */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Max Gas Price (gwei) <span className="text-gray-500">- optional fallback</span>
            </label>
            <input
              type="number"
              step="0.000001"
              value={maxGasPrice}
              onChange={(e) => setMaxGasPrice(e.target.value)}
              placeholder="0.001"
              className="w-full px-4 py-3 bg-gray-800 border border-gray-700 rounded-lg text-white font-mono focus:ring-2 focus:ring-cyan-500 focus:border-transparent"
            />
          </div>

          {/* Expiry */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              Expires After
            </label>
            <div className="grid grid-cols-3 gap-2">
              {EXPIRY_OPTIONS.map(({ value, label }) => (
                <button
                  key={value}
                  type="button"
                  onClick={() => setExpiryHours(value)}
                  className={`
                    px-3 py-2 rounded-lg text-sm transition-all
                    ${expiryHours === value
                      ? 'bg-cyan-500/20 border border-cyan-500 text-cyan-400'
                      : 'bg-gray-800 border border-transparent text-gray-300 hover:border-gray-600'
                    }
                  `}
                >
                  {label}
                </button>
              ))}
            </div>
          </div>

          {/* Notifications */}
          <div className="flex items-center justify-between p-4 bg-gray-800/50 rounded-lg">
            <div>
              <div className="text-sm font-medium text-gray-300">Push Notifications</div>
              <div className="text-xs text-gray-500">Get notified when target is reached</div>
            </div>
            <button
              type="button"
              onClick={() => setNotifyEnabled(!notifyEnabled)}
              className={`
                w-12 h-6 rounded-full transition-colors relative
                ${notifyEnabled ? 'bg-cyan-500' : 'bg-gray-700'}
              `}
            >
              <div className={`
                w-5 h-5 bg-white rounded-full absolute top-0.5 transition-transform
                ${notifyEnabled ? 'translate-x-6' : 'translate-x-0.5'}
              `} />
            </button>
          </div>

          {/* Summary */}
          <div className="p-4 bg-gray-800 rounded-lg">
            <div className="text-sm text-gray-400 mb-2">Summary</div>
            <div className="text-sm text-gray-300">
              Execute <span className="text-cyan-400">{txType}</span> on{' '}
              <span className="text-cyan-400">{enabledChains.find(c => c.id === chainId)?.name}</span> when
              gas drops to <span className="text-green-400">{targetGasPrice || '...'} gwei</span>
              {' '}(currently {currentGas.toFixed(6)} gwei)
            </div>
          </div>

          {/* Actions */}
          <div className="flex gap-3">
            <button
              type="button"
              onClick={onClose}
              className="flex-1 py-3 px-6 bg-gray-700 hover:bg-gray-600 text-white rounded-xl font-medium transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              className="flex-1 py-3 px-6 bg-cyan-500 hover:bg-cyan-600 text-white rounded-xl font-medium transition-colors"
            >
              Schedule Transaction
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default ScheduleTransactionModal;
