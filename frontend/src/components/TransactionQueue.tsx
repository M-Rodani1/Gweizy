import React, { useState, useEffect, useCallback } from 'react';
import { Clock, Plus, Trash2, X, Zap, CheckCircle, AlertCircle, Send } from 'lucide-react';
import { sendGasAlertNotification } from '../utils/pushNotifications';

interface QueuedTransaction {
  id: string;
  description: string;
  targetGwei: number;
  estimatedGasUnits: number;
  createdAt: Date;
  status: 'waiting' | 'ready' | 'executed';
}

interface TransactionQueueProps {
  currentGas: number;
  ethPrice: number;
}

// Transaction presets for common operations
const TRANSACTION_PRESETS = [
  { name: 'Token Swap', gasUnits: 150000 },
  { name: 'NFT Mint', gasUnits: 200000 },
  { name: 'Token Transfer', gasUnits: 65000 },
  { name: 'Contract Deploy', gasUnits: 500000 },
  { name: 'Bridge Transfer', gasUnits: 250000 },
  { name: 'Custom', gasUnits: 100000 }
];

const TransactionQueue: React.FC<TransactionQueueProps> = ({ currentGas, ethPrice }) => {
  const [queue, setQueue] = useState<QueuedTransaction[]>([]);
  const [showAddForm, setShowAddForm] = useState(false);
  const [notifiedTransactions, setNotifiedTransactions] = useState<Set<string>>(new Set());

  // Form state
  const [description, setDescription] = useState('');
  const [selectedPreset, setSelectedPreset] = useState(TRANSACTION_PRESETS[0]);
  const [customGasUnits, setCustomGasUnits] = useState('100000');
  const [targetGwei, setTargetGwei] = useState('');

  // Load queue from localStorage
  useEffect(() => {
    const saved = localStorage.getItem('transaction_queue');
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        setQueue(parsed.map((t: QueuedTransaction) => ({
          ...t,
          createdAt: new Date(t.createdAt)
        })));
      } catch (e) {
        console.error('Failed to parse saved queue:', e);
      }
    }
  }, []);

  // Save queue to localStorage
  useEffect(() => {
    localStorage.setItem('transaction_queue', JSON.stringify(queue));
  }, [queue]);

  // Check for ready transactions
  useEffect(() => {
    if (currentGas <= 0) return;

    queue.forEach(tx => {
      if (tx.status === 'waiting' && currentGas <= tx.targetGwei) {
        // Mark as ready
        setQueue(prev => prev.map(t =>
          t.id === tx.id ? { ...t, status: 'ready' as const } : t
        ));

        // Send notification if not already notified
        if (!notifiedTransactions.has(tx.id)) {
          sendGasAlertNotification(currentGas, {
            id: parseInt(tx.id),
            alert_type: 'below',
            threshold_gwei: tx.targetGwei,
            is_active: true
          });
          setNotifiedTransactions(prev => new Set([...prev, tx.id]));
        }
      } else if (tx.status === 'ready' && currentGas > tx.targetGwei * 1.1) {
        // Mark as waiting again if gas rose significantly
        setQueue(prev => prev.map(t =>
          t.id === tx.id ? { ...t, status: 'waiting' as const } : t
        ));
        setNotifiedTransactions(prev => {
          const next = new Set(prev);
          next.delete(tx.id);
          return next;
        });
      }
    });
  }, [currentGas, queue, notifiedTransactions]);

  const addTransaction = useCallback((e: React.FormEvent) => {
    e.preventDefault();

    const gasUnits = selectedPreset.name === 'Custom'
      ? parseInt(customGasUnits)
      : selectedPreset.gasUnits;

    const newTx: QueuedTransaction = {
      id: Date.now().toString(),
      description: description || selectedPreset.name,
      targetGwei: parseFloat(targetGwei),
      estimatedGasUnits: gasUnits,
      createdAt: new Date(),
      status: 'waiting'
    };

    setQueue(prev => [newTx, ...prev]);
    setShowAddForm(false);
    setDescription('');
    setTargetGwei('');
  }, [description, selectedPreset, customGasUnits, targetGwei]);

  const removeTransaction = useCallback((id: string) => {
    setQueue(prev => prev.filter(t => t.id !== id));
    setNotifiedTransactions(prev => {
      const next = new Set(prev);
      next.delete(id);
      return next;
    });
  }, []);

  const markAsExecuted = useCallback((id: string) => {
    setQueue(prev => prev.map(t =>
      t.id === id ? { ...t, status: 'executed' as const } : t
    ));
  }, []);

  const calculateCostUsd = (gasUnits: number, gweiPrice: number): number => {
    const ethCost = (gasUnits * gweiPrice) / 1e9;
    return ethCost * ethPrice;
  };

  const getSuggestedTarget = () => {
    if (currentGas <= 0) return '0.001';
    return (currentGas * 0.7).toFixed(6); // 30% below current
  };

  const waitingCount = queue.filter(t => t.status === 'waiting').length;
  const readyCount = queue.filter(t => t.status === 'ready').length;

  return (
    <div className="bg-gradient-to-br from-slate-800/50 to-slate-900/50 backdrop-blur-sm border border-slate-700 rounded-xl p-6 shadow-xl">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-orange-500/20 rounded-lg">
            <Clock className="w-5 h-5 text-orange-400" />
          </div>
          <div>
            <h3 className="text-lg font-bold text-white">Transaction Queue</h3>
            <p className="text-xs text-gray-400">Queue transactions for optimal gas timing</p>
          </div>
        </div>

        <div className="flex items-center gap-3">
          {readyCount > 0 && (
            <span className="px-3 py-1 bg-green-500/20 text-green-400 text-sm font-medium rounded-full border border-green-500/30 animate-pulse">
              {readyCount} ready!
            </span>
          )}
          <button
            onClick={() => setShowAddForm(!showAddForm)}
            className="px-4 py-2 bg-orange-500/20 hover:bg-orange-500/30 text-orange-400 rounded-lg text-sm font-medium transition-colors flex items-center gap-2"
          >
            {showAddForm ? <X className="w-4 h-4" /> : <Plus className="w-4 h-4" />}
            {showAddForm ? 'Cancel' : 'Add'}
          </button>
        </div>
      </div>

      {/* Add Form */}
      {showAddForm && (
        <form onSubmit={addTransaction} className="mb-6 p-4 bg-slate-700/30 rounded-lg border border-slate-600">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
            {/* Transaction Type */}
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">Transaction Type</label>
              <select
                value={selectedPreset.name}
                onChange={(e) => setSelectedPreset(TRANSACTION_PRESETS.find(p => p.name === e.target.value) || TRANSACTION_PRESETS[0])}
                className="w-full px-3 py-2 bg-slate-800 border border-slate-600 rounded-lg text-white focus:border-orange-500 focus:outline-none"
              >
                {TRANSACTION_PRESETS.map(preset => (
                  <option key={preset.name} value={preset.name}>{preset.name}</option>
                ))}
              </select>
            </div>

            {/* Custom Gas Units */}
            {selectedPreset.name === 'Custom' && (
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-2">Gas Units</label>
                <input
                  type="number"
                  value={customGasUnits}
                  onChange={(e) => setCustomGasUnits(e.target.value)}
                  className="w-full px-3 py-2 bg-slate-800 border border-slate-600 rounded-lg text-white focus:border-orange-500 focus:outline-none"
                  placeholder="100000"
                />
              </div>
            )}

            {/* Target Gas Price */}
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Target Gas (gwei)
                {currentGas > 0 && (
                  <button
                    type="button"
                    onClick={() => setTargetGwei(getSuggestedTarget())}
                    className="ml-2 text-xs text-orange-400 hover:text-orange-300"
                  >
                    Suggest: {getSuggestedTarget()}
                  </button>
                )}
              </label>
              <input
                type="number"
                step="0.000001"
                value={targetGwei}
                onChange={(e) => setTargetGwei(e.target.value)}
                className="w-full px-3 py-2 bg-slate-800 border border-slate-600 rounded-lg text-white focus:border-orange-500 focus:outline-none"
                placeholder="0.0005"
                required
              />
            </div>

            {/* Description */}
            <div className={selectedPreset.name === 'Custom' ? 'md:col-span-2' : ''}>
              <label className="block text-sm font-medium text-gray-300 mb-2">Description (optional)</label>
              <input
                type="text"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                className="w-full px-3 py-2 bg-slate-800 border border-slate-600 rounded-lg text-white focus:border-orange-500 focus:outline-none"
                placeholder={selectedPreset.name}
              />
            </div>
          </div>

          {/* Cost Preview */}
          {targetGwei && currentGas > 0 && (
            <div className="mb-4 p-3 bg-slate-800/50 rounded-lg grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="text-gray-400">Current cost: </span>
                <span className="text-white font-medium">
                  ${calculateCostUsd(selectedPreset.name === 'Custom' ? parseInt(customGasUnits) : selectedPreset.gasUnits, currentGas).toFixed(4)}
                </span>
              </div>
              <div>
                <span className="text-gray-400">Target cost: </span>
                <span className="text-green-400 font-medium">
                  ${calculateCostUsd(selectedPreset.name === 'Custom' ? parseInt(customGasUnits) : selectedPreset.gasUnits, parseFloat(targetGwei)).toFixed(4)}
                </span>
              </div>
            </div>
          )}

          <button
            type="submit"
            disabled={!targetGwei}
            className="w-full px-4 py-2 bg-orange-500 hover:bg-orange-600 disabled:bg-slate-600 disabled:cursor-not-allowed text-white rounded-lg font-medium transition-colors"
          >
            Add to Queue
          </button>
        </form>
      )}

      {/* Queue List */}
      {queue.length === 0 ? (
        <div className="text-center py-8 text-gray-400">
          <Clock className="w-12 h-12 mx-auto mb-3 opacity-50" />
          <p>No transactions queued</p>
          <p className="text-xs mt-2">Add transactions to get notified when gas is optimal</p>
        </div>
      ) : (
        <div className="space-y-3">
          {queue.map((tx) => (
            <div
              key={tx.id}
              className={`p-4 rounded-lg border transition-all ${
                tx.status === 'ready'
                  ? 'bg-green-500/10 border-green-500/30 animate-pulse'
                  : tx.status === 'executed'
                  ? 'bg-slate-800/30 border-slate-700 opacity-60'
                  : 'bg-slate-700/30 border-slate-600'
              }`}
            >
              <div className="flex items-center justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    {tx.status === 'ready' && <Zap className="w-4 h-4 text-green-400" />}
                    {tx.status === 'executed' && <CheckCircle className="w-4 h-4 text-gray-400" />}
                    {tx.status === 'waiting' && <Clock className="w-4 h-4 text-orange-400" />}
                    <span className={`font-semibold ${
                      tx.status === 'ready' ? 'text-green-400' :
                      tx.status === 'executed' ? 'text-gray-400' : 'text-white'
                    }`}>
                      {tx.description}
                    </span>
                    {tx.status === 'ready' && (
                      <span className="px-2 py-0.5 bg-green-500/20 text-green-400 text-xs rounded-full">
                        GAS IS LOW!
                      </span>
                    )}
                  </div>
                  <div className="text-xs text-gray-400 flex gap-4">
                    <span>Target: {tx.targetGwei.toFixed(6)} gwei</span>
                    <span>Est. gas: {tx.estimatedGasUnits.toLocaleString()}</span>
                    {currentGas > 0 && tx.status !== 'executed' && (
                      <span className={currentGas <= tx.targetGwei ? 'text-green-400' : 'text-yellow-400'}>
                        {currentGas <= tx.targetGwei
                          ? `Save $${(calculateCostUsd(tx.estimatedGasUnits, currentGas) - calculateCostUsd(tx.estimatedGasUnits, tx.targetGwei)).toFixed(4)}`
                          : `Waiting... (${((1 - tx.targetGwei / currentGas) * 100).toFixed(0)}% drop needed)`
                        }
                      </span>
                    )}
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  {tx.status === 'ready' && (
                    <button
                      onClick={() => markAsExecuted(tx.id)}
                      className="p-2 bg-green-500/20 text-green-400 hover:bg-green-500/30 rounded-lg transition-colors"
                      title="Mark as executed"
                    >
                      <Send className="w-4 h-4" />
                    </button>
                  )}
                  <button
                    onClick={() => removeTransaction(tx.id)}
                    className="p-2 bg-red-500/20 text-red-400 hover:bg-red-500/30 rounded-lg transition-colors"
                    title="Remove"
                  >
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Current Gas Info */}
      {currentGas > 0 && (
        <div className="mt-6 p-3 bg-orange-500/10 rounded-lg border border-orange-500/30">
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-400">Current Gas:</span>
            <span className="font-bold text-white">{currentGas.toFixed(6)} gwei</span>
          </div>
          {waitingCount > 0 && (
            <p className="text-xs text-gray-400 mt-2">
              {waitingCount} transaction{waitingCount > 1 ? 's' : ''} waiting for lower gas
            </p>
          )}
        </div>
      )}
    </div>
  );
};

export default TransactionQueue;
