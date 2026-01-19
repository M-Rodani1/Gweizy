import React, { useMemo, useState } from 'react';
import { Clock } from 'lucide-react';
import { useScheduler, ScheduledTransaction } from '../contexts/SchedulerContext';
import { useChain } from '../contexts/ChainContext';
import { SUPPORTED_CHAINS } from '../config/chains';
import ChainBadge from './ChainBadge';
import ScheduleTransactionModal from './ScheduleTransactionModal';
import ExecuteTransactionModal from './ExecuteTransactionModal';
import VirtualizedList from './ui/VirtualizedList';

const ScheduledTransactionsList: React.FC = () => {
  const { transactions, removeTransaction, markExecuted, markCancelled, pendingCount, readyCount } = useScheduler();
  const { multiChainGas } = useChain();
  const [showModal, setShowModal] = useState(false);
  const [executeTx, setExecuteTx] = useState<ScheduledTransaction | null>(null);
  const [filter, setFilter] = useState<'all' | 'pending' | 'ready' | 'completed'>('all');
  const itemHeight = 176;
  const maxListHeight = 384;

  const filteredTransactions = useMemo(() => {
    return transactions
      .filter(tx => {
        if (filter === 'all') return true;
        if (filter === 'pending') return tx.status === 'pending';
        if (filter === 'ready') return tx.status === 'ready';
        if (filter === 'completed') return ['executed', 'expired', 'cancelled'].includes(tx.status);
        return true;
      })
      .sort((a, b) => b.createdAt - a.createdAt);
  }, [filter, transactions]);

  const formatTimeRemaining = (expiresAt: number): string => {
    const remaining = expiresAt - Date.now();
    if (remaining <= 0) return 'Expired';

    const hours = Math.floor(remaining / (1000 * 60 * 60));
    const minutes = Math.floor((remaining % (1000 * 60 * 60)) / (1000 * 60));

    if (hours > 24) return `${Math.floor(hours / 24)}d ${hours % 24}h`;
    if (hours > 0) return `${hours}h ${minutes}m`;
    return `${minutes}m`;
  };

  const getStatusBadge = (status: ScheduledTransaction['status']) => {
    switch (status) {
      case 'pending':
        return <span className="px-2 py-0.5 text-xs bg-yellow-500/20 text-yellow-400 rounded-full">Waiting</span>;
      case 'ready':
        return <span className="px-2 py-0.5 text-xs bg-green-500/20 text-green-400 rounded-full animate-pulse">Ready!</span>;
      case 'executed':
        return <span className="px-2 py-0.5 text-xs bg-cyan-500/20 text-cyan-400 rounded-full">Executed</span>;
      case 'expired':
        return <span className="px-2 py-0.5 text-xs bg-gray-500/20 text-gray-400 rounded-full">Expired</span>;
      case 'cancelled':
        return <span className="px-2 py-0.5 text-xs bg-red-500/20 text-red-400 rounded-full">Cancelled</span>;
    }
  };

  return (
    <div className="bg-gray-800/50 border border-gray-700 rounded-xl overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 border-b border-gray-700/50 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Clock className="w-4 h-4 text-cyan-400" />
          <h3 className="font-semibold text-white">Scheduled Transactions</h3>
          {pendingCount > 0 && (
            <span className="px-2 py-0.5 text-xs bg-yellow-500/20 text-yellow-400 rounded-full">
              {pendingCount} waiting
            </span>
          )}
          {readyCount > 0 && (
            <span className="px-2 py-0.5 text-xs bg-green-500/20 text-green-400 rounded-full animate-pulse">
              {readyCount} ready!
            </span>
          )}
        </div>
        <button
          onClick={() => setShowModal(true)}
          className="px-3 py-1.5 bg-cyan-500 hover:bg-cyan-600 text-white text-sm rounded-lg transition-colors"
        >
          + Schedule
        </button>
      </div>

      {/* Filter tabs */}
      <div className="px-4 py-2 border-b border-gray-700/30 flex gap-2">
        {(['all', 'pending', 'ready', 'completed'] as const).map(f => (
          <button
            key={f}
            onClick={() => setFilter(f)}
            className={`
              px-3 py-1 text-sm rounded-lg transition-colors
              ${filter === f
                ? 'bg-gray-700 text-white'
                : 'text-gray-400 hover:text-white'
              }
            `}
          >
            {f.charAt(0).toUpperCase() + f.slice(1)}
          </button>
        ))}
      </div>

      {/* Transaction List */}
      {filteredTransactions.length === 0 ? (
        <div className="p-8 text-center">
          <div className="flex justify-center mb-3">
            <Clock className="w-8 h-8 text-gray-500" />
          </div>
          <div className="text-gray-400 mb-2">No scheduled transactions</div>
          <div className="text-sm text-gray-500">
            Schedule transactions to execute when gas prices drop
          </div>
          <button
            onClick={() => setShowModal(true)}
            className="mt-4 px-4 py-2 bg-cyan-500/20 text-cyan-400 rounded-lg hover:bg-cyan-500/30 transition-colors"
          >
            Schedule Your First Transaction
          </button>
        </div>
      ) : (
        <VirtualizedList
          items={filteredTransactions}
          itemHeight={itemHeight}
          maxHeight={maxListHeight}
          getKey={(tx) => tx.id}
          renderItem={(tx) => {
            const chain = SUPPORTED_CHAINS[tx.chainId];
            const currentGas = multiChainGas[tx.chainId]?.gasPrice || 0;
            const progress = currentGas > 0
              ? Math.min(100, Math.max(0, (1 - (currentGas - tx.targetGasPrice) / currentGas) * 100))
              : 0;

            return (
              <div className="h-full border-b border-gray-700/30 p-4 hover:bg-gray-700/20 transition-colors">
                <div className="flex items-start justify-between mb-2">
                  <div className="flex items-center gap-2">
                    {chain ? (
                      <ChainBadge chain={chain} size="sm" />
                    ) : (
                      <div className="w-6 h-6 rounded-full bg-gray-700 text-xs text-white flex items-center justify-center">
                        ?
                      </div>
                    )}
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="font-medium text-white">{tx.txType}</span>
                        {getStatusBadge(tx.status)}
                      </div>
                      <div className="text-xs text-gray-500">
                        {chain?.name} • Created {new Date(tx.createdAt).toLocaleDateString()}
                      </div>
                    </div>
                  </div>

                  <div className="text-right">
                    <div className="text-sm text-gray-400">Expires in</div>
                    <div className="text-sm font-medium text-white">
                      {formatTimeRemaining(tx.expiresAt)}
                    </div>
                  </div>
                </div>

                {/* Gas progress */}
                <div className="mb-3">
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-gray-500">
                      Current: <span className="text-white">{currentGas.toFixed(6)}</span> gwei
                    </span>
                    <span className="text-gray-500">
                      Target: <span className="text-green-400">{tx.targetGasPrice.toFixed(6)}</span> gwei
                    </span>
                  </div>
                  <div className="h-2 bg-gray-700 rounded-full overflow-hidden">
                    <div
                      className={`h-full transition-all ${
                        tx.status === 'ready' ? 'bg-green-500' :
                        progress > 80 ? 'bg-yellow-500' : 'bg-cyan-500'
                      }`}
                      style={{ width: `${progress}%` }}
                    />
                  </div>
                </div>

                {/* Actions */}
                <div className="flex gap-2">
                  {tx.status === 'ready' && (
                    <button
                      onClick={() => setExecuteTx(tx)}
                      className="flex-1 py-2 bg-green-500 hover:bg-green-600 text-white text-sm rounded-lg transition-colors"
                    >
                      Execute Now
                    </button>
                  )}
                  {tx.status === 'pending' && (
                    <button
                      onClick={() => markCancelled(tx.id)}
                      className="flex-1 py-2 bg-gray-700 hover:bg-gray-600 text-white text-sm rounded-lg transition-colors"
                    >
                      Cancel
                    </button>
                  )}
                  {['executed', 'expired', 'cancelled'].includes(tx.status) && (
                    <button
                      onClick={() => removeTransaction(tx.id)}
                      className="flex-1 py-2 bg-gray-700 hover:bg-gray-600 text-white text-sm rounded-lg transition-colors"
                    >
                      Remove
                    </button>
                  )}
                </div>
              </div>
            );
          }}
        />
      )}

      {/* Modal */}
      <ScheduleTransactionModal
        isOpen={showModal}
        onClose={() => setShowModal(false)}
      />

      <ExecuteTransactionModal
        isOpen={!!executeTx}
        onClose={() => setExecuteTx(null)}
        chainId={executeTx?.chainId || 8453}
        txType={executeTx?.txType}
        gasGwei={executeTx ? Math.min(multiChainGas[executeTx.chainId]?.gasPrice || executeTx.targetGasPrice, executeTx.maxGasPrice) : null}
        defaultToAddress={executeTx?.toAddress}
        defaultAmountEth={executeTx?.amount}
        defaultData={executeTx?.data}
        onExecuted={() => {
          if (executeTx) {
            markExecuted(executeTx.id);
          }
          setExecuteTx(null);
        }}
      />
    </div>
  );
};

export default React.memo(ScheduledTransactionsList);
