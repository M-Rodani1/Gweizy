import React, { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';
import { useVirtualizer } from '@tanstack/react-virtual';
import { TableRowData } from '../../types';
import { fetchTransactions } from '../api/gasApi';
import LoadingSpinner from './LoadingSpinner';

// Virtual scrolling threshold - use virtual scrolling for 50+ rows
const VIRTUAL_SCROLL_THRESHOLD = 50;
const ROW_HEIGHT = 60; // Estimated row height in pixels

// Memoized table row component to prevent unnecessary re-renders
const TableRow = React.memo<{ row: TableRowData; index: number }>(({ row }) => (
  <tr className="border-b border-gray-700 hover:bg-gray-700/50">
    <th scope="row" className="p-3 font-mono text-sm text-cyan-400 text-left">
      {row.txHash}
    </th>
    <td className="p-3">
      <span className="bg-cyan-600/50 text-cyan-200 text-xs font-semibold mr-2 px-2.5 py-0.5 rounded">
        {row.method}
      </span>
    </td>
    <td className="p-3 text-gray-300">{row.age}</td>
    <td className="p-3 text-right text-gray-300">
      {row.gasUsed !== undefined && row.gasUsed !== null ? row.gasUsed.toLocaleString() : 'N/A'}
    </td>
    <td className="p-3 text-right font-semibold text-teal-300">
      {row.gasPrice !== undefined && row.gasPrice !== null ? row.gasPrice.toFixed(4) : 'N/A'}
    </td>
  </tr>
));
TableRow.displayName = 'TableRow';

const GasPriceTable: React.FC = () => {
  const [transactions, setTransactions] = useState<TableRowData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  // Ref for the scrollable container
  const parentRef = useRef<HTMLDivElement>(null);

  // Memoize loadData to prevent unnecessary re-creations
  const loadData = useCallback(async () => {
    try {
      setError(null);
      const data = await fetchTransactions(5);
      setTransactions(data);
    } catch (err) {
      console.error('Error loading transactions:', err);
      setError(err instanceof Error ? err.message : 'Failed to load transactions');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData();
    
    // Auto-refresh every 30 seconds
    const interval = setInterval(loadData, 30000);
    return () => clearInterval(interval);
  }, [loadData]);

  // Memoize whether to use virtual scrolling
  const useVirtualScrolling = useMemo(
    () => transactions.length >= VIRTUAL_SCROLL_THRESHOLD,
    [transactions.length]
  );

  // Virtual scrolling setup
  const virtualizer = useVirtualizer({
    count: transactions.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => ROW_HEIGHT,
    overscan: 5, // Render 5 extra rows above/below visible area for smooth scrolling
    enabled: useVirtualScrolling,
  });

  // Memoize visible rows for regular rendering (no virtual scrolling)
  const visibleRows = useMemo(() => {
    if (useVirtualScrolling) {
      return null; // Virtual scrolling will handle rendering
    }
    return transactions;
  }, [transactions, useVirtualScrolling]);

  if (loading && transactions.length === 0) {
    return (
      <div className="bg-gray-800 p-4 sm:p-6 rounded-lg shadow-lg h-full overflow-x-auto">
        <LoadingSpinner message="Loading transactions..." />
      </div>
    );
  }

  if (error) {
    return (
      <div className="bg-gray-800 p-4 sm:p-6 rounded-lg shadow-lg h-full">
        <div className="text-center">
          <p className="text-red-400 mb-4 flex items-center justify-center gap-2">
            <AlertTriangle className="w-4 h-4" />
            {error}
          </p>
          <button
            onClick={loadData}
            className="px-4 py-2 bg-cyan-500 hover:bg-cyan-600 rounded-md transition-colors text-sm"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gray-800 p-4 sm:p-6 rounded-lg shadow-lg h-full flex flex-col">
      <div className="flex justify-between items-center mb-4 flex-shrink-0">
        <h2 className="text-xl font-semibold text-gray-200">
          Recent Base Transactions
          {useVirtualScrolling && (
            <span className="ml-2 text-sm text-gray-400 font-normal">
              ({transactions.length} total)
            </span>
          )}
        </h2>
        <button
          onClick={loadData}
          className="text-sm text-cyan-400 hover:text-cyan-300 transition-colors"
          title="Refresh"
        >
          <span className="inline-flex items-center gap-1">
            <RefreshCw className="w-3 h-3" />
            Refresh
          </span>
        </button>
      </div>
      
      {transactions.length === 0 && !loading ? (
        <div className="text-center py-8 text-gray-500">
          No recent transactions found
        </div>
      ) : (
        <div className="overflow-x-auto flex-1" ref={parentRef} style={{ maxHeight: '600px', overflowY: 'auto' }}>
          <table className="w-full text-left table-auto">
            <thead className="border-b-2 border-gray-600 sticky top-0 bg-gray-800 z-10">
              <tr className="text-gray-400">
                <th scope="col" className="p-3 text-left">Tx Hash</th>
                <th scope="col" className="p-3 text-left">Method</th>
                <th scope="col" className="p-3 text-left">Age</th>
                <th scope="col" className="p-3 text-right">Gas Used</th>
                <th scope="col" className="p-3 text-right">Gas Price (Gwei)</th>
              </tr>
            </thead>
            <tbody>
              {useVirtualScrolling ? (
                // Virtual scrolling mode
                <>
                  {/* Spacer for rows before visible area */}
                  <tr aria-hidden="true">
                    <td colSpan={5} style={{ height: `${virtualizer.getVirtualItems()[0]?.start ?? 0}px` }} />
                  </tr>
                  {/* Render only visible rows */}
                  {virtualizer.getVirtualItems().map((virtualRow) => {
                    const row = transactions[virtualRow.index];
                    return (
                      <tr
                        key={`${row.txHash}-${virtualRow.index}`}
                        data-index={virtualRow.index}
                        ref={virtualizer.measureElement}
                        className="border-b border-gray-700 hover:bg-gray-700/50"
                      >
                        <th scope="row" className="p-3 font-mono text-sm text-cyan-400 text-left">
                          {row.txHash}
                        </th>
                        <td className="p-3">
                          <span className="bg-cyan-600/50 text-cyan-200 text-xs font-semibold mr-2 px-2.5 py-0.5 rounded">
                            {row.method}
                          </span>
                        </td>
                        <td className="p-3 text-gray-300">{row.age}</td>
                        <td className="p-3 text-right text-gray-300">
                          {row.gasUsed !== undefined && row.gasUsed !== null ? row.gasUsed.toLocaleString() : 'N/A'}
                        </td>
                        <td className="p-3 text-right font-semibold text-teal-300">
                          {row.gasPrice !== undefined && row.gasPrice !== null ? row.gasPrice.toFixed(4) : 'N/A'}
                        </td>
                      </tr>
                    );
                  })}
                  {/* Spacer for rows after visible area */}
                  <tr aria-hidden="true">
                    <td colSpan={5} style={{ 
                      height: `${virtualizer.getTotalSize() - (virtualizer.getVirtualItems()[virtualizer.getVirtualItems().length - 1]?.end ?? virtualizer.getTotalSize())}px` 
                    }} />
                  </tr>
                </>
              ) : (
                // Regular rendering mode (no virtual scrolling)
                Array.isArray(visibleRows) && visibleRows.length > 0 ? visibleRows.map((row, index) => (
                  <TableRow key={`${row.txHash}-${index}`} row={row} index={index} />
                )) : null
              )}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
};

export default React.memo(GasPriceTable);
