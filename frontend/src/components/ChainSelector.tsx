import React, { useState, useRef, useEffect, useCallback, useId } from 'react';
import { useChain, useChainComparison } from '../contexts/ChainContext';
import { ChainConfig } from '../config/chains';
import ChainBadge from './ChainBadge';

interface ChainSelectorProps {
  showGasPrices?: boolean;
  compact?: boolean;
}

const ChainSelector: React.FC<ChainSelectorProps> = ({
  showGasPrices = true,
  compact = false
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [focusedIndex, setFocusedIndex] = useState(-1);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const buttonRef = useRef<HTMLButtonElement>(null);
  const listboxId = useId();

  const {
    selectedChain,
    setSelectedChainId,
    multiChainGas,
    bestChainForTx,
    isLoading
  } = useChain();

  const chainComparison = useChainComparison();

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setIsOpen(false);
        setFocusedIndex(-1);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Handle keyboard navigation
  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    const chainCount = chainComparison.length;

    switch (e.key) {
      case 'Escape':
        setIsOpen(false);
        setFocusedIndex(-1);
        buttonRef.current?.focus();
        e.preventDefault();
        break;

      case 'ArrowDown':
        e.preventDefault();
        if (!isOpen) {
          setIsOpen(true);
          setFocusedIndex(0);
        } else {
          setFocusedIndex(prev => (prev + 1) % chainCount);
        }
        break;

      case 'ArrowUp':
        e.preventDefault();
        if (!isOpen) {
          setIsOpen(true);
          setFocusedIndex(chainCount - 1);
        } else {
          setFocusedIndex(prev => (prev - 1 + chainCount) % chainCount);
        }
        break;

      case 'Enter':
      case ' ':
        e.preventDefault();
        if (isOpen && focusedIndex >= 0) {
          const selectedOption = chainComparison[focusedIndex];
          if (selectedOption) {
            setSelectedChainId(selectedOption.chain.id);
            setIsOpen(false);
            setFocusedIndex(-1);
            buttonRef.current?.focus();
          }
        } else if (!isOpen) {
          setIsOpen(true);
          setFocusedIndex(0);
        }
        break;

      case 'Home':
        if (isOpen) {
          e.preventDefault();
          setFocusedIndex(0);
        }
        break;

      case 'End':
        if (isOpen) {
          e.preventDefault();
          setFocusedIndex(chainCount - 1);
        }
        break;

      case 'Tab':
        if (isOpen) {
          setIsOpen(false);
          setFocusedIndex(-1);
        }
        break;
    }
  }, [isOpen, focusedIndex, chainComparison, setSelectedChainId]);

  const handleSelectChain = (chainId: number) => {
    setSelectedChainId(chainId);
    setIsOpen(false);
    setFocusedIndex(-1);
    buttonRef.current?.focus();
  };

  const formatGasPrice = (price: number | null | undefined): string => {
    if (!Number.isFinite(price)) return 'N/A';
    const safePrice = Number(price);
    if (safePrice < 0) return 'N/A';
    if (safePrice >= 1) return safePrice.toFixed(2);
    if (safePrice >= 0.01) return safePrice.toFixed(4);
    return safePrice.toFixed(6);
  };

  const currentGas = multiChainGas[selectedChain.id];

  return (
    <div className="relative" ref={dropdownRef} onKeyDown={handleKeyDown}>
      {/* Selected Chain Button */}
      <button
        ref={buttonRef}
        onClick={() => setIsOpen(!isOpen)}
        aria-haspopup="listbox"
        aria-expanded={isOpen}
        aria-controls={listboxId}
        aria-label={`Select network. Current: ${selectedChain.name}`}
        className={`
          flex items-center gap-2 px-3 py-2 rounded-lg
          bg-gray-800 hover:bg-gray-700 border border-gray-600
          transition-all duration-200
          focus:outline-none focus:ring-2 focus:ring-cyan-500
          ${isOpen ? 'ring-2 ring-cyan-500' : ''}
          ${compact ? 'text-sm' : ''}
        `}
      >
        <ChainBadge chain={selectedChain} size={compact ? 'sm' : 'md'} />
        <span className="font-medium text-white">{selectedChain.shortName}</span>

        {showGasPrices && currentGas && !currentGas.loading && (
          <span className="text-cyan-400 font-mono text-sm" aria-label={`Gas price: ${formatGasPrice(currentGas.gasPrice)} gwei`}>
            {formatGasPrice(currentGas.gasPrice)}
          </span>
        )}

        {isLoading && (
          <span className="w-4 h-4 border-2 border-gray-500 border-t-cyan-400 rounded-full animate-spin" aria-hidden="true" />
        )}

        <svg
          className={`w-4 h-4 text-gray-400 transition-transform ${isOpen ? 'rotate-180' : ''}`}
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
          aria-hidden="true"
        >
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {/* Dropdown Menu */}
      {isOpen && (
        <div
          id={listboxId}
          role="listbox"
          aria-label="Select network"
          className="absolute top-full left-0 mt-2 w-72 bg-gray-900 border border-gray-700 rounded-xl shadow-xl z-50 overflow-hidden"
        >
          {/* Header */}
          <div className="px-4 py-3 border-b border-gray-700 bg-gray-800/50">
            <div className="text-sm font-medium text-gray-300">Select Network</div>
            {bestChainForTx && bestChainForTx.chainId !== selectedChain.id && (
              <div className="text-xs text-green-400 mt-1">
                Tip: {bestChainForTx.reason}
              </div>
            )}
          </div>

          {/* Chain List */}
          <div className="max-h-80 overflow-y-auto">
            {chainComparison.map(({ chain, gas }, index) => (
              <ChainOption
                key={chain.id}
                chain={chain}
                gasPrice={gas?.gasPrice || 0}
                isSelected={chain.id === selectedChain.id}
                isBest={bestChainForTx?.chainId === chain.id}
                isLoading={gas?.loading || false}
                isFocused={index === focusedIndex}
                onClick={() => handleSelectChain(chain.id)}
                formatGasPrice={formatGasPrice}
              />
            ))}
          </div>

          {/* Footer */}
          <div className="px-4 py-2 border-t border-gray-700 bg-gray-800/30">
            <div className="text-xs text-gray-500 text-center">
              Gas prices update every 30s
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

interface ChainOptionProps {
  chain: ChainConfig;
  gasPrice: number;
  isSelected: boolean;
  isBest: boolean;
  isLoading: boolean;
  isFocused: boolean;
  onClick: () => void;
  formatGasPrice: (price: number) => string;
}

const ChainOption: React.FC<ChainOptionProps> = ({
  chain,
  gasPrice,
  isSelected,
  isBest,
  isLoading,
  isFocused,
  onClick,
  formatGasPrice
}) => {
  return (
    <div
      role="option"
      aria-selected={isSelected}
      onClick={onClick}
      className={`
        w-full px-4 py-3 flex items-center justify-between cursor-pointer
        hover:bg-gray-800 transition-colors
        ${isSelected ? 'bg-gray-800 border-l-2 border-cyan-500' : ''}
        ${isFocused ? 'bg-gray-700 outline-none ring-2 ring-inset ring-cyan-500' : ''}
      `}
    >
      <div className="flex items-center gap-3">
        <ChainBadge chain={chain} size="md" />
        <div className="text-left">
          <div className="flex items-center gap-2">
            <span className="font-medium text-white">{chain.name}</span>
            {isBest && (
              <span className="px-1.5 py-0.5 text-xs bg-green-500/20 text-green-400 rounded">
                Cheapest
              </span>
            )}
            {chain.isL2 && (
              <span className="px-1.5 py-0.5 text-xs bg-cyan-500/20 text-cyan-400 rounded">
                L2
              </span>
            )}
          </div>
          <div className="text-xs text-gray-500">
            Chain ID: {chain.id}
          </div>
        </div>
      </div>

      <div className="text-right">
        {isLoading ? (
          <div className="w-4 h-4 border-2 border-gray-500 border-t-cyan-400 rounded-full animate-spin" aria-hidden="true" />
        ) : (
          <>
            <div className="font-mono text-sm text-cyan-400">
              {formatGasPrice(gasPrice)} gwei
            </div>
            {isSelected && (
              <svg className="w-4 h-4 text-cyan-400 ml-auto" fill="currentColor" viewBox="0 0 20 20" aria-hidden="true">
                <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
              </svg>
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default ChainSelector;
