import React, { useCallback, useEffect, useId, useMemo, useRef, useState } from 'react';

export interface ComboboxOption {
  id: string;
  label: string;
  value: string;
}

interface ComboboxProps {
  label: string;
  options: ComboboxOption[];
  placeholder?: string;
  onSelect: (option: ComboboxOption) => void;
}

const Combobox: React.FC<ComboboxProps> = ({
  label,
  options,
  placeholder = 'Search...',
  onSelect,
}) => {
  const inputId = useId();
  const listboxId = useId();
  const containerRef = useRef<HTMLDivElement>(null);
  const [inputValue, setInputValue] = useState('');
  const [isOpen, setIsOpen] = useState(false);
  const [highlightedIndex, setHighlightedIndex] = useState(-1);

  const filteredOptions = useMemo(() => {
    const query = inputValue.toLowerCase();
    return options.filter((option) => option.label.toLowerCase().includes(query));
  }, [options, inputValue]);

  const selectOption = useCallback(
    (option: ComboboxOption) => {
      setInputValue(option.label);
      setIsOpen(false);
      setHighlightedIndex(-1);
      onSelect(option);
    },
    [onSelect]
  );

  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setIsOpen(false);
        setHighlightedIndex(-1);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  const handleKeyDown = (event: React.KeyboardEvent<HTMLInputElement>) => {
    if (!isOpen && (event.key === 'ArrowDown' || event.key === 'ArrowUp')) {
      setIsOpen(true);
      setHighlightedIndex(0);
      event.preventDefault();
      return;
    }

    switch (event.key) {
      case 'ArrowDown':
        event.preventDefault();
        setHighlightedIndex((prev) =>
          prev < filteredOptions.length - 1 ? prev + 1 : 0
        );
        break;
      case 'ArrowUp':
        event.preventDefault();
        setHighlightedIndex((prev) =>
          prev > 0 ? prev - 1 : filteredOptions.length - 1
        );
        break;
      case 'Enter':
        if (isOpen && highlightedIndex >= 0) {
          event.preventDefault();
          const option = filteredOptions[highlightedIndex];
          if (option) {
            selectOption(option);
          }
        }
        break;
      case 'Escape':
        setIsOpen(false);
        setHighlightedIndex(-1);
        break;
      default:
        break;
    }
  };

  const activeOptionId =
    highlightedIndex >= 0 ? `${inputId}-option-${highlightedIndex}` : undefined;

  return (
    <div ref={containerRef} className="relative w-full">
      <label htmlFor={inputId} className="block text-sm font-medium text-gray-300 mb-1">
        {label}
      </label>
      <input
        id={inputId}
        role="combobox"
        aria-autocomplete="list"
        aria-expanded={isOpen}
        aria-controls={listboxId}
        aria-activedescendant={activeOptionId}
        className="w-full rounded-lg border border-gray-700 bg-gray-900 px-3 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-cyan-500"
        value={inputValue}
        placeholder={placeholder}
        onChange={(event) => {
          setInputValue(event.target.value);
          setIsOpen(true);
          setHighlightedIndex(0);
        }}
        onKeyDown={handleKeyDown}
      />

      {isOpen && filteredOptions.length > 0 && (
        <ul
          id={listboxId}
          role="listbox"
          className="absolute z-50 mt-2 max-h-60 w-full overflow-y-auto rounded-lg border border-gray-700 bg-gray-900 shadow-xl"
        >
          {filteredOptions.map((option, index) => {
            const isActive = index === highlightedIndex;
            return (
              <li
                key={option.id}
                id={`${inputId}-option-${index}`}
                role="option"
                aria-selected={isActive}
                className={`cursor-pointer px-3 py-2 text-sm text-white ${
                  isActive ? 'bg-gray-700' : 'hover:bg-gray-800'
                }`}
                onMouseDown={(event) => {
                  event.preventDefault();
                  selectOption(option);
                }}
              >
                {option.label}
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
};

export default Combobox;
