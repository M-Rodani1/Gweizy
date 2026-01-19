import React, { useCallback, useEffect, useMemo, useState } from 'react';

interface VirtualizedListProps<T> {
  items: T[];
  itemHeight: number;
  maxHeight: number;
  overscan?: number;
  className?: string;
  contentClassName?: string;
  renderItem: (item: T, index: number) => React.ReactNode;
  getKey?: (item: T, index: number) => React.Key;
}

const VirtualizedList = <T,>({
  items,
  itemHeight,
  maxHeight,
  overscan = 3,
  className = '',
  contentClassName = '',
  renderItem,
  getKey
}: VirtualizedListProps<T>) => {
  const [scrollTop, setScrollTop] = useState(0);

  useEffect(() => {
    setScrollTop(0);
  }, [items.length]);

  const totalHeight = items.length * itemHeight;
  const height = Math.min(maxHeight, totalHeight);

  const { startIndex, endIndex } = useMemo(() => {
    if (items.length === 0) {
      return { startIndex: 0, endIndex: -1 };
    }
    const start = Math.max(0, Math.floor(scrollTop / itemHeight) - overscan);
    const end = Math.min(
      items.length - 1,
      Math.floor((scrollTop + height) / itemHeight) + overscan
    );
    return { startIndex: start, endIndex: end };
  }, [height, itemHeight, items.length, overscan, scrollTop]);

  const visibleItems = useMemo(() => {
    if (endIndex < startIndex) return [];
    return items.slice(startIndex, endIndex + 1);
  }, [endIndex, items, startIndex]);

  const handleScroll = useCallback((event: React.UIEvent<HTMLDivElement>) => {
    setScrollTop(event.currentTarget.scrollTop);
  }, []);

  if (items.length === 0) {
    return null;
  }

  return (
    <div
      className={className}
      style={{ height, overflowY: 'auto' }}
      onScroll={handleScroll}
    >
      <div
        className={contentClassName}
        style={{ height: totalHeight, position: 'relative' }}
      >
        {visibleItems.map((item, offset) => {
          const index = startIndex + offset;
          const key = getKey ? getKey(item, index) : index;
          return (
            <div
              key={key}
              style={{
                position: 'absolute',
                top: index * itemHeight,
                left: 0,
                right: 0,
                height: itemHeight
              }}
            >
              {renderItem(item, index)}
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default VirtualizedList;
