/**
 * Aria Live Region components for screen reader announcements.
 *
 * Provides accessible announcements for dynamic content updates
 * without requiring focus changes.
 *
 * @module components/ui/LiveRegion
 */

import React, { createContext, useContext, useState, useCallback, useRef, useEffect } from 'react';

/**
 * Live region politeness levels.
 */
export type Politeness = 'polite' | 'assertive' | 'off';

/**
 * Announcement to be made.
 */
export interface Announcement {
  message: string;
  politeness?: Politeness;
  id?: string;
}

/**
 * Live region context value.
 */
interface LiveRegionContextValue {
  announce: (message: string, politeness?: Politeness) => void;
  announcePolite: (message: string) => void;
  announceAssertive: (message: string) => void;
  clear: () => void;
}

const LiveRegionContext = createContext<LiveRegionContextValue | null>(null);

/**
 * Provider for the live region context.
 *
 * @example
 * ```tsx
 * <LiveRegionProvider>
 *   <App />
 * </LiveRegionProvider>
 * ```
 */
export const LiveRegionProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [politeMessage, setPoliteMessage] = useState('');
  const [assertiveMessage, setAssertiveMessage] = useState('');
  const clearTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const announce = useCallback((message: string, politeness: Politeness = 'polite') => {
    // Clear previous timeout
    if (clearTimeoutRef.current) {
      clearTimeout(clearTimeoutRef.current);
    }

    if (politeness === 'assertive') {
      setAssertiveMessage(message);
    } else if (politeness === 'polite') {
      setPoliteMessage(message);
    }

    // Auto-clear after announcement
    clearTimeoutRef.current = setTimeout(() => {
      setPoliteMessage('');
      setAssertiveMessage('');
    }, 5000);
  }, []);

  const announcePolite = useCallback((message: string) => {
    announce(message, 'polite');
  }, [announce]);

  const announceAssertive = useCallback((message: string) => {
    announce(message, 'assertive');
  }, [announce]);

  const clear = useCallback(() => {
    setPoliteMessage('');
    setAssertiveMessage('');
  }, []);

  useEffect(() => {
    return () => {
      if (clearTimeoutRef.current) {
        clearTimeout(clearTimeoutRef.current);
      }
    };
  }, []);

  return (
    <LiveRegionContext.Provider value={{ announce, announcePolite, announceAssertive, clear }}>
      {children}
      {/* Screen reader only live regions */}
      <div
        role="status"
        aria-live="polite"
        aria-atomic="true"
        className="sr-only"
      >
        {politeMessage}
      </div>
      <div
        role="alert"
        aria-live="assertive"
        aria-atomic="true"
        className="sr-only"
      >
        {assertiveMessage}
      </div>
    </LiveRegionContext.Provider>
  );
};

/**
 * Hook to access the live region context.
 *
 * @example
 * ```tsx
 * function PriceDisplay({ price }) {
 *   const { announcePolite } = useLiveRegion();
 *
 *   useEffect(() => {
 *     announcePolite(`Gas price updated to ${price} gwei`);
 *   }, [price]);
 *
 *   return <div>{price} gwei</div>;
 * }
 * ```
 */
export function useLiveRegion(): LiveRegionContextValue {
  const context = useContext(LiveRegionContext);
  if (!context) {
    // Return no-op functions if used outside provider
    return {
      announce: () => {},
      announcePolite: () => {},
      announceAssertive: () => {},
      clear: () => {},
    };
  }
  return context;
}

/**
 * Standalone live region component for simple use cases.
 *
 * @example
 * ```tsx
 * <LiveRegion politeness="polite">
 *   {status ? `Status: ${status}` : ''}
 * </LiveRegion>
 * ```
 */
export const LiveRegion: React.FC<{
  children: React.ReactNode;
  politeness?: Politeness;
  atomic?: boolean;
  relevant?: 'additions' | 'removals' | 'text' | 'all';
  className?: string;
}> = ({
  children,
  politeness = 'polite',
  atomic = true,
  relevant = 'additions text',
  className = '',
}) => {
  return (
    <div
      role={politeness === 'assertive' ? 'alert' : 'status'}
      aria-live={politeness}
      aria-atomic={atomic}
      aria-relevant={relevant}
      className={`sr-only ${className}`.trim()}
    >
      {children}
    </div>
  );
};

/**
 * Visible status indicator with live region.
 *
 * @example
 * ```tsx
 * <StatusAnnouncer status="success">
 *   Price updated successfully
 * </StatusAnnouncer>
 * ```
 */
export const StatusAnnouncer: React.FC<{
  children: React.ReactNode;
  status?: 'success' | 'error' | 'warning' | 'info';
  politeness?: Politeness;
  className?: string;
}> = ({
  children,
  status = 'info',
  politeness = 'polite',
  className = '',
}) => {
  const statusColors = {
    success: 'text-green-400',
    error: 'text-red-400',
    warning: 'text-yellow-400',
    info: 'text-cyan-400',
  };

  return (
    <div
      role={politeness === 'assertive' ? 'alert' : 'status'}
      aria-live={politeness}
      className={`${statusColors[status]} ${className}`.trim()}
    >
      {children}
    </div>
  );
};

/**
 * Hook for announcing value changes.
 *
 * @example
 * ```tsx
 * function GasPrice({ value }) {
 *   useAnnounceChange(value, (val) => `Gas price: ${val} gwei`);
 *   return <div>{value} gwei</div>;
 * }
 * ```
 */
export function useAnnounceChange<T>(
  value: T,
  formatter: (value: T) => string,
  options: { politeness?: Politeness; debounceMs?: number } = {}
): void {
  const { politeness = 'polite', debounceMs = 1000 } = options;
  const { announce } = useLiveRegion();
  const previousValue = useRef<T>(value);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (value !== previousValue.current) {
      // Debounce rapid changes
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }

      timeoutRef.current = setTimeout(() => {
        announce(formatter(value), politeness);
        previousValue.current = value;
      }, debounceMs);
    }

    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
      }
    };
  }, [value, formatter, announce, politeness, debounceMs]);
}

export default LiveRegion;
