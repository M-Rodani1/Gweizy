/**
 * Offline detection and handling utilities
 * Provides offline state management and queue for offline actions
 */

import { useState, useEffect } from 'react';

/**
 * Hook to detect online/offline status
 * @returns Object with isOnline status and lastOnline timestamp
 */
export function useOnlineStatus() {
  const [isOnline, setIsOnline] = useState(
    typeof navigator !== 'undefined' ? navigator.onLine : true
  );
  const [lastOnline, setLastOnline] = useState<number | null>(
    typeof navigator !== 'undefined' ? Date.now() : null
  );

  useEffect(() => {
    if (typeof navigator === 'undefined') return;

    const handleOnline = () => {
      setIsOnline(true);
      setLastOnline(Date.now());
    };

    const handleOffline = () => {
      setIsOnline(false);
    };

    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);

    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  return { isOnline, lastOnline };
}

/**
 * Queue for actions to perform when back online
 */
class OfflineQueue {
  private queue: Array<() => Promise<void>> = [];

  /**
   * Add action to queue
   */
  enqueue(action: () => Promise<void>): void {
    this.queue.push(action);
  }

  /**
   * Process all queued actions
   */
  async processQueue(): Promise<void> {
    while (this.queue.length > 0) {
      const action = this.queue.shift();
      if (action) {
        try {
          await action();
        } catch (error) {
          console.error('Failed to process offline queue action:', error);
          // Re-queue failed actions
          this.queue.push(action);
        }
      }
    }
  }

  /**
   * Clear queue
   */
  clear(): void {
    this.queue = [];
  }

  /**
   * Get queue length
   */
  getLength(): number {
    return this.queue.length;
  }
}

export const offlineQueue = new OfflineQueue();

// Process queue when back online
if (typeof window !== 'undefined') {
  window.addEventListener('online', () => {
    offlineQueue.processQueue();
  });
}
