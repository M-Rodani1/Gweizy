/**
 * Service Worker registration and utilities
 *
 * Handles service worker lifecycle, updates, and cache management.
 *
 * @module utils/serviceWorker
 */

export interface ServiceWorkerConfig {
  /** Callback when SW is successfully registered */
  onSuccess?: (registration: ServiceWorkerRegistration) => void;
  /** Callback when SW update is available */
  onUpdate?: (registration: ServiceWorkerRegistration) => void;
  /** Callback when content is cached for offline use */
  onOfflineReady?: () => void;
  /** Callback for registration errors */
  onError?: (error: Error) => void;
}

export interface CacheInfo {
  version: string;
  caches: string[];
  totalSize: number;
  entryCount: number;
}

/**
 * Register the service worker.
 *
 * @param config - Configuration options
 * @returns Promise that resolves to the registration, or null if unsupported
 *
 * @example
 * ```ts
 * await registerServiceWorker({
 *   onSuccess: () => console.log('SW registered'),
 *   onUpdate: () => showUpdateNotification(),
 * });
 * ```
 */
export async function registerServiceWorker(
  config: ServiceWorkerConfig = {}
): Promise<ServiceWorkerRegistration | null> {
  const { onSuccess, onUpdate, onOfflineReady, onError } = config;

  // Check if service workers are supported
  if (!('serviceWorker' in navigator)) {
    console.log('[SW] Service workers not supported');
    return null;
  }

  // Only register in production or when explicitly enabled
  if (import.meta.env.DEV && !import.meta.env.VITE_SW_DEV) {
    console.log('[SW] Skipping SW registration in development');
    return null;
  }

  try {
    const registration = await navigator.serviceWorker.register('/sw.js', {
      scope: '/',
    });

    console.log('[SW] Service Worker registered with scope:', registration.scope);

    // Check for updates on page load
    registration.addEventListener('updatefound', () => {
      const newWorker = registration.installing;
      if (!newWorker) return;

      newWorker.addEventListener('statechange', () => {
        if (newWorker.state === 'installed') {
          if (navigator.serviceWorker.controller) {
            // New content available
            console.log('[SW] New content available, update ready');
            onUpdate?.(registration);
          } else {
            // Content cached for offline use
            console.log('[SW] Content cached for offline use');
            onOfflineReady?.();
          }
        }
      });
    });

    // Registration successful
    onSuccess?.(registration);

    return registration;
  } catch (error) {
    console.error('[SW] Registration failed:', error);
    onError?.(error instanceof Error ? error : new Error(String(error)));
    return null;
  }
}

/**
 * Unregister all service workers.
 *
 * @returns Promise that resolves when all workers are unregistered
 */
export async function unregisterServiceWorker(): Promise<boolean> {
  if (!('serviceWorker' in navigator)) {
    return false;
  }

  try {
    const registrations = await navigator.serviceWorker.getRegistrations();
    await Promise.all(registrations.map((reg) => reg.unregister()));
    console.log('[SW] All service workers unregistered');
    return true;
  } catch (error) {
    console.error('[SW] Unregistration failed:', error);
    return false;
  }
}

/**
 * Force the waiting service worker to activate.
 *
 * @returns Promise that resolves when message is sent
 */
export async function skipWaiting(): Promise<void> {
  const registration = await navigator.serviceWorker.ready;
  const waitingWorker = registration.waiting;

  if (waitingWorker) {
    waitingWorker.postMessage({ type: 'SKIP_WAITING' });
    // Reload to get new content
    window.location.reload();
  }
}

/**
 * Clear the API cache.
 *
 * @returns Promise that resolves when message is sent
 */
export async function clearApiCache(): Promise<void> {
  const registration = await navigator.serviceWorker.ready;
  const activeWorker = registration.active;

  if (activeWorker) {
    activeWorker.postMessage({ type: 'CLEAR_API_CACHE' });
  }
}

/**
 * Get cache information from the service worker.
 *
 * @returns Promise that resolves with cache info
 */
export async function getCacheInfo(): Promise<CacheInfo | null> {
  if (!('serviceWorker' in navigator) || !navigator.serviceWorker.controller) {
    return null;
  }

  return new Promise((resolve) => {
    const channel = new MessageChannel();

    channel.port1.onmessage = (event) => {
      if (event.data?.type === 'CACHE_INFO') {
        resolve(event.data.payload);
      }
    };

    navigator.serviceWorker.controller.postMessage(
      { type: 'GET_CACHE_INFO' },
      [channel.port2]
    );

    // Timeout after 5 seconds
    setTimeout(() => resolve(null), 5000);
  });
}

/**
 * Check if the app is running with an active service worker.
 *
 * @returns True if a service worker is controlling the page
 */
export function isServiceWorkerActive(): boolean {
  return 'serviceWorker' in navigator && !!navigator.serviceWorker.controller;
}

/**
 * Check if the app is running offline.
 *
 * @returns True if the browser is offline
 */
export function isOffline(): boolean {
  return !navigator.onLine;
}

/**
 * Listen for online/offline status changes.
 *
 * @param onOnline - Callback when going online
 * @param onOffline - Callback when going offline
 * @returns Cleanup function to remove listeners
 */
export function listenForNetworkChanges(
  onOnline?: () => void,
  onOffline?: () => void
): () => void {
  const handleOnline = () => {
    console.log('[Network] Online');
    onOnline?.();
  };

  const handleOffline = () => {
    console.log('[Network] Offline');
    onOffline?.();
  };

  window.addEventListener('online', handleOnline);
  window.addEventListener('offline', handleOffline);

  return () => {
    window.removeEventListener('online', handleOnline);
    window.removeEventListener('offline', handleOffline);
  };
}

export default registerServiceWorker;
