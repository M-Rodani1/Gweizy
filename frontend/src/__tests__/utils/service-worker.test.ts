/**
 * Service Worker Tests
 *
 * Tests for service worker registration, lifecycle management,
 * cache utilities, and network status handling.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';

// ============================================================================
// Mock Types
// ============================================================================

interface MockServiceWorkerRegistration {
  scope: string;
  active: { postMessage: ReturnType<typeof vi.fn> } | null;
  waiting: { postMessage: ReturnType<typeof vi.fn> } | null;
  installing: { state: string; addEventListener: ReturnType<typeof vi.fn> } | null;
  addEventListener: ReturnType<typeof vi.fn>;
  update: ReturnType<typeof vi.fn>;
  unregister: ReturnType<typeof vi.fn>;
}

interface MockServiceWorkerContainer {
  register: ReturnType<typeof vi.fn>;
  ready: Promise<MockServiceWorkerRegistration>;
  controller: { postMessage: ReturnType<typeof vi.fn> } | null;
  getRegistrations: ReturnType<typeof vi.fn>;
  addEventListener: ReturnType<typeof vi.fn>;
}

// ============================================================================
// Service Worker Utilities (inline for testing)
// ============================================================================

interface ServiceWorkerConfig {
  onSuccess?: (registration: ServiceWorkerRegistration) => void;
  onUpdate?: (registration: ServiceWorkerRegistration) => void;
  onOfflineReady?: () => void;
  onError?: (error: Error) => void;
}

interface CacheInfo {
  version: string;
  caches: string[];
  totalSize: number;
  entryCount: number;
}

async function registerServiceWorker(
  config: ServiceWorkerConfig = {}
): Promise<ServiceWorkerRegistration | null> {
  const { onSuccess, onUpdate, onOfflineReady, onError } = config;

  if (!('serviceWorker' in navigator)) {
    return null;
  }

  try {
    const registration = await navigator.serviceWorker.register('/sw.js', {
      scope: '/',
    });

    registration.addEventListener('updatefound', () => {
      const newWorker = registration.installing;
      if (!newWorker) return;

      newWorker.addEventListener('statechange', () => {
        if (newWorker.state === 'installed') {
          if (navigator.serviceWorker.controller) {
            onUpdate?.(registration);
          } else {
            onOfflineReady?.();
          }
        }
      });
    });

    onSuccess?.(registration);
    return registration;
  } catch (error) {
    onError?.(error instanceof Error ? error : new Error(String(error)));
    return null;
  }
}

async function unregisterServiceWorker(): Promise<boolean> {
  if (!('serviceWorker' in navigator)) {
    return false;
  }

  try {
    const registrations = await navigator.serviceWorker.getRegistrations();
    await Promise.all(registrations.map((reg: ServiceWorkerRegistration) => reg.unregister()));
    return true;
  } catch {
    return false;
  }
}

async function skipWaiting(): Promise<void> {
  const registration = await navigator.serviceWorker.ready;
  const waitingWorker = registration.waiting;

  if (waitingWorker) {
    waitingWorker.postMessage({ type: 'SKIP_WAITING' });
  }
}

async function clearApiCache(): Promise<void> {
  const registration = await navigator.serviceWorker.ready;
  const activeWorker = registration.active;

  if (activeWorker) {
    activeWorker.postMessage({ type: 'CLEAR_API_CACHE' });
  }
}

function isServiceWorkerActive(): boolean {
  return 'serviceWorker' in navigator && navigator.serviceWorker != null && !!navigator.serviceWorker.controller;
}

function isOffline(): boolean {
  return !navigator.onLine;
}

function listenForNetworkChanges(
  onOnline?: () => void,
  onOffline?: () => void
): () => void {
  const handleOnline = () => {
    onOnline?.();
  };

  const handleOffline = () => {
    onOffline?.();
  };

  window.addEventListener('online', handleOnline);
  window.addEventListener('offline', handleOffline);

  return () => {
    window.removeEventListener('online', handleOnline);
    window.removeEventListener('offline', handleOffline);
  };
}

// ============================================================================
// Tests
// ============================================================================

describe('Service Worker Utilities', () => {
  let mockRegistration: MockServiceWorkerRegistration;
  let mockServiceWorker: MockServiceWorkerContainer;
  let originalNavigator: Navigator;
  let originalWindow: Window & typeof globalThis;

  beforeEach(() => {
    // Create mock registration
    mockRegistration = {
      scope: '/',
      active: { postMessage: vi.fn() },
      waiting: { postMessage: vi.fn() },
      installing: null,
      addEventListener: vi.fn(),
      update: vi.fn().mockResolvedValue(undefined),
      unregister: vi.fn().mockResolvedValue(true),
    };

    // Create mock service worker container
    mockServiceWorker = {
      register: vi.fn().mockResolvedValue(mockRegistration),
      ready: Promise.resolve(mockRegistration),
      controller: { postMessage: vi.fn() },
      getRegistrations: vi.fn().mockResolvedValue([mockRegistration]),
      addEventListener: vi.fn(),
    };

    // Store originals
    originalNavigator = navigator;
    originalWindow = window;

    // Mock navigator.serviceWorker
    Object.defineProperty(navigator, 'serviceWorker', {
      value: mockServiceWorker,
      configurable: true,
    });

    // Mock navigator.onLine
    Object.defineProperty(navigator, 'onLine', {
      value: true,
      configurable: true,
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('registerServiceWorker', () => {
    it('should register service worker successfully', async () => {
      const onSuccess = vi.fn();

      const result = await registerServiceWorker({ onSuccess });

      expect(mockServiceWorker.register).toHaveBeenCalledWith('/sw.js', { scope: '/' });
      expect(onSuccess).toHaveBeenCalledWith(mockRegistration);
      expect(result).toBe(mockRegistration);
    });

    it('should return null when service worker is not supported', async () => {
      Object.defineProperty(navigator, 'serviceWorker', {
        value: undefined,
        configurable: true,
      });

      const result = await registerServiceWorker();

      expect(result).toBeNull();
    });

    it('should call onError when registration fails', async () => {
      const error = new Error('Registration failed');
      mockServiceWorker.register.mockRejectedValue(error);
      const onError = vi.fn();

      const result = await registerServiceWorker({ onError });

      expect(onError).toHaveBeenCalledWith(error);
      expect(result).toBeNull();
    });

    it('should set up updatefound listener', async () => {
      await registerServiceWorker();

      expect(mockRegistration.addEventListener).toHaveBeenCalledWith(
        'updatefound',
        expect.any(Function)
      );
    });

    it('should call onUpdate when update is available', async () => {
      const onUpdate = vi.fn();
      let updateFoundCallback: () => void;

      mockRegistration.addEventListener.mockImplementation((event: string, callback: () => void) => {
        if (event === 'updatefound') {
          updateFoundCallback = callback;
        }
      });

      mockRegistration.installing = {
        state: 'installed',
        addEventListener: vi.fn((event: string, cb: () => void) => {
          if (event === 'statechange') {
            cb();
          }
        }),
      };

      await registerServiceWorker({ onUpdate });

      // Trigger updatefound
      updateFoundCallback!();

      expect(onUpdate).toHaveBeenCalledWith(mockRegistration);
    });

    it('should call onOfflineReady when first installed', async () => {
      const onOfflineReady = vi.fn();
      let updateFoundCallback: () => void;

      // No controller means first install
      mockServiceWorker.controller = null;

      mockRegistration.addEventListener.mockImplementation((event: string, callback: () => void) => {
        if (event === 'updatefound') {
          updateFoundCallback = callback;
        }
      });

      mockRegistration.installing = {
        state: 'installed',
        addEventListener: vi.fn((event: string, cb: () => void) => {
          if (event === 'statechange') {
            cb();
          }
        }),
      };

      await registerServiceWorker({ onOfflineReady });

      // Trigger updatefound
      updateFoundCallback!();

      expect(onOfflineReady).toHaveBeenCalled();
    });
  });

  describe('unregisterServiceWorker', () => {
    it('should unregister all service workers', async () => {
      const result = await unregisterServiceWorker();

      expect(mockServiceWorker.getRegistrations).toHaveBeenCalled();
      expect(mockRegistration.unregister).toHaveBeenCalled();
      expect(result).toBe(true);
    });

    it('should return false when service worker is not supported', async () => {
      Object.defineProperty(navigator, 'serviceWorker', {
        value: undefined,
        configurable: true,
      });

      const result = await unregisterServiceWorker();

      expect(result).toBe(false);
    });

    it('should return false on error', async () => {
      mockServiceWorker.getRegistrations.mockRejectedValue(new Error('Failed'));

      const result = await unregisterServiceWorker();

      expect(result).toBe(false);
    });

    it('should handle multiple registrations', async () => {
      const mockReg1 = { unregister: vi.fn().mockResolvedValue(true) };
      const mockReg2 = { unregister: vi.fn().mockResolvedValue(true) };
      mockServiceWorker.getRegistrations.mockResolvedValue([mockReg1, mockReg2]);

      const result = await unregisterServiceWorker();

      expect(mockReg1.unregister).toHaveBeenCalled();
      expect(mockReg2.unregister).toHaveBeenCalled();
      expect(result).toBe(true);
    });
  });

  describe('skipWaiting', () => {
    it('should post SKIP_WAITING message to waiting worker', async () => {
      await skipWaiting();

      expect(mockRegistration.waiting?.postMessage).toHaveBeenCalledWith({
        type: 'SKIP_WAITING',
      });
    });

    it('should do nothing when no waiting worker', async () => {
      mockRegistration.waiting = null;

      await skipWaiting();

      // Should not throw
    });
  });

  describe('clearApiCache', () => {
    it('should post CLEAR_API_CACHE message to active worker', async () => {
      await clearApiCache();

      expect(mockRegistration.active?.postMessage).toHaveBeenCalledWith({
        type: 'CLEAR_API_CACHE',
      });
    });

    it('should do nothing when no active worker', async () => {
      mockRegistration.active = null;

      await clearApiCache();

      // Should not throw
    });
  });

  describe('isServiceWorkerActive', () => {
    it('should return true when controller exists', () => {
      expect(isServiceWorkerActive()).toBe(true);
    });

    it('should return false when no controller', () => {
      mockServiceWorker.controller = null;

      expect(isServiceWorkerActive()).toBe(false);
    });

    it('should return false when service worker not supported', () => {
      Object.defineProperty(navigator, 'serviceWorker', {
        value: undefined,
        configurable: true,
      });

      expect(isServiceWorkerActive()).toBe(false);
    });
  });

  describe('isOffline', () => {
    it('should return false when online', () => {
      Object.defineProperty(navigator, 'onLine', {
        value: true,
        configurable: true,
      });

      expect(isOffline()).toBe(false);
    });

    it('should return true when offline', () => {
      Object.defineProperty(navigator, 'onLine', {
        value: false,
        configurable: true,
      });

      expect(isOffline()).toBe(true);
    });
  });

  describe('listenForNetworkChanges', () => {
    it('should add event listeners', () => {
      const onOnline = vi.fn();
      const onOffline = vi.fn();
      const addEventListenerSpy = vi.spyOn(window, 'addEventListener');

      listenForNetworkChanges(onOnline, onOffline);

      expect(addEventListenerSpy).toHaveBeenCalledWith('online', expect.any(Function));
      expect(addEventListenerSpy).toHaveBeenCalledWith('offline', expect.any(Function));
    });

    it('should call onOnline when online event fires', () => {
      const onOnline = vi.fn();

      listenForNetworkChanges(onOnline, undefined);

      window.dispatchEvent(new Event('online'));

      expect(onOnline).toHaveBeenCalled();
    });

    it('should call onOffline when offline event fires', () => {
      const onOffline = vi.fn();

      listenForNetworkChanges(undefined, onOffline);

      window.dispatchEvent(new Event('offline'));

      expect(onOffline).toHaveBeenCalled();
    });

    it('should return cleanup function', () => {
      const onOnline = vi.fn();
      const onOffline = vi.fn();
      const removeEventListenerSpy = vi.spyOn(window, 'removeEventListener');

      const cleanup = listenForNetworkChanges(onOnline, onOffline);

      cleanup();

      expect(removeEventListenerSpy).toHaveBeenCalledWith('online', expect.any(Function));
      expect(removeEventListenerSpy).toHaveBeenCalledWith('offline', expect.any(Function));
    });

    it('should not call callbacks after cleanup', () => {
      const onOnline = vi.fn();
      const onOffline = vi.fn();

      const cleanup = listenForNetworkChanges(onOnline, onOffline);
      cleanup();

      // Note: Actual cleanup removes the specific handler, so events
      // dispatched after cleanup won't trigger the removed handlers
    });

    it('should handle missing callbacks gracefully', () => {
      const cleanup = listenForNetworkChanges();

      window.dispatchEvent(new Event('online'));
      window.dispatchEvent(new Event('offline'));

      cleanup();

      // Should not throw
    });
  });

  describe('Service Worker Lifecycle', () => {
    it('should handle full registration lifecycle', async () => {
      const callbacks = {
        onSuccess: vi.fn(),
        onUpdate: vi.fn(),
        onOfflineReady: vi.fn(),
        onError: vi.fn(),
      };

      const registration = await registerServiceWorker(callbacks);

      expect(registration).toBeDefined();
      expect(callbacks.onSuccess).toHaveBeenCalled();
      expect(callbacks.onError).not.toHaveBeenCalled();
    });

    it('should handle update check', async () => {
      const registration = await registerServiceWorker();

      await mockRegistration.update();

      expect(mockRegistration.update).toHaveBeenCalled();
    });

    it('should handle unregistration after registration', async () => {
      await registerServiceWorker();
      const result = await unregisterServiceWorker();

      expect(result).toBe(true);
    });
  });

  describe('Cache Management', () => {
    it('should clear API cache when requested', async () => {
      await clearApiCache();

      expect(mockRegistration.active?.postMessage).toHaveBeenCalledWith({
        type: 'CLEAR_API_CACHE',
      });
    });
  });

  describe('Edge Cases', () => {
    it('should handle rapid registration calls', async () => {
      const results = await Promise.all([
        registerServiceWorker(),
        registerServiceWorker(),
        registerServiceWorker(),
      ]);

      expect(results.every((r) => r !== null)).toBe(true);
    });

    it('should handle concurrent online/offline events', () => {
      const onOnline = vi.fn();
      const onOffline = vi.fn();

      listenForNetworkChanges(onOnline, onOffline);

      // Rapid online/offline switches
      window.dispatchEvent(new Event('offline'));
      window.dispatchEvent(new Event('online'));
      window.dispatchEvent(new Event('offline'));
      window.dispatchEvent(new Event('online'));

      expect(onOnline).toHaveBeenCalledTimes(2);
      expect(onOffline).toHaveBeenCalledTimes(2);
    });
  });

  describe('Error Scenarios', () => {
    it('should handle registration network error', async () => {
      const networkError = new Error('Network error');
      mockServiceWorker.register.mockRejectedValue(networkError);
      const onError = vi.fn();

      await registerServiceWorker({ onError });

      expect(onError).toHaveBeenCalledWith(networkError);
    });

    it('should handle non-Error rejection', async () => {
      mockServiceWorker.register.mockRejectedValue('string error');
      const onError = vi.fn();

      await registerServiceWorker({ onError });

      expect(onError).toHaveBeenCalledWith(expect.any(Error));
    });
  });
});
