/**
 * Performance optimisation utilities for the Gweizy app
 */

/**
 * Debounce function - delays execution until after delay ms have passed since last call
 */
export function debounce<T extends (...args: unknown[]) => unknown>(
  func: T,
  delay: number
): (...args: Parameters<T>) => void {
  let timeoutId: ReturnType<typeof setTimeout>;
  
  return function (...args: Parameters<T>) {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => func(...args), delay);
  };
}

/**
 * Throttle function - limits execution to once per delay ms
 */
export function throttle<T extends (...args: unknown[]) => unknown>(
  func: T,
  delay: number
): (...args: Parameters<T>) => void {
  let lastCall = 0;
  
  return function (...args: Parameters<T>) {
    const now = Date.now();
    if (now - lastCall >= delay) {
      lastCall = now;
      func(...args);
    }
  };
}

/**
 * Request Idle Callback polyfill and wrapper
 * Schedules work to be done during browser idle time
 */
export function scheduleIdleWork(callback: () => void, timeout = 2000): void {
  if ('requestIdleCallback' in window) {
    window.requestIdleCallback(callback, { timeout });
  } else {
    setTimeout(callback, 100);
  }
}

/**
 * Preload a module for faster subsequent loading
 */
export function preloadModule(importFn: () => Promise<unknown>): void {
  scheduleIdleWork(() => {
    importFn().catch(() => {
      // Silently fail - this is just a preload hint
    });
  });
}

/**
 * Check if the user prefers reduced motion
 */
export function prefersReducedMotion(): boolean {
  if (typeof window === 'undefined') return false;
  return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
}

/**
 * Check if the device has a slow connection
 */
export function hasSlowConnection(): boolean {
  if (typeof navigator === 'undefined') return false;
  
  const connection = (navigator as Navigator & {
    connection?: { effectiveType?: string; saveData?: boolean };
  }).connection;
  
  if (!connection) return false;
  
  return connection.saveData || 
         connection.effectiveType === 'slow-2g' ||
         connection.effectiveType === '2g';
}

/**
 * Memoize a function's results based on its arguments
 */
export function memoize<T extends (...args: unknown[]) => unknown>(
  func: T,
  getKey: (...args: Parameters<T>) => string = (...args) => JSON.stringify(args)
): T {
  const cache = new Map<string, ReturnType<T>>();
  
  return function (...args: Parameters<T>): ReturnType<T> {
    const key = getKey(...args);
    
    if (cache.has(key)) {
      return cache.get(key) as ReturnType<T>;
    }
    
    const result = func(...args) as ReturnType<T>;
    cache.set(key, result);
    
    // Limit cache size to prevent memory leaks
    if (cache.size > 100) {
      const firstKey = cache.keys().next().value;
      if (firstKey !== undefined) {
        cache.delete(firstKey);
      }
    }
    
    return result;
  } as T;
}

/**
 * Format bytes to human readable string (memoized)
 */
export const formatBytes = (bytes: number, decimals = 2): string => {
  if (bytes === 0) return '0 Bytes';

  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];

  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
};

/**
 * Measure component render time (development only)
 */
export function measureRender(componentName: string): () => void {
  if (process.env.NODE_ENV !== 'development') {
    return () => {};
  }
  
  const start = performance.now();
  
  return () => {
    const end = performance.now();
    const duration = end - start;
    
    if (duration > 16) { // More than 1 frame at 60fps
      console.warn(`[Performance] ${componentName} took ${duration.toFixed(2)}ms to render`);
    }
  };
}

/**
 * Render performance monitor for tracking slow components in development.
 */
export interface RenderMetric {
  componentName: string;
  duration: number;
  threshold: number;
  timestamp: number;
}

const renderMetrics: RenderMetric[] = [];
let renderThresholdMs = 16;

export function setRenderThresholdMs(value: number): void {
  renderThresholdMs = value;
}

export function getRenderMetrics(): RenderMetric[] {
  return [...renderMetrics];
}

export function clearRenderMetrics(): void {
  renderMetrics.length = 0;
}

export function createRenderMonitor(componentName: string, thresholdMs = renderThresholdMs): () => void {
  if (process.env.NODE_ENV !== 'development') {
    return () => {};
  }

  const start = performance.now();

  return () => {
    const duration = performance.now() - start;
    if (duration < thresholdMs) {
      return;
    }

    const metric: RenderMetric = {
      componentName,
      duration,
      threshold: thresholdMs,
      timestamp: Date.now(),
    };
    renderMetrics.push(metric);
    console.warn(
      `[Performance] ${componentName} render ${duration.toFixed(2)}ms exceeded ${thresholdMs}ms`
    );
  };
}

export function withRenderMonitor<T extends (...args: any[]) => any>(
  componentName: string,
  fn: T,
  thresholdMs = renderThresholdMs
): T {
  return function (...args: Parameters<T>): ReturnType<T> {
    const stop = createRenderMonitor(componentName, thresholdMs);
    const result = fn(...args);
    stop();
    return result;
  } as T;
}

/**
 * Enforce HTTPS by redirecting non-localhost traffic.
 */
export function enforceHttps(): boolean {
  if (typeof window === 'undefined') return false;
  const { protocol, hostname, href } = window.location;
  if (protocol === 'https:') return false;
  if (hostname === 'localhost' || hostname === '127.0.0.1') return false;
  window.location.href = href.replace(/^http:/, 'https:');
  return true;
}
