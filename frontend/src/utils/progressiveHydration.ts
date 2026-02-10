import type { ReactElement } from 'react';
import { createRoot, hydrateRoot } from 'react-dom/client';

type IdleDeadline = { didTimeout: boolean; timeRemaining: () => number };

export interface ProgressiveHydrationOptions {
  deferHydration?: boolean;
  idleTimeout?: number;
}

export function scheduleHydration(callback: () => void, timeout = 2000): number {
  const ric = (globalThis as { requestIdleCallback?: (cb: (d: IdleDeadline) => void, opts?: { timeout: number }) => number })
    .requestIdleCallback;

  if (typeof ric === 'function') {
    return ric(() => callback(), { timeout });
  }

  const setTimeoutFn = (globalThis as { setTimeout?: (cb: () => void, ms: number) => number }).setTimeout;
  if (typeof setTimeoutFn === 'function') {
    return setTimeoutFn(callback, 1);
  }

  callback();
  return 0;
}

export function mountApp(
  rootElement: HTMLElement,
  element: ReactElement,
  options: ProgressiveHydrationOptions = {}
): { mode: 'hydrate' | 'render'; deferred: boolean } {
  const { deferHydration = true, idleTimeout = 2000 } = options;
  const shouldHydrate = rootElement.hasChildNodes();

  if (shouldHydrate) {
    if (deferHydration) {
      scheduleHydration(() => {
        hydrateRoot(rootElement, element);
      }, idleTimeout);
      return { mode: 'hydrate', deferred: true };
    }
    hydrateRoot(rootElement, element);
    return { mode: 'hydrate', deferred: false };
  }

  const root = createRoot(rootElement);
  root.render(element);
  return { mode: 'render', deferred: false };
}
