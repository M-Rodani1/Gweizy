import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import React from 'react';
import { mountApp } from '../../utils/progressiveHydration';

const { renderMock, hydrateRootMock, createRootMock } = vi.hoisted(() => {
  const render = vi.fn();
  return {
    renderMock: render,
    hydrateRootMock: vi.fn(),
    createRootMock: vi.fn(() => ({ render })),
  };
});

vi.mock('react-dom/client', () => ({
  createRoot: createRootMock,
  hydrateRoot: hydrateRootMock,
}));

describe('progressive hydration', () => {
  const originalRic = globalThis.requestIdleCallback;

  beforeEach(() => {
    renderMock.mockClear();
    hydrateRootMock.mockClear();
    createRootMock.mockClear();
  });

  afterEach(() => {
    if (originalRic) {
      globalThis.requestIdleCallback = originalRic;
    } else {
      delete (globalThis as { requestIdleCallback?: unknown }).requestIdleCallback;
    }
  });

  it('defers hydration when SSR markup is present', () => {
    const root = document.createElement('div');
    root.appendChild(document.createElement('span'));

    globalThis.requestIdleCallback = vi.fn((cb: () => void) => {
      cb();
      return 1;
    });

    const result = mountApp(root, <div>App</div>);

    expect(result).toEqual({ mode: 'hydrate', deferred: true });
    expect(globalThis.requestIdleCallback).toHaveBeenCalled();
    expect(hydrateRootMock).toHaveBeenCalledWith(root, <div>App</div>);
    expect(createRootMock).not.toHaveBeenCalled();
  });

  it('renders immediately when no SSR markup exists', () => {
    const root = document.createElement('div');

    const result = mountApp(root, <div>App</div>);

    expect(result).toEqual({ mode: 'render', deferred: false });
    expect(createRootMock).toHaveBeenCalledWith(root);
    expect(renderMock).toHaveBeenCalledWith(<div>App</div>);
    expect(hydrateRootMock).not.toHaveBeenCalled();
  });

  it('hydrates immediately when deferral is disabled', () => {
    const root = document.createElement('div');
    root.appendChild(document.createElement('span'));

    globalThis.requestIdleCallback = vi.fn();

    const result = mountApp(root, <div>App</div>, { deferHydration: false });

    expect(result).toEqual({ mode: 'hydrate', deferred: false });
    expect(hydrateRootMock).toHaveBeenCalledWith(root, <div>App</div>);
    expect(globalThis.requestIdleCallback).not.toHaveBeenCalled();
  });
});
