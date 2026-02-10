import { describe, it, expect, beforeEach } from 'vitest';
import {
  applyResourceHints,
  ensureResourceHint,
  preconnect,
  dnsPrefetch,
  preloadResource,
  prefetchResource,
} from '../../utils/resourceHints';

describe('resource hints', () => {
  beforeEach(() => {
    document.head.innerHTML = '';
  });

  it('adds a preload hint with attributes', () => {
    const link = ensureResourceHint(
      preloadResource('/assets/app.js', {
        as: 'script',
        fetchPriority: 'high',
      })
    );

    expect(link.rel).toBe('preload');
    expect(link.href).toContain('/assets/app.js');
    expect(link.as).toBe('script');
    expect(link.getAttribute('fetchpriority')).toBe('high');
  });

  it('adds prefetch and preconnect hints without duplicates', () => {
    const hints = [
      preconnect('https://api.example.com'),
      dnsPrefetch('https://api.example.com'),
      prefetchResource('/data.json', { as: 'fetch' }),
    ];

    applyResourceHints(hints);
    applyResourceHints(hints);

    const links = document.querySelectorAll('link');
    expect(links).toHaveLength(3);
    expect(document.querySelector('link[rel="preconnect"][href="https://api.example.com"]')).toBeTruthy();
    expect(document.querySelector('link[rel="dns-prefetch"][href="https://api.example.com"]')).toBeTruthy();
    expect(document.querySelector('link[rel="prefetch"][href="/data.json"]')).toBeTruthy();
  });
});
