import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, act } from '@testing-library/react';
import LazyImage from '../../components/ui/LazyImage';

let intersectionCallback: ((entries: IntersectionObserverEntry[]) => void) | null = null;

class MockIntersectionObserver {
  constructor(callback: IntersectionObserverCallback) {
    intersectionCallback = callback;
  }

  observe() {}
  unobserve() {}
  disconnect() {}
}

describe('LazyImage format fallbacks', () => {
  beforeEach(() => {
    vi.stubGlobal('IntersectionObserver', MockIntersectionObserver);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    intersectionCallback = null;
  });

  it('renders AVIF/WebP sources when the image is in view', () => {
    const { container } = render(
      <LazyImage
        src="/images/fallback.jpg"
        avifSrc="/images/fallback.avif"
        webpSrc="/images/fallback.webp"
        alt="Optimized asset"
      />
    );

    expect(container.querySelector('source[type="image/avif"]')).toBeNull();
    expect(container.querySelector('source[type="image/webp"]')).toBeNull();

    act(() => {
      intersectionCallback?.([{ isIntersecting: true } as IntersectionObserverEntry]);
    });

    const avifSource = container.querySelector('source[type="image/avif"]');
    const webpSource = container.querySelector('source[type="image/webp"]');

    expect(avifSource).toBeTruthy();
    expect(webpSource).toBeTruthy();
    expect(avifSource?.getAttribute('srcset')).toBe('/images/fallback.avif');
    expect(webpSource?.getAttribute('srcset')).toBe('/images/fallback.webp');
  });
});
