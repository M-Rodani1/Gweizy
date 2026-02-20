import { afterEach, describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import LazyImage from '../../components/ui/LazyImage';

class MockIntersectionObserver implements IntersectionObserver {
  readonly root: Element | Document | null = null;
  readonly rootMargin = '';
  readonly thresholds: ReadonlyArray<number> = [];

  disconnect(): void {}
  observe(): void {}
  takeRecords(): IntersectionObserverEntry[] {
    return [];
  }
  unobserve(): void {}
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('alt text validation', () => {
  it('requires alt text on LazyImage', () => {
    vi.stubGlobal(
      'IntersectionObserver',
      MockIntersectionObserver as unknown as typeof IntersectionObserver
    );

    render(<LazyImage src="/img.png" alt="Decorative image" />);
    const img = screen.getByRole('img', { name: 'Decorative image' });
    expect(img).toHaveAttribute('alt', 'Decorative image');
  });
});
