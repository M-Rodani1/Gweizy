import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { render, act } from '@testing-library/react';
import ScrollReveal from '../../components/ui/ScrollReveal';

let intersectionCallback: ((entries: IntersectionObserverEntry[]) => void) | null = null;

class MockIntersectionObserver {
  constructor(callback: IntersectionObserverCallback) {
    intersectionCallback = callback;
  }

  observe() {}
  unobserve() {}
  disconnect() {}
}

describe('ScrollReveal', () => {
  beforeEach(() => {
    vi.stubGlobal('IntersectionObserver', MockIntersectionObserver);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    intersectionCallback = null;
  });

  it('adds the visible class when intersecting', () => {
    const { container } = render(
      <ScrollReveal>
        <div>Content</div>
      </ScrollReveal>
    );

    const wrapper = container.firstElementChild as HTMLElement;
    expect(wrapper.className).toContain('scroll-reveal');
    expect(wrapper.className).not.toContain('scroll-reveal--visible');

    act(() => {
      intersectionCallback?.([{ isIntersecting: true } as IntersectionObserverEntry]);
    });

    expect(wrapper.className).toContain('scroll-reveal--visible');
  });

  it('reveals immediately when IntersectionObserver is unavailable', () => {
    vi.unstubAllGlobals();
    vi.stubGlobal('IntersectionObserver', undefined as unknown as typeof IntersectionObserver);

    const { container } = render(
      <ScrollReveal>
        <div>Content</div>
      </ScrollReveal>
    );

    const wrapper = container.firstElementChild as HTMLElement;
    expect(wrapper.className).toContain('scroll-reveal--visible');
  });
});
