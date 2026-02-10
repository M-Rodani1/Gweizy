import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import LazyImage from '../../components/ui/LazyImage';

describe('alt text validation', () => {
  it('requires alt text on LazyImage', () => {
    const intersectionObserver = class {
      observe() {}
      disconnect() {}
    };
    (globalThis as typeof globalThis & { IntersectionObserver?: unknown }).IntersectionObserver =
      intersectionObserver;

    render(<LazyImage src="/img.png" alt="Decorative image" />);
    const img = screen.getByRole('img', { name: 'Decorative image' });
    expect(img).toHaveAttribute('alt', 'Decorative image');
  });
});
