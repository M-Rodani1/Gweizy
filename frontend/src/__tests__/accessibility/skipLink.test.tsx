import { describe, it, expect } from 'vitest';
import { render, fireEvent, screen } from '@testing-library/react';
import { SkipLink } from '../../components/ui/SkipLink';

describe('SkipLink', () => {
  it('focuses the target element on click', () => {
    render(
      <div>
        <SkipLink targetId="main-content">Skip</SkipLink>
        <main id="main-content">Content</main>
      </div>
    );

    const link = screen.getByRole('link', { name: 'Skip' });
    const target = screen.getByText('Content');
    (target as HTMLElement).scrollIntoView = () => {};
    fireEvent.click(link);

    expect(target).toHaveAttribute('tabindex', '-1');
    expect(document.activeElement).toBe(target);
  });
});
