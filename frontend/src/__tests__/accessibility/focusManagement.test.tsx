import { describe, it, expect, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { getFocusableElements, trapFocus, focusFirst } from '../../utils/focusManagement';

describe('focus management utilities', () => {
  beforeEach(() => {
    document.body.innerHTML = '';
  });

  it('finds focusable elements', () => {
    render(
      <div>
        <button type="button">One</button>
        <button type="button" disabled>
          Two
        </button>
        <a href="/test">Link</a>
      </div>
    );

    const container = screen.getByText('One').closest('div') as HTMLElement;
    const focusables = getFocusableElements(container);
    expect(focusables).toHaveLength(2);
  });

  it('focuses the first focusable element', () => {
    render(
      <div>
        <button type="button">First</button>
        <button type="button">Second</button>
      </div>
    );

    const container = screen.getByText('First').closest('div') as HTMLElement;
    focusFirst(container);
    expect(document.activeElement).toBe(screen.getByText('First'));
  });

  it('traps focus within container', () => {
    render(
      <div>
        <button type="button">First</button>
        <button type="button">Last</button>
      </div>
    );

    const container = screen.getByText('First').closest('div') as HTMLElement;
    const first = screen.getByText('First');
    const last = screen.getByText('Last');

    (last as HTMLElement).focus();
    const event = new KeyboardEvent('keydown', { key: 'Tab' });
    trapFocus(container, event);
    expect(document.activeElement).toBe(first);
  });
});
