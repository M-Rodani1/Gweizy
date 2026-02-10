import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { LiveRegion, LiveRegionProvider, useLiveRegion } from '../../components/ui/LiveRegion';

function Announcer() {
  const { announcePolite } = useLiveRegion();
  return (
    <button type="button" onClick={() => announcePolite('Updated')}>
      Announce
    </button>
  );
}

describe('aria live regions', () => {
  it('renders polite and assertive live regions in the provider', () => {
    render(
      <LiveRegionProvider>
        <div>App</div>
      </LiveRegionProvider>
    );

    const polite = screen.getByRole('status');
    const assertive = screen.getByRole('alert');

    expect(polite).toHaveAttribute('aria-live', 'polite');
    expect(assertive).toHaveAttribute('aria-live', 'assertive');
  });

  it('announces messages through the provider', () => {
    render(
      <LiveRegionProvider>
        <Announcer />
      </LiveRegionProvider>
    );

    fireEvent.click(screen.getByRole('button', { name: 'Announce' }));

    const polite = screen.getByRole('status');
    expect(polite).toHaveTextContent('Updated');
  });

  it('configures standalone LiveRegion with assertive politeness', () => {
    render(<LiveRegion politeness="assertive">Warning</LiveRegion>);

    const region = screen.getByRole('alert');
    expect(region).toHaveAttribute('aria-live', 'assertive');
  });
});
