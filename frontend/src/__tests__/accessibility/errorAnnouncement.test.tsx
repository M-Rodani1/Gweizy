import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { LiveRegionProvider, useAnnounceError } from '../../components/ui/LiveRegion';

function ErrorAnnouncer({ error }: { error?: string | null }) {
  useAnnounceError(error ?? null);
  return null;
}

describe('error announcement', () => {
  it('announces errors with assertive live region', () => {
    const { rerender } = render(
      <LiveRegionProvider>
        <ErrorAnnouncer error={null} />
      </LiveRegionProvider>
    );

    rerender(
      <LiveRegionProvider>
        <ErrorAnnouncer error="Invalid email" />
      </LiveRegionProvider>
    );

    const alert = screen.getByRole('alert');
    expect(alert).toHaveAttribute('aria-live', 'assertive');
    expect(alert).toHaveTextContent('Error: Invalid email');
  });
});
