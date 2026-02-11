import { describe, it, expect } from 'vitest';
import { render } from '@testing-library/react';
import SandboxedIframe from '../../components/ui/SandboxedIframe';

describe('Iframe sandboxing', () => {
  it('applies a restrictive sandbox by default', () => {
    const { container } = render(
      <SandboxedIframe title="Embed" src="https://example.com/embed" />
    );

    const iframe = container.querySelector('iframe');
    expect(iframe).toBeTruthy();
    expect(iframe?.getAttribute('sandbox')).toBe('allow-scripts allow-same-origin');
  });

  it('allows custom sandbox policies', () => {
    const { container } = render(
      <SandboxedIframe
        title="Embed"
        src="https://example.com/embed"
        sandbox="allow-scripts"
      />
    );

    const iframe = container.querySelector('iframe');
    expect(iframe?.getAttribute('sandbox')).toBe('allow-scripts');
  });
});
