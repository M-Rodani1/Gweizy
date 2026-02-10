import { describe, it, expect, beforeEach } from 'vitest';
import { enforceHttps } from '../../utils/performanceOptimizations';

describe('HTTPS enforcement', () => {
  const originalLocation = window.location;

  beforeEach(() => {
    Object.defineProperty(window, 'location', {
      value: {
        protocol: 'http:',
        hostname: 'example.com',
        href: 'http://example.com/page',
      },
      writable: true,
    });
  });

  it('redirects to https for non-localhost http', () => {
    const redirected = enforceHttps();
    expect(redirected).toBe(true);
    expect(window.location.href).toBe('https://example.com/page');
  });

  it('does not redirect on localhost', () => {
    window.location.hostname = 'localhost';
    window.location.href = 'http://localhost:3000';
    const redirected = enforceHttps();
    expect(redirected).toBe(false);
    expect(window.location.href).toBe('http://localhost:3000');
  });

  it('does not redirect when already https', () => {
    window.location.protocol = 'https:';
    window.location.href = 'https://example.com/page';
    const redirected = enforceHttps();
    expect(redirected).toBe(false);
    expect(window.location.href).toBe('https://example.com/page');
  });

  afterEach(() => {
    Object.defineProperty(window, 'location', {
      value: originalLocation,
      writable: true,
    });
  });
});
