import { describe, it, expect, vi, beforeEach } from 'vitest';
import { setSentryRef, trackEvent } from '../../utils/analytics';

describe('analytics abstraction layer', () => {
  beforeEach(() => {
    setSentryRef(null);
  });

  it('sends events through configured sentry ref', () => {
    const captureMessage = vi.fn();
    setSentryRef({ captureMessage });

    trackEvent('test_event', { value: 1 });

    expect(captureMessage).toHaveBeenCalledWith('test_event', expect.objectContaining({
      level: 'info'
    }));
  });

  it('falls back to console.debug when no provider exists', () => {
    const spy = vi.spyOn(console, 'debug').mockImplementation(() => {});

    trackEvent('fallback_event', { enabled: true });

    expect(spy).toHaveBeenCalledWith('[analytics]', expect.objectContaining({
      event: 'fallback_event'
    }));
  });
});
