import { describe, it, expect, vi } from 'vitest';
import { logSecurityEvent } from '../../utils/securityAudit';
import { trackEvent } from '../../utils/analytics';

vi.mock('../../utils/analytics', () => ({
  trackEvent: vi.fn(),
}));

describe('security audit logging', () => {
  it('logs security events via analytics', () => {
    logSecurityEvent('csrf_token_missing', { route: '/api/test' });
    expect(trackEvent).toHaveBeenCalledWith('security:csrf_token_missing', {
      route: '/api/test',
    });
  });
});
