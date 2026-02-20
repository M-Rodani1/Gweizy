import { logSensitiveAction } from '../../utils/securityAudit';
import { trackEvent } from '../../utils/analytics';
import { vi } from 'vitest';

vi.mock('../../utils/analytics', () => ({
  trackEvent: vi.fn(),
}));

describe('sensitive action audit logging', () => {
  it('logs sensitive actions with metadata', () => {
    logSensitiveAction('wallet_connect', {
      route: '/connect',
      detail: 'user_confirmed',
      resourceId: 'wallet-1',
    });

    expect(trackEvent).toHaveBeenCalledWith('audit:sensitive_action', {
      action: 'wallet_connect',
      route: '/connect',
      detail: 'user_confirmed',
      resourceId: 'wallet-1',
    });
  });
});
