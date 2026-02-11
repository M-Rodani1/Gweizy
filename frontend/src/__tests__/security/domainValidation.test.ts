import { describe, it, expect } from 'vitest';
import { getDefaultTrustedDomains, isTrustedDomain } from '../../utils/domainValidation';

describe('Anti-phishing domain validation', () => {
  it('accepts trusted base domains and subdomains', () => {
    expect(isTrustedDomain('https://basegasoptimizer.com')).toBe(true);
    expect(isTrustedDomain('https://app.basegasoptimizer.com')).toBe(true);
    expect(isTrustedDomain('https://basescan.org/tx/0x123')).toBe(true);
  });

  it('rejects lookalike domains', () => {
    expect(isTrustedDomain('https://basegasoptimizer.com.evil.com')).toBe(false);
    expect(isTrustedDomain('https://basegasoptimizer.co')).toBe(false);
  });

  it('exposes the default allowlist', () => {
    const domains = getDefaultTrustedDomains();
    expect(domains).toContain('basegasoptimizer.com');
  });
});
