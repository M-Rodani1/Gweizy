import { sanitizeRedirectUrl } from '../../utils/redirectValidation';

describe('sanitizeRedirectUrl', () => {
  it('allows safe relative paths', () => {
    expect(sanitizeRedirectUrl('/dashboard')).toBe('/dashboard');
    expect(sanitizeRedirectUrl('/dashboard?tab=gas')).toBe('/dashboard?tab=gas');
  });

  it('allows hash and query-only redirects', () => {
    expect(sanitizeRedirectUrl('#section')).toBe('#section');
    expect(sanitizeRedirectUrl('?tab=gas')).toBe('?tab=gas');
  });

  it('blocks protocol-relative and backslash redirects', () => {
    expect(sanitizeRedirectUrl('//evil.com')).toBe('/');
    expect(sanitizeRedirectUrl('/\\evil.com')).toBe('/');
  });

  it('blocks disallowed origins', () => {
    expect(
      sanitizeRedirectUrl('https://evil.com/phish', ['https://gweizy.app'])
    ).toBe('/');
  });

  it('allows explicitly allowed origins', () => {
    const allowed = 'https://gweizy.app';
    expect(
      sanitizeRedirectUrl('https://gweizy.app/wallet', [allowed])
    ).toBe('https://gweizy.app/wallet');
  });

  it('blocks non-http(s) protocols', () => {
    expect(
      sanitizeRedirectUrl('javascript:alert(1)', ['https://gweizy.app'])
    ).toBe('/');
  });
});
