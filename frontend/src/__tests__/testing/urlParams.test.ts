import { describe, it, expect } from 'vitest';
import { getApiUrl, API_CONFIG } from '../../config/api';

describe('URL parameter parsing and serialization', () => {
  it('serializes query params for API URLs', () => {
    const url = getApiUrl(API_CONFIG.ENDPOINTS.CURRENT, {
      chainId: 1,
      includeHistory: true,
      search: 'gas price',
    });

    expect(url).toContain(`${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.CURRENT}?`);

    const parsed = new URL(url);
    expect(parsed.searchParams.get('chainId')).toBe('1');
    expect(parsed.searchParams.get('includeHistory')).toBe('true');
    expect(parsed.searchParams.get('search')).toBe('gas price');
  });

  it('returns a base URL when no params are provided', () => {
    const url = getApiUrl(API_CONFIG.ENDPOINTS.HEALTH);
    expect(url).toBe(`${API_CONFIG.BASE_URL}${API_CONFIG.ENDPOINTS.HEALTH}`);
  });

  it('encodes special characters during serialization', () => {
    const url = getApiUrl(API_CONFIG.ENDPOINTS.STATS, {
      filter: 'type:fast&sort=desc',
    });

    expect(url).toContain('filter=type%3Afast%26sort%3Ddesc');

    const parsed = new URL(url);
    expect(parsed.searchParams.get('filter')).toBe('type:fast&sort=desc');
  });
});
