import { describe, it, expect } from 'vitest';
import { getVersionedApiUrl } from '../../utils/apiVersioning';

describe('api versioning support', () => {
  it('builds versioned api urls', () => {
    const url = getVersionedApiUrl('https://example.com/api', '/health', 'v2');
    expect(url).toBe('https://example.com/api/v2/health');
  });
});
