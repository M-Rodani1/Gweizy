import { describe, it, expect } from 'vitest';
import { initializeApiKeys, getApiKeyHeader, rotateApiKey, resetApiKeys } from '../../utils/apiKeys';

describe('api key rotation', () => {
  it('returns headers for configured keys', () => {
    initializeApiKeys(['key-a', 'key-b']);
    const header = getApiKeyHeader();
    expect(header).toEqual({ 'X-API-Key': 'key-a' });
    resetApiKeys();
  });

  it('rotates to the next key', () => {
    initializeApiKeys(['key-a', 'key-b']);
    rotateApiKey();
    const header = getApiKeyHeader();
    expect(header).toEqual({ 'X-API-Key': 'key-b' });
    resetApiKeys();
  });
});
