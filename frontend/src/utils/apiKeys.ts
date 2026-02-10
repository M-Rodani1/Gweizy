/**
 * API key rotation helper for client-side requests.
 */

interface ApiKeyConfig {
  keys: string[];
  headerName?: string;
}

const DEFAULT_HEADER = 'X-API-Key';

let config: ApiKeyConfig = {
  keys: [],
  headerName: DEFAULT_HEADER,
};

let index = 0;

export function initializeApiKeys(keys: string[], headerName = DEFAULT_HEADER): void {
  config = { keys: [...keys], headerName };
  index = 0;
}

export function getApiKeyHeader(): Record<string, string> {
  if (!config.keys.length) {
    return {};
  }
  const key = config.keys[index % config.keys.length];
  return { [config.headerName ?? DEFAULT_HEADER]: key };
}

export function rotateApiKey(): void {
  if (!config.keys.length) return;
  index = (index + 1) % config.keys.length;
}

export function resetApiKeys(): void {
  config = { keys: [], headerName: DEFAULT_HEADER };
  index = 0;
}
