export const DEFAULT_API_VERSION = 'v1';

const normalizeBaseUrl = (baseUrl: string) => baseUrl.replace(/\/+$/, '');

export const getVersionedApiUrl = (
  baseUrl: string,
  endpoint: string,
  version: string = DEFAULT_API_VERSION
): string => {
  const normalized = normalizeBaseUrl(baseUrl);
  const path = endpoint.startsWith('/') ? endpoint : `/${endpoint}`;
  return `${normalized}/${version}${path}`;
};
