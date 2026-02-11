const DEFAULT_ALLOWED_DOMAINS = [
  'basegasoptimizer.com',
  'basegasfeesml-production.up.railway.app',
  'basescan.org',
  'mainnet.base.org',
];

const isSubdomain = (host: string, domain: string): boolean => {
  return host === domain || host.endsWith(`.${domain}`);
};

export function isTrustedDomain(url: string, allowedDomains = DEFAULT_ALLOWED_DOMAINS): boolean {
  try {
    const parsed = new URL(url);
    const host = parsed.hostname.toLowerCase();

    return allowedDomains.some((domain) => isSubdomain(host, domain.toLowerCase()));
  } catch {
    return false;
  }
}

export function getDefaultTrustedDomains(): string[] {
  return [...DEFAULT_ALLOWED_DOMAINS];
}
