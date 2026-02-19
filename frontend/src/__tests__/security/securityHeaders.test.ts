import { describe, it, expect } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';

function extractHeaders(headersContent: string): Record<string, string> {
  const lines = headersContent.split('\n');
  const headers: Record<string, string> = {};
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed || trimmed.startsWith('/*')) continue;
    const [name, ...valueParts] = trimmed.split(':');
    if (!name || valueParts.length === 0) continue;
    headers[name.trim()] = valueParts.join(':').trim();
  }
  return headers;
}

function extractMetaHeaders(htmlContent: string): Record<string, string> {
  const metaRegex = /<meta http-equiv="([^"]+)" content="([^"]+)"[^>]*>/gi;
  const headers: Record<string, string> = {};
  let match: RegExpExecArray | null;
  while ((match = metaRegex.exec(htmlContent)) !== null) {
    headers[match[1]] = match[2];
  }
  return headers;
}

describe('security headers', () => {
  it('defines core security headers in _headers', () => {
    const headersPath = path.resolve(process.cwd(), 'public/_headers');
    const headersContent = fs.readFileSync(headersPath, 'utf8');
    const headers = extractHeaders(headersContent);

    expect(headers['Content-Security-Policy']).toBeTruthy();
    expect(headers['Referrer-Policy']).toBe('strict-origin-when-cross-origin');
    expect(headers['X-Content-Type-Options']).toBe('nosniff');
    expect(headers['X-Frame-Options']).toBe('DENY');
    expect(headers['Permissions-Policy']).toBe('geolocation=(), microphone=(), camera=()');
    expect(headers['Strict-Transport-Security']).toBe('max-age=31536000; includeSubDomains; preload');
  });

  it('does not set security-only response headers via meta tags', () => {
    const htmlPath = path.resolve(process.cwd(), 'index.html');

    const htmlContent = fs.readFileSync(htmlPath, 'utf8');
    const metaHeaders = extractMetaHeaders(htmlContent);
    expect(metaHeaders['Content-Security-Policy']).toBeUndefined();
    expect(metaHeaders['Cross-Origin-Opener-Policy']).toBeUndefined();
    expect(metaHeaders['Cross-Origin-Embedder-Policy']).toBeUndefined();
    expect(metaHeaders['X-Frame-Options']).toBeUndefined();
    expect(metaHeaders['Strict-Transport-Security']).toBeUndefined();
  });

  it('enables subresource integrity in Vite config', () => {
    const configPath = path.resolve(process.cwd(), 'vite.config.ts');
    const configContent = fs.readFileSync(configPath, 'utf8');
    expect(configContent).toContain("sri(");
  });
});
