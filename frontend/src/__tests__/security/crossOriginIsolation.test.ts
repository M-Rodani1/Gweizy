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

describe('Cross-origin isolation headers', () => {
  it('sets COOP and COEP in headers and meta tags', () => {
    const headersPath = path.resolve(process.cwd(), 'public/_headers');
    const htmlPath = path.resolve(process.cwd(), 'index.html');

    const headersContent = fs.readFileSync(headersPath, 'utf8');
    const htmlContent = fs.readFileSync(htmlPath, 'utf8');

    const headers = extractHeaders(headersContent);
    const metaHeaders = extractMetaHeaders(htmlContent);

    expect(headers['Cross-Origin-Opener-Policy']).toBe('same-origin');
    expect(headers['Cross-Origin-Embedder-Policy']).toBe('require-corp');
    expect(metaHeaders['Cross-Origin-Opener-Policy']).toBe(headers['Cross-Origin-Opener-Policy']);
    expect(metaHeaders['Cross-Origin-Embedder-Policy']).toBe(headers['Cross-Origin-Embedder-Policy']);
  });
});
