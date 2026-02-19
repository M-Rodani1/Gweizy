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

describe('Cross-origin isolation headers', () => {
  it('sets COOP and COEP in HTTP headers only', () => {
    const headersPath = path.resolve(process.cwd(), 'public/_headers');
    const htmlPath = path.resolve(process.cwd(), 'index.html');

    const headersContent = fs.readFileSync(headersPath, 'utf8');
    const htmlContent = fs.readFileSync(htmlPath, 'utf8');

    const headers = extractHeaders(headersContent);

    expect(headers['Cross-Origin-Opener-Policy']).toBe('same-origin');
    expect(headers['Cross-Origin-Embedder-Policy']).toBe('require-corp');
    expect(htmlContent).not.toMatch(/http-equiv="Cross-Origin-Opener-Policy"/i);
    expect(htmlContent).not.toMatch(/http-equiv="Cross-Origin-Embedder-Policy"/i);
  });
});
