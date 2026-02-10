import { describe, it, expect } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';

function extractCspFromHeaders(headersContent: string): string | null {
  const lines = headersContent.split('\n');
  const cspLine = lines.find((line) => line.trimStart().startsWith('Content-Security-Policy:'));
  if (!cspLine) return null;
  return cspLine.split('Content-Security-Policy:')[1]?.trim() ?? null;
}

function extractCspFromHtml(htmlContent: string): string | null {
  const match = htmlContent.match(/http-equiv="Content-Security-Policy"\s+content="([^"]+)"/i);
  return match?.[1] ?? null;
}

describe('CSP headers configuration', () => {
  it('should define CSP in headers and index.html meta tag', () => {
    const headersPath = path.resolve(process.cwd(), 'public/_headers');
    const htmlPath = path.resolve(process.cwd(), 'index.html');

    const headersContent = fs.readFileSync(headersPath, 'utf8');
    const htmlContent = fs.readFileSync(htmlPath, 'utf8');

    const headerCsp = extractCspFromHeaders(headersContent);
    const metaCsp = extractCspFromHtml(htmlContent);

    expect(headerCsp).toBeTruthy();
    expect(metaCsp).toBeTruthy();
  });

  it('should keep CSP header and meta tag in sync', () => {
    const headersPath = path.resolve(process.cwd(), 'public/_headers');
    const htmlPath = path.resolve(process.cwd(), 'index.html');

    const headersContent = fs.readFileSync(headersPath, 'utf8');
    const htmlContent = fs.readFileSync(htmlPath, 'utf8');

    const headerCsp = extractCspFromHeaders(headersContent);
    const metaCsp = extractCspFromHtml(htmlContent);

    expect(headerCsp).toBe(metaCsp);
  });

  it('should include core directives', () => {
    const headersPath = path.resolve(process.cwd(), 'public/_headers');
    const headersContent = fs.readFileSync(headersPath, 'utf8');
    const headerCsp = extractCspFromHeaders(headersContent) ?? '';

    expect(headerCsp).toContain("default-src 'self'");
    expect(headerCsp).toContain("script-src 'self'");
    expect(headerCsp).toContain("style-src 'self'");
    expect(headerCsp).toContain("img-src 'self'");
    expect(headerCsp).toContain("connect-src 'self'");
    expect(headerCsp).toContain("frame-ancestors 'none'");
  });
});
