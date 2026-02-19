import { describe, it, expect } from 'vitest';
import fs from 'node:fs';
import path from 'node:path';

function extractCspFromHeaders(headersContent: string): string | null {
  const lines = headersContent.split('\n');
  const cspLine = lines.find((line) => line.trimStart().startsWith('Content-Security-Policy:'));
  if (!cspLine) return null;
  return cspLine.split('Content-Security-Policy:')[1]?.trim() ?? null;
}

describe('CSP headers configuration', () => {
  it('should define CSP in headers', () => {
    const headersPath = path.resolve(process.cwd(), 'public/_headers');

    const headersContent = fs.readFileSync(headersPath, 'utf8');
    const headerCsp = extractCspFromHeaders(headersContent);

    expect(headerCsp).toBeTruthy();
  });

  it('should not set CSP via meta tags', () => {
    const htmlPath = path.resolve(process.cwd(), 'index.html');
    const htmlContent = fs.readFileSync(htmlPath, 'utf8');
    expect(htmlContent).not.toMatch(/http-equiv="Content-Security-Policy"/i);
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
