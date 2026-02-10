import { describe, it, expect } from 'vitest';
import {
  escapeHtml,
  sanitizeString,
  sanitizeObject,
  sanitizeUrl,
  sanitizeSearchQuery,
} from '../../utils/sanitize';

describe('XSS prevention utilities', () => {
  it('escapes HTML special characters', () => {
    const input = `<script>alert("xss")</script>`;
    expect(escapeHtml(input)).toBe(
      '&lt;script&gt;alert(&quot;xss&quot;)&lt;&#x2F;script&gt;'
    );
  });

  it('removes script tags and event handlers', () => {
    const input = `<img src="x" onerror="alert(1)" /><script>evil()</script>`;
    const sanitized = sanitizeString(input);
    expect(sanitized).toContain('<img src="x" />');
    expect(sanitized).not.toContain('onerror');
    expect(sanitized).not.toContain('script');
  });

  it('sanitizes nested object strings', () => {
    const input = {
      title: '<b>Safe</b>',
      nested: {
        value: '<script>bad()</script>',
      },
    };

    const sanitized = sanitizeObject(input);
    expect(sanitized.title).toBe('<b>Safe</b>');
    expect(sanitized.nested.value).toBe('');
  });

  it('rejects dangerous URLs and allows safe ones', () => {
    expect(sanitizeUrl('javascript:alert(1)')).toBeNull();
    expect(sanitizeUrl('data:text/html;base64,evil')).toBeNull();
    expect(sanitizeUrl('https://example.com')).toBe('https://example.com');
    expect(sanitizeUrl('/relative/path')).toBe('/relative/path');
  });

  it('sanitizes search queries', () => {
    const input = '  hello <script>bad</script> world  ';
    expect(sanitizeSearchQuery(input)).toBe('hello scriptbad/script world');
  });
});
