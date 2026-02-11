import { sanitizeString } from './sanitize';

type DomPurifyConfig = Record<string, unknown>;

type DomPurifyLike = {
  sanitize: (html: string, config?: DomPurifyConfig) => string;
};

function getDomPurify(): DomPurifyLike | null {
  if (typeof globalThis !== 'undefined') {
    const candidate = (globalThis as { DOMPurify?: DomPurifyLike }).DOMPurify;
    if (candidate && typeof candidate.sanitize === 'function') {
      return candidate;
    }
  }

  return null;
}

export function sanitizeHtml(html: string, config?: DomPurifyConfig): string {
  if (typeof html !== 'string') {
    return '';
  }

  const domPurify = getDomPurify();
  if (domPurify) {
    return domPurify.sanitize(html, config);
  }

  return sanitizeString(html);
}

export function createSafeMarkup(
  html: string,
  config?: DomPurifyConfig
): { __html: string } {
  return { __html: sanitizeHtml(html, config) };
}
