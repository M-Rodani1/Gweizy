type ResourceHintRel = 'preload' | 'prefetch' | 'preconnect' | 'dns-prefetch' | 'modulepreload';

export interface ResourceHint {
  rel: ResourceHintRel;
  href: string;
  as?: string;
  type?: string;
  crossOrigin?: '' | 'anonymous' | 'use-credentials';
  fetchPriority?: 'high' | 'low' | 'auto';
}

export function ensureResourceHint(hint: ResourceHint, doc: Document = document): HTMLLinkElement {
  const existing = doc.querySelector<HTMLLinkElement>(
    `link[rel="${hint.rel}"][href="${hint.href}"]`
  );
  if (existing) {
    return existing;
  }

  const link = doc.createElement('link');
  link.rel = hint.rel;
  link.href = hint.href;

  if (hint.as) {
    link.as = hint.as;
  }
  if (hint.type) {
    link.type = hint.type;
  }
  if (hint.crossOrigin !== undefined) {
    link.crossOrigin = hint.crossOrigin;
  }
  if (hint.fetchPriority) {
    link.setAttribute('fetchpriority', hint.fetchPriority);
  }

  doc.head.appendChild(link);
  return link;
}

export function applyResourceHints(hints: ResourceHint[], doc: Document = document): HTMLLinkElement[] {
  return hints.map((hint) => ensureResourceHint(hint, doc));
}

export function preconnect(href: string, crossOrigin: ResourceHint['crossOrigin'] = 'anonymous'): ResourceHint {
  return { rel: 'preconnect', href, crossOrigin };
}

export function dnsPrefetch(href: string): ResourceHint {
  return { rel: 'dns-prefetch', href };
}

export function preloadResource(
  href: string,
  options: Pick<ResourceHint, 'as' | 'type' | 'crossOrigin' | 'fetchPriority'> = {}
): ResourceHint {
  return { rel: 'preload', href, ...options };
}

export function prefetchResource(
  href: string,
  options: Pick<ResourceHint, 'as' | 'type' | 'crossOrigin' | 'fetchPriority'> = {}
): ResourceHint {
  return { rel: 'prefetch', href, ...options };
}

