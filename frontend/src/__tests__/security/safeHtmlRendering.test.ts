import { createSafeMarkup, sanitizeHtml } from '../../utils/safeHtml';

describe('safe HTML rendering', () => {
  afterEach(() => {
    delete (globalThis as { DOMPurify?: unknown }).DOMPurify;
  });

  it('uses DOMPurify when available', () => {
    const sanitize = vi.fn((value: string) => `clean:${value}`);
    (globalThis as { DOMPurify?: { sanitize: typeof sanitize } }).DOMPurify = {
      sanitize,
    };

    const config = { ALLOWED_TAGS: ['strong'] };
    const result = sanitizeHtml('<strong>ok</strong>', config);

    expect(result).toBe('clean:<strong>ok</strong>');
    expect(sanitize).toHaveBeenCalledWith('<strong>ok</strong>', config);
  });

  it('falls back to sanitizeString when DOMPurify is unavailable', () => {
    const result = sanitizeHtml('<script>alert(1)</script> Safe');
    expect(result).toBe('Safe');
  });

  it('creates safe markup for dangerouslySetInnerHTML', () => {
    const markup = createSafeMarkup('<img onerror="evil()" src="x" />');
    expect(markup).toEqual({ __html: '<img src="x" />' });
  });
});
