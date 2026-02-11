import { describe, it, expect } from 'vitest';
import ShareResults from '../../components/ShareResults';

describe('React.memo audit', () => {
  it('avoids re-rendering memoized components with identical props', () => {
    const memoSymbol = Symbol.for('react.memo');
    const shareResultsMemo = (ShareResults as unknown as { $$typeof?: symbol }).$$typeof;

    expect(shareResultsMemo).toBe(memoSymbol);
  });
});
