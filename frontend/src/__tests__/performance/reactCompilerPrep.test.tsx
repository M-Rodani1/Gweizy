import { describe, it, expect } from 'vitest';
import { compilerMemo } from '../../utils/reactCompiler';

describe('React Compiler preparation', () => {
  it('wraps components with React.memo and preserves displayName', () => {
    const Sample = () => null;
    const Memoized = compilerMemo(Sample, 'Sample');

    expect((Memoized as unknown as { $$typeof?: symbol }).$$typeof).toBe(Symbol.for('react.memo'));
    expect(Memoized.displayName).toBe('Sample');
  });
});
