import { describe, it, expect } from 'vitest';
import { handlers } from '../../mocks/handlers';

describe('mock server improvements', () => {
  it('exports default msw handlers', () => {
    expect(handlers.length).toBeGreaterThan(0);
  });
});
