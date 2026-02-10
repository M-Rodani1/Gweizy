import { describe, it, expect } from 'vitest';
import { render } from '@testing-library/react';
import { validateHeadingHierarchy } from '../../utils/headingHierarchy';

describe('heading hierarchy validation', () => {
  it('accepts sequential heading levels', () => {
    const { container } = render(
      <main>
        <h1>Dashboard</h1>
        <section>
          <h2>Summary</h2>
          <h3>Details</h3>
        </section>
        <section>
          <h2>Insights</h2>
          <h3>Trends</h3>
        </section>
      </main>
    );

    const result = validateHeadingHierarchy(container);
    expect(result.isValid).toBe(true);
    expect(result.issues).toHaveLength(0);
  });

  it('flags skipped heading levels', () => {
    const { container } = render(
      <main>
        <h1>Dashboard</h1>
        <section>
          <h3>Skipped</h3>
        </section>
      </main>
    );

    const result = validateHeadingHierarchy(container);
    expect(result.isValid).toBe(false);
    expect(result.issues[0]?.message).toContain('Heading level skipped');
  });
});
