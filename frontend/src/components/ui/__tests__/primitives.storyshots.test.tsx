import { describe, it, expect } from 'vitest';
import { render } from '@testing-library/react';
import { composeStories } from '@storybook/react';
import * as stories from '../Primitives.stories';

const composed = composeStories(stories);

describe('Primitives stories', () => {
  for (const storyName of Object.keys(composed)) {
    const Story = (composed as any)[storyName];
    it(`${storyName} matches story snapshot`, () => {
      const { container } = render(<Story />);
      expect(container).toMatchSnapshot();
    });
  }
});
