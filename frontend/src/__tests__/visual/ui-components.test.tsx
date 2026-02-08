/**
 * Visual regression tests for UI components using DOM snapshots
 * These tests verify the rendered structure remains consistent
 */

import React from 'react';
import { describe, it, expect } from 'vitest';
import { render } from '@testing-library/react';
import { Button } from '../../components/ui/Button';
import { Card } from '../../components/ui/Card';
import { Badge } from '../../components/ui/Badge';
import { Stat } from '../../components/ui/Stat';
import { Chip } from '../../components/ui/Chip';
import { SectionHeader } from '../../components/ui/SectionHeader';

describe('UI Components Visual Regression', () => {
  describe('Button', () => {
    it('renders primary variant correctly', () => {
      const { container } = render(<Button variant="primary">Primary</Button>);
      expect(container.innerHTML).toMatchSnapshot();
    });

    it('renders secondary variant correctly', () => {
      const { container } = render(<Button variant="secondary">Secondary</Button>);
      expect(container.innerHTML).toMatchSnapshot();
    });

    it('renders outline variant correctly', () => {
      const { container } = render(<Button variant="outline">Outline</Button>);
      expect(container.innerHTML).toMatchSnapshot();
    });

    it('renders all sizes correctly', () => {
      const { container } = render(
        <div>
          <Button size="sm">Small</Button>
          <Button size="md">Medium</Button>
          <Button size="lg">Large</Button>
        </div>
      );
      expect(container.innerHTML).toMatchSnapshot();
    });

    it('renders disabled state correctly', () => {
      const { container } = render(<Button disabled>Disabled</Button>);
      expect(container.innerHTML).toMatchSnapshot();
    });

    it('renders with icon correctly', () => {
      const { container } = render(
        <Button icon={<span data-testid="icon">★</span>}>With Icon</Button>
      );
      expect(container.innerHTML).toMatchSnapshot();
    });
  });

  describe('Card', () => {
    it('renders basic card correctly', () => {
      const { container } = render(
        <Card>
          <p>Card content</p>
        </Card>
      );
      expect(container.innerHTML).toMatchSnapshot();
    });

    it('renders card with title correctly', () => {
      const { container } = render(
        <Card title="Card Title">
          <p>Card content</p>
        </Card>
      );
      expect(container.innerHTML).toMatchSnapshot();
    });

    it('renders card with title and subtitle correctly', () => {
      const { container } = render(
        <Card title="Card Title" subtitle="Card subtitle">
          <p>Card content</p>
        </Card>
      );
      expect(container.innerHTML).toMatchSnapshot();
    });

    it('renders card with action correctly', () => {
      const { container } = render(
        <Card title="Card Title" action={<Button size="sm">Action</Button>}>
          <p>Card content</p>
        </Card>
      );
      expect(container.innerHTML).toMatchSnapshot();
    });

    it('renders all padding sizes correctly', () => {
      const { container } = render(
        <div>
          <Card padding="sm"><p>Small padding</p></Card>
          <Card padding="md"><p>Medium padding</p></Card>
          <Card padding="lg"><p>Large padding</p></Card>
        </div>
      );
      expect(container.innerHTML).toMatchSnapshot();
    });
  });

  describe('Badge', () => {
    it('renders all variants correctly', () => {
      const { container } = render(
        <div>
          <Badge variant="accent">Accent</Badge>
          <Badge variant="success">Success</Badge>
          <Badge variant="warning">Warning</Badge>
          <Badge variant="danger">Danger</Badge>
          <Badge variant="neutral">Neutral</Badge>
        </div>
      );
      expect(container.innerHTML).toMatchSnapshot();
    });
  });

  describe('Stat', () => {
    it('renders basic stat correctly', () => {
      const { container } = render(
        <Stat label="Gas Saved" value="$52K+" />
      );
      expect(container.innerHTML).toMatchSnapshot();
    });

    it('renders stat with helper text correctly', () => {
      const { container } = render(
        <Stat label="Accuracy" value="82%" helper="rolling 30d" />
      );
      expect(container.innerHTML).toMatchSnapshot();
    });

    it('renders stat with trends correctly', () => {
      const { container } = render(
        <div>
          <Stat label="Up Trend" value="100" trend="up" />
          <Stat label="Down Trend" value="100" trend="down" />
          <Stat label="Neutral Trend" value="100" trend="neutral" />
        </div>
      );
      expect(container.innerHTML).toMatchSnapshot();
    });
  });

  describe('Chip', () => {
    it('renders basic chip correctly', () => {
      const { container } = render(<Chip label="Chip Label" />);
      expect(container.innerHTML).toMatchSnapshot();
    });
  });

  describe('SectionHeader', () => {
    it('renders basic header correctly', () => {
      const { container } = render(
        <SectionHeader title="Section Title" />
      );
      expect(container.innerHTML).toMatchSnapshot();
    });

    it('renders header with all props correctly', () => {
      const { container } = render(
        <SectionHeader
          eyebrow="Overview"
          title="AI Transaction Pilot"
          description="Live optimisation powered by DQN models."
          action={<Button size="sm">View</Button>}
        />
      );
      expect(container.innerHTML).toMatchSnapshot();
    });

    it('renders centered header correctly', () => {
      const { container } = render(
        <SectionHeader
          align="center"
          title="Centered Title"
          description="Centered description text"
        />
      );
      expect(container.innerHTML).toMatchSnapshot();
    });
  });
});
