import type { Meta, StoryObj } from '@storybook/react';
import { Button } from './Button';
import { Card } from './Card';
import { Badge } from './Badge';
import { Stat } from './Stat';
import { Pill } from './Pill';
import { Chip } from './Chip';
import { SectionHeader } from './SectionHeader';

const meta: Meta = {
  title: 'Primitives',
  parameters: {
    layout: 'padded'
  }
};

export default meta;

export const Buttons: StoryObj = {
  render: () => (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-3">
        <Button variant="primary">Primary</Button>
        <Button variant="secondary">Secondary</Button>
        <Button variant="outline">Outline</Button>
        <Button variant="ghost">Ghost</Button>
        <Button variant="danger">Danger</Button>
        <Button variant="success">Success</Button>
        <Button variant="link">Link Button</Button>
      </div>
      <div className="flex flex-wrap gap-3">
        <Button size="sm">Small</Button>
        <Button size="md">Medium</Button>
        <Button size="lg">Large</Button>
      </div>
    </div>
  )
};

export const BadgesAndPills: StoryObj = {
  render: () => (
    <div className="flex flex-wrap gap-3 items-center">
      <Badge variant="accent">Accent</Badge>
      <Badge variant="success">Success</Badge>
      <Badge variant="warning">Warning</Badge>
      <Badge variant="danger">Danger</Badge>
      <Badge variant="neutral">Neutral</Badge>
      <Pill color="cyan">Pill</Pill>
      <Chip label="Chip" />
    </div>
  )
};

export const Stats: StoryObj = {
  render: () => (
    <div className="grid grid-cols-3 gap-4">
      <Stat label="Gas Saved" value="$52K+" helper="+3.2% today" trend="up" />
      <Stat label="Accuracy" value="82%" helper="rolling 30d" />
      <Stat label="Predictions" value="15K+" trend="neutral" />
    </div>
  )
};

export const Cards: StoryObj = {
  render: () => (
    <div className="grid grid-cols-2 gap-4">
      <Card title="Simple Card" subtitle="Use for content blocks">
        <p className="text-sm text-[var(--text-secondary)]">Card body content goes here.</p>
      </Card>
      <Card
        title="With Action"
        action={<Button size="sm" variant="outline">Action</Button>}
      >
        <p className="text-sm text-[var(--text-secondary)]">A card with a right-aligned action button.</p>
      </Card>
    </div>
  )
};

export const SectionHeaders: StoryObj = {
  render: () => (
    <div className="space-y-6">
      <SectionHeader
        eyebrow="Overview"
        title="AI Transaction Pilot"
        description="Live optimisation powered by DQN models."
        action={<Button size="sm" variant="outline">View</Button>}
      />
      <SectionHeader
        align="center"
        title="Analytics"
        description="Charts, patterns, and model insights."
      />
    </div>
  )
};
