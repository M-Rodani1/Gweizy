import { describe, it, expect } from 'vitest';
import { render } from '@testing-library/react';
import { Button } from '../Button';
import { Card } from '../Card';
import { Badge } from '../Badge';
import { Stat } from '../Stat';
import { Pill } from '../Pill';
import { Chip } from '../Chip';
import { SectionHeader } from '../SectionHeader';

describe('UI primitives snapshots', () => {
  it('Button variants match snapshot', () => {
    const { container } = render(
      <>
        <Button variant="primary">Primary</Button>
        <Button variant="secondary">Secondary</Button>
        <Button variant="outline">Outline</Button>
        <Button variant="ghost">Ghost</Button>
      </>
    );
    expect(container).toMatchSnapshot();
  });

  it('Card with header matches snapshot', () => {
    const { container } = render(
      <Card title="Card Title" subtitle="Subtitle" action={<Button size="sm">Action</Button>}>
        <p>Body</p>
      </Card>
    );
    expect(container).toMatchSnapshot();
  });

  it('Badges and pills match snapshot', () => {
    const { container } = render(
      <>
        <Badge variant="accent">Accent</Badge>
        <Badge variant="success">Success</Badge>
        <Pill color="cyan">Cyan</Pill>
        <Chip label="Chip" />
      </>
    );
    expect(container).toMatchSnapshot();
  });

  it('Stat renders with helper', () => {
    const { container } = render(<Stat label="Accuracy" value="82%" helper="Rolling 30d" trend="up" />);
    expect(container).toMatchSnapshot();
  });

  it('SectionHeader renders alignments', () => {
    const { container } = render(
      <>
        <SectionHeader eyebrow="Overview" title="AI Pilot" description="Details" action={<Button size="sm">View</Button>} />
        <SectionHeader align="center" title="Analytics" description="Charts" />
      </>
    );
    expect(container).toMatchSnapshot();
  });
});
