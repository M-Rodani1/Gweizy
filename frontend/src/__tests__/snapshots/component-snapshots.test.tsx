/**
 * Component Snapshot Tests
 *
 * Captures the rendered output of complex components to detect
 * unintended changes. Run `npm test -- -u` to update snapshots.
 */

import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render } from '@testing-library/react';
import React from 'react';

// Mock framer-motion to avoid animation issues in snapshots
vi.mock('framer-motion', () => ({
  motion: {
    div: ({ children, ...props }: any) => <div {...props}>{children}</div>,
    span: ({ children, ...props }: any) => <span {...props}>{children}</span>,
    button: ({ children, ...props }: any) => <button {...props}>{children}</button>,
    p: ({ children, ...props }: any) => <p {...props}>{children}</p>,
  },
  AnimatePresence: ({ children }: any) => <>{children}</>,
}));

// Mock lucide-react icons to prevent snapshot noise
vi.mock('lucide-react', () => ({
  TrendingUp: () => <span data-testid="icon-trending-up">↑</span>,
  TrendingDown: () => <span data-testid="icon-trending-down">↓</span>,
  Clock: () => <span data-testid="icon-clock">⏰</span>,
  RefreshCw: () => <span data-testid="icon-refresh">↻</span>,
  AlertCircle: () => <span data-testid="icon-alert">⚠</span>,
  Check: () => <span data-testid="icon-check">✓</span>,
  X: () => <span data-testid="icon-x">✕</span>,
  ChevronDown: () => <span data-testid="icon-chevron-down">▼</span>,
  ChevronUp: () => <span data-testid="icon-chevron-up">▲</span>,
  Info: () => <span data-testid="icon-info">ℹ</span>,
  Zap: () => <span data-testid="icon-zap">⚡</span>,
  Fuel: () => <span data-testid="icon-fuel">⛽</span>,
  Wallet: () => <span data-testid="icon-wallet">💰</span>,
  Settings: () => <span data-testid="icon-settings">⚙</span>,
  Menu: () => <span data-testid="icon-menu">☰</span>,
  Share2: () => <span data-testid="icon-share">⤴</span>,
  Download: () => <span data-testid="icon-download">↓</span>,
  ExternalLink: () => <span data-testid="icon-external">↗</span>,
  Copy: () => <span data-testid="icon-copy">📋</span>,
  Calendar: () => <span data-testid="icon-calendar">📅</span>,
  Layers: () => <span data-testid="icon-layers">◫</span>,
  Activity: () => <span data-testid="icon-activity">📈</span>,
  Bell: () => <span data-testid="icon-bell">🔔</span>,
  Inbox: () => <span data-testid="icon-inbox">📥</span>,
  Sun: () => <span data-testid="icon-sun">☀</span>,
  Moon: () => <span data-testid="icon-moon">🌙</span>,
}));

// Import components after mocks
import { Button } from '../../components/ui/Button';
import { Badge } from '../../components/ui/Badge';
import { Card } from '../../components/ui/Card';
import { Stat } from '../../components/ui/Stat';
import { Chip } from '../../components/ui/Chip';
import { SectionHeader } from '../../components/ui/SectionHeader';
import Skeleton, {
  SkeletonCard,
  SkeletonList,
  SkeletonMetrics,
  SkeletonChart,
} from '../../components/ui/Skeleton';
import EmptyState from '../../components/EmptyState';
import LoadingSpinner from '../../components/LoadingSpinner';

describe('UI Component Snapshots', () => {
  describe('Button', () => {
    it('should match snapshot for primary variant', () => {
      const { container } = render(<Button variant="primary">Primary Button</Button>);
      expect(container).toMatchSnapshot();
    });

    it('should match snapshot for secondary variant', () => {
      const { container } = render(<Button variant="secondary">Secondary Button</Button>);
      expect(container).toMatchSnapshot();
    });

    it('should match snapshot for outline variant', () => {
      const { container } = render(<Button variant="outline">Outline Button</Button>);
      expect(container).toMatchSnapshot();
    });

    it('should match snapshot for all sizes', () => {
      const { container } = render(
        <div>
          <Button size="sm">Small</Button>
          <Button size="md">Medium</Button>
          <Button size="lg">Large</Button>
        </div>
      );
      expect(container).toMatchSnapshot();
    });

    it('should match snapshot when disabled', () => {
      const { container } = render(<Button disabled>Disabled</Button>);
      expect(container).toMatchSnapshot();
    });

    it('should match snapshot when loading', () => {
      const { container } = render(<Button loading>Loading</Button>);
      expect(container).toMatchSnapshot();
    });
  });

  describe('Badge', () => {
    it('should match snapshot for all variants', () => {
      const { container } = render(
        <div>
          <Badge variant="accent">Accent</Badge>
          <Badge variant="success">Success</Badge>
          <Badge variant="warning">Warning</Badge>
          <Badge variant="danger">Danger</Badge>
          <Badge variant="neutral">Neutral</Badge>
        </div>
      );
      expect(container).toMatchSnapshot();
    });
  });

  describe('Card', () => {
    it('should match snapshot with title and content', () => {
      const { container } = render(
        <Card title="Card Title" subtitle="Card subtitle">
          <p>Card content goes here</p>
        </Card>
      );
      expect(container).toMatchSnapshot();
    });

    it('should match snapshot with action button', () => {
      const { container } = render(
        <Card
          title="Card with Action"
          action={<Button size="sm">Action</Button>}
        >
          <p>Content</p>
        </Card>
      );
      expect(container).toMatchSnapshot();
    });
  });

  describe('Stat', () => {
    it('should match snapshot with all props', () => {
      const { container } = render(
        <div>
          <Stat label="Gas Saved" value="$52K+" helper="+3.2% today" trend="up" />
          <Stat label="Accuracy" value="82%" helper="rolling 30d" trend="neutral" />
          <Stat label="Predictions" value="15K+" trend="down" />
        </div>
      );
      expect(container).toMatchSnapshot();
    });
  });

  describe('Chip', () => {
    it('should match snapshot', () => {
      const { container } = render(
        <div>
          <Chip label="Default" />
          <Chip label="Active" active />
          <Chip label="With Count" count={5} />
        </div>
      );
      expect(container).toMatchSnapshot();
    });
  });

  describe('SectionHeader', () => {
    it('should match snapshot with all props', () => {
      const { container } = render(
        <SectionHeader
          eyebrow="Overview"
          title="Dashboard Title"
          description="This is a description of the section"
          action={<Button size="sm">View All</Button>}
        />
      );
      expect(container).toMatchSnapshot();
    });

    it('should match snapshot with center alignment', () => {
      const { container } = render(
        <SectionHeader
          align="center"
          title="Centered Title"
          description="Centered description"
        />
      );
      expect(container).toMatchSnapshot();
    });
  });
});

describe('Skeleton Component Snapshots', () => {
  it('should match snapshot for basic skeleton', () => {
    const { container } = render(
      <div>
        <Skeleton variant="text" width="100%" />
        <Skeleton variant="rect" width={200} height={100} />
        <Skeleton variant="circle" width={40} height={40} />
      </div>
    );
    expect(container).toMatchSnapshot();
  });

  it('should match snapshot for SkeletonCard', () => {
    const { container } = render(<SkeletonCard />);
    expect(container).toMatchSnapshot();
  });

  it('should match snapshot for SkeletonList', () => {
    const { container } = render(<SkeletonList count={3} />);
    expect(container).toMatchSnapshot();
  });

  it('should match snapshot for SkeletonMetrics', () => {
    const { container } = render(<SkeletonMetrics />);
    expect(container).toMatchSnapshot();
  });

  it('should match snapshot for SkeletonChart', () => {
    const { container } = render(<SkeletonChart height={200} />);
    expect(container).toMatchSnapshot();
  });
});

describe('Feedback Component Snapshots', () => {
  describe('EmptyState', () => {
    it('should match snapshot for no-data type', () => {
      const { container } = render(<EmptyState type="no-data" />);
      expect(container).toMatchSnapshot();
    });

    it('should match snapshot for no-wallet type', () => {
      const { container } = render(<EmptyState type="no-wallet" />);
      expect(container).toMatchSnapshot();
    });

    it('should match snapshot for error type', () => {
      const { container } = render(<EmptyState type="error" />);
      expect(container).toMatchSnapshot();
    });

    it('should match snapshot with custom content', () => {
      const { container } = render(
        <EmptyState
          title="Custom Title"
          description="Custom description text"
          action={<Button>Custom Action</Button>}
        />
      );
      expect(container).toMatchSnapshot();
    });
  });

  describe('LoadingSpinner', () => {
    it('should match snapshot with default message', () => {
      const { container } = render(<LoadingSpinner />);
      expect(container).toMatchSnapshot();
    });

    it('should match snapshot with custom message', () => {
      const { container } = render(<LoadingSpinner message="Fetching data..." />);
      expect(container).toMatchSnapshot();
    });
  });
});

describe('Complex Layout Snapshots', () => {
  it('should match snapshot for stats grid', () => {
    const { container } = render(
      <div className="grid grid-cols-3 gap-4">
        <Stat label="Gas Saved" value="$52K+" trend="up" />
        <Stat label="Accuracy" value="82%" trend="neutral" />
        <Stat label="Predictions" value="15K+" trend="down" />
      </div>
    );
    expect(container).toMatchSnapshot();
  });

  it('should match snapshot for card grid', () => {
    const { container } = render(
      <div className="grid grid-cols-2 gap-4">
        <Card title="Card 1">Content 1</Card>
        <Card title="Card 2">Content 2</Card>
        <Card title="Card 3">Content 3</Card>
        <Card title="Card 4">Content 4</Card>
      </div>
    );
    expect(container).toMatchSnapshot();
  });

  it('should match snapshot for badge collection', () => {
    const { container } = render(
      <div className="flex gap-2 flex-wrap">
        <Badge variant="success">Live</Badge>
        <Badge variant="warning">Pending</Badge>
        <Badge variant="danger">Error</Badge>
        <Chip label="Ethereum" />
        <Chip label="Base" active />
        <Chip label="Polygon" />
      </div>
    );
    expect(container).toMatchSnapshot();
  });
});

describe('Snapshot Stability', () => {
  it('should produce consistent snapshots across renders', () => {
    const Component = () => (
      <Card title="Consistent">
        <Stat label="Value" value="100" />
        <Badge variant="success">Active</Badge>
      </Card>
    );

    const { container: container1 } = render(<Component />);
    const { container: container2 } = render(<Component />);

    expect(container1.innerHTML).toBe(container2.innerHTML);
  });
});
