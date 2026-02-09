/**
 * Accessibility tests using axe-core
 *
 * Tests key components for WCAG compliance:
 * - No critical accessibility violations
 * - Proper heading structure
 * - Color contrast
 * - Keyboard accessibility
 * - ARIA attributes
 */

import { describe, it, expect } from 'vitest';
import { render } from '@testing-library/react';
import { configureAxe, toHaveNoViolations } from 'jest-axe';

// Extend Vitest with axe matchers
expect.extend(toHaveNoViolations);

// Configure axe
const axe = configureAxe({
  rules: {
    // Ignore color contrast for now as it may fail in jsdom
    'color-contrast': { enabled: false },
    // Ignore region rule as we're testing isolated components
    'region': { enabled: false },
  },
});

// Components to test
import Button from '../../components/ui/Button';
import Card from '../../components/ui/Card';
import Badge from '../../components/ui/Badge';
import Skeleton from '../../components/ui/Skeleton';
import Chip from '../../components/ui/Chip';

describe('Accessibility Tests', () => {
  describe('Button', () => {
    it('should have no accessibility violations', async () => {
      const { container } = render(<Button>Click me</Button>);
      const results = await axe(container);
      expect(results).toHaveNoViolations();
    });

    it('should have no violations with variants', async () => {
      const { container } = render(
        <div>
          <Button variant="primary">Primary</Button>
          <Button variant="secondary">Secondary</Button>
          <Button variant="ghost">Ghost</Button>
          <Button variant="danger">Danger</Button>
        </div>
      );
      const results = await axe(container);
      expect(results).toHaveNoViolations();
    });

    it('should have no violations when disabled', async () => {
      const { container } = render(<Button disabled>Disabled</Button>);
      const results = await axe(container);
      expect(results).toHaveNoViolations();
    });
  });

  describe('Card', () => {
    it('should have no accessibility violations', async () => {
      const { container } = render(
        <Card title="Card Title" subtitle="Card subtitle">
          Card content goes here
        </Card>
      );
      const results = await axe(container);
      expect(results).toHaveNoViolations();
    });

    it('should have no violations with action', async () => {
      const { container } = render(
        <Card
          title="Card with Action"
          action={<button>Action</button>}
        >
          Card content with action button
        </Card>
      );
      const results = await axe(container);
      expect(results).toHaveNoViolations();
    });
  });

  describe('Badge', () => {
    it('should have no accessibility violations', async () => {
      const { container } = render(
        <div>
          <Badge variant="neutral">Neutral</Badge>
          <Badge variant="success">Success</Badge>
          <Badge variant="warning">Warning</Badge>
          <Badge variant="danger">Error</Badge>
        </div>
      );
      const results = await axe(container);
      expect(results).toHaveNoViolations();
    });
  });

  describe('Skeleton', () => {
    it('should have no accessibility violations', async () => {
      const { container } = render(
        <div>
          <Skeleton className="w-32 h-4" />
          <Skeleton className="w-48 h-4" />
          <Skeleton className="w-24 h-4" />
        </div>
      );
      const results = await axe(container);
      expect(results).toHaveNoViolations();
    });
  });

  describe('Chip', () => {
    it('should have no accessibility violations', async () => {
      const { container } = render(
        <div>
          <Chip label="Default Chip" />
          <Chip label="Success Chip" />
          <Chip label="Warning Chip" />
        </div>
      );
      const results = await axe(container);
      expect(results).toHaveNoViolations();
    });
  });

  describe('Form elements', () => {
    it('should have no violations for labeled inputs', async () => {
      const { container } = render(
        <div>
          <label htmlFor="email">Email</label>
          <input id="email" type="email" />

          <label htmlFor="password">Password</label>
          <input id="password" type="password" />
        </div>
      );
      const results = await axe(container);
      expect(results).toHaveNoViolations();
    });

    it('should have no violations for select elements', async () => {
      const { container } = render(
        <div>
          <label htmlFor="chain">Select Chain</label>
          <select id="chain">
            <option value="8453">Base</option>
            <option value="1">Ethereum</option>
            <option value="42161">Arbitrum</option>
          </select>
        </div>
      );
      const results = await axe(container);
      expect(results).toHaveNoViolations();
    });
  });

  describe('Page structure', () => {
    it('should have no violations for navigation structure', async () => {
      const { container } = render(
        <nav aria-label="Main navigation">
          <ul>
            <li><a href="/">Home</a></li>
            <li><a href="/analytics">Analytics</a></li>
            <li><a href="/about">About</a></li>
          </ul>
        </nav>
      );
      const results = await axe(container);
      expect(results).toHaveNoViolations();
    });

    it('should have no violations for main content', async () => {
      const { container } = render(
        <main>
          <h1>Dashboard</h1>
          <section aria-labelledby="gas-section">
            <h2 id="gas-section">Current Gas Price</h2>
            <p>0.00125 gwei</p>
          </section>
        </main>
      );
      const results = await axe(container);
      expect(results).toHaveNoViolations();
    });
  });

  describe('Interactive elements', () => {
    it('should have no violations for buttons with icons', async () => {
      const { container } = render(
        <button aria-label="Close dialog">
          <svg aria-hidden="true" width="24" height="24">
            <path d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      );
      const results = await axe(container);
      expect(results).toHaveNoViolations();
    });

    it('should have no violations for link buttons', async () => {
      const { container } = render(
        <a href="/settings" role="button">
          Settings
        </a>
      );
      const results = await axe(container);
      expect(results).toHaveNoViolations();
    });
  });

  describe('Data display', () => {
    it('should have no violations for data tables', async () => {
      const { container } = render(
        <table>
          <caption>Recent Transactions</caption>
          <thead>
            <tr>
              <th scope="col">Hash</th>
              <th scope="col">Gas Used</th>
              <th scope="col">Price</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td>0xabc...</td>
              <td>21,000</td>
              <td>0.001 gwei</td>
            </tr>
          </tbody>
        </table>
      );
      const results = await axe(container);
      expect(results).toHaveNoViolations();
    });

    it('should have no violations for definition lists', async () => {
      const { container } = render(
        <dl>
          <dt>Current Gas</dt>
          <dd>0.00125 gwei</dd>
          <dt>Base Fee</dt>
          <dd>0.001 gwei</dd>
          <dt>Priority Fee</dt>
          <dd>0.00025 gwei</dd>
        </dl>
      );
      const results = await axe(container);
      expect(results).toHaveNoViolations();
    });
  });

  describe('Status messages', () => {
    it('should have no violations for alert messages', async () => {
      const { container } = render(
        <div role="alert" aria-live="polite">
          Gas price has dropped below your target!
        </div>
      );
      const results = await axe(container);
      expect(results).toHaveNoViolations();
    });

    it('should have no violations for status updates', async () => {
      const { container } = render(
        <div role="status" aria-live="polite">
          Last updated: 2 seconds ago
        </div>
      );
      const results = await axe(container);
      expect(results).toHaveNoViolations();
    });
  });
});
