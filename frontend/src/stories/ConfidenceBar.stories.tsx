/**
 * Storybook stories for ConfidenceBar component
 */

import type { Meta, StoryObj } from '@storybook/react';
import ConfidenceBar from '../components/ui/ConfidenceBar';

const meta: Meta<typeof ConfidenceBar> = {
  title: 'Components/ConfidenceBar',
  component: ConfidenceBar,
  parameters: {
    layout: 'padded',
    backgrounds: {
      default: 'dark',
      values: [{ name: 'dark', value: '#111827' }],
    },
  },
  tags: ['autodocs'],
  argTypes: {
    showLabels: {
      control: 'select',
      options: ['action', 'classification'],
    },
  },
  decorators: [
    (Story) => (
      <div className="max-w-md">
        <Story />
      </div>
    ),
  ],
};

export default meta;
type Story = StoryObj<typeof meta>;

export const Default: Story = {
  args: {
    probs: {
      wait: 0.6,
      normal: 0.3,
      urgent: 0.1,
    },
    showLabels: 'action',
  },
};

export const HighWaitProbability: Story = {
  args: {
    probs: {
      wait: 0.85,
      normal: 0.1,
      urgent: 0.05,
    },
    showLabels: 'action',
  },
  parameters: {
    docs: {
      description: {
        story: 'When gas prices are likely to drop, the "Wait" probability is high. Great time to delay transactions.',
      },
    },
  },
};

export const UrgentSituation: Story = {
  args: {
    probs: {
      wait: 0.1,
      normal: 0.2,
      urgent: 0.7,
    },
    showLabels: 'action',
  },
  parameters: {
    docs: {
      description: {
        story: 'When gas prices are expected to rise, urgency is high. Execute transactions now!',
      },
    },
  },
};

export const BalancedProbabilities: Story = {
  args: {
    probs: {
      wait: 0.35,
      normal: 0.35,
      urgent: 0.3,
    },
    showLabels: 'action',
  },
  parameters: {
    docs: {
      description: {
        story: 'Uncertain market conditions with roughly equal probabilities for each action.',
      },
    },
  },
};

export const MostlyNormal: Story = {
  args: {
    probs: {
      wait: 0.15,
      normal: 0.7,
      urgent: 0.15,
    },
    showLabels: 'action',
  },
  parameters: {
    docs: {
      description: {
        story: 'Stable market conditions - normal gas prices expected to continue.',
      },
    },
  },
};

export const ClassificationLabels: Story = {
  args: {
    probs: {
      wait: 0.2,
      normal: 0.5,
      urgent: 0.3,
    },
    showLabels: 'classification',
  },
  parameters: {
    docs: {
      description: {
        story: 'Uses "Elevated/Normal/Spike" labels instead of "Wait/Normal/Urgent" for classification context.',
      },
    },
  },
};

export const AllScenarios: Story = {
  render: () => (
    <div className="space-y-8">
      <div>
        <h3 className="text-white text-sm font-medium mb-3">Best Time to Wait</h3>
        <ConfidenceBar probs={{ wait: 0.75, normal: 0.2, urgent: 0.05 }} showLabels="action" />
      </div>

      <div>
        <h3 className="text-white text-sm font-medium mb-3">Execute Now</h3>
        <ConfidenceBar probs={{ wait: 0.05, normal: 0.15, urgent: 0.8 }} showLabels="action" />
      </div>

      <div>
        <h3 className="text-white text-sm font-medium mb-3">Stable Conditions</h3>
        <ConfidenceBar probs={{ wait: 0.2, normal: 0.6, urgent: 0.2 }} showLabels="action" />
      </div>

      <div>
        <h3 className="text-white text-sm font-medium mb-3">Market Uncertainty</h3>
        <ConfidenceBar probs={{ wait: 0.33, normal: 0.34, urgent: 0.33 }} showLabels="action" />
      </div>

      <div>
        <h3 className="text-white text-sm font-medium mb-3">Classification Mode</h3>
        <ConfidenceBar probs={{ wait: 0.4, normal: 0.4, urgent: 0.2 }} showLabels="classification" />
      </div>
    </div>
  ),
};

export const InCard: Story = {
  render: () => (
    <div className="bg-gray-800/50 border border-gray-700 rounded-2xl p-6 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-white font-semibold">Transaction Timing</h3>
        <span className="text-green-400 text-sm font-medium">Recommendation: Wait</span>
      </div>
      <ConfidenceBar probs={{ wait: 0.72, normal: 0.2, urgent: 0.08 }} showLabels="action" />
      <p className="text-gray-400 text-sm">
        Gas prices are expected to drop in the next hour. Consider waiting for better rates.
      </p>
    </div>
  ),
};
