/**
 * Storybook stories for Sparkline component
 */

import type { Meta, StoryObj } from '@storybook/react';
import Sparkline from '../components/ui/Sparkline';

const meta: Meta<typeof Sparkline> = {
  title: 'Components/Sparkline',
  component: Sparkline,
  parameters: {
    layout: 'centered',
    backgrounds: {
      default: 'dark',
      values: [{ name: 'dark', value: '#111827' }],
    },
  },
  tags: ['autodocs'],
  argTypes: {
    color: {
      control: 'color',
    },
    width: {
      control: { type: 'range', min: 40, max: 200, step: 10 },
    },
    height: {
      control: { type: 'range', min: 15, max: 60, step: 5 },
    },
    strokeWidth: {
      control: { type: 'range', min: 0.5, max: 4, step: 0.5 },
    },
  },
};

export default meta;
type Story = StoryObj<typeof meta>;

// Sample data sets
const upTrend = [10, 12, 15, 14, 18, 22, 25, 28, 32, 35];
const downTrend = [35, 32, 28, 25, 22, 18, 14, 15, 12, 10];
const volatile = [20, 35, 15, 40, 25, 45, 20, 50, 30, 35];
const stable = [22, 23, 21, 22, 23, 22, 21, 22, 23, 22];
const gasHistory = [45.2, 42.1, 48.3, 51.2, 49.8, 46.3, 43.1, 41.5, 38.9, 40.2];

export const Default: Story = {
  args: {
    data: upTrend,
    width: 80,
    height: 24,
    color: '#06b6d4',
  },
};

export const UpTrend: Story = {
  args: {
    data: upTrend,
    width: 100,
    height: 30,
    color: '#22c55e',
    showDot: true,
  },
};

export const DownTrend: Story = {
  args: {
    data: downTrend,
    width: 100,
    height: 30,
    color: '#ef4444',
    showDot: true,
  },
};

export const WithGradient: Story = {
  args: {
    data: volatile,
    width: 120,
    height: 40,
    color: '#06b6d4',
    showGradient: true,
    showDot: true,
  },
};

export const TrendColorUp: Story = {
  args: {
    data: upTrend,
    width: 100,
    height: 30,
    useTrendColor: true,
    showDot: true,
  },
  parameters: {
    docs: {
      description: {
        story: 'Uses automatic trend coloring - red for up (bad for gas prices), green for down (good).',
      },
    },
  },
};

export const TrendColorDown: Story = {
  args: {
    data: downTrend,
    width: 100,
    height: 30,
    useTrendColor: true,
    showDot: true,
  },
};

export const GasPriceHistory: Story = {
  args: {
    data: gasHistory,
    width: 120,
    height: 35,
    color: '#f59e0b',
    showGradient: true,
    showDot: true,
    strokeWidth: 2,
  },
};

export const StablePrice: Story = {
  args: {
    data: stable,
    width: 80,
    height: 24,
    color: '#6b7280',
  },
};

export const EmptyData: Story = {
  args: {
    data: [],
    width: 80,
    height: 24,
  },
  parameters: {
    docs: {
      description: {
        story: 'Shows a dashed line when no data is available.',
      },
    },
  },
};

export const InContext: Story = {
  render: () => (
    <div className="bg-gray-800 rounded-lg p-4 space-y-4 w-64">
      <div className="flex items-center justify-between">
        <span className="text-gray-400 text-sm">Gas Price</span>
        <div className="flex items-center gap-2">
          <Sparkline data={downTrend} width={60} height={20} useTrendColor showDot />
          <span className="text-green-400 text-sm">-12%</span>
        </div>
      </div>
      <div className="flex items-center justify-between">
        <span className="text-gray-400 text-sm">Network Load</span>
        <div className="flex items-center gap-2">
          <Sparkline data={upTrend} width={60} height={20} useTrendColor showDot />
          <span className="text-red-400 text-sm">+28%</span>
        </div>
      </div>
      <div className="flex items-center justify-between">
        <span className="text-gray-400 text-sm">Transactions</span>
        <div className="flex items-center gap-2">
          <Sparkline data={volatile} width={60} height={20} color="#a855f7" showDot />
          <span className="text-purple-400 text-sm">~</span>
        </div>
      </div>
    </div>
  ),
};

export const AllSizes: Story = {
  render: () => (
    <div className="space-y-4">
      <div className="flex items-center gap-4">
        <span className="text-gray-400 w-20">Small</span>
        <Sparkline data={volatile} width={40} height={15} color="#06b6d4" />
      </div>
      <div className="flex items-center gap-4">
        <span className="text-gray-400 w-20">Medium</span>
        <Sparkline data={volatile} width={80} height={24} color="#06b6d4" />
      </div>
      <div className="flex items-center gap-4">
        <span className="text-gray-400 w-20">Large</span>
        <Sparkline data={volatile} width={150} height={40} color="#06b6d4" showGradient />
      </div>
      <div className="flex items-center gap-4">
        <span className="text-gray-400 w-20">XL</span>
        <Sparkline data={volatile} width={200} height={60} color="#06b6d4" showGradient showDot strokeWidth={2} />
      </div>
    </div>
  ),
};
