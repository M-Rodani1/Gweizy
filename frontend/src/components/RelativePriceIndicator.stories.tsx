/**
 * Storybook story for RelativePriceIndicator
 */

import type { Meta, StoryObj } from '@storybook/react';
import RelativePriceIndicator from './RelativePriceIndicator';

const meta: Meta<typeof RelativePriceIndicator> = {
  title: 'Components/RelativePriceIndicator',
  component: RelativePriceIndicator,
  parameters: {
    layout: 'centered',
  },
  tags: ['autodocs'],
};

export default meta;
type Story = StoryObj<typeof RelativePriceIndicator>;

export const Excellent: Story = {
  args: {
    currentGas: 0.001,
    className: '',
  },
};

export const Good: Story = {
  args: {
    currentGas: 0.002,
    className: '',
  },
};

export const Average: Story = {
  args: {
    currentGas: 0.0025,
    className: '',
  },
};

export const High: Story = {
  args: {
    currentGas: 0.0035,
    className: '',
  },
};

export const VeryHigh: Story = {
  args: {
    currentGas: 0.005,
    className: '',
  },
};
