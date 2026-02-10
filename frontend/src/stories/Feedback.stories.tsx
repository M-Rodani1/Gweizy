/**
 * Storybook stories for Feedback components (LoadingSpinner, EmptyState)
 */

import type { Meta, StoryObj } from '@storybook/react';
import LoadingSpinner from '../components/LoadingSpinner';
import EmptyState from '../components/EmptyState';
import { Button } from '../components/ui/Button';

// LoadingSpinner Meta
export default {
  title: 'Components/Feedback',
  parameters: {
    layout: 'padded',
    backgrounds: {
      default: 'dark',
      values: [{ name: 'dark', value: '#111827' }],
    },
  },
} as Meta;

// LoadingSpinner Stories
export const Spinner: StoryObj = {
  render: () => <LoadingSpinner />,
};

export const SpinnerWithMessage: StoryObj = {
  render: () => <LoadingSpinner message="Fetching gas prices..." />,
};

export const SpinnerInCard: StoryObj = {
  render: () => (
    <div className="bg-gray-800 rounded-lg p-6 w-80">
      <LoadingSpinner message="Loading predictions..." />
    </div>
  ),
};

// EmptyState Stories
export const NoData: StoryObj = {
  render: () => <EmptyState type="no-data" />,
};

export const NoWallet: StoryObj = {
  render: () => (
    <EmptyState
      type="no-wallet"
      action={<Button variant="primary">Connect Wallet</Button>}
    />
  ),
};

export const NoPredictions: StoryObj = {
  render: () => <EmptyState type="no-predictions" />,
};

export const ErrorState: StoryObj = {
  render: () => (
    <EmptyState
      type="error"
      action={<Button variant="secondary">Try Again</Button>}
    />
  ),
};

export const LoadingState: StoryObj = {
  render: () => <EmptyState type="loading" />,
};

export const CustomContent: StoryObj = {
  render: () => (
    <EmptyState
      type="no-data"
      title="No Transactions Found"
      description="You haven't made any transactions in the last 30 days. Start transacting to see your history here."
      action={
        <div className="flex gap-3">
          <Button variant="outline">View All Time</Button>
          <Button variant="primary">Make Transaction</Button>
        </div>
      }
    />
  ),
};

export const InCardContext: StoryObj = {
  render: () => (
    <div className="grid grid-cols-2 gap-4">
      <div className="bg-gray-800/50 border border-gray-700 rounded-2xl">
        <div className="p-4 border-b border-gray-700">
          <h3 className="text-white font-semibold">Transaction History</h3>
        </div>
        <EmptyState
          type="no-wallet"
          title="Connect to View"
          description="Link your wallet to see your transaction history."
          action={<Button size="sm" variant="primary">Connect</Button>}
        />
      </div>
      <div className="bg-gray-800/50 border border-gray-700 rounded-2xl">
        <div className="p-4 border-b border-gray-700">
          <h3 className="text-white font-semibold">Predictions</h3>
        </div>
        <EmptyState
          type="no-predictions"
          title="Generating..."
          description="Our AI is analyzing current market conditions."
        />
      </div>
    </div>
  ),
};

export const AllStates: StoryObj = {
  render: () => (
    <div className="grid grid-cols-2 gap-6">
      <div className="bg-gray-800/50 rounded-lg p-4">
        <h4 className="text-gray-400 text-sm mb-4">No Data</h4>
        <EmptyState type="no-data" />
      </div>
      <div className="bg-gray-800/50 rounded-lg p-4">
        <h4 className="text-gray-400 text-sm mb-4">No Wallet</h4>
        <EmptyState type="no-wallet" />
      </div>
      <div className="bg-gray-800/50 rounded-lg p-4">
        <h4 className="text-gray-400 text-sm mb-4">No Predictions</h4>
        <EmptyState type="no-predictions" />
      </div>
      <div className="bg-gray-800/50 rounded-lg p-4">
        <h4 className="text-gray-400 text-sm mb-4">Error</h4>
        <EmptyState type="error" />
      </div>
    </div>
  ),
};
