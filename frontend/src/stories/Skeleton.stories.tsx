/**
 * Storybook stories for Skeleton components
 */

import type { Meta, StoryObj } from '@storybook/react';
import Skeleton, {
  SkeletonCard,
  SkeletonList,
  SkeletonMetrics,
  SkeletonChart,
  SkeletonTable,
  SkeletonGasPrediction,
  SkeletonAccuracyMetrics,
  SkeletonMultiChain,
  SkeletonGasHero,
  SkeletonNetworkIntel,
  SkeletonForecast,
  SkeletonProfile,
  SkeletonHeatmap,
  ErrorFallback,
} from '../components/ui/Skeleton';

const meta: Meta<typeof Skeleton> = {
  title: 'Components/Skeleton',
  component: Skeleton,
  parameters: {
    layout: 'padded',
    backgrounds: {
      default: 'dark',
      values: [{ name: 'dark', value: '#111827' }],
    },
  },
  tags: ['autodocs'],
};

export default meta;
type Story = StoryObj<typeof meta>;

export const BasicVariants: Story = {
  render: () => (
    <div className="space-y-4">
      <div>
        <p className="text-gray-400 text-sm mb-2">Text</p>
        <Skeleton variant="text" width="60%" />
      </div>
      <div>
        <p className="text-gray-400 text-sm mb-2">Rectangle</p>
        <Skeleton variant="rect" width={200} height={100} />
      </div>
      <div>
        <p className="text-gray-400 text-sm mb-2">Circle</p>
        <Skeleton variant="circle" width={48} height={48} />
      </div>
      <div>
        <p className="text-gray-400 text-sm mb-2">Card</p>
        <Skeleton variant="card" width={300} height={150} />
      </div>
    </div>
  ),
};

export const CardLayout: Story = {
  render: () => <SkeletonCard />,
};

export const ListLayout: Story = {
  render: () => <SkeletonList count={5} />,
};

export const MetricsLayout: Story = {
  render: () => <SkeletonMetrics />,
};

export const ChartLayout: Story = {
  render: () => <SkeletonChart height={250} />,
};

export const TableLayout: Story = {
  render: () => <SkeletonTable rows={5} cols={4} />,
};

export const GasPredictionSkeleton: Story = {
  render: () => (
    <div className="grid grid-cols-3 gap-4">
      <SkeletonGasPrediction />
      <SkeletonGasPrediction />
      <SkeletonGasPrediction />
    </div>
  ),
};

export const AccuracyMetricsSkeleton: Story = {
  render: () => <SkeletonAccuracyMetrics />,
};

export const MultiChainSkeleton: Story = {
  render: () => <SkeletonMultiChain count={5} />,
};

export const GasHeroSkeleton: Story = {
  render: () => <SkeletonGasHero />,
};

export const NetworkIntelSkeleton: Story = {
  render: () => <SkeletonNetworkIntel />,
};

export const ForecastSkeleton: Story = {
  render: () => <SkeletonForecast />,
};

export const ProfileSkeleton: Story = {
  render: () => <SkeletonProfile />,
};

export const HeatmapSkeleton: Story = {
  render: () => <SkeletonHeatmap />,
};

export const ErrorState: Story = {
  render: () => (
    <ErrorFallback
      error="Unable to connect to the gas price API. Please check your network connection."
      onRetry={() => alert('Retry clicked!')}
    />
  ),
};

export const AllSkeletons: Story = {
  render: () => (
    <div className="space-y-8">
      <section>
        <h3 className="text-white text-lg font-semibold mb-4">Gas Predictions</h3>
        <div className="grid grid-cols-3 gap-4">
          <SkeletonGasPrediction />
          <SkeletonGasPrediction />
          <SkeletonGasPrediction />
        </div>
      </section>

      <section>
        <h3 className="text-white text-lg font-semibold mb-4">Forecast & Profile</h3>
        <div className="grid grid-cols-2 gap-4">
          <SkeletonForecast />
          <SkeletonProfile />
        </div>
      </section>

      <section>
        <h3 className="text-white text-lg font-semibold mb-4">Analytics</h3>
        <div className="grid grid-cols-2 gap-4">
          <SkeletonNetworkIntel />
          <SkeletonChart height={180} />
        </div>
      </section>

      <section>
        <h3 className="text-white text-lg font-semibold mb-4">Data Display</h3>
        <div className="grid grid-cols-2 gap-4">
          <SkeletonMultiChain count={4} />
          <SkeletonHeatmap />
        </div>
      </section>
    </div>
  ),
};
