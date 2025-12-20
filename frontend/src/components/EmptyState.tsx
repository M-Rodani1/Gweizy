import React from 'react';
import { Inbox, TrendingUp, Activity, Wallet, AlertCircle } from 'lucide-react';

interface EmptyStateProps {
  type?: 'no-data' | 'no-wallet' | 'no-predictions' | 'error' | 'loading';
  title?: string;
  description?: string;
  action?: React.ReactNode;
  className?: string;
}

const EmptyState: React.FC<EmptyStateProps> = ({
  type = 'no-data',
  title,
  description,
  action,
  className = ''
}) => {
  const getIcon = () => {
    switch (type) {
      case 'no-wallet':
        return <Wallet className="w-16 h-16 text-cyan-400/50" />;
      case 'no-predictions':
        return <TrendingUp className="w-16 h-16 text-purple-400/50" />;
      case 'error':
        return <AlertCircle className="w-16 h-16 text-red-400/50" />;
      case 'loading':
        return <Activity className="w-16 h-16 text-cyan-400/50 animate-pulse" />;
      default:
        return <Inbox className="w-16 h-16 text-gray-400/50" />;
    }
  };

  const getDefaultTitle = () => {
    switch (type) {
      case 'no-wallet':
        return 'No Wallet Connected';
      case 'no-predictions':
        return 'No Predictions Available';
      case 'error':
        return 'Something Went Wrong';
      case 'loading':
        return 'Loading...';
      default:
        return 'No Data Available';
    }
  };

  const getDefaultDescription = () => {
    switch (type) {
      case 'no-wallet':
        return 'Connect your wallet to view your transaction history and personalized insights.';
      case 'no-predictions':
        return 'Predictions are being generated. Check back in a few moments.';
      case 'error':
        return 'We encountered an issue loading this data. Please try again later.';
      case 'loading':
        return 'Fetching the latest data from the network...';
      default:
        return 'There is no data to display at this time.';
    }
  };

  return (
    <div className={`flex flex-col items-center justify-center py-12 px-6 ${className}`}>
      <div className="relative mb-6">
        {/* Animated background circle */}
        <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/10 to-purple-500/10 rounded-full blur-2xl animate-pulse" />

        {/* Icon */}
        <div className="relative">
          {getIcon()}
        </div>
      </div>

      <h3 className="text-xl font-semibold text-gray-100 mb-2 text-center">
        {title || getDefaultTitle()}
      </h3>

      <p className="text-sm text-gray-400 text-center max-w-md mb-6">
        {description || getDefaultDescription()}
      </p>

      {action && (
        <div className="mt-2">
          {action}
        </div>
      )}
    </div>
  );
};

export default EmptyState;
