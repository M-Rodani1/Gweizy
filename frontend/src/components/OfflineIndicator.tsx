/**
 * Offline Indicator Component
 * Shows when user is offline
 */

import React from 'react';
import { useOnlineStatus } from '../utils/offline';

export const OfflineIndicator: React.FC = () => {
  const { isOnline } = useOnlineStatus();

  if (isOnline) return null;

  return (
    <div className="fixed top-0 left-0 right-0 bg-yellow-600 text-yellow-900 px-4 py-2 text-center z-50">
      <p className="text-sm font-semibold">
        ⚠️ You're offline. Some features may be limited.
      </p>
    </div>
  );
};
