/**
 * React Query Provider
 * Wraps the app with React Query for data fetching, caching, and synchronization
 */

import React, { ReactNode, lazy, Suspense } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

// Lazy load devtools - only in dev mode and if available
const ReactQueryDevtools = import.meta.env.DEV
  ? lazy(() =>
      import('@tanstack/react-query-devtools')
        .then((mod) => ({ default: mod.ReactQueryDevtools }))
        .catch(() => ({ default: () => null }))
    )
  : null;

// Create a client with optimized default options for performance
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // Increase staleTime for better caching - data is considered fresh for 30 seconds
      staleTime: 30000, // 30 seconds (increased from 10s)
      // Keep cached data for 10 minutes (increased from 5 minutes)
      gcTime: 600000, // 10 minutes (formerly cacheTime)
      // Retry configuration
      retry: 2, // Reduced from 3 for faster failure feedback
      retryDelay: (attemptIndex) => Math.min(1000 * 2 ** attemptIndex, 30000),
      // Prevent unnecessary refetches
      refetchOnWindowFocus: false,
      refetchOnReconnect: true,
      refetchOnMount: true, // Only refetch on mount if data is stale
      // Network mode - use cache when offline
      networkMode: 'online',
    },
    mutations: {
      retry: 1,
      // Optimistic updates can be configured per mutation
    },
  },
});

interface QueryProviderProps {
  children: ReactNode;
}

export const QueryProvider: React.FC<QueryProviderProps> = ({ children }) => {
  return (
    <QueryClientProvider client={queryClient}>
      {children}
      {import.meta.env.DEV && ReactQueryDevtools && (
        <Suspense fallback={null}>
          <ReactQueryDevtools initialIsOpen={false} />
        </Suspense>
      )}
    </QueryClientProvider>
  );
};

export { queryClient };
