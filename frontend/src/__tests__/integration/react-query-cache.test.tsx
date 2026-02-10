/**
 * React Query Cache Integration Tests
 *
 * Tests cache behavior, invalidation, optimistic updates,
 * and data synchronization patterns.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, waitFor, act } from '@testing-library/react';
import { QueryClient, QueryClientProvider, useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import React from 'react';

// ============================================================================
// Test Utilities
// ============================================================================

function createTestQueryClient() {
  return new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        gcTime: Infinity,
        staleTime: 0,
      },
      mutations: {
        retry: false,
      },
    },
  });
}

function createWrapper(queryClient: QueryClient) {
  return function Wrapper({ children }: { children: React.ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        {children}
      </QueryClientProvider>
    );
  };
}

// ============================================================================
// Cache Behavior Tests
// ============================================================================

describe('React Query Cache Integration', () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    queryClient = createTestQueryClient();
  });

  afterEach(() => {
    queryClient.clear();
  });

  describe('Basic Cache Operations', () => {
    it('should cache query results', async () => {
      const fetchFn = vi.fn().mockResolvedValue({ data: 'test' });

      const { result, rerender } = renderHook(
        () => useQuery({
          queryKey: ['test'],
          queryFn: fetchFn,
        }),
        { wrapper: createWrapper(queryClient) }
      );

      await waitFor(() => expect(result.current.isSuccess).toBe(true));
      expect(fetchFn).toHaveBeenCalledTimes(1);

      // Rerender should use cached data
      rerender();
      expect(result.current.data).toEqual({ data: 'test' });
    });

    it('should use separate cache for different query keys', async () => {
      const fetchFn1 = vi.fn().mockResolvedValue({ key: 'first' });
      const fetchFn2 = vi.fn().mockResolvedValue({ key: 'second' });

      const { result: result1 } = renderHook(
        () => useQuery({
          queryKey: ['query1'],
          queryFn: fetchFn1,
        }),
        { wrapper: createWrapper(queryClient) }
      );

      const { result: result2 } = renderHook(
        () => useQuery({
          queryKey: ['query2'],
          queryFn: fetchFn2,
        }),
        { wrapper: createWrapper(queryClient) }
      );

      await waitFor(() => {
        expect(result1.current.isSuccess).toBe(true);
        expect(result2.current.isSuccess).toBe(true);
      });

      expect(result1.current.data).toEqual({ key: 'first' });
      expect(result2.current.data).toEqual({ key: 'second' });
    });

    it('should handle parameterized query keys', async () => {
      const fetchFn = vi.fn().mockImplementation((id: number) =>
        Promise.resolve({ id, name: `Item ${id}` })
      );

      const { result: result1 } = renderHook(
        () => useQuery({
          queryKey: ['item', 1],
          queryFn: () => fetchFn(1),
        }),
        { wrapper: createWrapper(queryClient) }
      );

      const { result: result2 } = renderHook(
        () => useQuery({
          queryKey: ['item', 2],
          queryFn: () => fetchFn(2),
        }),
        { wrapper: createWrapper(queryClient) }
      );

      await waitFor(() => {
        expect(result1.current.isSuccess).toBe(true);
        expect(result2.current.isSuccess).toBe(true);
      });

      expect(result1.current.data).toEqual({ id: 1, name: 'Item 1' });
      expect(result2.current.data).toEqual({ id: 2, name: 'Item 2' });
      expect(fetchFn).toHaveBeenCalledTimes(2);
    });
  });

  describe('Cache Invalidation', () => {
    it('should refetch on invalidation', async () => {
      let callCount = 0;
      const fetchFn = vi.fn().mockImplementation(() => {
        callCount++;
        return Promise.resolve({ count: callCount });
      });

      const { result } = renderHook(
        () => {
          const query = useQuery({
            queryKey: ['counter'],
            queryFn: fetchFn,
          });
          const client = useQueryClient();
          return { query, client };
        },
        { wrapper: createWrapper(queryClient) }
      );

      await waitFor(() => expect(result.current.query.isSuccess).toBe(true));
      expect(result.current.query.data).toEqual({ count: 1 });

      // Invalidate and wait for refetch
      await act(async () => {
        await result.current.client.invalidateQueries({ queryKey: ['counter'] });
      });

      await waitFor(() => expect(result.current.query.data).toEqual({ count: 2 }));
      expect(fetchFn).toHaveBeenCalledTimes(2);
    });

    it('should invalidate matching queries with prefix', async () => {
      const fetchUser = vi.fn().mockResolvedValue({ type: 'user' });
      const fetchUserPosts = vi.fn().mockResolvedValue({ type: 'posts' });
      const fetchOther = vi.fn().mockResolvedValue({ type: 'other' });

      const { result } = renderHook(
        () => {
          const user = useQuery({
            queryKey: ['user', 1],
            queryFn: fetchUser,
          });
          const posts = useQuery({
            queryKey: ['user', 1, 'posts'],
            queryFn: fetchUserPosts,
          });
          const other = useQuery({
            queryKey: ['other'],
            queryFn: fetchOther,
          });
          const client = useQueryClient();
          return { user, posts, other, client };
        },
        { wrapper: createWrapper(queryClient) }
      );

      await waitFor(() => {
        expect(result.current.user.isSuccess).toBe(true);
        expect(result.current.posts.isSuccess).toBe(true);
        expect(result.current.other.isSuccess).toBe(true);
      });

      // Invalidate all user-related queries
      await act(async () => {
        await result.current.client.invalidateQueries({ queryKey: ['user'] });
      });

      // User queries should refetch
      await waitFor(() => {
        expect(fetchUser).toHaveBeenCalledTimes(2);
        expect(fetchUserPosts).toHaveBeenCalledTimes(2);
      });

      // Other query should not refetch
      expect(fetchOther).toHaveBeenCalledTimes(1);
    });
  });

  describe('Stale Time Behavior', () => {
    it('should not refetch when data is fresh (staleTime)', async () => {
      const fetchFn = vi.fn().mockResolvedValue({ fresh: true });

      // First render
      const { result, unmount } = renderHook(
        () => useQuery({
          queryKey: ['stale-test'],
          queryFn: fetchFn,
          staleTime: Infinity, // Data never becomes stale
        }),
        { wrapper: createWrapper(queryClient) }
      );

      await waitFor(() => expect(result.current.isSuccess).toBe(true));
      expect(fetchFn).toHaveBeenCalledTimes(1);

      unmount();

      // Second render - should use cached data
      const { result: result2 } = renderHook(
        () => useQuery({
          queryKey: ['stale-test'],
          queryFn: fetchFn,
          staleTime: Infinity,
        }),
        { wrapper: createWrapper(queryClient) }
      );

      // Should have data immediately without refetch
      expect(result2.current.data).toEqual({ fresh: true });
      expect(fetchFn).toHaveBeenCalledTimes(1);
    });
  });

  describe('Optimistic Updates', () => {
    interface Todo {
      id: string;
      text: string;
      done: boolean;
    }

    it('should apply and rollback optimistic updates', async () => {
      const initialTodos: Todo[] = [
        { id: '1', text: 'First', done: false },
      ];

      queryClient.setQueryData(['todos'], initialTodos);

      // Test the optimistic update pattern directly via queryClient
      const previousData = queryClient.getQueryData<Todo[]>(['todos']);
      expect(previousData).toEqual(initialTodos);

      // Apply optimistic update
      queryClient.setQueryData<Todo[]>(['todos'], (old) =>
        old?.map((t) => (t.id === '1' ? { ...t, done: true } : t))
      );

      // Verify optimistic update applied
      const optimisticData = queryClient.getQueryData<Todo[]>(['todos']);
      expect(optimisticData?.[0].done).toBe(true);

      // Rollback
      queryClient.setQueryData(['todos'], previousData);

      // Verify rollback
      const rolledBackData = queryClient.getQueryData<Todo[]>(['todos']);
      expect(rolledBackData?.[0].done).toBe(false);
    });
  });

  describe('Query State Management', () => {
    it('should handle enabled/disabled queries', async () => {
      const fetchFn = vi.fn().mockResolvedValue({ data: 'test' });

      const { result, rerender } = renderHook(
        ({ enabled }) => useQuery({
          queryKey: ['enabled-test'],
          queryFn: fetchFn,
          enabled,
        }),
        {
          wrapper: createWrapper(queryClient),
          initialProps: { enabled: false },
        }
      );

      // Should not fetch when disabled
      expect(result.current.fetchStatus).toBe('idle');
      expect(fetchFn).not.toHaveBeenCalled();

      // Enable the query
      rerender({ enabled: true });

      await waitFor(() => expect(result.current.isSuccess).toBe(true));
      expect(fetchFn).toHaveBeenCalledTimes(1);
    });

    it('should handle query errors', async () => {
      const error = new Error('Fetch failed');
      const fetchFn = vi.fn().mockRejectedValue(error);

      const { result } = renderHook(
        () => useQuery({
          queryKey: ['error-test'],
          queryFn: fetchFn,
        }),
        { wrapper: createWrapper(queryClient) }
      );

      await waitFor(() => expect(result.current.isError).toBe(true));

      expect(result.current.error).toEqual(error);
    });
  });

  describe('Manual Cache Updates', () => {
    it('should update cache with setQueryData', () => {
      queryClient.setQueryData(['manual'], { initial: true });

      const data = queryClient.getQueryData(['manual']);
      expect(data).toEqual({ initial: true });

      // Update
      queryClient.setQueryData(['manual'], { updated: true });

      const updatedData = queryClient.getQueryData(['manual']);
      expect(updatedData).toEqual({ updated: true });
    });

    it('should get cached data with getQueryData', () => {
      const testData = { id: 1, name: 'Test' };
      queryClient.setQueryData(['get-test'], testData);

      const cachedData = queryClient.getQueryData(['get-test']);
      expect(cachedData).toEqual(testData);
    });

    it('should remove cache entry', () => {
      queryClient.setQueryData(['remove-test'], { data: 'to-remove' });

      expect(queryClient.getQueryData(['remove-test'])).toBeDefined();

      queryClient.removeQueries({ queryKey: ['remove-test'] });

      expect(queryClient.getQueryData(['remove-test'])).toBeUndefined();
    });

    it('should clear all queries', () => {
      queryClient.setQueryData(['a'], 1);
      queryClient.setQueryData(['b'], 2);
      queryClient.setQueryData(['c'], 3);

      queryClient.clear();

      expect(queryClient.getQueryData(['a'])).toBeUndefined();
      expect(queryClient.getQueryData(['b'])).toBeUndefined();
      expect(queryClient.getQueryData(['c'])).toBeUndefined();
    });
  });

  describe('Prefetching', () => {
    it('should prefetch data before component mounts', async () => {
      const fetchFn = vi.fn().mockResolvedValue({ prefetched: true });

      // Prefetch
      await queryClient.prefetchQuery({
        queryKey: ['prefetch-test'],
        queryFn: fetchFn,
      });

      expect(fetchFn).toHaveBeenCalledTimes(1);

      // Data should be in cache
      const cachedData = queryClient.getQueryData(['prefetch-test']);
      expect(cachedData).toEqual({ prefetched: true });
    });

    it('should prefetch multiple queries in parallel', async () => {
      const fetchFn1 = vi.fn().mockResolvedValue({ id: 1 });
      const fetchFn2 = vi.fn().mockResolvedValue({ id: 2 });
      const fetchFn3 = vi.fn().mockResolvedValue({ id: 3 });

      await Promise.all([
        queryClient.prefetchQuery({
          queryKey: ['parallel', 1],
          queryFn: fetchFn1,
        }),
        queryClient.prefetchQuery({
          queryKey: ['parallel', 2],
          queryFn: fetchFn2,
        }),
        queryClient.prefetchQuery({
          queryKey: ['parallel', 3],
          queryFn: fetchFn3,
        }),
      ]);

      expect(queryClient.getQueryData(['parallel', 1])).toEqual({ id: 1 });
      expect(queryClient.getQueryData(['parallel', 2])).toEqual({ id: 2 });
      expect(queryClient.getQueryData(['parallel', 3])).toEqual({ id: 3 });
    });
  });

  describe('Initial Data', () => {
    it('should use initialData and not refetch if staleTime is Infinity', async () => {
      const fetchFn = vi.fn().mockResolvedValue({ data: 'fetched' });
      const initialData = { data: 'initial' };

      const { result } = renderHook(
        () => useQuery({
          queryKey: ['initial-data-test'],
          queryFn: fetchFn,
          initialData,
          staleTime: Infinity,
        }),
        { wrapper: createWrapper(queryClient) }
      );

      // Should use initial data
      expect(result.current.data).toEqual(initialData);
      expect(result.current.isSuccess).toBe(true);
      expect(fetchFn).not.toHaveBeenCalled();
    });
  });

  describe('Query Matching', () => {
    it('should match queries by exact key', () => {
      queryClient.setQueryData(['exact', 'key'], { exact: true });
      queryClient.setQueryData(['exact', 'key', 'nested'], { nested: true });

      const queries = queryClient.getQueriesData({ queryKey: ['exact', 'key'], exact: true });
      expect(queries).toHaveLength(1);
      expect(queries[0][1]).toEqual({ exact: true });
    });

    it('should match queries by prefix', () => {
      queryClient.setQueryData(['prefix', 1], { id: 1 });
      queryClient.setQueryData(['prefix', 2], { id: 2 });
      queryClient.setQueryData(['other'], { other: true });

      const queries = queryClient.getQueriesData({ queryKey: ['prefix'] });
      expect(queries).toHaveLength(2);
    });
  });

  describe('Mutation Integration', () => {
    it('should handle mutation success', async () => {
      const mutateFn = vi.fn().mockResolvedValue({ success: true });

      const { result } = renderHook(
        () => useMutation({
          mutationFn: mutateFn,
        }),
        { wrapper: createWrapper(queryClient) }
      );

      await act(async () => {
        await result.current.mutateAsync({ id: 1 });
      });

      await waitFor(() => {
        expect(result.current.isSuccess).toBe(true);
      });
      expect(result.current.data).toEqual({ success: true });
    });

    it('should handle mutation error', async () => {
      const error = new Error('Mutation failed');
      const mutateFn = vi.fn().mockRejectedValue(error);

      const { result } = renderHook(
        () => useMutation({
          mutationFn: mutateFn,
        }),
        { wrapper: createWrapper(queryClient) }
      );

      await act(async () => {
        try {
          await result.current.mutateAsync({ id: 1 });
        } catch {
          // Expected
        }
      });

      await waitFor(() => {
        expect(result.current.isError).toBe(true);
      });
      expect(result.current.error).toEqual(error);
    });

    it('should invalidate queries on mutation success', async () => {
      const fetchFn = vi.fn().mockResolvedValue({ count: 1 });
      queryClient.setQueryData(['invalidate-on-mutate'], { count: 0 });

      const { result } = renderHook(
        () => {
          const query = useQuery({
            queryKey: ['invalidate-on-mutate'],
            queryFn: fetchFn,
          });
          const client = useQueryClient();
          const mutation = useMutation({
            mutationFn: () => Promise.resolve({ success: true }),
            onSuccess: () => {
              client.invalidateQueries({ queryKey: ['invalidate-on-mutate'] });
            },
          });
          return { query, mutation };
        },
        { wrapper: createWrapper(queryClient) }
      );

      await act(async () => {
        await result.current.mutation.mutateAsync({});
      });

      await waitFor(() => {
        expect(fetchFn).toHaveBeenCalled();
      });
    });
  });

  describe('Cache State', () => {
    it('should track query state correctly', () => {
      // No data initially
      const state1 = queryClient.getQueryState(['state-test']);
      expect(state1).toBeUndefined();

      // After setting data
      queryClient.setQueryData(['state-test'], { data: 'test' });
      const state2 = queryClient.getQueryState(['state-test']);
      expect(state2?.data).toEqual({ data: 'test' });
      expect(state2?.status).toBe('success');
    });

    it('should support query defaults', async () => {
      const defaultFn = vi.fn().mockResolvedValue({ default: true });

      queryClient.setQueryDefaults(['with-defaults'], {
        queryFn: defaultFn,
        staleTime: Infinity,
      });

      const { result } = renderHook(
        () => useQuery({
          queryKey: ['with-defaults'],
        }),
        { wrapper: createWrapper(queryClient) }
      );

      await waitFor(() => expect(result.current.isSuccess).toBe(true));
      expect(result.current.data).toEqual({ default: true });
    });
  });
});
