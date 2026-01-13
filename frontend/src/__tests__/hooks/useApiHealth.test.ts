/**
 * Tests for useApiHealth hook
 *
 * Tests the API health checking hook that monitors backend connectivity.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useApiHealth } from '../../hooks/useApiHealth';
import * as gasApi from '../../api/gasApi';
import React from 'react';

// Mock the gasApi module
vi.mock('../../api/gasApi', () => ({
  checkHealth: vi.fn()
}));

// Mock constants
vi.mock('../../constants', () => ({
  REFRESH_INTERVALS: {
    API_HEALTH: 30000
  }
}));

describe('useApiHealth', () => {
  let queryClient: QueryClient;

  beforeEach(() => {
    vi.clearAllMocks();
    queryClient = new QueryClient({
      defaultOptions: {
        queries: {
          retry: false,
          gcTime: 0,
        },
      },
    });
  });

  afterEach(() => {
    queryClient.clear();
  });

  const wrapper = ({ children }: { children: React.ReactNode }) =>
    React.createElement(QueryClientProvider, { client: queryClient }, children);

  describe('Initial Load', () => {
    it('should start in loading state', () => {
      vi.mocked(gasApi.checkHealth).mockImplementation(
        () => new Promise(() => {})
      );

      const { result } = renderHook(() => useApiHealth(), { wrapper });

      expect(result.current.isLoading).toBe(true);
    });

    it('should call checkHealth on mount', async () => {
      vi.mocked(gasApi.checkHealth).mockResolvedValue(true);

      renderHook(() => useApiHealth(), { wrapper });

      await waitFor(() => {
        expect(gasApi.checkHealth).toHaveBeenCalled();
      });
    });
  });

  describe('Success State', () => {
    it('should return true when API is healthy', async () => {
      vi.mocked(gasApi.checkHealth).mockResolvedValue(true);

      const { result } = renderHook(() => useApiHealth(), { wrapper });

      await waitFor(() => {
        expect(result.current.data).toBe(true);
      });
    });

    it('should not be loading after successful fetch', async () => {
      vi.mocked(gasApi.checkHealth).mockResolvedValue(true);

      const { result } = renderHook(() => useApiHealth(), { wrapper });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });
    });

    it('should not have error after successful fetch', async () => {
      vi.mocked(gasApi.checkHealth).mockResolvedValue(true);

      const { result } = renderHook(() => useApiHealth(), { wrapper });

      await waitFor(() => {
        expect(result.current.isError).toBe(false);
      });
    });
  });

  describe('Error State', () => {
    it('should return false when API is unhealthy', async () => {
      vi.mocked(gasApi.checkHealth).mockResolvedValue(false);

      const { result } = renderHook(() => useApiHealth(), { wrapper });

      await waitFor(() => {
        expect(result.current.data).toBe(false);
      });
    });

    it('should set isError when request fails', async () => {
      vi.mocked(gasApi.checkHealth).mockRejectedValue(new Error('Network error'));

      const { result } = renderHook(() => useApiHealth(), { wrapper });

      await waitFor(() => {
        expect(result.current.isError).toBe(true);
      });
    });

    it('should include error object when request fails', async () => {
      const error = new Error('Network error');
      vi.mocked(gasApi.checkHealth).mockRejectedValue(error);

      const { result } = renderHook(() => useApiHealth(), { wrapper });

      await waitFor(() => {
        expect(result.current.error).toBeTruthy();
      });
    });
  });

  describe('Refetch Functionality', () => {
    it('should provide refetch function', async () => {
      vi.mocked(gasApi.checkHealth).mockResolvedValue(true);

      const { result } = renderHook(() => useApiHealth(), { wrapper });

      await waitFor(() => {
        expect(typeof result.current.refetch).toBe('function');
      });
    });

    it('should refetch when refetch is called', async () => {
      vi.mocked(gasApi.checkHealth).mockResolvedValue(true);

      const { result } = renderHook(() => useApiHealth(), { wrapper });

      await waitFor(() => {
        expect(result.current.data).toBe(true);
      });

      vi.mocked(gasApi.checkHealth).mockClear();

      await result.current.refetch();

      expect(gasApi.checkHealth).toHaveBeenCalled();
    });
  });

  describe('Query Configuration', () => {
    it('should use correct query key', async () => {
      vi.mocked(gasApi.checkHealth).mockResolvedValue(true);

      renderHook(() => useApiHealth(), { wrapper });

      await waitFor(() => {
        const queryState = queryClient.getQueryState(['apiHealth']);
        expect(queryState).toBeTruthy();
      });
    });
  });

  describe('State Transitions', () => {
    it('should transition from loading to success', async () => {
      vi.mocked(gasApi.checkHealth).mockResolvedValue(true);

      const { result } = renderHook(() => useApiHealth(), { wrapper });

      // Initially loading
      expect(result.current.isLoading).toBe(true);

      // After fetch completes
      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
        expect(result.current.data).toBe(true);
      });
    });

    it('should transition from loading to error', async () => {
      vi.mocked(gasApi.checkHealth).mockRejectedValue(new Error('Failed'));

      const { result } = renderHook(() => useApiHealth(), { wrapper });

      // Initially loading
      expect(result.current.isLoading).toBe(true);

      // After fetch fails
      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
        expect(result.current.isError).toBe(true);
      });
    });
  });

  describe('Multiple Calls', () => {
    it('should cache results between renders', async () => {
      vi.mocked(gasApi.checkHealth).mockResolvedValue(true);

      const { result, rerender } = renderHook(() => useApiHealth(), { wrapper });

      await waitFor(() => {
        expect(result.current.data).toBe(true);
      });

      const callCount = vi.mocked(gasApi.checkHealth).mock.calls.length;

      rerender();

      // Should use cached data, not refetch
      expect(vi.mocked(gasApi.checkHealth).mock.calls.length).toBe(callCount);
    });
  });

  describe('Timeout Handling', () => {
    it('should handle slow responses', async () => {
      vi.mocked(gasApi.checkHealth).mockImplementation(
        () => new Promise(resolve => setTimeout(() => resolve(true), 100))
      );

      const { result } = renderHook(() => useApiHealth(), { wrapper });

      expect(result.current.isLoading).toBe(true);

      await waitFor(() => {
        expect(result.current.data).toBe(true);
      }, { timeout: 200 });
    });
  });
});
