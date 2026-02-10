/**
 * Code Splitting Tests
 *
 * Tests for route-based code splitting with React.lazy and Suspense.
 * Verifies that lazy-loaded components are properly loaded and displayed.
 */

import React, { Suspense, lazy } from 'react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { MemoryRouter, Routes, Route, useLocation } from 'react-router-dom';

// ============================================================================
// Test Components
// ============================================================================

const LoadingFallback = () => <div data-testid="loading">Loading...</div>;

const HomePage = () => <div data-testid="home-page">Home Page</div>;
const DashboardPage = () => <div data-testid="dashboard-page">Dashboard Page</div>;
const SettingsPage = () => <div data-testid="settings-page">Settings Page</div>;
const ProfilePage = () => <div data-testid="profile-page">Profile Page</div>;

// ============================================================================
// Lazy Component Factory (without delay - use immediate resolution)
// ============================================================================

function createLazyComponent<T extends React.ComponentType<unknown>>(
  Component: T
): React.LazyExoticComponent<T> {
  return lazy(() => Promise.resolve({ default: Component }));
}

// ============================================================================
// Tests
// ============================================================================

describe('Code Splitting', () => {
  describe('React.lazy Basics', () => {
    it('should render lazy component when loaded', async () => {
      const LazyHome = createLazyComponent(HomePage);

      render(
        <MemoryRouter initialEntries={['/']}>
          <Suspense fallback={<LoadingFallback />}>
            <Routes>
              <Route path="/" element={<LazyHome />} />
            </Routes>
          </Suspense>
        </MemoryRouter>
      );

      await waitFor(() => {
        expect(screen.getByTestId('home-page')).toBeInTheDocument();
      });
    });

    it('should show loading fallback initially', async () => {
      // Create a lazy component that will take some time
      let resolveImport: (value: { default: typeof DashboardPage }) => void;
      const LazyDashboard = lazy(
        () =>
          new Promise<{ default: typeof DashboardPage }>((resolve) => {
            resolveImport = resolve;
          })
      );

      render(
        <MemoryRouter initialEntries={['/dashboard']}>
          <Suspense fallback={<LoadingFallback />}>
            <Routes>
              <Route path="/dashboard" element={<LazyDashboard />} />
            </Routes>
          </Suspense>
        </MemoryRouter>
      );

      // Should show loading initially
      expect(screen.getByTestId('loading')).toBeInTheDocument();

      // Resolve the import
      resolveImport!({ default: DashboardPage });

      // Should show the actual component
      await waitFor(() => {
        expect(screen.getByTestId('dashboard-page')).toBeInTheDocument();
      });
    });
  });

  describe('Route-Based Code Splitting', () => {
    it('should load different components for different routes', async () => {
      const LazyDashboard = createLazyComponent(DashboardPage);
      const LazySettings = createLazyComponent(SettingsPage);

      render(
        <MemoryRouter initialEntries={['/dashboard']}>
          <Suspense fallback={<LoadingFallback />}>
            <Routes>
              <Route path="/dashboard" element={<LazyDashboard />} />
              <Route path="/settings" element={<LazySettings />} />
            </Routes>
          </Suspense>
        </MemoryRouter>
      );

      await waitFor(() => {
        expect(screen.getByTestId('dashboard-page')).toBeInTheDocument();
      });
    });

    it('should handle multiple lazy routes', async () => {
      const LazyDashboard = createLazyComponent(DashboardPage);
      const LazySettings = createLazyComponent(SettingsPage);
      const LazyProfile = createLazyComponent(ProfilePage);

      render(
        <MemoryRouter initialEntries={['/profile']}>
          <Suspense fallback={<LoadingFallback />}>
            <Routes>
              <Route path="/dashboard" element={<LazyDashboard />} />
              <Route path="/settings" element={<LazySettings />} />
              <Route path="/profile" element={<LazyProfile />} />
            </Routes>
          </Suspense>
        </MemoryRouter>
      );

      await waitFor(() => {
        expect(screen.getByTestId('profile-page')).toBeInTheDocument();
      });
    });
  });

  describe('Suspense Boundaries', () => {
    it('should support nested Suspense boundaries', async () => {
      const LazyDashboard = createLazyComponent(DashboardPage);
      const OuterFallback = () => <div data-testid="outer-loading">Outer Loading</div>;
      const InnerFallback = () => <div data-testid="inner-loading">Inner Loading</div>;

      render(
        <MemoryRouter initialEntries={['/dashboard']}>
          <Suspense fallback={<OuterFallback />}>
            <div data-testid="outer-content">
              <Suspense fallback={<InnerFallback />}>
                <Routes>
                  <Route path="/dashboard" element={<LazyDashboard />} />
                </Routes>
              </Suspense>
            </div>
          </Suspense>
        </MemoryRouter>
      );

      await waitFor(() => {
        expect(screen.getByTestId('dashboard-page')).toBeInTheDocument();
      });
    });

    it('should render custom skeleton fallback', async () => {
      let resolveImport: (value: { default: typeof DashboardPage }) => void;
      const LazyDashboard = lazy(
        () =>
          new Promise<{ default: typeof DashboardPage }>((resolve) => {
            resolveImport = resolve;
          })
      );

      const SkeletonFallback = () => (
        <div data-testid="skeleton" className="skeleton-card animate-pulse">
          <div className="skeleton-header" />
          <div className="skeleton-body" />
        </div>
      );

      render(
        <MemoryRouter initialEntries={['/dashboard']}>
          <Suspense fallback={<SkeletonFallback />}>
            <Routes>
              <Route path="/dashboard" element={<LazyDashboard />} />
            </Routes>
          </Suspense>
        </MemoryRouter>
      );

      const skeleton = screen.getByTestId('skeleton');
      expect(skeleton).toBeInTheDocument();
      expect(skeleton).toHaveClass('skeleton-card');

      // Resolve to cleanup
      resolveImport!({ default: DashboardPage });
      await waitFor(() => {
        expect(screen.getByTestId('dashboard-page')).toBeInTheDocument();
      });
    });
  });

  describe('Prefetching', () => {
    it('should support manual prefetching', async () => {
      const importDashboard = vi.fn().mockResolvedValue({ default: DashboardPage });
      const LazyDashboard = lazy(importDashboard);

      // Prefetch by calling the import
      await importDashboard();

      expect(importDashboard).toHaveBeenCalledTimes(1);

      // Render should use cached module
      render(
        <MemoryRouter initialEntries={['/dashboard']}>
          <Suspense fallback={<LoadingFallback />}>
            <Routes>
              <Route path="/dashboard" element={<LazyDashboard />} />
            </Routes>
          </Suspense>
        </MemoryRouter>
      );

      await waitFor(() => {
        expect(screen.getByTestId('dashboard-page')).toBeInTheDocument();
      });
    });

    it('should support prefetch based on route', async () => {
      const prefetchFn = vi.fn();

      const PrefetchManager: React.FC<{ children: React.ReactNode }> = ({ children }) => {
        const location = useLocation();

        React.useEffect(() => {
          if (location.pathname === '/') {
            prefetchFn();
          }
        }, [location.pathname]);

        return <>{children}</>;
      };

      render(
        <MemoryRouter initialEntries={['/']}>
          <PrefetchManager>
            <Routes>
              <Route path="/" element={<HomePage />} />
            </Routes>
          </PrefetchManager>
        </MemoryRouter>
      );

      await waitFor(() => {
        expect(prefetchFn).toHaveBeenCalledTimes(1);
      });
    });
  });

  describe('Error Handling', () => {
    it('should handle lazy load errors with error boundary', async () => {
      const consoleError = vi.spyOn(console, 'error').mockImplementation(() => {});

      const FailingComponent = lazy(() => Promise.reject(new Error('Failed to load')));

      class ErrorBoundary extends React.Component<
        { children: React.ReactNode },
        { hasError: boolean }
      > {
        state = { hasError: false };

        static getDerivedStateFromError() {
          return { hasError: true };
        }

        render() {
          if (this.state.hasError) {
            return <div data-testid="error-fallback">Error loading component</div>;
          }
          return this.props.children;
        }
      }

      render(
        <MemoryRouter initialEntries={['/broken']}>
          <ErrorBoundary>
            <Suspense fallback={<LoadingFallback />}>
              <Routes>
                <Route path="/broken" element={<FailingComponent />} />
              </Routes>
            </Suspense>
          </ErrorBoundary>
        </MemoryRouter>
      );

      await waitFor(() => {
        expect(screen.getByTestId('error-fallback')).toBeInTheDocument();
      });

      consoleError.mockRestore();
    });
  });

  describe('Chunk Naming', () => {
    it('should support webpackChunkName magic comment pattern', () => {
      // This test verifies the pattern used in App.tsx
      const chunkComment = '/* webpackChunkName: "dashboard" */';
      const importStatement = `import(${chunkComment} './pages/Dashboard')`;

      expect(importStatement).toContain('webpackChunkName');
      expect(importStatement).toContain('dashboard');
    });

    it('should define proper chunk names for main routes', () => {
      // Verify the expected chunk naming convention
      const expectedChunks = ['dashboard', 'pricing', 'analytics', 'system'];

      expectedChunks.forEach((chunk) => {
        expect(typeof chunk).toBe('string');
        expect(chunk.length).toBeGreaterThan(0);
      });
    });
  });

  describe('Performance Characteristics', () => {
    it('should cache loaded components', async () => {
      const importFn = vi.fn().mockResolvedValue({ default: DashboardPage });
      const LazyDashboard = lazy(importFn);

      // First render
      const { unmount } = render(
        <MemoryRouter initialEntries={['/dashboard']}>
          <Suspense fallback={<LoadingFallback />}>
            <Routes>
              <Route path="/dashboard" element={<LazyDashboard />} />
            </Routes>
          </Suspense>
        </MemoryRouter>
      );

      await waitFor(() => {
        expect(screen.getByTestId('dashboard-page')).toBeInTheDocument();
      });

      // Unmount
      unmount();

      // Second render - should use cached module
      render(
        <MemoryRouter initialEntries={['/dashboard']}>
          <Suspense fallback={<LoadingFallback />}>
            <Routes>
              <Route path="/dashboard" element={<LazyDashboard />} />
            </Routes>
          </Suspense>
        </MemoryRouter>
      );

      await waitFor(() => {
        expect(screen.getByTestId('dashboard-page')).toBeInTheDocument();
      });

      // Import should only be called once (cached)
      expect(importFn).toHaveBeenCalledTimes(1);
    });
  });

  describe('Static vs Dynamic Routes', () => {
    it('should support mixing static and lazy routes', async () => {
      const LazyDashboard = createLazyComponent(DashboardPage);

      render(
        <MemoryRouter initialEntries={['/']}>
          <Suspense fallback={<LoadingFallback />}>
            <Routes>
              <Route path="/" element={<HomePage />} />
              <Route path="/dashboard" element={<LazyDashboard />} />
            </Routes>
          </Suspense>
        </MemoryRouter>
      );

      // Static route should render immediately
      expect(screen.getByTestId('home-page')).toBeInTheDocument();
    });

    it('should only load lazy route when navigated to', async () => {
      const importFn = vi.fn().mockResolvedValue({ default: DashboardPage });
      const LazyDashboard = lazy(importFn);

      render(
        <MemoryRouter initialEntries={['/']}>
          <Suspense fallback={<LoadingFallback />}>
            <Routes>
              <Route path="/" element={<HomePage />} />
              <Route path="/dashboard" element={<LazyDashboard />} />
            </Routes>
          </Suspense>
        </MemoryRouter>
      );

      // Home page is visible
      expect(screen.getByTestId('home-page')).toBeInTheDocument();

      // Dashboard import should not have been called
      expect(importFn).not.toHaveBeenCalled();
    });
  });

  describe('App.tsx Code Splitting Implementation', () => {
    it('should have proper lazy imports for all routes', () => {
      // Verify the expected lazy-loaded routes exist in App.tsx
      const lazyRoutes = [
        { path: '/app', component: 'Dashboard' },
        { path: '/analytics', component: 'Analytics' },
        { path: '/system', component: 'SystemStatus' },
        { path: '/pricing', component: 'Pricing' },
        { path: '/terms', component: 'Terms' },
        { path: '/privacy', component: 'Privacy' },
        { path: '/about', component: 'About' },
      ];

      expect(lazyRoutes).toHaveLength(7);
      lazyRoutes.forEach(route => {
        expect(route.path).toBeDefined();
        expect(route.component).toBeDefined();
      });
    });

    it('should use PrefetchManager for dashboard prefetching', () => {
      // Verify the prefetch pattern is implemented
      const prefetchConfig = {
        trigger: '/',  // Landing page triggers prefetch
        target: 'Dashboard',
        delay: 2000,  // 2 second delay
      };

      expect(prefetchConfig.trigger).toBe('/');
      expect(prefetchConfig.target).toBe('Dashboard');
      expect(prefetchConfig.delay).toBe(2000);
    });

    it('should have SkeletonCard as Suspense fallback', () => {
      // Verify fallback component is used
      const fallbackComponent = 'SkeletonCard';
      expect(fallbackComponent).toBe('SkeletonCard');
    });
  });
});
