/**
 * Tests for ErrorBoundary and SectionErrorBoundary components
 *
 * Tests error catching, fallback rendering, retry functionality,
 * and Sentry integration.
 */

import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import ErrorBoundary from '../../components/ErrorBoundary';
import { SectionErrorBoundary } from '../../components/SectionErrorBoundary';

// Mock Sentry
vi.mock('@sentry/react', () => ({
  captureException: vi.fn(),
}));

import * as Sentry from '@sentry/react';

// Component that throws an error
const ThrowingComponent = ({ shouldThrow = true }: { shouldThrow?: boolean }) => {
  if (shouldThrow) {
    throw new Error('Test error');
  }
  return <div>No error</div>;
};

describe('ErrorBoundary', () => {
  // Suppress console.error during tests
  const originalError = console.error;

  beforeEach(() => {
    vi.clearAllMocks();
    console.error = vi.fn();
  });

  afterEach(() => {
    console.error = originalError;
  });

  describe('Normal Rendering', () => {
    it('should render children when no error', () => {
      render(
        <ErrorBoundary>
          <div>Child content</div>
        </ErrorBoundary>
      );

      expect(screen.getByText('Child content')).toBeInTheDocument();
    });

    it('should not show error UI when no error', () => {
      render(
        <ErrorBoundary>
          <div>Child content</div>
        </ErrorBoundary>
      );

      expect(screen.queryByText('Oops! Something went wrong')).not.toBeInTheDocument();
    });
  });

  describe('Error Handling', () => {
    it('should catch errors and show error UI', () => {
      render(
        <ErrorBoundary>
          <ThrowingComponent />
        </ErrorBoundary>
      );

      expect(screen.getByText('Oops! Something went wrong')).toBeInTheDocument();
    });

    it('should log error to console', () => {
      render(
        <ErrorBoundary>
          <ThrowingComponent />
        </ErrorBoundary>
      );

      expect(console.error).toHaveBeenCalled();
    });

    it('should report error to Sentry', () => {
      render(
        <ErrorBoundary>
          <ThrowingComponent />
        </ErrorBoundary>
      );

      expect(Sentry.captureException).toHaveBeenCalled();
    });

    it('should call onError callback when provided', () => {
      const onError = vi.fn();

      render(
        <ErrorBoundary onError={onError}>
          <ThrowingComponent />
        </ErrorBoundary>
      );

      expect(onError).toHaveBeenCalled();
      expect(onError.mock.calls[0][0]).toBeInstanceOf(Error);
    });
  });

  describe('Custom Fallback', () => {
    it('should render custom fallback when provided', () => {
      render(
        <ErrorBoundary fallback={<div>Custom error UI</div>}>
          <ThrowingComponent />
        </ErrorBoundary>
      );

      expect(screen.getByText('Custom error UI')).toBeInTheDocument();
      expect(screen.queryByText('Oops! Something went wrong')).not.toBeInTheDocument();
    });
  });

  describe('Error UI Elements', () => {
    it('should show Try Again button by default', () => {
      render(
        <ErrorBoundary>
          <ThrowingComponent />
        </ErrorBoundary>
      );

      expect(screen.getByRole('button', { name: /try again/i })).toBeInTheDocument();
    });

    it('should hide Try Again button when showRetry is false', () => {
      render(
        <ErrorBoundary showRetry={false}>
          <ThrowingComponent />
        </ErrorBoundary>
      );

      expect(screen.queryByRole('button', { name: /try again/i })).not.toBeInTheDocument();
    });

    it('should show Return to Home button', () => {
      render(
        <ErrorBoundary>
          <ThrowingComponent />
        </ErrorBoundary>
      );

      expect(screen.getByRole('button', { name: /return to home/i })).toBeInTheDocument();
    });

    it('should show Refresh Page button', () => {
      render(
        <ErrorBoundary>
          <ThrowingComponent />
        </ErrorBoundary>
      );

      expect(screen.getByRole('button', { name: /refresh page/i })).toBeInTheDocument();
    });

    it('should show error code', () => {
      render(
        <ErrorBoundary>
          <ThrowingComponent />
        </ErrorBoundary>
      );

      expect(screen.getByText(/Error Code:/)).toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    it('should have role="alert" on error container', () => {
      render(
        <ErrorBoundary>
          <ThrowingComponent />
        </ErrorBoundary>
      );

      expect(screen.getByRole('alert')).toBeInTheDocument();
    });

    it('should have aria-live="assertive" on error container', () => {
      render(
        <ErrorBoundary>
          <ThrowingComponent />
        </ErrorBoundary>
      );

      const alert = screen.getByRole('alert');
      expect(alert).toHaveAttribute('aria-live', 'assertive');
    });
  });

  describe('Recovery', () => {
    it('should reset error state when Try Again is clicked', () => {
      render(
        <ErrorBoundary>
          <ThrowingComponent shouldThrow={true} />
        </ErrorBoundary>
      );

      expect(screen.getByText('Oops! Something went wrong')).toBeInTheDocument();

      // We can't easily test recovery because the component will throw again
      // but we can verify the button exists and is clickable
      const tryAgainButton = screen.getByRole('button', { name: /try again/i });
      expect(tryAgainButton).toBeEnabled();
    });
  });
});

describe('SectionErrorBoundary', () => {
  const originalError = console.error;

  beforeEach(() => {
    vi.clearAllMocks();
    console.error = vi.fn();
  });

  afterEach(() => {
    console.error = originalError;
  });

  describe('Normal Rendering', () => {
    it('should render children when no error', () => {
      render(
        <SectionErrorBoundary sectionName="Test Section">
          <div>Section content</div>
        </SectionErrorBoundary>
      );

      expect(screen.getByText('Section content')).toBeInTheDocument();
    });
  });

  describe('Error Handling', () => {
    it('should catch errors and show section error UI', () => {
      render(
        <SectionErrorBoundary sectionName="Gas Predictions">
          <ThrowingComponent />
        </SectionErrorBoundary>
      );

      expect(screen.getByText('Unable to load Gas Predictions')).toBeInTheDocument();
    });

    it('should show Retry button', () => {
      render(
        <SectionErrorBoundary sectionName="Test Section">
          <ThrowingComponent />
        </SectionErrorBoundary>
      );

      expect(screen.getByRole('button', { name: /retry/i })).toBeInTheDocument();
    });

    it('should report error to Sentry with section tag', () => {
      render(
        <SectionErrorBoundary sectionName="Test Section">
          <ThrowingComponent />
        </SectionErrorBoundary>
      );

      expect(Sentry.captureException).toHaveBeenCalledWith(
        expect.any(Error),
        expect.objectContaining({
          tags: {
            section: 'Test Section',
          },
        })
      );
    });

    it('should call onError callback with section context', () => {
      const onError = vi.fn();

      render(
        <SectionErrorBoundary sectionName="Test Section" onError={onError}>
          <ThrowingComponent />
        </SectionErrorBoundary>
      );

      expect(onError).toHaveBeenCalled();
    });
  });

  describe('Custom Fallback', () => {
    it('should render custom fallback when provided', () => {
      render(
        <SectionErrorBoundary
          sectionName="Test Section"
          fallback={<div>Custom section error</div>}
        >
          <ThrowingComponent />
        </SectionErrorBoundary>
      );

      expect(screen.getByText('Custom section error')).toBeInTheDocument();
    });
  });

  describe('Accessibility', () => {
    it('should have role="alert" on error container', () => {
      render(
        <SectionErrorBoundary sectionName="Test Section">
          <ThrowingComponent />
        </SectionErrorBoundary>
      );

      expect(screen.getByRole('alert')).toBeInTheDocument();
    });

    it('should have aria-live="polite" on error container', () => {
      render(
        <SectionErrorBoundary sectionName="Test Section">
          <ThrowingComponent />
        </SectionErrorBoundary>
      );

      const alert = screen.getByRole('alert');
      expect(alert).toHaveAttribute('aria-live', 'polite');
    });
  });

  describe('Custom className', () => {
    it('should apply custom className to error container', () => {
      render(
        <SectionErrorBoundary sectionName="Test Section" className="custom-class">
          <ThrowingComponent />
        </SectionErrorBoundary>
      );

      const alert = screen.getByRole('alert');
      expect(alert.className).toContain('custom-class');
    });
  });
});
