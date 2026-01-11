/**
 * ErrorBoundary Tests
 */

import React from 'react';
import { render, fireEvent } from '@testing-library/react-native';
import { Text, View } from 'react-native';
import { ErrorBoundary } from '../ErrorBoundary';

// Component that throws an error
function ThrowError({ shouldThrow }: { shouldThrow: boolean }): React.JSX.Element {
  if (shouldThrow) {
    throw new Error('Test error message');
  }
  return <Text testID="child-content">Hello World</Text>;
}

// Suppress console.error for error boundary tests
const originalConsoleError = console.error;
beforeAll(() => {
  console.error = jest.fn();
});
afterAll(() => {
  console.error = originalConsoleError;
});

describe('ErrorBoundary', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('rendering', () => {
    it('renders children when no error', () => {
      const { getByTestId } = render(
        <ErrorBoundary>
          <ThrowError shouldThrow={false} />
        </ErrorBoundary>
      );

      expect(getByTestId('child-content')).toBeTruthy();
    });

    it('renders fallback UI when error occurs', () => {
      const { getByTestId, getByText } = render(
        <ErrorBoundary>
          <ThrowError shouldThrow={true} />
        </ErrorBoundary>
      );

      expect(getByTestId('error-boundary-fallback')).toBeTruthy();
      expect(getByText('Something went wrong')).toBeTruthy();
      expect(getByText('Test error message')).toBeTruthy();
    });

    it('renders default message when error has no message', () => {
      // Create a component that throws an error without a message
      const ThrowEmptyError = () => {
        throw new Error();
      };

      const { getByText } = render(
        <ErrorBoundary>
          <ThrowEmptyError />
        </ErrorBoundary>
      );

      expect(getByText('An unexpected error occurred')).toBeTruthy();
    });

    it('renders custom fallback when provided', () => {
      const customFallback = (
        <View testID="custom-fallback">
          <Text>Custom Error UI</Text>
        </View>
      );

      const { getByTestId, queryByTestId } = render(
        <ErrorBoundary fallback={customFallback}>
          <ThrowError shouldThrow={true} />
        </ErrorBoundary>
      );

      expect(getByTestId('custom-fallback')).toBeTruthy();
      expect(queryByTestId('error-boundary-fallback')).toBeNull();
    });
  });

  describe('callbacks', () => {
    it('calls onError callback when error is caught', () => {
      const onError = jest.fn();

      render(
        <ErrorBoundary onError={onError}>
          <ThrowError shouldThrow={true} />
        </ErrorBoundary>
      );

      expect(onError).toHaveBeenCalledTimes(1);
      expect(onError).toHaveBeenCalledWith(
        expect.any(Error),
        expect.objectContaining({ componentStack: expect.any(String) })
      );
    });

    it('does not call onError when no error', () => {
      const onError = jest.fn();

      render(
        <ErrorBoundary onError={onError}>
          <ThrowError shouldThrow={false} />
        </ErrorBoundary>
      );

      expect(onError).not.toHaveBeenCalled();
    });

    it('calls onRetry callback when retry button pressed', () => {
      const onRetry = jest.fn();

      const { getByTestId } = render(
        <ErrorBoundary onRetry={onRetry}>
          <ThrowError shouldThrow={true} />
        </ErrorBoundary>
      );

      fireEvent.press(getByTestId('error-boundary-retry'));

      expect(onRetry).toHaveBeenCalledTimes(1);
    });
  });

  describe('retry functionality', () => {
    it('clears error state when retry is pressed', () => {
      let shouldThrow = true;

      const DynamicComponent = () => {
        if (shouldThrow) {
          throw new Error('Dynamic error');
        }
        return <Text testID="recovered-content">Recovered!</Text>;
      };

      const { getByTestId, queryByTestId, rerender } = render(
        <ErrorBoundary>
          <DynamicComponent />
        </ErrorBoundary>
      );

      // Initially shows error fallback
      expect(getByTestId('error-boundary-fallback')).toBeTruthy();

      // Fix the error
      shouldThrow = false;

      // Press retry
      fireEvent.press(getByTestId('error-boundary-retry'));

      // Force re-render
      rerender(
        <ErrorBoundary>
          <DynamicComponent />
        </ErrorBoundary>
      );

      // Should show recovered content
      expect(queryByTestId('error-boundary-fallback')).toBeNull();
      expect(getByTestId('recovered-content')).toBeTruthy();
    });

    it('renders retry button with correct text', () => {
      const { getByText } = render(
        <ErrorBoundary>
          <ThrowError shouldThrow={true} />
        </ErrorBoundary>
      );

      expect(getByText('Try Again')).toBeTruthy();
    });
  });

  describe('development logging', () => {
    it('logs error to console in development', () => {
      // __DEV__ is true in test environment
      render(
        <ErrorBoundary>
          <ThrowError shouldThrow={true} />
        </ErrorBoundary>
      );

      expect(console.error).toHaveBeenCalled();
    });
  });
});
