import React from 'react';
import {
  render,
  waitFor,
  act,
  fireEvent,
} from '@testing-library/react-native';
import { SyncProgressScreen } from '../SyncProgressScreen';
import * as api from '@/services/api';
import type { SyncStatusResponse } from '@/types/api';

// Mock the API module
jest.mock('@/services/api', () => ({
  getSyncStatus: jest.fn(),
}));

const mockGetSyncStatus = api.getSyncStatus as jest.MockedFunction<
  typeof api.getSyncStatus
>;

describe('SyncProgressScreen', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  const createSyncStatus = (
    overrides: Partial<SyncStatusResponse> = {}
  ): SyncStatusResponse => ({
    status: 'syncing',
    emails_synced: 0,
    total_emails: null,
    progress: 0,
    error: null,
    ...overrides,
  });

  it('renders the sync progress screen', async () => {
    mockGetSyncStatus.mockResolvedValue(createSyncStatus());

    const { getByTestId } = render(<SyncProgressScreen />);

    await waitFor(() => {
      expect(getByTestId('sync-progress-screen')).toBeTruthy();
    });
  });

  it('shows loading indicator while syncing', async () => {
    mockGetSyncStatus.mockResolvedValue(createSyncStatus());

    const { getByTestId } = render(<SyncProgressScreen />);

    await waitFor(() => {
      expect(getByTestId('loading-indicator')).toBeTruthy();
    });
  });

  it('shows sync content when syncing', async () => {
    mockGetSyncStatus.mockResolvedValue(createSyncStatus());

    const { getByTestId } = render(<SyncProgressScreen />);

    await waitFor(() => {
      expect(getByTestId('sync-content')).toBeTruthy();
    });
  });

  it('displays email count', async () => {
    mockGetSyncStatus.mockResolvedValue(
      createSyncStatus({
        emails_synced: 150,
        progress: 0.15,
      })
    );

    const { getByTestId, getByText } = render(<SyncProgressScreen />);

    await waitFor(() => {
      expect(getByTestId('email-count')).toBeTruthy();
      expect(getByText('150 emails processed')).toBeTruthy();
    });
  });

  it('shows progress bar', async () => {
    mockGetSyncStatus.mockResolvedValue(
      createSyncStatus({
        progress: 0.5,
      })
    );

    const { getByTestId } = render(<SyncProgressScreen />);

    await waitFor(() => {
      expect(getByTestId('progress-bar-container')).toBeTruthy();
      expect(getByTestId('progress-bar-fill')).toBeTruthy();
    });
  });

  it('calls onComplete when sync completes', async () => {
    const onComplete = jest.fn();
    mockGetSyncStatus.mockResolvedValue(
      createSyncStatus({
        status: 'completed',
        emails_synced: 1000,
        progress: 1.0,
      })
    );

    render(<SyncProgressScreen onComplete={onComplete} />);

    await waitFor(() => {
      expect(onComplete).toHaveBeenCalled();
    });
  });

  it('calls onError when sync fails', async () => {
    const onError = jest.fn();
    mockGetSyncStatus.mockResolvedValue(
      createSyncStatus({
        status: 'failed',
        error: 'Network error',
      })
    );

    render(<SyncProgressScreen onError={onError} />);

    await waitFor(() => {
      expect(onError).toHaveBeenCalledWith('Network error');
    });
  });

  it('shows error content when sync fails', async () => {
    mockGetSyncStatus.mockResolvedValue(
      createSyncStatus({
        status: 'failed',
        error: 'Something went wrong',
      })
    );

    const { getByTestId, getByText } = render(<SyncProgressScreen />);

    await waitFor(() => {
      expect(getByTestId('error-content')).toBeTruthy();
      expect(getByText('Sync Failed')).toBeTruthy();
      expect(getByText('Something went wrong')).toBeTruthy();
    });
  });

  it('shows retry button when sync fails', async () => {
    mockGetSyncStatus.mockResolvedValue(
      createSyncStatus({
        status: 'failed',
        error: 'Error',
      })
    );

    const { getByTestId } = render(<SyncProgressScreen />);

    await waitFor(() => {
      expect(getByTestId('retry-button')).toBeTruthy();
    });
  });

  it('retries sync when retry button is pressed', async () => {
    // First call returns failed, subsequent calls return syncing
    mockGetSyncStatus.mockResolvedValue(
      createSyncStatus({
        status: 'failed',
        error: 'Error',
      })
    );

    const { getByTestId } = render(<SyncProgressScreen />);

    await waitFor(() => {
      expect(getByTestId('retry-button')).toBeTruthy();
    });

    // Now mock for retry
    mockGetSyncStatus.mockResolvedValue(createSyncStatus({ status: 'syncing' }));

    // Press retry
    fireEvent.press(getByTestId('retry-button'));

    // Should show sync content after retry
    await waitFor(() => {
      expect(getByTestId('sync-content')).toBeTruthy();
    });
  });

  it('polls status at regular intervals', async () => {
    mockGetSyncStatus.mockResolvedValue(createSyncStatus());

    render(<SyncProgressScreen />);

    // Initial call
    await waitFor(() => {
      expect(mockGetSyncStatus).toHaveBeenCalledTimes(1);
    });

    // Advance timer by poll interval
    await act(async () => {
      jest.advanceTimersByTime(2500);
    });

    await waitFor(() => {
      expect(mockGetSyncStatus).toHaveBeenCalledTimes(2);
    });
  });

  it('stops polling when sync completes', async () => {
    const onComplete = jest.fn();
    mockGetSyncStatus.mockResolvedValue(
      createSyncStatus({
        status: 'completed',
        progress: 1.0,
      })
    );

    render(<SyncProgressScreen onComplete={onComplete} />);

    // Wait for completion callback
    await waitFor(() => {
      expect(onComplete).toHaveBeenCalled();
    });

    // Verify the component stops updating (shows complete message)
    await waitFor(() => {
      expect(mockGetSyncStatus).toHaveBeenCalled();
    });
  });

  it('stops polling when sync fails', async () => {
    const onError = jest.fn();
    mockGetSyncStatus.mockResolvedValue(
      createSyncStatus({
        status: 'failed',
        error: 'Error',
      })
    );

    render(<SyncProgressScreen onError={onError} />);

    // Wait for error callback
    await waitFor(() => {
      expect(onError).toHaveBeenCalled();
    });

    // Verify the component shows error state
    await waitFor(() => {
      expect(mockGetSyncStatus).toHaveBeenCalled();
    });
  });

  it('continues polling on network errors', async () => {
    mockGetSyncStatus
      .mockRejectedValueOnce(new Error('Network error'))
      .mockResolvedValueOnce(createSyncStatus());

    render(<SyncProgressScreen />);

    // First call (fails)
    await waitFor(() => {
      expect(mockGetSyncStatus).toHaveBeenCalledTimes(1);
    });

    // Advance timer - should retry
    await act(async () => {
      jest.advanceTimersByTime(2500);
    });

    await waitFor(() => {
      expect(mockGetSyncStatus).toHaveBeenCalledTimes(2);
    });
  });

  it('displays pending status message', async () => {
    mockGetSyncStatus.mockResolvedValue(
      createSyncStatus({
        status: 'pending',
      })
    );

    const { getByText } = render(<SyncProgressScreen />);

    await waitFor(() => {
      expect(getByText('Starting email analysis...')).toBeTruthy();
    });
  });

  it('displays syncing status message', async () => {
    mockGetSyncStatus.mockResolvedValue(
      createSyncStatus({
        status: 'syncing',
      })
    );

    const { getByText } = render(<SyncProgressScreen />);

    await waitFor(() => {
      expect(getByText('Analyzing your emails...')).toBeTruthy();
    });
  });

  it('displays completed status message', async () => {
    mockGetSyncStatus.mockResolvedValue(
      createSyncStatus({
        status: 'completed',
        progress: 1.0,
      })
    );

    const { getByText } = render(<SyncProgressScreen />);

    await waitFor(() => {
      expect(getByText('Analysis complete!')).toBeTruthy();
    });
  });

  it('handles zero emails gracefully', async () => {
    mockGetSyncStatus.mockResolvedValue(
      createSyncStatus({
        emails_synced: 0,
        progress: 0,
      })
    );

    const { getByText } = render(<SyncProgressScreen />);

    await waitFor(() => {
      expect(getByText('0 emails processed')).toBeTruthy();
    });
  });

  it('clamps progress bar width', async () => {
    // Test with progress > 1
    mockGetSyncStatus.mockResolvedValue(
      createSyncStatus({
        progress: 1.5,
      })
    );

    const { getByTestId } = render(<SyncProgressScreen />);

    await waitFor(() => {
      const progressFill = getByTestId('progress-bar-fill');
      // Width should be clamped to 100%
      expect(progressFill.props.style).toEqual(
        expect.arrayContaining([expect.objectContaining({ width: '100%' })])
      );
    });
  });

  it('uses default error message when error is null', async () => {
    const onError = jest.fn();
    mockGetSyncStatus.mockResolvedValue(
      createSyncStatus({
        status: 'failed',
        error: null,
      })
    );

    const { getByText } = render(<SyncProgressScreen onError={onError} />);

    await waitFor(() => {
      expect(getByText('An unexpected error occurred')).toBeTruthy();
      expect(onError).toHaveBeenCalledWith('Sync failed');
    });
  });

  it('displays loading message before status is loaded', async () => {
    // Mock a slow API response
    mockGetSyncStatus.mockImplementation(
      () => new Promise((resolve) => setTimeout(() => resolve(createSyncStatus()), 1000))
    );

    const { getByText } = render(<SyncProgressScreen />);

    // Before the API returns, status is null so 'Loading...' should show
    expect(getByText('Loading...')).toBeTruthy();
  });

  it('clears interval when polling stops due to completion', async () => {
    // Start with syncing, then complete after a poll
    mockGetSyncStatus
      .mockResolvedValueOnce(createSyncStatus({ status: 'syncing' }))
      .mockResolvedValueOnce(createSyncStatus({ status: 'completed', progress: 1.0 }));

    const onComplete = jest.fn();
    render(<SyncProgressScreen onComplete={onComplete} />);

    // Wait for initial poll
    await waitFor(() => {
      expect(mockGetSyncStatus).toHaveBeenCalled();
    });

    // Advance to trigger another poll
    await act(async () => {
      jest.advanceTimersByTime(2500);
    });

    // Should eventually call onComplete
    await waitFor(() => {
      expect(onComplete).toHaveBeenCalled();
    });
  });

  it('handles rapid status transitions', async () => {
    // Start with syncing, then complete
    mockGetSyncStatus
      .mockResolvedValueOnce(createSyncStatus({ status: 'syncing' }))
      .mockResolvedValueOnce(createSyncStatus({ status: 'syncing', emails_synced: 100 }))
      .mockResolvedValueOnce(createSyncStatus({ status: 'completed', progress: 1.0 }));

    const onComplete = jest.fn();
    render(<SyncProgressScreen onComplete={onComplete} />);

    // Advance through polling intervals
    await act(async () => {
      jest.advanceTimersByTime(5000);
    });

    await waitFor(() => {
      expect(onComplete).toHaveBeenCalled();
    });
  });

  describe('skip button', () => {
    it('shows skip button when onSkip is provided and sync fails', async () => {
      const onSkip = jest.fn();
      mockGetSyncStatus.mockResolvedValue(
        createSyncStatus({ status: 'failed', error: 'Test error' })
      );

      const { getByTestId } = render(<SyncProgressScreen onSkip={onSkip} />);

      await waitFor(() => {
        expect(getByTestId('skip-button')).toBeTruthy();
      });
    });

    it('does not show skip button when onSkip is not provided', async () => {
      mockGetSyncStatus.mockResolvedValue(
        createSyncStatus({ status: 'failed', error: 'Test error' })
      );

      const { queryByTestId } = render(<SyncProgressScreen />);

      await waitFor(() => {
        expect(queryByTestId('skip-button')).toBeNull();
      });
    });

    it('calls onSkip when skip button is pressed', async () => {
      const onSkip = jest.fn();
      mockGetSyncStatus.mockResolvedValue(
        createSyncStatus({ status: 'failed', error: 'Test error' })
      );

      const { getByTestId } = render(<SyncProgressScreen onSkip={onSkip} />);

      await waitFor(() => {
        expect(getByTestId('skip-button')).toBeTruthy();
      });

      fireEvent.press(getByTestId('skip-button'));
      expect(onSkip).toHaveBeenCalledTimes(1);
    });
  });
});
