import React, { useEffect, useState, useCallback, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ActivityIndicator,
  TouchableOpacity,
} from 'react-native';
import { colors, typography, spacing, borderRadius } from '@/theme';
import { getSyncStatus } from '@/services/api';
import type { SyncStatusResponse, SyncStatus } from '@/types/api';

const POLL_INTERVAL_MS = 2500;

interface SyncProgressScreenProps {
  onComplete?: () => void;
  onError?: (error: string) => void;
}

/**
 * SyncProgressScreen shows email sync progress for first-time users
 *
 * This screen:
 * - Polls the backend every 2.5 seconds for sync status
 * - Shows a progress indicator and email count
 * - Auto-navigates when sync completes
 * - Shows error state if sync fails
 */
export function SyncProgressScreen({
  onComplete,
  onError,
}: SyncProgressScreenProps): React.JSX.Element {
  const [status, setStatus] = useState<SyncStatusResponse | null>(null);
  const [isPolling, setIsPolling] = useState(true);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const pollStatus = useCallback(async () => {
    try {
      const syncStatus = await getSyncStatus();
      setStatus(syncStatus);

      if (syncStatus.status === 'completed') {
        setIsPolling(false);
        onComplete?.();
      } else if (syncStatus.status === 'failed') {
        setIsPolling(false);
        onError?.(syncStatus.error || 'Sync failed');
      }
    } catch {
      // Continue polling on network errors
    }
  }, [onComplete, onError]);

  useEffect(() => {
    // Don't poll if not active
    if (!isPolling) {
      return;
    }

    // Initial poll
    pollStatus();

    // Set up interval
    const interval = setInterval(pollStatus, POLL_INTERVAL_MS);

    return () => {
      clearInterval(interval);
    };
  }, [isPolling, pollStatus]);

  const getStatusMessage = (syncStatus: SyncStatus | undefined): string => {
    switch (syncStatus) {
      case 'pending':
        return 'Starting email analysis...';
      case 'syncing':
        return 'Analyzing your emails...';
      case 'completed':
        return 'Analysis complete!';
      default:
        // Handles undefined (loading) and 'failed' (shows error UI instead)
        return 'Loading...';
    }
  };

  const renderProgressBar = (): React.JSX.Element => {
    const progress = status?.progress ?? 0;
    // Use percentage as DimensionValue (cast needed for template literal)
    const progressPercent =
      `${Math.min(Math.max(progress * 100, 0), 100)}%` as `${number}%`;

    return (
      <View style={styles.progressBarContainer} testID="progress-bar-container">
        <View
          style={[styles.progressBarFill, { width: progressPercent }]}
          testID="progress-bar-fill"
        />
      </View>
    );
  };

  const renderContent = (): React.JSX.Element => {
    if (status?.status === 'failed') {
      return (
        <View style={styles.errorContent} testID="error-content">
          <Text style={styles.errorIcon}>⚠️</Text>
          <Text style={styles.errorTitle}>Sync Failed</Text>
          <Text style={styles.errorMessage}>
            {status.error || 'An unexpected error occurred'}
          </Text>
          <TouchableOpacity
            style={styles.retryButton}
            onPress={() => {
              setIsPolling(true);
              pollStatus();
            }}
            testID="retry-button"
          >
            <Text style={styles.retryButtonText}>Try Again</Text>
          </TouchableOpacity>
        </View>
      );
    }

    return (
      <View style={styles.syncContent} testID="sync-content">
        <ActivityIndicator
          size="large"
          color={colors.primary[500]}
          testID="loading-indicator"
        />
        <Text style={styles.statusMessage}>
          {getStatusMessage(status?.status)}
        </Text>
        {renderProgressBar()}
        <Text style={styles.emailCount} testID="email-count">
          {status?.emails_synced ?? 0} emails processed
        </Text>
        <Text style={styles.helpText}>
          This usually takes a minute or two.{'\n'}
          You can leave this screen open.
        </Text>
      </View>
    );
  };

  return (
    <View style={styles.container} testID="sync-progress-screen">
      <View style={styles.content}>{renderContent()}</View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background.primary,
  },
  content: {
    flex: 1,
    justifyContent: 'center',
    paddingHorizontal: spacing[4],
  },
  syncContent: {
    alignItems: 'center',
  },
  statusMessage: {
    fontFamily: typography.fontFamily.sans,
    fontSize: typography.fontSize.xl,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
    textAlign: 'center',
    marginTop: spacing[4],
    marginBottom: spacing[3],
  },
  progressBarContainer: {
    width: '100%',
    height: 8,
    backgroundColor: colors.gray[200],
    borderRadius: borderRadius.full,
    overflow: 'hidden',
    marginVertical: spacing[2],
  },
  progressBarFill: {
    height: '100%',
    backgroundColor: colors.primary[500],
    borderRadius: borderRadius.full,
  },
  emailCount: {
    fontFamily: typography.fontFamily.sans,
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.medium,
    color: colors.primary[600],
    textAlign: 'center',
    marginTop: spacing[2],
  },
  helpText: {
    fontFamily: typography.fontFamily.sans,
    fontSize: typography.fontSize.sm,
    color: colors.text.secondary,
    textAlign: 'center',
    marginTop: spacing[4],
    lineHeight: typography.fontSize.sm * typography.lineHeight.relaxed,
  },
  errorContent: {
    alignItems: 'center',
  },
  errorIcon: {
    fontSize: 48,
    marginBottom: spacing[2],
  },
  errorTitle: {
    fontFamily: typography.fontFamily.sans,
    fontSize: typography.fontSize.xl,
    fontWeight: typography.fontWeight.bold,
    color: colors.error,
    textAlign: 'center',
    marginBottom: spacing[2],
  },
  errorMessage: {
    fontFamily: typography.fontFamily.sans,
    fontSize: typography.fontSize.base,
    color: colors.text.secondary,
    textAlign: 'center',
    marginBottom: spacing[4],
    paddingHorizontal: spacing[4],
  },
  retryButton: {
    height: 48,
    paddingHorizontal: spacing[6],
    backgroundColor: colors.primary[500],
    borderRadius: borderRadius.md,
    justifyContent: 'center',
    alignItems: 'center',
  },
  retryButtonText: {
    fontFamily: typography.fontFamily.sans,
    fontSize: typography.fontSize.base,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.inverse,
  },
});
