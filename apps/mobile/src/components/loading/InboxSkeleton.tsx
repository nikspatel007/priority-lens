/**
 * InboxSkeleton
 *
 * Loading skeleton for the inbox/email list view
 */

import React from 'react';
import { View, StyleSheet } from 'react-native';
import { Skeleton } from './Skeleton';
import { colors, spacing, borderRadius } from '@/theme';

export interface InboxSkeletonProps {
  /** Number of email skeleton items to show */
  count?: number;
  /** Test ID for testing */
  testID?: string;
}

/**
 * EmailItemSkeleton renders a single email placeholder
 */
function EmailItemSkeleton({ testID }: { testID: string }): React.JSX.Element {
  return (
    <View style={styles.emailItem} testID={testID}>
      {/* Avatar */}
      <Skeleton
        width={40}
        height={40}
        variant="circle"
        testID={`${testID}-avatar`}
      />

      {/* Content */}
      <View style={styles.emailContent}>
        {/* Sender and time row */}
        <View style={styles.emailHeader}>
          <Skeleton
            width={120}
            height={16}
            variant="text"
            testID={`${testID}-sender`}
          />
          <Skeleton
            width={40}
            height={12}
            variant="text"
            testID={`${testID}-time`}
          />
        </View>

        {/* Subject */}
        <Skeleton
          width="90%"
          height={14}
          variant="text"
          style={styles.subject}
          testID={`${testID}-subject`}
        />

        {/* Preview */}
        <Skeleton
          width="70%"
          height={12}
          variant="text"
          style={styles.preview}
          testID={`${testID}-preview`}
        />
      </View>
    </View>
  );
}

/**
 * InboxSkeleton displays loading placeholders for the email inbox
 */
export function InboxSkeleton({
  count = 5,
  testID = 'inbox-skeleton',
}: InboxSkeletonProps): React.JSX.Element {
  return (
    <View testID={testID}>
      {Array.from({ length: count }).map((_, index) => (
        <EmailItemSkeleton key={index} testID={`${testID}-item-${index}`} />
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  emailItem: {
    flexDirection: 'row',
    padding: spacing[4],
    backgroundColor: colors.backgrounds.primary,
    borderBottomWidth: 1,
    borderBottomColor: colors.border.light,
  },
  emailContent: {
    flex: 1,
    marginLeft: spacing[3],
  },
  emailHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: spacing[1],
  },
  subject: {
    marginBottom: spacing[1],
  },
  preview: {
    marginTop: spacing[1],
  },
});
