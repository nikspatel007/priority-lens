/**
 * TaskSkeleton
 *
 * Loading skeleton for the task list view
 */

import React from 'react';
import { View, StyleSheet } from 'react-native';
import { Skeleton } from './Skeleton';
import { colors, spacing, borderRadius } from '@/theme';

export interface TaskSkeletonProps {
  /** Number of task skeleton items to show */
  count?: number;
  /** Test ID for testing */
  testID?: string;
}

/**
 * TaskItemSkeleton renders a single task placeholder
 */
function TaskItemSkeleton({ testID }: { testID: string }): React.JSX.Element {
  return (
    <View style={styles.taskItem} testID={testID}>
      {/* Checkbox placeholder */}
      <Skeleton
        width={24}
        height={24}
        variant="circle"
        testID={`${testID}-checkbox`}
      />

      {/* Content */}
      <View style={styles.taskContent}>
        {/* Title */}
        <Skeleton
          width="85%"
          height={16}
          variant="text"
          testID={`${testID}-title`}
        />

        {/* Meta row (due date, priority) */}
        <View style={styles.taskMeta}>
          <Skeleton
            width={60}
            height={12}
            variant="text"
            testID={`${testID}-due`}
          />
          <Skeleton
            width={50}
            height={20}
            variant="rectangle"
            style={styles.priorityBadge}
            testID={`${testID}-priority`}
          />
        </View>
      </View>
    </View>
  );
}

/**
 * TaskSkeleton displays loading placeholders for the task list
 */
export function TaskSkeleton({
  count = 4,
  testID = 'task-skeleton',
}: TaskSkeletonProps): React.JSX.Element {
  return (
    <View testID={testID}>
      {Array.from({ length: count }).map((_, index) => (
        <TaskItemSkeleton key={index} testID={`${testID}-item-${index}`} />
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  taskItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    padding: spacing[4],
    backgroundColor: colors.backgrounds.primary,
    borderBottomWidth: 1,
    borderBottomColor: colors.border.light,
  },
  taskContent: {
    flex: 1,
    marginLeft: spacing[3],
  },
  taskMeta: {
    flexDirection: 'row',
    alignItems: 'center',
    marginTop: spacing[2],
    gap: spacing[2],
  },
  priorityBadge: {
    borderRadius: borderRadius.full,
  },
});
