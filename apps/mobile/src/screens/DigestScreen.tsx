import React, { useEffect, useState, useCallback } from 'react';
import {
  View,
  Text,
  StyleSheet,
  ScrollView,
  RefreshControl,
  TouchableOpacity,
  ActivityIndicator,
} from 'react-native';
import { colors, typography, spacing, borderRadius } from '@/theme';
import { getDigest } from '@/services/api';
import type {
  DigestResponse,
  DigestTodoItem,
  DigestTopicItem,
  UrgencyLevel,
} from '@/types/api';

interface DigestScreenProps {
  /** Navigate to conversation screen */
  onConversationPress?: () => void;
  /** Handle action on a digest item */
  onAction?: (actionType: string, itemId: string) => void;
}

/**
 * DigestScreen - Smart AI Inbox View
 *
 * Displays personalized daily digest with:
 * - Time-appropriate greeting
 * - Suggested to-dos (actionable items)
 * - Topics to catch up on (grouped conversations)
 *
 * This is the "AI Inbox" experience for Priority Lens Phase 6.
 */
export function DigestScreen({
  onConversationPress,
  onAction,
}: DigestScreenProps): React.JSX.Element {
  const [digest, setDigest] = useState<DigestResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchDigest = useCallback(async () => {
    try {
      const data = await getDigest(5, 5);
      setDigest(data);
      setError(null);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to load digest';
      setError(message);
    }
  }, []);

  useEffect(() => {
    const loadDigest = async () => {
      setIsLoading(true);
      await fetchDigest();
      setIsLoading(false);
    };
    loadDigest();
  }, [fetchDigest]);

  const handleRefresh = useCallback(async () => {
    setIsRefreshing(true);
    await fetchDigest();
    setIsRefreshing(false);
  }, [fetchDigest]);

  const handleAction = useCallback(
    (actionType: string, itemId: string) => {
      onAction?.(actionType, itemId);
    },
    [onAction]
  );

  const getUrgencyColor = (urgency: UrgencyLevel): string => {
    switch (urgency) {
      case 'high':
        return colors.error;
      case 'medium':
        return colors.warning;
      case 'low':
        return colors.success;
      default:
        return colors.gray[500];
    }
  };

  const renderTodoItem = (item: DigestTodoItem): React.JSX.Element => (
    <TouchableOpacity
      key={item.id}
      style={styles.todoCard}
      onPress={() => handleAction('open', item.id)}
      testID={`todo-item-${item.id}`}
    >
      <View style={styles.todoHeader}>
        <View
          style={[
            styles.urgencyDot,
            { backgroundColor: getUrgencyColor(item.urgency) },
          ]}
        />
        <Text style={styles.todoTitle} numberOfLines={2}>
          {item.title}
        </Text>
      </View>
      <Text style={styles.todoSource} numberOfLines={1}>
        {item.source}
      </Text>
      {item.due && (
        <Text
          style={[
            styles.todoDue,
            item.due === 'Overdue' && styles.todoDueOverdue,
          ]}
        >
          {item.due}
        </Text>
      )}
      <View style={styles.todoActions}>
        {item.actions.slice(0, 2).map((action) => (
          <TouchableOpacity
            key={action.id}
            style={[
              styles.actionButton,
              action.type === 'reply' && styles.actionButtonPrimary,
            ]}
            onPress={() => handleAction(action.type, item.id)}
            testID={`action-${action.id}`}
          >
            <Text
              style={[
                styles.actionButtonText,
                action.type === 'reply' && styles.actionButtonTextPrimary,
              ]}
            >
              {action.label}
            </Text>
          </TouchableOpacity>
        ))}
      </View>
    </TouchableOpacity>
  );

  const renderTopicItem = (item: DigestTopicItem): React.JSX.Element => (
    <TouchableOpacity
      key={item.id}
      style={styles.topicCard}
      onPress={() => handleAction('open_topic', item.id)}
      testID={`topic-item-${item.id}`}
    >
      <View style={styles.topicHeader}>
        <Text style={styles.topicTitle} numberOfLines={2}>
          {item.title}
        </Text>
        <View style={styles.topicBadge}>
          <Text style={styles.topicBadgeText}>{item.email_count}</Text>
        </View>
      </View>
      <Text style={styles.topicParticipants} numberOfLines={1}>
        {item.participants.join(', ')}
      </Text>
      <Text style={styles.topicActivity}>{item.last_activity}</Text>
    </TouchableOpacity>
  );

  if (isLoading) {
    return (
      <View style={styles.loadingContainer} testID="loading-container">
        <ActivityIndicator size="large" color={colors.primary[500]} />
        <Text style={styles.loadingText}>Loading your digest...</Text>
      </View>
    );
  }

  if (error) {
    return (
      <View style={styles.errorContainer} testID="error-container">
        <Text style={styles.errorIcon}>!</Text>
        <Text style={styles.errorText}>{error}</Text>
        <TouchableOpacity
          style={styles.retryButton}
          onPress={handleRefresh}
          testID="retry-button"
        >
          <Text style={styles.retryButtonText}>Try Again</Text>
        </TouchableOpacity>
      </View>
    );
  }

  return (
    <ScrollView
      style={styles.container}
      contentContainerStyle={styles.contentContainer}
      refreshControl={
        <RefreshControl
          refreshing={isRefreshing}
          onRefresh={handleRefresh}
          tintColor={colors.primary[500]}
        />
      }
      testID="digest-screen"
    >
      {/* Header/Greeting */}
      <View style={styles.header}>
        <Text style={styles.greeting} testID="greeting">
          {digest?.greeting}
        </Text>
        <Text style={styles.subtitle} testID="subtitle">
          {digest?.subtitle}
        </Text>
      </View>

      {/* Suggested To-Dos Section */}
      {digest?.suggested_todos && digest.suggested_todos.length > 0 && (
        <View style={styles.section}>
          <View style={styles.sectionHeader}>
            <Text style={styles.sectionTitle}>Suggested To-Dos</Text>
            <Text style={styles.sectionIcon}>✓</Text>
          </View>
          {digest.suggested_todos.map(renderTodoItem)}
        </View>
      )}

      {/* Topics to Catch Up On Section */}
      {digest?.topics_to_catchup && digest.topics_to_catchup.length > 0 && (
        <View style={styles.section}>
          <View style={styles.sectionHeader}>
            <Text style={styles.sectionTitle}>Topics to Catch Up On</Text>
            <Text style={styles.sectionIcon}>📁</Text>
          </View>
          {digest.topics_to_catchup.map(renderTopicItem)}
        </View>
      )}

      {/* Empty State */}
      {(!digest?.suggested_todos || digest.suggested_todos.length === 0) &&
        (!digest?.topics_to_catchup || digest.topics_to_catchup.length === 0) && (
          <View style={styles.emptyState} testID="empty-state">
            <Text style={styles.emptyStateIcon}>🎉</Text>
            <Text style={styles.emptyStateText}>All caught up!</Text>
            <Text style={styles.emptyStateSubtext}>
              Nothing needs your attention right now.
            </Text>
          </View>
        )}

      {/* Voice Assistant Button */}
      {onConversationPress && (
        <TouchableOpacity
          style={styles.voiceButton}
          onPress={onConversationPress}
          testID="voice-button"
        >
          <Text style={styles.voiceButtonIcon}>🎤</Text>
          <Text style={styles.voiceButtonText}>Ask Lenso</Text>
        </TouchableOpacity>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.backgrounds.primary,
  },
  contentContainer: {
    padding: spacing[4],
    paddingBottom: spacing[8],
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: colors.backgrounds.primary,
  },
  loadingText: {
    fontFamily: typography.fontFamily.sans,
    fontSize: typography.fontSize.base,
    color: colors.text.secondary,
    marginTop: spacing[3],
  },
  errorContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: colors.backgrounds.primary,
    padding: spacing[4],
  },
  errorIcon: {
    fontSize: 48,
    color: colors.error,
    marginBottom: spacing[2],
  },
  errorText: {
    fontFamily: typography.fontFamily.sans,
    fontSize: typography.fontSize.base,
    color: colors.text.secondary,
    textAlign: 'center',
    marginBottom: spacing[4],
  },
  retryButton: {
    paddingHorizontal: spacing[6],
    paddingVertical: spacing[3],
    backgroundColor: colors.primary[500],
    borderRadius: borderRadius.md,
  },
  retryButtonText: {
    fontFamily: typography.fontFamily.sans,
    fontSize: typography.fontSize.base,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.inverse,
  },
  header: {
    marginBottom: spacing[6],
  },
  greeting: {
    fontFamily: typography.fontFamily.serif,
    fontSize: typography.fontSize['2xl'],
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
    marginBottom: spacing[1],
  },
  subtitle: {
    fontFamily: typography.fontFamily.sans,
    fontSize: typography.fontSize.base,
    color: colors.text.secondary,
  },
  section: {
    marginBottom: spacing[6],
  },
  sectionHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: spacing[3],
  },
  sectionTitle: {
    fontFamily: typography.fontFamily.sans,
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
    flex: 1,
  },
  sectionIcon: {
    fontSize: typography.fontSize.lg,
    marginLeft: spacing[2],
  },
  todoCard: {
    backgroundColor: colors.backgrounds.card,
    borderRadius: borderRadius.lg,
    padding: spacing[4],
    marginBottom: spacing[3],
    shadowColor: colors.gray[900],
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 2,
    elevation: 1,
  },
  todoHeader: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: spacing[2],
  },
  urgencyDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginTop: 6,
    marginRight: spacing[2],
  },
  todoTitle: {
    flex: 1,
    fontFamily: typography.fontFamily.sans,
    fontSize: typography.fontSize.base,
    fontWeight: typography.fontWeight.medium,
    color: colors.text.primary,
    lineHeight: typography.fontSize.base * typography.lineHeight.normal,
  },
  todoSource: {
    fontFamily: typography.fontFamily.sans,
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
    marginBottom: spacing[1],
    marginLeft: spacing[4],
  },
  todoDue: {
    fontFamily: typography.fontFamily.sans,
    fontSize: typography.fontSize.sm,
    color: colors.warning,
    marginLeft: spacing[4],
    marginBottom: spacing[2],
  },
  todoDueOverdue: {
    color: colors.error,
  },
  todoActions: {
    flexDirection: 'row',
    marginTop: spacing[2],
    marginLeft: spacing[4],
  },
  actionButton: {
    paddingHorizontal: spacing[3],
    paddingVertical: spacing[2],
    backgroundColor: colors.gray[100],
    borderRadius: borderRadius.sm,
    marginRight: spacing[2],
  },
  actionButtonPrimary: {
    backgroundColor: colors.primary[500],
  },
  actionButtonText: {
    fontFamily: typography.fontFamily.sans,
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.medium,
    color: colors.text.secondary,
  },
  actionButtonTextPrimary: {
    color: colors.text.inverse,
  },
  topicCard: {
    backgroundColor: colors.backgrounds.card,
    borderRadius: borderRadius.lg,
    padding: spacing[4],
    marginBottom: spacing[3],
    shadowColor: colors.gray[900],
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 2,
    elevation: 1,
  },
  topicHeader: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: spacing[2],
  },
  topicTitle: {
    flex: 1,
    fontFamily: typography.fontFamily.sans,
    fontSize: typography.fontSize.base,
    fontWeight: typography.fontWeight.medium,
    color: colors.text.primary,
    lineHeight: typography.fontSize.base * typography.lineHeight.normal,
  },
  topicBadge: {
    backgroundColor: colors.primary[100],
    borderRadius: borderRadius.full,
    paddingHorizontal: spacing[2],
    paddingVertical: spacing[1],
    marginLeft: spacing[2],
  },
  topicBadgeText: {
    fontFamily: typography.fontFamily.sans,
    fontSize: typography.fontSize.xs,
    fontWeight: typography.fontWeight.semibold,
    color: colors.primary[700],
  },
  topicParticipants: {
    fontFamily: typography.fontFamily.sans,
    fontSize: typography.fontSize.sm,
    color: colors.text.tertiary,
    marginBottom: spacing[1],
  },
  topicActivity: {
    fontFamily: typography.fontFamily.sans,
    fontSize: typography.fontSize.xs,
    color: colors.text.tertiary,
  },
  emptyState: {
    alignItems: 'center',
    paddingVertical: spacing[8],
  },
  emptyStateIcon: {
    fontSize: 48,
    marginBottom: spacing[3],
  },
  emptyStateText: {
    fontFamily: typography.fontFamily.serif,
    fontSize: typography.fontSize.xl,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
    marginBottom: spacing[1],
  },
  emptyStateSubtext: {
    fontFamily: typography.fontFamily.sans,
    fontSize: typography.fontSize.base,
    color: colors.text.secondary,
  },
  voiceButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: colors.primary[500],
    borderRadius: borderRadius.full,
    paddingVertical: spacing[4],
    paddingHorizontal: spacing[6],
    marginTop: spacing[4],
    shadowColor: colors.primary[900],
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.2,
    shadowRadius: 8,
    elevation: 4,
  },
  voiceButtonIcon: {
    fontSize: typography.fontSize.lg,
    marginRight: spacing[2],
  },
  voiceButtonText: {
    fontFamily: typography.fontFamily.sans,
    fontSize: typography.fontSize.base,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.inverse,
  },
});
