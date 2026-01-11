/**
 * AppHeader Component
 *
 * Top header with:
 * - View toggle (cards/list) on left
 * - App name in center
 * - Profile icon on right
 */

import React from 'react';
import { View, StyleSheet, TouchableOpacity, Text, Image, type ViewStyle } from 'react-native';
import { colors, spacing, typography, borderRadius, shadows } from '../../theme';

// Icons
const ICONS = {
  grid: require('../../../assets/icons/grid.png'),
  list: require('../../../assets/icons/list.png'),
  user: require('../../../assets/icons/user.png'),
};

export type ViewMode = 'cards' | 'list';

export interface AppHeaderProps {
  /** Current view mode */
  viewMode?: ViewMode;
  /** Callback when view mode changes */
  onViewModeChange?: (mode: ViewMode) => void;
  /** Callback when profile is pressed */
  onProfilePress?: () => void;
  /** Whether to show the view toggle */
  showViewToggle?: boolean;
  /** Safe area top inset */
  topInset?: number;
  /** Optional style overrides */
  style?: ViewStyle;
  /** Test ID for testing */
  testID?: string;
}

export function AppHeader({
  viewMode = 'cards',
  onViewModeChange,
  onProfilePress,
  showViewToggle = false,
  topInset = 0,
  style,
  testID,
}: AppHeaderProps) {
  return (
    <View
      style={[styles.container, { paddingTop: topInset }, style]}
      testID={testID}
    >
      <View style={styles.content}>
        {/* Left: View toggle or empty space */}
        {showViewToggle && onViewModeChange ? (
          <View style={styles.viewToggle} testID={testID ? `${testID}-view-toggle` : undefined}>
            <TouchableOpacity
              style={[styles.toggleBtn, viewMode === 'cards' && styles.toggleBtnActive]}
              onPress={() => onViewModeChange('cards')}
              testID={testID ? `${testID}-cards-btn` : undefined}
            >
              <Image
                source={ICONS.grid}
                style={[
                  styles.toggleIcon,
                  { tintColor: viewMode === 'cards' ? colors.primary[600] : colors.gray[400] },
                ]}
                resizeMode="contain"
              />
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.toggleBtn, viewMode === 'list' && styles.toggleBtnActive]}
              onPress={() => onViewModeChange('list')}
              testID={testID ? `${testID}-list-btn` : undefined}
            >
              <Image
                source={ICONS.list}
                style={[
                  styles.toggleIcon,
                  { tintColor: viewMode === 'list' ? colors.primary[600] : colors.gray[400] },
                ]}
                resizeMode="contain"
              />
            </TouchableOpacity>
          </View>
        ) : (
          <View style={styles.spacer} />
        )}

        {/* Center: App name */}
        <Text style={styles.appName} testID={testID ? `${testID}-title` : undefined}>
          Priority Lens
        </Text>

        {/* Right: Profile */}
        {onProfilePress ? (
          <TouchableOpacity
            style={styles.profileButton}
            onPress={onProfilePress}
            activeOpacity={0.7}
            testID={testID ? `${testID}-profile` : undefined}
          >
            <Image
              source={ICONS.user}
              style={styles.profileIcon}
              resizeMode="contain"
            />
          </TouchableOpacity>
        ) : (
          <View style={styles.spacer} />
        )}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    backgroundColor: colors.background,
  },
  content: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: spacing[4],
    paddingVertical: spacing[2],
  },
  spacer: {
    width: 72, // Match toggle width for centering
  },
  appName: {
    fontFamily: typography.serif.fontFamily,
    fontSize: typography.size.lg,
    fontWeight: typography.weight.semibold,
    color: colors.text.primary,
  },
  // View toggle
  viewToggle: {
    flexDirection: 'row',
    backgroundColor: colors.gray[100],
    borderRadius: borderRadius.lg,
    padding: 3,
    gap: 2,
  },
  toggleBtn: {
    padding: spacing[2],
    borderRadius: borderRadius.md,
  },
  toggleBtnActive: {
    backgroundColor: colors.background,
    ...shadows.sm,
  },
  toggleIcon: {
    width: 20,
    height: 20,
  },
  // Profile button
  profileButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    backgroundColor: colors.gray[100],
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: colors.gray[200],
  },
  profileIcon: {
    width: 22,
    height: 22,
    tintColor: colors.gray[600],
  },
});
