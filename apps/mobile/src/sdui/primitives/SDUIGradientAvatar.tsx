/**
 * SDUI Gradient Avatar Component
 *
 * Avatar with gradient background and initials.
 * Falls back to solid color if gradient fails.
 */

import React from 'react';
import {
  View,
  Text,
  Image,
  StyleSheet,
  ViewStyle,
  ImageStyle,
} from 'react-native';
import { colors, typography, shadows } from '../../theme';

export interface GradientAvatarProps {
  initials?: string;
  imageUrl?: string;
  gradientColors?: [string, string];
  size?: number;
  style?: ViewStyle | ImageStyle;
}

export function SDUIGradientAvatar({
  initials = '?',
  imageUrl,
  gradientColors,
  size = 56,
  style,
}: GradientAvatarProps) {
  // Get background color - use first gradient color or generate from initials
  const backgroundColor = gradientColors?.[0] || getColorFromInitials(initials);

  if (imageUrl) {
    return (
      <Image
        source={{ uri: imageUrl }}
        style={[
          {
            width: size,
            height: size,
            borderRadius: size / 2,
          },
          style as ImageStyle,
        ]}
        testID="sdui-gradient-avatar-image"
      />
    );
  }

  return (
    <View
      style={[
        styles.avatar,
        {
          width: size,
          height: size,
          borderRadius: size / 2,
          backgroundColor,
        },
        style,
      ]}
      testID="sdui-gradient-avatar"
    >
      <Text style={[styles.initials, { fontSize: size * 0.38 }]}>
        {initials}
      </Text>
    </View>
  );
}

// Generate consistent color from initials
function getColorFromInitials(initials: string): string {
  const avatarColors = [
    '#667EEA', // Indigo-purple
    '#764BA2', // Purple
    '#F093FB', // Pink
    '#F5576C', // Rose
    '#4FACFE', // Blue
    '#00F2FE', // Cyan
    '#43E97B', // Green
    '#38F9D7', // Teal
    '#FA709A', // Salmon
    '#FEE140', // Yellow
  ];

  let hash = 0;
  for (let i = 0; i < initials.length; i++) {
    hash = initials.charCodeAt(i) + ((hash << 5) - hash);
  }

  return avatarColors[Math.abs(hash) % avatarColors.length];
}

const styles = StyleSheet.create({
  avatar: {
    alignItems: 'center',
    justifyContent: 'center',
    ...shadows.md,
  },
  initials: {
    fontFamily: typography.serif.fontFamily,
    fontWeight: typography.weight.semibold,
    color: colors.text.inverse,
    textAlign: 'center',
  },
});
