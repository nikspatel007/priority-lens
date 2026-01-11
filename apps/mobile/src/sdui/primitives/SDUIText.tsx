/**
 * SDUI Text Component
 *
 * Typography primitive for SDUI rendering.
 */

import React from 'react';
import { Text, StyleSheet } from 'react-native';
import { TextProps } from '../types';
import { colors, typography } from '../../theme';

const variantStyles = {
  body: {
    fontSize: 16,
    lineHeight: 24,
    fontFamily: typography.sans.fontFamily,
  },
  heading: {
    fontSize: 20,
    lineHeight: 28,
    fontFamily: typography.serif.fontFamily,
    fontWeight: '600' as const,
  },
  title: {
    fontSize: 28,
    lineHeight: 34,
    fontFamily: typography.serif.fontFamily,
    fontWeight: '700' as const,
  },
  caption: {
    fontSize: 13,
    lineHeight: 18,
    fontFamily: typography.sans.fontFamily,
    color: colors.text.secondary,
  },
  label: {
    fontSize: 14,
    lineHeight: 20,
    fontFamily: typography.sans.fontFamily,
    fontWeight: '500' as const,
  },
};

const weightStyles = {
  normal: { fontWeight: '400' as const },
  medium: { fontWeight: '500' as const },
  semibold: { fontWeight: '600' as const },
  bold: { fontWeight: '700' as const },
};

export function SDUIText({
  value,
  variant = 'body',
  color,
  weight,
  align = 'left',
  numberOfLines,
}: TextProps) {
  return (
    <Text
      style={[
        styles.base,
        variantStyles[variant],
        weight && weightStyles[weight],
        color && { color },
        { textAlign: align },
      ]}
      numberOfLines={numberOfLines}
    >
      {value}
    </Text>
  );
}

const styles = StyleSheet.create({
  base: {
    color: colors.text.primary,
  },
});
