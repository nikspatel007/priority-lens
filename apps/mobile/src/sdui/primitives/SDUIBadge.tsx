/**
 * SDUI Badge Component
 *
 * Status indicator badges.
 */

import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { BadgeProps } from '../types';

const variantStyles = {
  default: {
    backgroundColor: '#E5E7EB',
    color: '#374151',
  },
  success: {
    backgroundColor: '#D1FAE5',
    color: '#065F46',
  },
  warning: {
    backgroundColor: '#FEF3C7',
    color: '#92400E',
  },
  error: {
    backgroundColor: '#FEE2E2',
    color: '#991B1B',
  },
  info: {
    backgroundColor: '#DBEAFE',
    color: '#1E40AF',
  },
};

export function SDUIBadge({ value, variant = 'default' }: BadgeProps) {
  const style = variantStyles[variant];

  return (
    <View
      style={[styles.container, { backgroundColor: style.backgroundColor }]}
      testID="sdui-badge"
    >
      <Text style={[styles.text, { color: style.color }]}>{value}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
    alignSelf: 'flex-start',
  },
  text: {
    fontSize: 12,
    fontWeight: '500',
  },
});
