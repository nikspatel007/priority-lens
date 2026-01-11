/**
 * SDUI Action Item Component
 *
 * Checklist item with optional assignee.
 */

import React from 'react';
import { TouchableOpacity, StyleSheet, View } from 'react-native';
import { ActionItemProps } from '../types';
import { SDUIText } from '../primitives/SDUIText';
import { SDUIAvatar } from '../primitives/SDUIAvatar';
import { SDUIStack } from '../layout/SDUIStack';
import { colors } from '../../theme';

interface SDUIActionItemProps extends ActionItemProps {
  onToggle?: () => void;
}

export function SDUIActionItem({
  text,
  checked,
  assignee,
  onToggle,
}: SDUIActionItemProps) {
  return (
    <TouchableOpacity
      style={styles.container}
      onPress={onToggle}
      activeOpacity={0.7}
      testID="sdui-action-item"
    >
      <SDUIStack direction="horizontal" gap={12} align="center">
        {/* Checkbox */}
        <View style={[styles.checkbox, checked && styles.checkboxChecked]}>
          {checked && <View style={styles.checkmark} />}
        </View>

        {/* Text */}
        <SDUIText
          value={text}
          variant="body"
          color={checked ? colors.text.secondary : colors.text.primary}
          numberOfLines={2}
        />

        {/* Assignee */}
        {assignee && (
          <View style={styles.assignee}>
            <SDUIAvatar name={assignee} size={24} />
          </View>
        )}
      </SDUIStack>
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  container: {
    paddingVertical: 12,
    paddingHorizontal: 4,
  },
  checkbox: {
    width: 22,
    height: 22,
    borderRadius: 6,
    borderWidth: 2,
    borderColor: colors.gray[300],
    justifyContent: 'center',
    alignItems: 'center',
  },
  checkboxChecked: {
    backgroundColor: colors.primary[500],
    borderColor: colors.primary[500],
  },
  checkmark: {
    width: 8,
    height: 4,
    borderLeftWidth: 2,
    borderBottomWidth: 2,
    borderColor: '#FFFFFF',
    transform: [{ rotate: '-45deg' }, { translateY: -1 }],
  },
  assignee: {
    marginLeft: 'auto',
  },
});
