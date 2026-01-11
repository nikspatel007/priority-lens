/**
 * SDUI View Toggle
 *
 * A segmented control for switching between Cards and List view.
 */

import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity } from 'react-native';
import { colors, typography, spacing, borderRadius } from '../../theme';

export type ViewMode = 'cards' | 'list';

interface SDUIViewToggleProps {
  mode: ViewMode;
  onChange: (mode: ViewMode) => void;
}

export function SDUIViewToggle({ mode, onChange }: SDUIViewToggleProps) {
  const isCards = mode === 'cards';

  return (
    <View style={styles.container} testID="sdui-view-toggle">
      <TouchableOpacity
        style={[styles.option, isCards && styles.optionActive]}
        onPress={() => onChange('cards')}
        activeOpacity={0.7}
        testID="sdui-view-toggle-cards"
      >
        <CardIcon active={isCards} />
        <Text style={[styles.optionText, isCards && styles.optionTextActive]}>
          Cards
        </Text>
      </TouchableOpacity>

      <TouchableOpacity
        style={[styles.option, !isCards && styles.optionActive]}
        onPress={() => onChange('list')}
        activeOpacity={0.7}
        testID="sdui-view-toggle-list"
      >
        <ListIcon active={!isCards} />
        <Text style={[styles.optionText, !isCards && styles.optionTextActive]}>
          List
        </Text>
      </TouchableOpacity>
    </View>
  );
}

// Card icon component
function CardIcon({ active }: { active: boolean }) {
  return (
    <View style={styles.iconContainer}>
      <View
        style={[
          styles.cardIconBox,
          { borderColor: active ? colors.gray[900] : colors.gray[400] },
        ]}
      />
    </View>
  );
}

// List icon component
function ListIcon({ active }: { active: boolean }) {
  const lineColor = active ? colors.gray[900] : colors.gray[400];
  return (
    <View style={styles.iconContainer}>
      <View style={[styles.listLine, { backgroundColor: lineColor }]} />
      <View style={[styles.listLine, { backgroundColor: lineColor }]} />
      <View style={[styles.listLine, { backgroundColor: lineColor }]} />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    backgroundColor: colors.gray[100],
    borderRadius: borderRadius.lg,
    padding: 3,
  },
  option: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: spacing[3],
    paddingVertical: spacing[2],
    borderRadius: borderRadius.md,
    gap: spacing[1],
  },
  optionActive: {
    backgroundColor: colors.background,
    shadowColor: colors.gray[900],
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
    elevation: 2,
  },
  optionText: {
    fontFamily: typography.sans.fontFamily,
    fontSize: typography.size.xs,
    fontWeight: typography.weight.medium,
    color: colors.gray[500],
  },
  optionTextActive: {
    color: colors.gray[900],
  },
  iconContainer: {
    width: 14,
    height: 14,
    justifyContent: 'center',
    alignItems: 'center',
  },
  cardIconBox: {
    width: 12,
    height: 12,
    borderWidth: 1.5,
    borderRadius: 3,
  },
  listLine: {
    width: 12,
    height: 2,
    borderRadius: 1,
    marginVertical: 1,
  },
});
