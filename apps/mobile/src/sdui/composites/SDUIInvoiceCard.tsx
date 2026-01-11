/**
 * SDUI Invoice Card Component
 *
 * Displays invoice/payment information.
 */

import React from 'react';
import { View, StyleSheet } from 'react-native';
import { InvoiceCardProps } from '../types';
import { SDUIText } from '../primitives/SDUIText';
import { SDUIBadge } from '../primitives/SDUIBadge';
import { SDUICard } from '../layout/SDUICard';
import { SDUIStack } from '../layout/SDUIStack';
import { colors } from '../../theme';

export function SDUIInvoiceCard({
  vendor,
  description,
  amount,
  dueDate,
  status = 'pending',
}: InvoiceCardProps) {
  const statusVariant = {
    pending: 'warning',
    paid: 'success',
    overdue: 'error',
  }[status] as 'warning' | 'success' | 'error';

  const statusLabel = {
    pending: 'Pending',
    paid: 'Paid',
    overdue: 'Overdue',
  }[status];

  return (
    <SDUICard variant="elevated">
      <SDUIStack direction="vertical" gap={12}>
        {/* Header */}
        <SDUIStack direction="horizontal" justify="between" align="start">
          <SDUIStack direction="vertical" gap={4}>
            <SDUIText value={vendor} variant="heading" />
            {description && <SDUIText value={description} variant="caption" />}
          </SDUIStack>
          <SDUIBadge value={statusLabel} variant={statusVariant} />
        </SDUIStack>

        {/* Details */}
        <View style={styles.divider} />

        <SDUIStack direction="horizontal" justify="between">
          <SDUIStack direction="vertical" gap={2}>
            <SDUIText value="Amount" variant="caption" />
            <SDUIText value={amount} variant="title" weight="bold" />
          </SDUIStack>

          {dueDate && (
            <SDUIStack direction="vertical" gap={2} align="end">
              <SDUIText value="Due" variant="caption" />
              <SDUIText value={dueDate} variant="label" />
            </SDUIStack>
          )}
        </SDUIStack>
      </SDUIStack>
    </SDUICard>
  );
}

const styles = StyleSheet.create({
  divider: {
    height: 1,
    backgroundColor: colors.gray[200],
    marginVertical: 4,
  },
});
