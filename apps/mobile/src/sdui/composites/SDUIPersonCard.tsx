/**
 * SDUI Person Card Component
 *
 * Displays contact/person information.
 */

import React from 'react';
import { PersonCardProps } from '../types';
import { SDUIText } from '../primitives/SDUIText';
import { SDUIAvatar } from '../primitives/SDUIAvatar';
import { SDUICard } from '../layout/SDUICard';
import { SDUIStack } from '../layout/SDUIStack';

export function SDUIPersonCard({
  name,
  title,
  email,
  avatar,
  compact = false,
}: PersonCardProps) {
  if (compact) {
    return (
      <SDUIStack direction="horizontal" gap={12} align="center">
        <SDUIAvatar name={name} src={avatar} size={40} />
        <SDUIStack direction="vertical" gap={2}>
          <SDUIText value={name} variant="label" weight="medium" />
          {title && <SDUIText value={title} variant="caption" />}
        </SDUIStack>
      </SDUIStack>
    );
  }

  return (
    <SDUICard variant="outlined">
      <SDUIStack direction="horizontal" gap={16} align="center">
        <SDUIAvatar name={name} src={avatar} size={56} />
        <SDUIStack direction="vertical" gap={4} layout={{ flex: 1 }}>
          <SDUIText value={name} variant="heading" />
          {title && <SDUIText value={title} variant="body" />}
          {email && <SDUIText value={email} variant="caption" />}
        </SDUIStack>
      </SDUIStack>
    </SDUICard>
  );
}
