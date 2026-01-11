/**
 * SDUI Card Component
 *
 * Container with card styling.
 */

import React from 'react';
import { View, ViewStyle } from 'react-native';
import { CardProps, LayoutProps } from '../types';
import { colors } from '../../theme';

interface SDUICardProps extends CardProps {
  layout?: LayoutProps;
  children?: React.ReactNode;
}

export function SDUICard({
  variant = 'default',
  backgroundColor,
  layout,
  children,
}: SDUICardProps) {
  const containerStyle: ViewStyle = {
    backgroundColor: backgroundColor || colors.surface,
    borderRadius: 12,
    overflow: 'hidden',
  };

  // Apply variant styles
  if (variant === 'elevated') {
    containerStyle.shadowColor = '#000';
    containerStyle.shadowOffset = { width: 0, height: 2 };
    containerStyle.shadowOpacity = 0.1;
    containerStyle.shadowRadius = 8;
    containerStyle.elevation = 4;
  } else if (variant === 'outlined') {
    containerStyle.borderWidth = 1;
    containerStyle.borderColor = colors.gray[200];
  }

  // Apply layout props
  if (layout) {
    if (layout.padding) {
      if (typeof layout.padding === 'number') {
        containerStyle.padding = layout.padding;
      } else {
        containerStyle.paddingTop = layout.padding[0];
        containerStyle.paddingRight = layout.padding[1];
        containerStyle.paddingBottom = layout.padding[2];
        containerStyle.paddingLeft = layout.padding[3];
      }
    } else {
      // Default padding for cards
      containerStyle.padding = 16;
    }
    if (layout.margin) {
      if (typeof layout.margin === 'number') {
        containerStyle.margin = layout.margin;
      } else {
        containerStyle.marginTop = layout.margin[0];
        containerStyle.marginRight = layout.margin[1];
        containerStyle.marginBottom = layout.margin[2];
        containerStyle.marginLeft = layout.margin[3];
      }
    }
    if (layout.width) {
      containerStyle.width = layout.width as number;
    }
    if (layout.maxWidth) {
      containerStyle.maxWidth = layout.maxWidth;
    }
    if (layout.height) {
      containerStyle.height = layout.height as number;
    }
  } else {
    containerStyle.padding = 16;
  }

  return (
    <View style={containerStyle} testID="sdui-card">
      {children}
    </View>
  );
}
