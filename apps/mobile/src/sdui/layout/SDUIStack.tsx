/**
 * SDUI Stack Component
 *
 * Flexbox-based layout component for horizontal/vertical stacking.
 */

import React from 'react';
import { View, ViewStyle } from 'react-native';
import { StackProps, LayoutProps } from '../types';

interface SDUIStackProps extends StackProps {
  layout?: LayoutProps;
  children?: React.ReactNode;
}

const alignMap = {
  start: 'flex-start',
  center: 'center',
  end: 'flex-end',
  stretch: 'stretch',
} as const;

const justifyMap = {
  start: 'flex-start',
  center: 'center',
  end: 'flex-end',
  between: 'space-between',
  around: 'space-around',
} as const;

export function SDUIStack({
  direction = 'vertical',
  gap = 0,
  align = 'stretch',
  justify = 'start',
  wrap = false,
  layout,
  children,
}: SDUIStackProps) {
  const containerStyle: ViewStyle = {
    flexDirection: direction === 'horizontal' ? 'row' : 'column',
    gap,
    alignItems: alignMap[align],
    justifyContent: justifyMap[justify],
    flexWrap: wrap ? 'wrap' : 'nowrap',
  };

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
    if (layout.flex) {
      containerStyle.flex = layout.flex;
    }
    if (layout.width) {
      containerStyle.width = layout.width as number;
    }
    if (layout.height) {
      containerStyle.height = layout.height as number;
    }
    if (layout.minHeight) {
      containerStyle.minHeight = layout.minHeight;
    }
    if (layout.maxWidth) {
      containerStyle.maxWidth = layout.maxWidth;
    }
  }

  return (
    <View style={containerStyle} testID="sdui-stack">
      {children}
    </View>
  );
}
