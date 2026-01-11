/**
 * Skeleton
 *
 * Base skeleton component with animated shimmer effect
 * for loading state placeholders.
 */

import React, { useEffect, useRef } from 'react';
import { View, Animated, StyleSheet, ViewStyle, DimensionValue } from 'react-native';
import { colors, borderRadius } from '@/theme';

export interface SkeletonProps {
  /** Width of the skeleton */
  width?: DimensionValue;
  /** Height of the skeleton */
  height?: number;
  /** Border radius variant */
  variant?: 'rectangle' | 'circle' | 'text';
  /** Custom style */
  style?: ViewStyle;
  /** Test ID for testing */
  testID?: string;
}

/**
 * Skeleton displays a shimmering placeholder while content loads
 */
export function Skeleton({
  width = '100%',
  height = 16,
  variant = 'rectangle',
  style,
  testID = 'skeleton',
}: SkeletonProps): React.JSX.Element {
  const animatedValue = useRef(new Animated.Value(0)).current;

  useEffect(() => {
    const animation = Animated.loop(
      Animated.sequence([
        Animated.timing(animatedValue, {
          toValue: 1,
          duration: 1000,
          useNativeDriver: true,
        }),
        Animated.timing(animatedValue, {
          toValue: 0,
          duration: 1000,
          useNativeDriver: true,
        }),
      ])
    );

    animation.start();

    return () => {
      animation.stop();
    };
  }, [animatedValue]);

  const opacity = animatedValue.interpolate({
    inputRange: [0, 1],
    outputRange: [0.3, 0.7],
  });

  const getBorderRadius = (): number => {
    switch (variant) {
      case 'circle':
        // istanbul ignore next: height is always a number in our usage
        return typeof height === 'number' ? height / 2 : 0;
      case 'text':
        return borderRadius.sm;
      default:
        return borderRadius.md;
    }
  };

  return (
    <Animated.View
      testID={testID}
      style={[
        styles.skeleton,
        {
          width,
          height,
          borderRadius: getBorderRadius(),
          opacity,
        },
        style,
      ]}
    />
  );
}

/**
 * SkeletonGroup renders multiple skeleton elements
 */
export interface SkeletonGroupProps {
  /** Number of skeleton items to render */
  count?: number;
  /** Spacing between items */
  spacing?: number;
  /** Props to pass to each skeleton */
  skeletonProps?: Omit<SkeletonProps, 'testID'>;
  /** Test ID for testing */
  testID?: string;
}

export function SkeletonGroup({
  count = 3,
  spacing = 8,
  skeletonProps = {},
  testID = 'skeleton-group',
}: SkeletonGroupProps): React.JSX.Element {
  return (
    <View testID={testID}>
      {Array.from({ length: count }).map((_, index) => (
        <View key={index} style={{ marginBottom: index < count - 1 ? spacing : 0 }}>
          <Skeleton {...skeletonProps} testID={`${testID}-item-${index}`} />
        </View>
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  skeleton: {
    backgroundColor: colors.gray[200],
  },
});
