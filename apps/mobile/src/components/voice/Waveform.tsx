/**
 * Waveform Component
 *
 * Animated waveform visualization that shows speech direction.
 * Animates from left (agent) or right (user) based on who is speaking.
 */

import React, { useEffect } from 'react';
import { View, StyleSheet, type ViewStyle } from 'react-native';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withRepeat,
  withTiming,
  withDelay,
  withSequence,
  Easing,
} from 'react-native-reanimated';
import { colors } from '../../theme';

export type SpeakingSource = 'agent' | 'user' | 'none';

export interface WaveformProps {
  /** Who is currently speaking */
  speakingSource: SpeakingSource;
  /** Number of bars to display */
  barCount?: number;
  /** Optional style overrides */
  style?: ViewStyle;
  /** Test ID for testing */
  testID?: string;
}

interface WaveformBarProps {
  index: number;
  isActive: boolean;
  direction: 'left' | 'right' | 'none';
  totalBars: number;
}

function WaveformBar({ index, isActive, direction, totalBars }: WaveformBarProps) {
  const height = useSharedValue(0.3);

  useEffect(() => {
    if (isActive) {
      // Calculate delay based on direction
      const delay =
        direction === 'left'
          ? index * 50 // Wave flows left to right (agent speaking)
          : direction === 'right'
            ? (totalBars - index - 1) * 50 // Wave flows right to left (user speaking)
            : 0;

      height.value = withDelay(
        delay,
        withRepeat(
          withSequence(
            withTiming(0.6 + Math.random() * 0.4, {
              duration: 150 + Math.random() * 100,
              easing: Easing.inOut(Easing.ease),
            }),
            withTiming(0.2 + Math.random() * 0.2, {
              duration: 150 + Math.random() * 100,
              easing: Easing.inOut(Easing.ease),
            })
          ),
          -1,
          false
        )
      );
    } else {
      height.value = withTiming(0.3, { duration: 300 });
    }
  }, [isActive, direction, index, totalBars, height]);

  const animatedStyle = useAnimatedStyle(() => ({
    height: `${height.value * 100}%`,
  }));

  const barColor =
    direction === 'left'
      ? colors.voice.agent
      : direction === 'right'
        ? colors.voice.user
        : colors.voice.waveform;

  return (
    <View style={styles.barContainer}>
      <Animated.View
        style={[
          styles.bar,
          animatedStyle,
          { backgroundColor: isActive ? barColor : colors.voice.waveform },
        ]}
      />
    </View>
  );
}

export function Waveform({
  speakingSource,
  barCount = 20,
  style,
  testID,
}: WaveformProps) {
  const isActive = speakingSource !== 'none';
  const direction =
    speakingSource === 'agent'
      ? 'left'
      : speakingSource === 'user'
        ? 'right'
        : 'none';

  return (
    <View style={[styles.container, style]} testID={testID}>
      {Array.from({ length: barCount }).map((_, index) => (
        <WaveformBar
          key={index}
          index={index}
          isActive={isActive}
          direction={direction}
          totalBars={barCount}
        />
      ))}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    height: 60,
    paddingHorizontal: 8,
    flex: 1,
  },
  barContainer: {
    flex: 1,
    height: '100%',
    justifyContent: 'center',
    alignItems: 'center',
    marginHorizontal: 2,
  },
  bar: {
    width: 4,
    borderRadius: 2,
    minHeight: 8,
  },
});
