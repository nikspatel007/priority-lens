/**
 * VoiceOrb Component
 *
 * Animated orb that represents either the agent (Lenso) or user
 * in voice mode. Pulsates when speaking.
 */

import React, { useEffect } from 'react';
import { View, StyleSheet, type ViewStyle } from 'react-native';
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withRepeat,
  withTiming,
  withSequence,
  Easing,
  interpolate,
} from 'react-native-reanimated';
import { colors } from '../../theme';

export type OrbType = 'agent' | 'user';

export interface VoiceOrbProps {
  /** Which entity this orb represents */
  type: OrbType;
  /** Whether this source is currently speaking */
  isSpeaking: boolean;
  /** Size of the orb in pixels */
  size?: number;
  /** Whether the orb should appear dimmed (e.g., mic off) */
  dimmed?: boolean;
  /** Optional style overrides */
  style?: ViewStyle;
  /** Test ID for testing */
  testID?: string;
}

export function VoiceOrb({
  type,
  isSpeaking,
  size = 80,
  dimmed = false,
  style,
  testID,
}: VoiceOrbProps) {
  const pulseScale = useSharedValue(1);
  const glowOpacity = useSharedValue(0.3);

  const orbColor = type === 'agent' ? colors.voice.agent : colors.voice.user;
  const glowColor = type === 'agent' ? colors.voice.agentGlow : colors.voice.userGlow;

  useEffect(() => {
    if (isSpeaking) {
      // Pulsating animation when speaking
      pulseScale.value = withRepeat(
        withSequence(
          withTiming(1.15, { duration: 400, easing: Easing.inOut(Easing.ease) }),
          withTiming(1, { duration: 400, easing: Easing.inOut(Easing.ease) })
        ),
        -1,
        false
      );
      glowOpacity.value = withRepeat(
        withSequence(
          withTiming(0.6, { duration: 400, easing: Easing.inOut(Easing.ease) }),
          withTiming(0.3, { duration: 400, easing: Easing.inOut(Easing.ease) })
        ),
        -1,
        false
      );
    } else {
      // Return to idle state
      pulseScale.value = withTiming(1, { duration: 300 });
      glowOpacity.value = withTiming(0.3, { duration: 300 });
    }
  }, [isSpeaking, pulseScale, glowOpacity]);

  const animatedOrbStyle = useAnimatedStyle(() => ({
    transform: [{ scale: pulseScale.value }],
  }));

  const animatedGlowStyle = useAnimatedStyle(() => ({
    opacity: glowOpacity.value,
    transform: [{ scale: interpolate(pulseScale.value, [1, 1.15], [1, 1.3]) }],
  }));

  return (
    <View
      style={[
        styles.container,
        { width: size * 1.5, height: size * 1.5, opacity: dimmed ? 0.4 : 1 },
        style,
      ]}
      testID={testID}
    >
      {/* Outer glow */}
      <Animated.View
        style={[
          styles.glow,
          animatedGlowStyle,
          {
            width: size * 1.4,
            height: size * 1.4,
            borderRadius: size * 0.7,
            backgroundColor: glowColor,
          },
        ]}
        testID={testID ? `${testID}-glow` : undefined}
      />
      {/* Inner orb */}
      <Animated.View
        style={[
          styles.orb,
          animatedOrbStyle,
          {
            width: size,
            height: size,
            borderRadius: size / 2,
            backgroundColor: orbColor,
          },
        ]}
        testID={testID ? `${testID}-orb` : undefined}
      />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    justifyContent: 'center',
    alignItems: 'center',
  },
  glow: {
    position: 'absolute',
  },
  orb: {
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 8,
  },
});
