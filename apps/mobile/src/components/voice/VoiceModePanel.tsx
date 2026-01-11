/**
 * VoiceModePanel Component
 *
 * The bottom panel shown in voice mode containing:
 * - Agent (Lenso) orb on the left
 * - Waveform visualization in the center
 * - User orb on the right
 * - Toggle to switch to text mode
 */

import React from 'react';
import { View, StyleSheet, TouchableOpacity, Text, Image, type ViewStyle } from 'react-native';
import { VoiceOrb } from './VoiceOrb';
import { Waveform, type SpeakingSource } from './Waveform';
import { colors, spacing, typography, borderRadius, shadows } from '../../theme';

// Icons
const ICONS = {
  keyboard: require('../../../assets/icons/keyboard.png'),
  mic: require('../../../assets/icons/mic.png'),
  micOff: require('../../../assets/icons/mic-off.png'),
};

export interface VoiceModePanelProps {
  /** Current speaking source */
  speakingSource: SpeakingSource;
  /** Callback when mode toggle is pressed */
  onToggleMode: () => void;
  /** Whether connected to voice room */
  isConnected: boolean;
  /** Whether currently connecting */
  isConnecting: boolean;
  /** Whether microphone is enabled */
  isMicrophoneEnabled: boolean;
  /** Callback when mic toggle is pressed */
  onToggleMicrophone: () => void;
  /** Optional style overrides */
  style?: ViewStyle;
  /** Test ID for testing */
  testID?: string;
}

export function VoiceModePanel({
  speakingSource,
  onToggleMode,
  isConnected,
  isConnecting,
  isMicrophoneEnabled,
  onToggleMicrophone,
  style,
  testID,
}: VoiceModePanelProps) {
  const statusText = isConnecting
    ? 'Connecting to Lenso...'
    : !isConnected
      ? 'Not connected'
      : isMicrophoneEnabled
        ? 'Tap to end your turn'
        : 'Tap to speak';

  return (
    <View style={[styles.container, style]} testID={testID}>
      {/* Connection status */}
      <Text style={styles.statusText} testID={testID ? `${testID}-status` : undefined}>
        {statusText}
      </Text>

      {/* Voice visualization row - tappable to toggle mic */}
      <TouchableOpacity
        style={styles.voiceRow}
        onPress={onToggleMicrophone}
        activeOpacity={0.7}
        disabled={!isConnected}
        testID={testID ? `${testID}-voice-row` : undefined}
      >
        {/* Agent orb */}
        <View style={styles.orbWrapper}>
          <VoiceOrb
            type="agent"
            isSpeaking={speakingSource === 'agent'}
            size={64}
            testID={testID ? `${testID}-agent-orb` : undefined}
          />
          <Text style={styles.label}>Lenso</Text>
        </View>

        {/* Waveform */}
        <View style={styles.waveformContainer}>
          <Waveform
            speakingSource={speakingSource}
            barCount={16}
            testID={testID ? `${testID}-waveform` : undefined}
          />
          {/* Mic indicator */}
          <View style={[styles.micIndicator, !isMicrophoneEnabled && styles.micIndicatorOff]}>
            <Image
              source={isMicrophoneEnabled ? ICONS.mic : ICONS.micOff}
              style={[
                styles.micIconImage,
                { tintColor: isMicrophoneEnabled ? colors.primary[600] : colors.gray[500] },
              ]}
              resizeMode="contain"
            />
          </View>
        </View>

        {/* User orb */}
        <View style={styles.orbWrapper}>
          <VoiceOrb
            type="user"
            isSpeaking={speakingSource === 'user'}
            size={64}
            dimmed={!isMicrophoneEnabled}
            testID={testID ? `${testID}-user-orb` : undefined}
          />
          <Text style={styles.label}>You</Text>
        </View>
      </TouchableOpacity>

      {/* Mode toggle */}
      <TouchableOpacity
        style={styles.toggleButton}
        onPress={onToggleMode}
        activeOpacity={0.7}
        testID={testID ? `${testID}-toggle-button` : undefined}
      >
        <Image source={ICONS.keyboard} style={styles.toggleIcon} resizeMode="contain" />
        <Text style={styles.toggleText}>Switch to text</Text>
      </TouchableOpacity>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    backgroundColor: colors.background,
    paddingVertical: spacing[4],
    paddingHorizontal: spacing[4],
    borderTopWidth: 1,
    borderTopColor: colors.gray[200],
  },
  statusText: {
    fontSize: typography.size.sm,
    fontFamily: typography.sans.fontFamily,
    color: colors.text.secondary,
    textAlign: 'center',
    marginBottom: spacing[3],
  },
  voiceRow: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: spacing[4],
  },
  orbWrapper: {
    alignItems: 'center',
  },
  label: {
    marginTop: spacing[1],
    fontSize: typography.size.xs,
    color: colors.text.secondary,
    fontFamily: typography.sans.fontFamily,
  },
  waveformContainer: {
    flex: 1,
    marginHorizontal: spacing[2],
  },
  toggleButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: spacing[3],
    paddingHorizontal: spacing[5],
    backgroundColor: colors.gray[100],
    borderRadius: borderRadius.xl,
    borderWidth: 1,
    borderColor: colors.gray[200],
  },
  toggleIcon: {
    width: 20,
    height: 20,
    tintColor: colors.gray[500],
  },
  toggleText: {
    marginLeft: spacing[2],
    fontSize: typography.size.sm,
    fontWeight: typography.weight.medium,
    color: colors.text.secondary,
    fontFamily: typography.sans.fontFamily,
  },
  micIndicator: {
    position: 'absolute',
    bottom: -12,
    alignSelf: 'center',
    paddingHorizontal: spacing[3],
    paddingVertical: spacing[1],
    borderRadius: borderRadius.full,
    backgroundColor: colors.primary[50],
    borderWidth: 1,
    borderColor: colors.primary[200],
    ...shadows.sm,
  },
  micIndicatorOff: {
    backgroundColor: colors.gray[100],
    borderColor: colors.gray[300],
  },
  micIconImage: {
    width: 18,
    height: 18,
  },
});
