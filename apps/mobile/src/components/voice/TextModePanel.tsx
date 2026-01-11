/**
 * TextModePanel Component
 *
 * The bottom panel shown in text mode containing:
 * - Voice mode toggle button
 * - Text input field
 * - Send button
 */

import React, { useState, useRef, useEffect } from 'react';
import {
  View,
  StyleSheet,
  TextInput,
  TouchableOpacity,
  Image,
  Pressable,
  Keyboard,
  type ViewStyle,
} from 'react-native';
import { colors, spacing, typography, borderRadius, shadows } from '../../theme';

// Icons
const ICONS = {
  mic: require('../../../assets/icons/mic.png'),
  send: require('../../../assets/icons/send.png'),
  arrowDown: require('../../../assets/icons/arrow-down.png'),
};

export interface TextModePanelProps {
  /** Callback when message is sent */
  onSendMessage: (message: string) => void;
  /** Callback when mode toggle is pressed */
  onToggleMode: () => void;
  /** Placeholder text */
  placeholder?: string;
  /** Whether sending is in progress */
  isSending?: boolean;
  /** Optional style overrides */
  style?: ViewStyle;
  /** Test ID for testing */
  testID?: string;
}

export function TextModePanel({
  onSendMessage,
  onToggleMode,
  placeholder = 'Message Lenso...',
  isSending = false,
  style,
  testID,
}: TextModePanelProps) {
  const [message, setMessage] = useState('');
  const [isKeyboardVisible, setIsKeyboardVisible] = useState(false);
  const inputRef = useRef<TextInput>(null);

  // Track keyboard visibility
  useEffect(() => {
    const showSubscription = Keyboard.addListener('keyboardDidShow', () => {
      setIsKeyboardVisible(true);
    });
    const hideSubscription = Keyboard.addListener('keyboardDidHide', () => {
      setIsKeyboardVisible(false);
    });

    return () => {
      showSubscription?.remove();
      hideSubscription?.remove();
    };
  }, []);

  const focusInput = () => {
    inputRef.current?.focus();
  };

  const dismissKeyboard = () => {
    Keyboard.dismiss();
  };

  const handleSend = () => {
    if (message.trim() && !isSending) {
      onSendMessage(message.trim());
      setMessage('');
    }
  };

  const canSend = message.trim() && !isSending;

  return (
    <View style={[styles.container, style]} testID={testID}>
      <View style={styles.inputRow}>
        {/* Voice mode toggle */}
        <TouchableOpacity
          style={styles.voiceButton}
          onPress={onToggleMode}
          activeOpacity={0.7}
          testID={testID ? `${testID}-voice-toggle` : undefined}
        >
          <Image
            source={ICONS.mic}
            style={styles.voiceIcon}
            resizeMode="contain"
          />
        </TouchableOpacity>

        {/* Text input */}
        <Pressable
          style={styles.inputContainer}
          onPress={focusInput}
          testID={testID ? `${testID}-input-container` : undefined}
        >
          <TextInput
            ref={inputRef}
            style={styles.input}
            value={message}
            onChangeText={setMessage}
            placeholder={placeholder}
            placeholderTextColor={colors.text.tertiary}
            multiline
            maxLength={1000}
            blurOnSubmit={false}
            returnKeyType="send"
            onSubmitEditing={handleSend}
            editable={!isSending}
            testID={testID ? `${testID}-input` : undefined}
          />
        </Pressable>

        {/* Send button - or dismiss keyboard button when keyboard is visible and no message */}
        {isKeyboardVisible && !message.trim() ? (
          <TouchableOpacity
            style={styles.dismissButton}
            onPress={dismissKeyboard}
            activeOpacity={0.7}
            testID={testID ? `${testID}-dismiss` : undefined}
          >
            <Image
              source={ICONS.arrowDown}
              style={styles.dismissIcon}
              resizeMode="contain"
            />
          </TouchableOpacity>
        ) : (
          <TouchableOpacity
            style={[
              styles.sendButton,
              canSend ? styles.sendButtonActive : styles.sendButtonInactive,
            ]}
            onPress={handleSend}
            disabled={!canSend}
            activeOpacity={0.7}
            testID={testID ? `${testID}-send` : undefined}
          >
            <Image
              source={ICONS.send}
              style={[
                styles.sendIcon,
                { tintColor: canSend ? colors.background : colors.gray[400] },
              ]}
              resizeMode="contain"
            />
          </TouchableOpacity>
        )}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    backgroundColor: colors.background,
    paddingVertical: spacing[3],
    paddingHorizontal: spacing[4],
    borderTopWidth: 1,
    borderTopColor: colors.gray[200],
  },
  inputRow: {
    flexDirection: 'row',
    alignItems: 'flex-end',
    gap: spacing[2],
  },
  voiceButton: {
    width: 46,
    height: 46,
    borderRadius: 23,
    backgroundColor: colors.voice.agent,
    justifyContent: 'center',
    alignItems: 'center',
    ...shadows.sm,
  },
  voiceIcon: {
    width: 22,
    height: 22,
    tintColor: colors.background,
  },
  inputContainer: {
    flex: 1,
    backgroundColor: colors.gray[100],
    borderRadius: borderRadius.xl,
    paddingHorizontal: spacing[4],
    paddingVertical: spacing[2],
    minHeight: 46,
    maxHeight: 120,
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: colors.gray[200],
  },
  input: {
    fontSize: typography.size.base,
    color: colors.text.primary,
    fontFamily: typography.sans.fontFamily,
    maxHeight: 100,
  },
  sendButton: {
    width: 46,
    height: 46,
    borderRadius: 23,
    justifyContent: 'center',
    alignItems: 'center',
    ...shadows.sm,
  },
  sendButtonActive: {
    backgroundColor: colors.primary[500],
  },
  sendButtonInactive: {
    backgroundColor: colors.gray[200],
  },
  sendIcon: {
    width: 20,
    height: 20,
  },
  dismissButton: {
    width: 46,
    height: 46,
    borderRadius: 23,
    backgroundColor: colors.gray[100],
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: colors.gray[200],
  },
  dismissIcon: {
    width: 20,
    height: 20,
    tintColor: colors.gray[500],
  },
});
