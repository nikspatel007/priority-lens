/**
 * ConversationScreen
 *
 * Main app screen for voice/text conversations with the AI assistant.
 * Features:
 * - Header with profile access
 * - Dynamic content area for SDUI components
 * - Bottom panel for text or voice input
 */

import React, { useCallback, useEffect, useState } from 'react';
import {
  View,
  Text,
  ScrollView,
  StyleSheet,
  KeyboardAvoidingView,
  Platform,
  ActivityIndicator,
  Pressable,
} from 'react-native';
import { useLiveKit } from '@/context/LiveKitContext';
import { useAuthContext } from '@/context/AuthContext';
import { AppHeader } from '@/components/header';
import { VoiceModePanel, TextModePanel } from '@/components/voice';
import { SDUIRenderer } from '@/sdui/SDUIRenderer';
import { createThread, createSession, submitTurn, executeAction } from '@/services/api';
import { colors, typography, spacing, borderRadius } from '@/theme';
import type { UIAction, UIBlock } from '@/types/api';
import type { UIBlock as SDUIBlock, UIAction as SDUIAction } from '@/sdui/types';

export type InputMode = 'voice' | 'text';

export interface ConversationScreenProps {
  /** Callback when settings/profile button is pressed */
  onSettingsPress?: () => void;
  /** Safe area top inset */
  topInset?: number;
  /** Safe area bottom inset */
  bottomInset?: number;
}

/**
 * ConversationScreen provides the main conversation interface
 *
 * Features:
 * - Voice mode with LiveKit integration
 * - Text mode with chat input
 * - SDUI block rendering from agent
 * - Mode toggle between voice and text
 */
export function ConversationScreen({
  onSettingsPress,
  topInset = 0,
  bottomInset = 0,
}: ConversationScreenProps): React.JSX.Element {
  const { user, isLoading: authLoading, getToken } = useAuthContext();
  const {
    isConnected,
    isConnecting,
    speakingSource,
    isMicrophoneEnabled,
    sduiBlocks,
    agentText,
    error: liveKitError,
    connect,
    disconnect,
    toggleMicrophone,
    clearSduiBlocks,
  } = useLiveKit();

  const [inputMode, setInputMode] = useState<InputMode>('voice');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [threadId, setThreadId] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isInitializing, setIsInitializing] = useState(true);

  // Initialize thread and session
  const initializeConversation = useCallback(async () => {
    try {
      setError(null);
      setIsInitializing(true);

      console.log('Creating thread...');
      const thread = await createThread({});
      console.log('Thread created:', thread.id);
      setThreadId(thread.id);

      console.log('Creating session...');
      const session = await createSession(thread.id, { mode: inputMode });
      console.log('Session created:', session.id);
      setSessionId(session.id);

      // If voice mode, connect to LiveKit
      // istanbul ignore else: default mode is voice
      if (inputMode === 'voice') {
        console.log('Connecting to LiveKit...');
        await connect(thread.id, session.id);
        console.log('LiveKit connected');
      }
    } catch (err) {
      console.error('Conversation init error:', err);
      // Get more details from the error
      let message = 'Failed to start conversation';
      if (err instanceof Error) {
        message = err.message;
        // Check for network errors
        if (message.includes('Network request failed')) {
          message = 'Cannot connect to server. Is the backend running?';
        }
      }
      // Log the full error for debugging
      console.error('Full error:', JSON.stringify(err, Object.getOwnPropertyNames(err)));
      setError(message);
    } finally {
      setIsInitializing(false);
    }
  }, [inputMode, connect]);

  // Initialize on mount (after auth is ready)
  useEffect(() => {
    // Wait for auth to be fully ready (loaded AND user signed in)
    if (authLoading || !user) {
      console.log('Waiting for auth...', { authLoading, hasUser: !!user });
      return;
    }

    // Small delay to ensure AuthProvider's token setter effect has run
    // This is needed because React runs child effects before parent effects
    const timer = setTimeout(() => {
      console.log('Auth ready, initializing conversation...');
      initializeConversation();
    }, 50);

    return () => {
      clearTimeout(timer);
      disconnect();
    };
  }, [authLoading, user]);

  // Handle mode toggle
  const handleModeToggle = useCallback(async () => {
    const newMode = inputMode === 'voice' ? 'text' : 'voice';
    setInputMode(newMode);

    // Disconnect from voice if switching to text
    if (newMode === 'text' && isConnected) {
      await disconnect();
    }

    // Create new session for new mode
    // istanbul ignore else: defensive - threadId should always exist after init
    if (threadId) {
      try {
        const session = await createSession(threadId, { mode: newMode });
        setSessionId(session.id);

        // Connect to voice if switching to voice mode
        if (newMode === 'voice') {
          await connect(threadId, session.id);
        }
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to switch mode';
        setError(message);
      }
    }
  }, [inputMode, threadId, isConnected, disconnect, connect]);

  // Handle text submission
  const handleTextSubmit = useCallback(async (text: string) => {
    // istanbul ignore next: defensive check for edge cases
    if (!text.trim() || !threadId || !sessionId || isSubmitting) {
      return;
    }

    setIsSubmitting(true);
    setError(null);

    try {
      await submitTurn(threadId, {
        session_id: sessionId,
        input: {
          type: 'text',
          text: text.trim(),
        },
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to send message';
      setError(message);
    } finally {
      setIsSubmitting(false);
    }
  }, [threadId, sessionId, isSubmitting]);

  // Handle SDUI actions
  const handleAction = useCallback(async (action: UIAction) => {
    // istanbul ignore next: defensive - threadId/sessionId should exist
    if (!threadId) return;

    // istanbul ignore next: payload defaults to empty object
    const actionPayload = action.payload ?? {};
    try {
      await executeAction({
        id: `action-${Date.now()}`,
        type: action.type,
        thread_id: threadId,
        session_id: sessionId,
        payload: actionPayload,
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Action failed';
      setError(message);
    }
  }, [threadId, sessionId]);

  const displayError = error || liveKitError;

  // istanbul ignore next: platform-specific behavior
  const keyboardBehavior = Platform.OS === 'ios' ? 'padding' : 'height';

  // Get time-based greeting
  const getGreeting = () => {
    const hour = new Date().getHours();
    if (hour < 12) return 'Good morning';
    if (hour < 17) return 'Good afternoon';
    return 'Good evening';
  };

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={keyboardBehavior}
      testID="conversation-screen"
    >
      {/* Header */}
      <AppHeader
        onProfilePress={onSettingsPress}
        topInset={topInset}
        testID="app-header"
      />

      {/* Error display */}
      {displayError && (
        <View style={styles.errorContainer} testID="error-container">
          <Text style={styles.errorText}>{displayError}</Text>
        </View>
      )}

      {/* Content area */}
      {isInitializing ? (
        <View style={styles.loadingContainer} testID="loading-container">
          <ActivityIndicator size="large" color={colors.primary[500]} />
          <Text style={styles.loadingText}>Connecting...</Text>
        </View>
      ) : (
        <ScrollView
          style={styles.content}
          contentContainerStyle={styles.contentContainer}
          testID="content-scroll"
          keyboardDismissMode="on-drag"
          keyboardShouldPersistTaps="handled"
        >
          {/* Agent text response */}
          {agentText && (
            <View style={styles.agentTextContainer} testID="agent-text">
              <Text style={styles.agentText}>{agentText}</Text>
            </View>
          )}

          {/* SDUI blocks */}
          {sduiBlocks.length > 0 && (
            <View style={styles.sduiContainer} testID="sdui-container">
              {sduiBlocks.map((block: UIBlock) => (
                <SDUIRenderer
                  key={block.id}
                  block={block as unknown as SDUIBlock}
                  onAction={handleAction as unknown as (action: SDUIAction) => void}
                />
              ))}
            </View>
          )}

          {/* Empty state - welcome message */}
          {!agentText && sduiBlocks.length === 0 && (
            <View style={styles.welcomeContainer} testID="empty-state">
              <Text style={styles.welcomeTitle}>
                {getGreeting()}
                {user?.firstName ? `, ${user.firstName}` : ''}
              </Text>
              <Text style={styles.welcomeSubtitle}>
                {inputMode === 'voice'
                  ? isConnected
                    ? 'Tap the mic button to start speaking'
                    : isConnecting
                      ? 'Connecting to voice...'
                      : 'Tap below to switch to text mode'
                  : 'Type a message to get started'}
              </Text>

              {/* Voice active indicator */}
              {inputMode === 'voice' && isConnected && (
                <View style={styles.voiceIndicator}>
                  <View
                    style={[
                      styles.voiceDot,
                      speakingSource !== 'none' && styles.voiceDotActive,
                      speakingSource === 'agent' && styles.voiceDotAgent,
                    ]}
                  />
                  <Text style={styles.voiceStatusText}>
                    {speakingSource === 'agent'
                      ? 'Lenso is speaking...'
                      : speakingSource === 'user'
                        ? 'Listening...'
                        : 'Ready to listen'}
                  </Text>
                </View>
              )}

              {/* Voice connection failed - show retry */}
              {inputMode === 'voice' && !isConnected && !isConnecting && (
                <View style={styles.retryContainer}>
                  <Pressable
                    style={styles.retryButton}
                    onPress={initializeConversation}
                    testID="retry-connection"
                  >
                    <Text style={styles.retryButtonText}>Retry Connection</Text>
                  </Pressable>
                </View>
              )}
            </View>
          )}
        </ScrollView>
      )}

      {/* Bottom input panel */}
      <View style={{ paddingBottom: bottomInset }}>
        {inputMode === 'voice' ? (
          <VoiceModePanel
            speakingSource={speakingSource}
            isMicrophoneEnabled={isMicrophoneEnabled}
            isConnected={isConnected}
            isConnecting={isConnecting}
            onToggleMicrophone={toggleMicrophone}
            onToggleMode={handleModeToggle}
            testID="voice-panel"
          />
        ) : (
          <TextModePanel
            onSendMessage={handleTextSubmit}
            onToggleMode={handleModeToggle}
            isSending={isSubmitting}
            testID="text-panel"
          />
        )}
      </View>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    gap: spacing[4],
  },
  loadingText: {
    fontSize: typography.size.lg,
    fontFamily: typography.serif.fontFamily,
    color: colors.text.secondary,
  },
  errorContainer: {
    backgroundColor: colors.error + '20',
    padding: spacing[3],
    marginHorizontal: spacing[4],
    marginTop: spacing[2],
    borderRadius: borderRadius.md,
  },
  errorText: {
    fontFamily: typography.sans.fontFamily,
    fontSize: typography.size.sm,
    color: colors.error,
    textAlign: 'center',
  },
  content: {
    flex: 1,
  },
  contentContainer: {
    padding: spacing[4],
    flexGrow: 1,
  },
  agentTextContainer: {
    backgroundColor: colors.gray[100],
    padding: spacing[3],
    borderRadius: borderRadius.lg,
    marginBottom: spacing[3],
  },
  agentText: {
    fontFamily: typography.sans.fontFamily,
    fontSize: typography.size.base,
    color: colors.text.primary,
    lineHeight: typography.size.base * typography.lineHeight.relaxed,
  },
  sduiContainer: {
    gap: spacing[3],
  },
  welcomeContainer: {
    flex: 1,
    justifyContent: 'center',
  },
  welcomeTitle: {
    fontSize: typography.size['3xl'],
    fontFamily: typography.serif.fontFamily,
    fontWeight: typography.weight.bold,
    color: colors.text.primary,
    marginBottom: spacing[2],
  },
  welcomeSubtitle: {
    fontSize: typography.size.lg,
    fontFamily: typography.sans.fontFamily,
    color: colors.text.secondary,
    marginBottom: spacing[8],
  },
  voiceIndicator: {
    alignItems: 'center',
    gap: spacing[3],
  },
  voiceDot: {
    width: 60,
    height: 60,
    borderRadius: 30,
    backgroundColor: colors.gray[300],
  },
  voiceDotActive: {
    backgroundColor: colors.primary[500],
  },
  voiceDotAgent: {
    backgroundColor: colors.success,
  },
  voiceStatusText: {
    fontSize: typography.size.base,
    fontFamily: typography.sans.fontFamily,
    color: colors.text.secondary,
    textAlign: 'center',
  },
  retryContainer: {
    alignItems: 'center',
    marginTop: spacing[4],
  },
  retryButton: {
    backgroundColor: colors.primary[500],
    paddingVertical: spacing[2],
    paddingHorizontal: spacing[4],
    borderRadius: borderRadius.md,
  },
  retryButtonText: {
    fontSize: typography.size.base,
    fontFamily: typography.sans.fontFamily,
    fontWeight: typography.weight.semibold,
    color: colors.text.inverse,
  },
});
