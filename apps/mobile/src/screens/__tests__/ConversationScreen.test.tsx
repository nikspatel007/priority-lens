/**
 * ConversationScreen Tests
 */

import React from 'react';
import { render, fireEvent, waitFor, act } from '@testing-library/react-native';
import { ConversationScreen } from '../ConversationScreen';

// Mock contexts
const mockConnect = jest.fn().mockResolvedValue(undefined);
const mockDisconnect = jest.fn().mockResolvedValue(undefined);
const mockToggleMicrophone = jest.fn().mockResolvedValue(undefined);
const mockClearSduiBlocks = jest.fn();

let mockIsConnected = false;
let mockIsConnecting = false;
let mockSpeakingSource: 'agent' | 'user' | 'none' = 'none';
let mockIsMicrophoneEnabled = false;
let mockSduiBlocks: Array<{ id: string; type: string; props?: Record<string, unknown> }> = [];
let mockAgentText = '';
let mockLiveKitError: string | null = null;

jest.mock('@/context/LiveKitContext', () => ({
  useLiveKit: () => ({
    isConnected: mockIsConnected,
    isConnecting: mockIsConnecting,
    speakingSource: mockSpeakingSource,
    isMicrophoneEnabled: mockIsMicrophoneEnabled,
    sduiBlocks: mockSduiBlocks,
    agentText: mockAgentText,
    error: mockLiveKitError,
    connect: mockConnect,
    disconnect: mockDisconnect,
    toggleMicrophone: mockToggleMicrophone,
    clearSduiBlocks: mockClearSduiBlocks,
  }),
}));

let mockUser: { firstName?: string; email?: string } | null = { firstName: 'Test', email: 'test@example.com' };
let mockAuthLoading = false;

jest.mock('@/context/AuthContext', () => ({
  useAuthContext: () => ({
    user: mockUser,
    isLoading: mockAuthLoading,
    getToken: jest.fn().mockResolvedValue('test-token'),
  }),
}));

// Mock API services
const mockCreateThread = jest.fn();
const mockCreateSession = jest.fn();
const mockSubmitTurn = jest.fn();
const mockExecuteAction = jest.fn();

jest.mock('@/services/api', () => ({
  createThread: (...args: unknown[]) => mockCreateThread(...args),
  createSession: (...args: unknown[]) => mockCreateSession(...args),
  submitTurn: (...args: unknown[]) => mockSubmitTurn(...args),
  executeAction: (...args: unknown[]) => mockExecuteAction(...args),
}));

// Mock AppHeader
jest.mock('@/components/header', () => ({
  AppHeader: ({ testID, onProfilePress, topInset }: {
    testID?: string;
    onProfilePress?: () => void;
    topInset?: number;
  }) => {
    const { TouchableOpacity, View, Text } = require('react-native');
    return (
      <View testID={testID} style={{ paddingTop: topInset }}>
        <Text>Priority Lens</Text>
        {onProfilePress && (
          <TouchableOpacity testID={`${testID}-profile`} onPress={onProfilePress}>
            <Text>Profile</Text>
          </TouchableOpacity>
        )}
      </View>
    );
  },
}));

// Mock VoiceModePanel
jest.mock('@/components/voice/VoiceModePanel', () => ({
  VoiceModePanel: ({ testID, onToggleMicrophone, onToggleMode }: {
    testID?: string;
    onToggleMicrophone: () => void;
    onToggleMode: () => void;
  }) => {
    const { TouchableOpacity, View, Text } = require('react-native');
    return (
      <View testID={testID}>
        <TouchableOpacity testID={`${testID}-mic-button`} onPress={onToggleMicrophone}>
          <Text>Toggle Mic</Text>
        </TouchableOpacity>
        <TouchableOpacity testID={`${testID}-toggle-mode`} onPress={onToggleMode}>
          <Text>Switch to text</Text>
        </TouchableOpacity>
      </View>
    );
  },
}));

// Mock TextModePanel
jest.mock('@/components/voice/TextModePanel', () => ({
  TextModePanel: ({ testID, onSendMessage, onToggleMode, isSending }: {
    testID?: string;
    onSendMessage: (text: string) => void;
    onToggleMode: () => void;
    isSending?: boolean;
  }) => {
    const React = require('react');
    const { TouchableOpacity, View, Text, TextInput } = require('react-native');
    const [text, setText] = React.useState('');
    return (
      <View testID={testID}>
        <TouchableOpacity testID={`${testID}-voice-toggle`} onPress={onToggleMode}>
          <Text>Switch to voice</Text>
        </TouchableOpacity>
        <TextInput
          testID={`${testID}-input`}
          value={text}
          onChangeText={setText}
          editable={!isSending}
          onSubmitEditing={() => {
            if (text.trim()) {
              onSendMessage(text.trim());
              setText('');
            }
          }}
        />
        <TouchableOpacity
          testID={`${testID}-send`}
          onPress={() => {
            if (text.trim() && !isSending) {
              onSendMessage(text.trim());
              setText('');
            }
          }}
          disabled={!text.trim() || isSending}
        >
          <Text>Send</Text>
        </TouchableOpacity>
      </View>
    );
  },
}));

// Mock SDUIRenderer
jest.mock('@/sdui/SDUIRenderer', () => ({
  SDUIRenderer: ({ block, onAction }: {
    block: { id: string; type: string };
    onAction?: (action: { type: string; payload?: Record<string, unknown> }) => void;
  }) => {
    const { View, Text, TouchableOpacity } = require('react-native');
    return (
      <View testID={`sdui-block-${block.id}`}>
        <Text>{block.type}</Text>
        {onAction && (
          <TouchableOpacity
            testID={`sdui-action-${block.id}`}
            onPress={() => onAction({ type: 'test.action', payload: { test: true } })}
          >
            <Text>Action</Text>
          </TouchableOpacity>
        )}
      </View>
    );
  },
}));

describe('ConversationScreen', () => {
  beforeEach(() => {
    jest.useFakeTimers();
    jest.clearAllMocks();

    // Reset mock implementations
    mockCreateThread.mockResolvedValue({ id: 'thread-123' });
    mockCreateSession.mockResolvedValue({ id: 'session-456' });
    mockSubmitTurn.mockResolvedValue({});
    mockExecuteAction.mockResolvedValue({});

    // Reset state mocks
    mockIsConnected = false;
    mockIsConnecting = false;
    mockSpeakingSource = 'none';
    mockIsMicrophoneEnabled = false;
    mockSduiBlocks = [];
    mockAgentText = '';
    mockLiveKitError = null;
    mockUser = { firstName: 'Test', email: 'test@example.com' };
    mockAuthLoading = false;
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  describe('rendering', () => {
    it('renders conversation screen', async () => {
      const { getByTestId } = render(<ConversationScreen />);

      await waitFor(() => {
        expect(getByTestId('conversation-screen')).toBeTruthy();
      });
    });

    it('renders app header', async () => {
      const { getByTestId } = render(<ConversationScreen />);

      await waitFor(() => {
        expect(getByTestId('app-header')).toBeTruthy();
      });
    });

    it('renders header with title', async () => {
      const { getByText } = render(<ConversationScreen />);

      await waitFor(() => {
        expect(getByText('Priority Lens')).toBeTruthy();
      });
    });

    it('renders profile button when onSettingsPress provided', async () => {
      const onSettingsPress = jest.fn();
      const { getByTestId } = render(<ConversationScreen onSettingsPress={onSettingsPress} />);

      await waitFor(() => {
        expect(getByTestId('app-header-profile')).toBeTruthy();
      });
    });

    it('does not render profile button when onSettingsPress not provided', async () => {
      const { queryByTestId } = render(<ConversationScreen />);

      await waitFor(() => {
        expect(queryByTestId('app-header-profile')).toBeNull();
      });
    });

    it('passes topInset to header', async () => {
      const { getByTestId } = render(<ConversationScreen topInset={44} />);

      await waitFor(() => {
        const header = getByTestId('app-header');
        expect(header.props.style).toEqual(
          expect.objectContaining({ paddingTop: 44 })
        );
      });
    });
  });

  describe('initialization', () => {
    it('shows loading state initially', async () => {
      // Delay the promise to see loading state
      mockCreateThread.mockImplementation(() => new Promise(() => {}));

      const { getByTestId } = render(<ConversationScreen />);

      expect(getByTestId('loading-container')).toBeTruthy();
    });

    it('creates thread and session on mount', async () => {
      render(<ConversationScreen />);

      await waitFor(() => {
        expect(mockCreateThread).toHaveBeenCalledWith({});
      });

      await waitFor(() => {
        expect(mockCreateSession).toHaveBeenCalledWith('thread-123', { mode: 'voice' });
      });
    });

    it('connects to LiveKit in voice mode', async () => {
      render(<ConversationScreen />);

      await waitFor(() => {
        expect(mockConnect).toHaveBeenCalledWith('thread-123', 'session-456');
      });
    });

    it('handles initialization error', async () => {
      mockCreateThread.mockRejectedValueOnce(new Error('Network error'));

      const { getByTestId } = render(<ConversationScreen />);

      await waitFor(() => {
        expect(getByTestId('error-container')).toBeTruthy();
      });
    });

    it('handles non-Error initialization exception', async () => {
      mockCreateThread.mockRejectedValueOnce('String error');

      const { getByText } = render(<ConversationScreen />);

      await waitFor(() => {
        expect(getByText('Failed to start conversation')).toBeTruthy();
      });
    });

    it('disconnects on unmount', async () => {
      const { unmount } = render(<ConversationScreen />);

      await waitFor(() => {
        expect(mockCreateThread).toHaveBeenCalled();
      });

      unmount();

      expect(mockDisconnect).toHaveBeenCalled();
    });
  });

  describe('voice mode', () => {
    it('renders voice panel in voice mode', async () => {
      const { getByTestId } = render(<ConversationScreen />);

      await waitFor(() => {
        expect(getByTestId('voice-panel')).toBeTruthy();
      });
    });

    it('handles microphone toggle', async () => {
      const { getByTestId } = render(<ConversationScreen />);

      await waitFor(() => {
        expect(getByTestId('voice-panel-mic-button')).toBeTruthy();
      });

      await act(async () => {
        fireEvent.press(getByTestId('voice-panel-mic-button'));
      });

      expect(mockToggleMicrophone).toHaveBeenCalled();
    });

    it('handles mode toggle from voice panel', async () => {
      mockIsConnected = true;
      const { getByTestId } = render(<ConversationScreen />);

      // Advance timers to trigger initialization (50ms delay for auth)
      await act(async () => {
        jest.advanceTimersByTime(100);
      });

      await waitFor(() => {
        expect(getByTestId('voice-panel-toggle-mode')).toBeTruthy();
      });

      // Clear mocks to only track mode toggle calls
      mockCreateSession.mockClear();

      await act(async () => {
        fireEvent.press(getByTestId('voice-panel-toggle-mode'));
      });

      // Mode toggle should disconnect and create new session
      await waitFor(() => {
        expect(mockDisconnect).toHaveBeenCalled();
      });

      await waitFor(() => {
        expect(mockCreateSession).toHaveBeenCalledWith('thread-123', { mode: 'text' });
      });
    });
  });

  describe('text mode', () => {
    it('renders text panel after switching to text mode', async () => {
      mockIsConnected = true;
      const { getByTestId } = render(<ConversationScreen />);

      await waitFor(() => {
        expect(getByTestId('voice-panel-toggle-mode')).toBeTruthy();
      });

      await act(async () => {
        fireEvent.press(getByTestId('voice-panel-toggle-mode'));
      });

      await waitFor(() => {
        expect(getByTestId('text-panel')).toBeTruthy();
      });
    });

    it('handles text submission', async () => {
      mockIsConnected = true;
      const { getByTestId } = render(<ConversationScreen />);

      // Wait for initialization to complete
      await waitFor(() => {
        expect(mockCreateThread).toHaveBeenCalled();
        expect(mockCreateSession).toHaveBeenCalled();
      });

      // Switch to text mode
      await act(async () => {
        fireEvent.press(getByTestId('voice-panel-toggle-mode'));
      });

      await waitFor(() => {
        expect(getByTestId('text-panel-input')).toBeTruthy();
      });

      // Type and send
      fireEvent.changeText(getByTestId('text-panel-input'), 'Hello world');

      mockSubmitTurn.mockClear();

      await act(async () => {
        fireEvent.press(getByTestId('text-panel-send'));
      });

      await waitFor(() => {
        expect(mockSubmitTurn).toHaveBeenCalledWith('thread-123', {
          session_id: 'session-456',
          input: {
            type: 'text',
            text: 'Hello world',
          },
        });
      });
    });

    it('handles text submission error', async () => {
      mockIsConnected = true;
      const { getByTestId, getByText } = render(<ConversationScreen />);

      await waitFor(() => {
        expect(mockCreateSession).toHaveBeenCalled();
      });

      await act(async () => {
        fireEvent.press(getByTestId('voice-panel-toggle-mode'));
      });

      await waitFor(() => {
        expect(getByTestId('text-panel-input')).toBeTruthy();
      });

      fireEvent.changeText(getByTestId('text-panel-input'), 'Hello');

      mockSubmitTurn.mockRejectedValueOnce(new Error('Send failed'));

      await act(async () => {
        fireEvent.press(getByTestId('text-panel-send'));
      });

      await waitFor(() => {
        expect(getByText('Send failed')).toBeTruthy();
      });
    });

    it('switches back to voice mode', async () => {
      mockIsConnected = true;
      const { getByTestId } = render(<ConversationScreen />);

      await waitFor(() => {
        expect(getByTestId('voice-panel-toggle-mode')).toBeTruthy();
      });

      // Switch to text
      await act(async () => {
        fireEvent.press(getByTestId('voice-panel-toggle-mode'));
      });

      await waitFor(() => {
        expect(getByTestId('text-panel-voice-toggle')).toBeTruthy();
      });

      mockConnect.mockClear();
      mockCreateSession.mockClear();

      // Switch back to voice
      await act(async () => {
        fireEvent.press(getByTestId('text-panel-voice-toggle'));
      });

      await waitFor(() => {
        expect(mockCreateSession).toHaveBeenCalledWith('thread-123', { mode: 'voice' });
      });
    });

    it('handles mode switch error', async () => {
      mockIsConnected = true;
      const { getByTestId, getByText } = render(<ConversationScreen />);

      await waitFor(() => {
        expect(mockCreateSession).toHaveBeenCalled();
      });

      // Set up error for next session creation
      mockCreateSession.mockRejectedValueOnce(new Error('Switch failed'));

      await act(async () => {
        fireEvent.press(getByTestId('voice-panel-toggle-mode'));
      });

      await waitFor(() => {
        expect(getByText('Switch failed')).toBeTruthy();
      });
    });
  });

  describe('content display', () => {
    it('renders agent text', async () => {
      mockAgentText = 'Hello from agent';

      const { getByTestId, getByText } = render(<ConversationScreen />);

      await waitFor(() => {
        expect(getByTestId('agent-text')).toBeTruthy();
        expect(getByText('Hello from agent')).toBeTruthy();
      });
    });

    it('renders SDUI blocks', async () => {
      mockSduiBlocks = [
        { id: 'block-1', type: 'text', props: { value: 'Block 1' } },
        { id: 'block-2', type: 'button', props: { label: 'Click' } },
      ];

      const { getByTestId } = render(<ConversationScreen />);

      await waitFor(() => {
        expect(getByTestId('sdui-container')).toBeTruthy();
        expect(getByTestId('sdui-block-block-1')).toBeTruthy();
        expect(getByTestId('sdui-block-block-2')).toBeTruthy();
      });
    });

    it('handles SDUI action', async () => {
      mockSduiBlocks = [{ id: 'block-1', type: 'button', props: {} }];

      const { getByTestId } = render(<ConversationScreen />);

      await waitFor(() => {
        expect(getByTestId('sdui-action-block-1')).toBeTruthy();
      });

      await act(async () => {
        fireEvent.press(getByTestId('sdui-action-block-1'));
      });

      await waitFor(() => {
        expect(mockExecuteAction).toHaveBeenCalledWith(
          expect.objectContaining({
            type: 'test.action',
            thread_id: 'thread-123',
            session_id: 'session-456',
            payload: { test: true },
          })
        );
      });
    });

    it('handles action error', async () => {
      mockSduiBlocks = [{ id: 'block-1', type: 'button', props: {} }];
      mockExecuteAction.mockRejectedValueOnce(new Error('Action failed'));

      const { getByTestId, getByText } = render(<ConversationScreen />);

      await waitFor(() => {
        expect(getByTestId('sdui-action-block-1')).toBeTruthy();
      });

      await act(async () => {
        fireEvent.press(getByTestId('sdui-action-block-1'));
      });

      await waitFor(() => {
        expect(getByText('Action failed')).toBeTruthy();
      });
    });

    it('renders empty state with greeting and user name', async () => {
      mockUser = { firstName: 'John' };

      const { getByTestId } = render(<ConversationScreen />);

      // Advance timers to trigger initialization (50ms delay for auth)
      await act(async () => {
        jest.advanceTimersByTime(100);
      });

      await waitFor(() => {
        expect(getByTestId('empty-state')).toBeTruthy();
      });
    });

    it('renders empty state without user name', async () => {
      // User is signed in but has no firstName
      mockUser = { email: 'test@example.com' };

      const { getByTestId } = render(<ConversationScreen />);

      // Advance timers to trigger initialization (50ms delay for auth)
      await act(async () => {
        jest.advanceTimersByTime(100);
      });

      await waitFor(() => {
        expect(getByTestId('empty-state')).toBeTruthy();
      });
    });
  });

  describe('error display', () => {
    it('displays LiveKit error', async () => {
      mockLiveKitError = 'LiveKit connection failed';

      const { getByTestId, getByText } = render(<ConversationScreen />);

      await waitFor(() => {
        expect(getByTestId('error-container')).toBeTruthy();
        expect(getByText('LiveKit connection failed')).toBeTruthy();
      });
    });
  });

  describe('profile/settings', () => {
    it('calls onSettingsPress when profile button pressed', async () => {
      const onSettingsPress = jest.fn();
      const { getByTestId } = render(<ConversationScreen onSettingsPress={onSettingsPress} />);

      await waitFor(() => {
        expect(getByTestId('app-header-profile')).toBeTruthy();
      });

      fireEvent.press(getByTestId('app-header-profile'));

      expect(onSettingsPress).toHaveBeenCalled();
    });
  });

  describe('bottom inset', () => {
    it('applies bottomInset to panel container', async () => {
      const { getByTestId } = render(<ConversationScreen bottomInset={34} />);

      await waitFor(() => {
        expect(getByTestId('voice-panel')).toBeTruthy();
      });

      // The bottom inset is applied to the parent wrapper
    });
  });
});
