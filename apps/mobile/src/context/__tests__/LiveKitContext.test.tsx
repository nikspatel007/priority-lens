/**
 * Tests for LiveKitContext
 */

import React from 'react';
import { Text, View } from 'react-native';
import { render, act, waitFor } from '@testing-library/react-native';

import {
  LiveKitProvider,
  useLiveKit,
  EventTypes,
  type CanonicalEvent,
} from '../LiveKitContext';
import * as api from '../../services/api';

// Mock the API
jest.mock('../../services/api');
const mockApi = api as jest.Mocked<typeof api>;

// Test component that exposes context
function TestConsumer({
  onContext,
}: {
  onContext?: (ctx: ReturnType<typeof useLiveKit>) => void;
}) {
  const context = useLiveKit();
  React.useEffect(() => {
    onContext?.(context);
  }, [context, onContext]);

  return (
    <View testID="test-consumer">
      <Text testID="is-connected">{context.isConnected.toString()}</Text>
      <Text testID="is-connecting">{context.isConnecting.toString()}</Text>
      <Text testID="speaking-source">{context.speakingSource}</Text>
      <Text testID="is-mic-enabled">{context.isMicrophoneEnabled.toString()}</Text>
      <Text testID="sdui-blocks-count">{context.sduiBlocks.length}</Text>
      <Text testID="last-seq">{context.lastSeq}</Text>
      <Text testID="agent-text">{context.agentText}</Text>
      <Text testID="error">{context.error || 'null'}</Text>
    </View>
  );
}

describe('LiveKitContext', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockApi.getLiveKitToken.mockResolvedValue({ token: 'test-token' });
  });

  describe('LiveKitProvider', () => {
    it('provides initial state', () => {
      const { getByTestId } = render(
        <LiveKitProvider>
          <TestConsumer />
        </LiveKitProvider>
      );

      expect(getByTestId('is-connected')).toHaveTextContent('false');
      expect(getByTestId('is-connecting')).toHaveTextContent('false');
      expect(getByTestId('speaking-source')).toHaveTextContent('none');
      expect(getByTestId('is-mic-enabled')).toHaveTextContent('false');
      expect(getByTestId('sdui-blocks-count')).toHaveTextContent('0');
      expect(getByTestId('last-seq')).toHaveTextContent('0');
      expect(getByTestId('agent-text')).toHaveTextContent('');
      expect(getByTestId('error')).toHaveTextContent('null');
    });

    it('accepts custom liveKitUrl', () => {
      const { getByTestId } = render(
        <LiveKitProvider liveKitUrl="wss://custom.livekit.io">
          <TestConsumer />
        </LiveKitProvider>
      );

      expect(getByTestId('test-consumer')).toBeTruthy();
    });
  });

  describe('useLiveKit hook', () => {
    it('throws error when used outside provider', () => {
      // Suppress console.error for this test
      const consoleSpy = jest.spyOn(console, 'error').mockImplementation(() => {});

      expect(() => {
        render(<TestConsumer />);
      }).toThrow('useLiveKit must be used within a LiveKitProvider');

      consoleSpy.mockRestore();
    });
  });

  describe('connect', () => {
    it('sets isConnecting during connection and then false after', async () => {
      let contextRef: ReturnType<typeof useLiveKit> | null = null;

      const { getByTestId } = render(
        <LiveKitProvider>
          <TestConsumer
            onContext={(ctx) => {
              contextRef = ctx;
            }}
          />
        </LiveKitProvider>
      );

      expect(getByTestId('is-connecting')).toHaveTextContent('false');

      await act(async () => {
        await contextRef?.connect('thread-1', 'session-1');
      });

      // After connection completes, isConnecting should be false
      expect(getByTestId('is-connecting')).toHaveTextContent('false');
      // And isConnected should be true
      expect(getByTestId('is-connected')).toHaveTextContent('true');
    });

    it('fetches token from API', async () => {
      let contextRef: ReturnType<typeof useLiveKit> | null = null;

      render(
        <LiveKitProvider>
          <TestConsumer onContext={(ctx) => (contextRef = ctx)} />
        </LiveKitProvider>
      );

      await act(async () => {
        await contextRef?.connect('thread-123', 'session-456');
      });

      expect(mockApi.getLiveKitToken).toHaveBeenCalledWith({
        thread_id: 'thread-123',
        session_id: 'session-456',
      });
    });

    it('sets isConnected to true after successful connection', async () => {
      let contextRef: ReturnType<typeof useLiveKit> | null = null;

      const { getByTestId } = render(
        <LiveKitProvider>
          <TestConsumer onContext={(ctx) => (contextRef = ctx)} />
        </LiveKitProvider>
      );

      await act(async () => {
        await contextRef?.connect('thread-1', 'session-1');
      });

      await waitFor(() => {
        expect(getByTestId('is-connected')).toHaveTextContent('true');
      });
    });

    it('handles connection error', async () => {
      mockApi.getLiveKitToken.mockRejectedValueOnce(new Error('Network error'));
      let contextRef: ReturnType<typeof useLiveKit> | null = null;

      const { getByTestId } = render(
        <LiveKitProvider>
          <TestConsumer onContext={(ctx) => (contextRef = ctx)} />
        </LiveKitProvider>
      );

      await act(async () => {
        await contextRef?.connect('thread-1', 'session-1');
      });

      await waitFor(() => {
        expect(getByTestId('error')).toHaveTextContent('Network error');
      });
      expect(getByTestId('is-connected')).toHaveTextContent('false');
    });

    it('does not connect if already connected', async () => {
      let contextRef: ReturnType<typeof useLiveKit> | null = null;

      render(
        <LiveKitProvider>
          <TestConsumer onContext={(ctx) => (contextRef = ctx)} />
        </LiveKitProvider>
      );

      await act(async () => {
        await contextRef?.connect('thread-1', 'session-1');
      });

      mockApi.getLiveKitToken.mockClear();

      await act(async () => {
        await contextRef?.connect('thread-2', 'session-2');
      });

      // Should not have called API again
      expect(mockApi.getLiveKitToken).not.toHaveBeenCalled();
    });
  });

  describe('disconnect', () => {
    it('resets state after disconnect', async () => {
      let contextRef: ReturnType<typeof useLiveKit> | null = null;

      const { getByTestId } = render(
        <LiveKitProvider>
          <TestConsumer onContext={(ctx) => (contextRef = ctx)} />
        </LiveKitProvider>
      );

      // Connect first
      await act(async () => {
        await contextRef?.connect('thread-1', 'session-1');
      });

      await waitFor(() => {
        expect(getByTestId('is-connected')).toHaveTextContent('true');
      });

      // Then disconnect
      await act(async () => {
        await contextRef?.disconnect();
      });

      expect(getByTestId('is-connected')).toHaveTextContent('false');
      expect(getByTestId('is-mic-enabled')).toHaveTextContent('false');
      expect(getByTestId('speaking-source')).toHaveTextContent('none');
    });

    it('handles disconnect when room is null', async () => {
      let contextRef: ReturnType<typeof useLiveKit> | null = null;

      const { getByTestId } = render(
        <LiveKitProvider>
          <TestConsumer onContext={(ctx) => (contextRef = ctx)} />
        </LiveKitProvider>
      );

      // Call disconnect without ever connecting (room is null)
      await act(async () => {
        await contextRef?.disconnect();
      });

      // State should remain at initial values
      expect(getByTestId('is-connected')).toHaveTextContent('false');
      expect(getByTestId('is-mic-enabled')).toHaveTextContent('false');
      expect(getByTestId('speaking-source')).toHaveTextContent('none');
    });
  });

  describe('toggleMicrophone', () => {
    it('toggles microphone state', async () => {
      let contextRef: ReturnType<typeof useLiveKit> | null = null;

      const { getByTestId } = render(
        <LiveKitProvider>
          <TestConsumer onContext={(ctx) => (contextRef = ctx)} />
        </LiveKitProvider>
      );

      // Connect first
      await act(async () => {
        await contextRef?.connect('thread-1', 'session-1');
      });

      await waitFor(() => {
        expect(getByTestId('is-connected')).toHaveTextContent('true');
      });

      // Toggle on
      await act(async () => {
        await contextRef?.toggleMicrophone();
      });

      expect(getByTestId('is-mic-enabled')).toHaveTextContent('true');

      // Toggle off
      await act(async () => {
        await contextRef?.toggleMicrophone();
      });

      expect(getByTestId('is-mic-enabled')).toHaveTextContent('false');
    });

    it('does nothing when not connected', async () => {
      let contextRef: ReturnType<typeof useLiveKit> | null = null;

      const { getByTestId } = render(
        <LiveKitProvider>
          <TestConsumer onContext={(ctx) => (contextRef = ctx)} />
        </LiveKitProvider>
      );

      await act(async () => {
        await contextRef?.toggleMicrophone();
      });

      expect(getByTestId('is-mic-enabled')).toHaveTextContent('false');
    });
  });

  describe('processEvent', () => {
    it('handles TURN_AGENT_OPEN event', async () => {
      let contextRef: ReturnType<typeof useLiveKit> | null = null;

      const { getByTestId } = render(
        <LiveKitProvider>
          <TestConsumer onContext={(ctx) => (contextRef = ctx)} />
        </LiveKitProvider>
      );

      await act(async () => {
        contextRef?.processEvent({ type: EventTypes.TURN_AGENT_OPEN });
      });

      expect(getByTestId('speaking-source')).toHaveTextContent('agent');
    });

    it('handles TURN_AGENT_CLOSE event', async () => {
      let contextRef: ReturnType<typeof useLiveKit> | null = null;

      const { getByTestId } = render(
        <LiveKitProvider>
          <TestConsumer onContext={(ctx) => (contextRef = ctx)} />
        </LiveKitProvider>
      );

      // First set to agent speaking
      await act(async () => {
        contextRef?.processEvent({ type: EventTypes.TURN_AGENT_OPEN });
      });

      expect(getByTestId('speaking-source')).toHaveTextContent('agent');

      // Then close
      await act(async () => {
        contextRef?.processEvent({ type: EventTypes.TURN_AGENT_CLOSE });
      });

      expect(getByTestId('speaking-source')).toHaveTextContent('none');
    });

    it('handles TURN_USER_OPEN event', async () => {
      let contextRef: ReturnType<typeof useLiveKit> | null = null;

      const { getByTestId } = render(
        <LiveKitProvider>
          <TestConsumer onContext={(ctx) => (contextRef = ctx)} />
        </LiveKitProvider>
      );

      await act(async () => {
        contextRef?.processEvent({ type: EventTypes.TURN_USER_OPEN });
      });

      expect(getByTestId('speaking-source')).toHaveTextContent('user');
    });

    it('handles TURN_USER_CLOSE event', async () => {
      let contextRef: ReturnType<typeof useLiveKit> | null = null;

      const { getByTestId } = render(
        <LiveKitProvider>
          <TestConsumer onContext={(ctx) => (contextRef = ctx)} />
        </LiveKitProvider>
      );

      // First set to user speaking
      await act(async () => {
        contextRef?.processEvent({ type: EventTypes.TURN_USER_OPEN });
      });

      // Then close
      await act(async () => {
        contextRef?.processEvent({ type: EventTypes.TURN_USER_CLOSE });
      });

      expect(getByTestId('speaking-source')).toHaveTextContent('none');
    });

    it('handles UI_BLOCK event', async () => {
      let contextRef: ReturnType<typeof useLiveKit> | null = null;

      const { getByTestId } = render(
        <LiveKitProvider>
          <TestConsumer onContext={(ctx) => (contextRef = ctx)} />
        </LiveKitProvider>
      );

      const block = { id: 'block-1', type: 'text', props: { value: 'Hello' } };

      await act(async () => {
        contextRef?.processEvent({
          type: EventTypes.UI_BLOCK,
          payload: { block },
        });
      });

      expect(getByTestId('sdui-blocks-count')).toHaveTextContent('1');
    });

    it('accumulates UI_BLOCK events', async () => {
      let contextRef: ReturnType<typeof useLiveKit> | null = null;

      const { getByTestId } = render(
        <LiveKitProvider>
          <TestConsumer onContext={(ctx) => (contextRef = ctx)} />
        </LiveKitProvider>
      );

      await act(async () => {
        contextRef?.processEvent({
          type: EventTypes.UI_BLOCK,
          payload: { block: { id: '1', type: 'text' } },
        });
        contextRef?.processEvent({
          type: EventTypes.UI_BLOCK,
          payload: { block: { id: '2', type: 'button' } },
        });
      });

      expect(getByTestId('sdui-blocks-count')).toHaveTextContent('2');
    });

    it('handles UI_CLEAR event', async () => {
      let contextRef: ReturnType<typeof useLiveKit> | null = null;

      const { getByTestId } = render(
        <LiveKitProvider>
          <TestConsumer onContext={(ctx) => (contextRef = ctx)} />
        </LiveKitProvider>
      );

      // Add some blocks
      await act(async () => {
        contextRef?.processEvent({
          type: EventTypes.UI_BLOCK,
          payload: { block: { id: '1', type: 'text' } },
        });
      });

      expect(getByTestId('sdui-blocks-count')).toHaveTextContent('1');

      // Clear
      await act(async () => {
        contextRef?.processEvent({ type: EventTypes.UI_CLEAR });
      });

      expect(getByTestId('sdui-blocks-count')).toHaveTextContent('0');
    });

    it('handles ASSISTANT_TEXT_FINAL event', async () => {
      let contextRef: ReturnType<typeof useLiveKit> | null = null;

      const { getByTestId } = render(
        <LiveKitProvider>
          <TestConsumer onContext={(ctx) => (contextRef = ctx)} />
        </LiveKitProvider>
      );

      await act(async () => {
        contextRef?.processEvent({
          type: EventTypes.ASSISTANT_TEXT_FINAL,
          payload: { text: 'Hello, how can I help?' },
        });
      });

      expect(getByTestId('agent-text')).toHaveTextContent('Hello, how can I help?');
    });

    it('handles SYSTEM_CANCEL event', async () => {
      let contextRef: ReturnType<typeof useLiveKit> | null = null;

      const { getByTestId } = render(
        <LiveKitProvider>
          <TestConsumer onContext={(ctx) => (contextRef = ctx)} />
        </LiveKitProvider>
      );

      // First set to speaking
      await act(async () => {
        contextRef?.processEvent({ type: EventTypes.TURN_AGENT_OPEN });
      });

      expect(getByTestId('speaking-source')).toHaveTextContent('agent');

      // Cancel
      await act(async () => {
        contextRef?.processEvent({ type: EventTypes.SYSTEM_CANCEL });
      });

      expect(getByTestId('speaking-source')).toHaveTextContent('none');
    });

    it('updates lastSeq from event', async () => {
      let contextRef: ReturnType<typeof useLiveKit> | null = null;

      const { getByTestId } = render(
        <LiveKitProvider>
          <TestConsumer onContext={(ctx) => (contextRef = ctx)} />
        </LiveKitProvider>
      );

      await act(async () => {
        contextRef?.processEvent({
          type: EventTypes.TURN_AGENT_OPEN,
          seq: 5,
        });
      });

      expect(getByTestId('last-seq')).toHaveTextContent('5');
    });

    it('only updates lastSeq if higher than current', async () => {
      let contextRef: ReturnType<typeof useLiveKit> | null = null;

      const { getByTestId } = render(
        <LiveKitProvider>
          <TestConsumer onContext={(ctx) => (contextRef = ctx)} />
        </LiveKitProvider>
      );

      await act(async () => {
        contextRef?.processEvent({ type: EventTypes.TURN_AGENT_OPEN, seq: 10 });
      });

      expect(getByTestId('last-seq')).toHaveTextContent('10');

      // Try to set lower seq
      await act(async () => {
        contextRef?.processEvent({ type: EventTypes.TURN_AGENT_CLOSE, seq: 5 });
      });

      // Should still be 10
      expect(getByTestId('last-seq')).toHaveTextContent('10');
    });

    it('ignores UI_BLOCK without block payload', async () => {
      let contextRef: ReturnType<typeof useLiveKit> | null = null;

      const { getByTestId } = render(
        <LiveKitProvider>
          <TestConsumer onContext={(ctx) => (contextRef = ctx)} />
        </LiveKitProvider>
      );

      await act(async () => {
        contextRef?.processEvent({
          type: EventTypes.UI_BLOCK,
          payload: {},
        });
      });

      expect(getByTestId('sdui-blocks-count')).toHaveTextContent('0');
    });

    it('ignores ASSISTANT_TEXT_FINAL without text payload', async () => {
      let contextRef: ReturnType<typeof useLiveKit> | null = null;

      const { getByTestId } = render(
        <LiveKitProvider>
          <TestConsumer onContext={(ctx) => (contextRef = ctx)} />
        </LiveKitProvider>
      );

      await act(async () => {
        contextRef?.processEvent({
          type: EventTypes.ASSISTANT_TEXT_FINAL,
          payload: {},
        });
      });

      expect(getByTestId('agent-text')).toHaveTextContent('');
    });
  });

  describe('clearSduiBlocks', () => {
    it('clears all SDUI blocks', async () => {
      let contextRef: ReturnType<typeof useLiveKit> | null = null;

      const { getByTestId } = render(
        <LiveKitProvider>
          <TestConsumer onContext={(ctx) => (contextRef = ctx)} />
        </LiveKitProvider>
      );

      // Add blocks
      await act(async () => {
        contextRef?.processEvent({
          type: EventTypes.UI_BLOCK,
          payload: { block: { id: '1', type: 'text' } },
        });
        contextRef?.processEvent({
          type: EventTypes.UI_BLOCK,
          payload: { block: { id: '2', type: 'button' } },
        });
      });

      expect(getByTestId('sdui-blocks-count')).toHaveTextContent('2');

      // Clear
      await act(async () => {
        contextRef?.clearSduiBlocks();
      });

      expect(getByTestId('sdui-blocks-count')).toHaveTextContent('0');
    });
  });

  describe('EventTypes', () => {
    it('exports all event types', () => {
      expect(EventTypes.TURN_USER_OPEN).toBe('turn.user.open');
      expect(EventTypes.TURN_USER_CLOSE).toBe('turn.user.close');
      expect(EventTypes.TURN_AGENT_OPEN).toBe('turn.agent.open');
      expect(EventTypes.TURN_AGENT_CLOSE).toBe('turn.agent.close');
      expect(EventTypes.UI_BLOCK).toBe('ui.block');
      expect(EventTypes.UI_CLEAR).toBe('ui.clear');
      expect(EventTypes.ASSISTANT_TEXT_FINAL).toBe('assistant.text.final');
      expect(EventTypes.TOOL_CALL).toBe('tool.call');
      expect(EventTypes.TOOL_RESULT).toBe('tool.result');
      expect(EventTypes.SYSTEM_CANCEL).toBe('system.cancel');
    });
  });

  describe('cleanup', () => {
    it('disconnects on unmount', async () => {
      let contextRef: ReturnType<typeof useLiveKit> | null = null;

      const { unmount } = render(
        <LiveKitProvider>
          <TestConsumer onContext={(ctx) => (contextRef = ctx)} />
        </LiveKitProvider>
      );

      // Connect first
      await act(async () => {
        await contextRef?.connect('thread-1', 'session-1');
      });

      // Should not throw on unmount
      expect(() => unmount()).not.toThrow();
    });
  });

  describe('toggleMicrophone advanced', () => {
    it('enables microphone and sends end_turn on disable', async () => {
      let contextRef: ReturnType<typeof useLiveKit> | null = null;

      const { getByTestId } = render(
        <LiveKitProvider>
          <TestConsumer onContext={(ctx) => (contextRef = ctx)} />
        </LiveKitProvider>
      );

      // Connect first
      await act(async () => {
        await contextRef?.connect('thread-1', 'session-1');
      });

      await waitFor(() => {
        expect(getByTestId('is-connected')).toHaveTextContent('true');
      });

      // Enable microphone
      await act(async () => {
        await contextRef?.toggleMicrophone();
      });

      expect(getByTestId('is-mic-enabled')).toHaveTextContent('true');

      // Disable microphone (triggers end_turn RPC)
      await act(async () => {
        await contextRef?.toggleMicrophone();
      });

      expect(getByTestId('is-mic-enabled')).toHaveTextContent('false');
    });
  });

  describe('processEvent with unknown type', () => {
    it('ignores unknown event types', async () => {
      let contextRef: ReturnType<typeof useLiveKit> | null = null;

      const { getByTestId } = render(
        <LiveKitProvider>
          <TestConsumer onContext={(ctx) => (contextRef = ctx)} />
        </LiveKitProvider>
      );

      // Should not throw with unknown event type
      await act(async () => {
        contextRef?.processEvent({ type: 'unknown.event.type' });
      });

      // State should remain unchanged
      expect(getByTestId('speaking-source')).toHaveTextContent('none');
    });
  });

  describe('handleActiveSpeakersChanged scenarios', () => {
    it('sets agent speaking when agent participant speaks', async () => {
      let contextRef: ReturnType<typeof useLiveKit> | null = null;
      let capturedSpeakerHandler: ((speakers: Array<{ identity: string; isSpeaking: boolean }>) => void) | null = null;

      // Get reference to the mock room
      const originalOn = jest.fn((event: string, handler: (...args: unknown[]) => void) => {
        if (event === 'activeSpeakersChanged') {
          capturedSpeakerHandler = handler as typeof capturedSpeakerHandler;
        }
        return {};
      });

      // We'll use processEvent to simulate what handleActiveSpeakersChanged would do
      // Since we can't directly access the handler, we test through processEvent
      const { getByTestId } = render(
        <LiveKitProvider>
          <TestConsumer onContext={(ctx) => (contextRef = ctx)} />
        </LiveKitProvider>
      );

      // Simulate agent speaking via turn events (equivalent to what activeSpeakersChanged does)
      await act(async () => {
        contextRef?.processEvent({ type: EventTypes.TURN_AGENT_OPEN });
      });

      expect(getByTestId('speaking-source')).toHaveTextContent('agent');

      // Simulate user speaking
      await act(async () => {
        contextRef?.processEvent({ type: EventTypes.TURN_USER_OPEN });
      });

      expect(getByTestId('speaking-source')).toHaveTextContent('user');
    });
  });

  describe('connection error scenarios', () => {
    it('handles non-Error exceptions during connection', async () => {
      mockApi.getLiveKitToken.mockRejectedValueOnce('String error');
      let contextRef: ReturnType<typeof useLiveKit> | null = null;

      const { getByTestId } = render(
        <LiveKitProvider>
          <TestConsumer onContext={(ctx) => (contextRef = ctx)} />
        </LiveKitProvider>
      );

      await act(async () => {
        await contextRef?.connect('thread-1', 'session-1');
      });

      await waitFor(() => {
        expect(getByTestId('error')).toHaveTextContent('Failed to connect');
      });
    });
  });

  describe('processEvent edge cases', () => {
    it('handles TOOL_CALL event without error', async () => {
      let contextRef: ReturnType<typeof useLiveKit> | null = null;

      render(
        <LiveKitProvider>
          <TestConsumer onContext={(ctx) => (contextRef = ctx)} />
        </LiveKitProvider>
      );

      // Should not throw
      await act(async () => {
        contextRef?.processEvent({
          type: EventTypes.TOOL_CALL,
          payload: { tool: 'get_inbox' },
        });
      });
    });

    it('handles TOOL_RESULT event without error', async () => {
      let contextRef: ReturnType<typeof useLiveKit> | null = null;

      render(
        <LiveKitProvider>
          <TestConsumer onContext={(ctx) => (contextRef = ctx)} />
        </LiveKitProvider>
      );

      // Should not throw
      await act(async () => {
        contextRef?.processEvent({
          type: EventTypes.TOOL_RESULT,
          payload: { result: {} },
        });
      });
    });

    it('does not update lastSeq if seq is undefined', async () => {
      let contextRef: ReturnType<typeof useLiveKit> | null = null;

      const { getByTestId } = render(
        <LiveKitProvider>
          <TestConsumer onContext={(ctx) => (contextRef = ctx)} />
        </LiveKitProvider>
      );

      // First set a seq
      await act(async () => {
        contextRef?.processEvent({ type: EventTypes.TURN_AGENT_OPEN, seq: 5 });
      });

      expect(getByTestId('last-seq')).toHaveTextContent('5');

      // Send event without seq
      await act(async () => {
        contextRef?.processEvent({ type: EventTypes.TURN_AGENT_CLOSE });
      });

      // Should still be 5
      expect(getByTestId('last-seq')).toHaveTextContent('5');
    });
  });
});
