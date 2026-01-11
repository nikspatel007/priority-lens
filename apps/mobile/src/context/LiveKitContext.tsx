/**
 * LiveKit Voice Context
 *
 * Provides voice conversation capabilities via LiveKit.
 * Handles room connection, microphone control, and canonical event parsing.
 */

import React, {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from 'react';

import { getLiveKitToken } from '../services/api';
import type { UIBlock } from '../types/api';

// Event types from Priority Lens backend
export const EventTypes = {
  TURN_USER_OPEN: 'turn.user.open',
  TURN_USER_CLOSE: 'turn.user.close',
  TURN_AGENT_OPEN: 'turn.agent.open',
  TURN_AGENT_CLOSE: 'turn.agent.close',
  UI_BLOCK: 'ui.block',
  UI_CLEAR: 'ui.clear',
  ASSISTANT_TEXT_FINAL: 'assistant.text.final',
  TOOL_CALL: 'tool.call',
  TOOL_RESULT: 'tool.result',
  SYSTEM_CANCEL: 'system.cancel',
} as const;

export type SpeakingSource = 'agent' | 'user' | 'none';

export interface CanonicalEvent {
  type: string;
  payload?: Record<string, unknown>;
  seq?: number;
}

export interface LiveKitContextType {
  /** Whether connected to a LiveKit room */
  isConnected: boolean;
  /** Whether currently connecting */
  isConnecting: boolean;
  /** Current speaking source */
  speakingSource: SpeakingSource;
  /** Whether microphone is enabled */
  isMicrophoneEnabled: boolean;
  /** SDUI blocks received from agent */
  sduiBlocks: UIBlock[];
  /** Last event sequence number */
  lastSeq: number;
  /** Latest agent text response */
  agentText: string;
  /** Error message if any */
  error: string | null;
  /** Connect to a LiveKit room */
  connect: (threadId: string, sessionId: string) => Promise<void>;
  /** Disconnect from the room */
  disconnect: () => Promise<void>;
  /** Toggle microphone on/off */
  toggleMicrophone: () => Promise<void>;
  /** Clear SDUI blocks */
  clearSduiBlocks: () => void;
  /** Process a canonical event (for testing/manual events) */
  processEvent: (event: CanonicalEvent) => void;
}

const LiveKitContext = createContext<LiveKitContextType | null>(null);

// Mock Room class for when LiveKit is not available
/* istanbul ignore next */
class MockRoom {
  private _connected = false;
  private _localParticipant = {
    setMicrophoneEnabled: async (_enabled: boolean) => {},
  };

  get state() {
    return this._connected ? 'connected' : 'disconnected';
  }

  get localParticipant() {
    return this._localParticipant;
  }

  async connect(_url: string, _token: string) {
    this._connected = true;
  }

  async disconnect() {
    this._connected = false;
  }

  on(_event: string, _handler: (...args: unknown[]) => void) {
    return this;
  }

  off(_event: string, _handler: (...args: unknown[]) => void) {
    return this;
  }

  async performRpc(_params: { destinationIdentity: string; method: string; payload: string }) {
    return '';
  }
}

// RoomEvent type with known event names
interface RoomEventType {
  Connected: string;
  Disconnected: string;
  DataReceived: string;
  ParticipantConnected: string;
  ActiveSpeakersChanged: string;
}

// Try to import real LiveKit, fall back to mock
let RoomClass: typeof MockRoom;
let RoomEvent: RoomEventType;
/* istanbul ignore next */
try {
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const livekit = require('livekit-client');
  // eslint-disable-next-line @typescript-eslint/no-require-imports
  const livekitReactNative = require('@livekit/react-native');

  // Register WebRTC globals for React Native
  if (livekitReactNative.registerGlobals) {
    livekitReactNative.registerGlobals();
    console.log('LiveKit: WebRTC globals registered');
  }

  RoomClass = livekit.Room;
  RoomEvent = livekit.RoomEvent;
} catch (e) {
  console.log('LiveKit: Using mock (native not available)', e);
  RoomClass = MockRoom;
  RoomEvent = {
    Connected: 'connected',
    Disconnected: 'disconnected',
    DataReceived: 'dataReceived',
    ParticipantConnected: 'participantConnected',
    ActiveSpeakersChanged: 'activeSpeakersChanged',
  };
}

interface LiveKitProviderProps {
  children: ReactNode;
  /** LiveKit server URL */
  liveKitUrl?: string;
}

export function LiveKitProvider({
  children,
  liveKitUrl = process.env.EXPO_PUBLIC_LIVEKIT_URL || 'wss://localhost:7880',
}: LiveKitProviderProps) {
  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [speakingSource, setSpeakingSource] = useState<SpeakingSource>('none');
  const [isMicrophoneEnabled, setIsMicrophoneEnabled] = useState(false);
  const [sduiBlocks, setSduiBlocks] = useState<UIBlock[]>([]);
  const [lastSeq, setLastSeq] = useState(0);
  const [agentText, setAgentText] = useState('');
  const [error, setError] = useState<string | null>(null);

  const roomRef = useRef<InstanceType<typeof RoomClass> | null>(null);
  const currentThreadId = useRef<string | null>(null);
  const currentSessionId = useRef<string | null>(null);

  // Process canonical events from the agent
  const processEvent = useCallback((event: CanonicalEvent) => {
    const { type, payload, seq } = event;

    // Update sequence number
    if (seq !== undefined && seq > lastSeq) {
      setLastSeq(seq);
    }

    switch (type) {
      case EventTypes.TURN_AGENT_OPEN:
        setSpeakingSource('agent');
        break;

      case EventTypes.TURN_AGENT_CLOSE:
        setSpeakingSource('none');
        break;

      case EventTypes.TURN_USER_OPEN:
        setSpeakingSource('user');
        break;

      case EventTypes.TURN_USER_CLOSE:
        setSpeakingSource('none');
        break;

      case EventTypes.UI_BLOCK:
        if (payload?.block) {
          setSduiBlocks((prev) => [...prev, payload.block as UIBlock]);
        }
        break;

      case EventTypes.UI_CLEAR:
        setSduiBlocks([]);
        break;

      case EventTypes.ASSISTANT_TEXT_FINAL:
        if (payload?.text) {
          setAgentText(payload.text as string);
        }
        break;

      case EventTypes.SYSTEM_CANCEL:
        setSpeakingSource('none');
        break;
    }
  }, [lastSeq]);

  // Handle data received from LiveKit
  /* istanbul ignore next - callback only called by real LiveKit room */
  const handleDataReceived = useCallback(
    (payload: Uint8Array) => {
      try {
        const decoder = new TextDecoder();
        const jsonStr = decoder.decode(payload);
        const event = JSON.parse(jsonStr) as CanonicalEvent;
        processEvent(event);
      } catch (err) {
        console.error('Failed to parse LiveKit data:', err);
      }
    },
    [processEvent]
  );

  // Handle active speakers changed
  /* istanbul ignore next - callback only called by real LiveKit room */
  const handleActiveSpeakersChanged = useCallback(
    (speakers: Array<{ identity: string; isSpeaking: boolean }>) => {
      const agentSpeaking = speakers.some(
        (s) => s.identity?.startsWith('agent') && s.isSpeaking
      );
      const userSpeaking = speakers.some(
        (s) => !s.identity?.startsWith('agent') && s.isSpeaking
      );

      if (agentSpeaking) {
        setSpeakingSource('agent');
      } else if (userSpeaking) {
        setSpeakingSource('user');
      } else {
        setSpeakingSource('none');
      }
    },
    []
  );

  // Connect to LiveKit room
  const connect = useCallback(
    async (threadId: string, sessionId: string) => {
      if (isConnecting || isConnected) {
        console.log('LiveKit: Already connecting or connected');
        return;
      }

      setIsConnecting(true);
      setError(null);

      try {
        // Get token from backend
        console.log('LiveKit: Getting token for thread:', threadId, 'session:', sessionId);
        const { token } = await getLiveKitToken({
          thread_id: threadId,
          session_id: sessionId,
        });
        console.log('LiveKit: Token received');

        // Create and configure room
        const room = new RoomClass();
        roomRef.current = room;
        currentThreadId.current = threadId;
        currentSessionId.current = sessionId;

        // Set up event handlers
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        room.on(RoomEvent.DataReceived, handleDataReceived as any);
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        room.on(RoomEvent.ActiveSpeakersChanged, handleActiveSpeakersChanged as any);
        /* istanbul ignore next - callback only called by real LiveKit room */
        room.on(RoomEvent.Disconnected, () => {
          console.log('LiveKit: Disconnected');
          setIsConnected(false);
          setSpeakingSource('none');
        });

        // Connect to the room
        console.log('LiveKit: Connecting to', liveKitUrl);
        await room.connect(liveKitUrl, token);
        console.log('LiveKit: Connected successfully');

        setIsConnected(true);
        setIsConnecting(false);
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'Failed to connect';
        console.error('LiveKit: Connection error:', errorMessage);
        setError(errorMessage);
        setIsConnecting(false);
        roomRef.current = null;
      }
    },
    [isConnecting, isConnected, liveKitUrl, handleDataReceived, handleActiveSpeakersChanged]
  );

  // Disconnect from LiveKit room
  const disconnect = useCallback(async () => {
    const room = roomRef.current;
    if (room) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      room.off(RoomEvent.DataReceived, handleDataReceived as any);
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      room.off(RoomEvent.ActiveSpeakersChanged, handleActiveSpeakersChanged as any);
      await room.disconnect();
      roomRef.current = null;
    }

    setIsConnected(false);
    setIsConnecting(false);
    setSpeakingSource('none');
    setIsMicrophoneEnabled(false);
    currentThreadId.current = null;
    currentSessionId.current = null;
  }, [handleDataReceived, handleActiveSpeakersChanged]);

  // Toggle microphone
  const toggleMicrophone = useCallback(async () => {
    const room = roomRef.current;
    if (!room || !isConnected) {
      return;
    }

    const newState = !isMicrophoneEnabled;

    try {
      await room.localParticipant.setMicrophoneEnabled(newState);
      setIsMicrophoneEnabled(newState);

      // Send end_turn RPC when disabling microphone
      if (!newState) {
        try {
          await room.performRpc({
            destinationIdentity: 'agent',
            method: 'end_turn',
            payload: JSON.stringify({}),
          });
        } catch {
          // Ignore RPC errors - agent may not support it
        }
      }
    } catch (err) {
      /* istanbul ignore next - only triggered with real LiveKit room errors */
      setError(err instanceof Error ? err.message : 'Failed to toggle microphone');
    }
  }, [isConnected, isMicrophoneEnabled]);

  // Clear SDUI blocks
  const clearSduiBlocks = useCallback(() => {
    setSduiBlocks([]);
  }, []);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (roomRef.current) {
        roomRef.current.disconnect();
        roomRef.current = null;
      }
    };
  }, []);

  const value = useMemo<LiveKitContextType>(
    () => ({
      isConnected,
      isConnecting,
      speakingSource,
      isMicrophoneEnabled,
      sduiBlocks,
      lastSeq,
      agentText,
      error,
      connect,
      disconnect,
      toggleMicrophone,
      clearSduiBlocks,
      processEvent,
    }),
    [
      isConnected,
      isConnecting,
      speakingSource,
      isMicrophoneEnabled,
      sduiBlocks,
      lastSeq,
      agentText,
      error,
      connect,
      disconnect,
      toggleMicrophone,
      clearSduiBlocks,
      processEvent,
    ]
  );

  return <LiveKitContext.Provider value={value}>{children}</LiveKitContext.Provider>;
}

export function useLiveKit(): LiveKitContextType {
  const context = useContext(LiveKitContext);
  if (!context) {
    throw new Error('useLiveKit must be used within a LiveKitProvider');
  }
  return context;
}

export { LiveKitContext };
