/* eslint-disable @typescript-eslint/no-require-imports */
/**
 * Jest setup file for React Native Testing Library
 *
 * This file runs before each test file and sets up the testing environment.
 */

// Mock expo module
jest.mock('expo', () => ({
  registerRootComponent: jest.fn(),
}));

// Mock expo-linking to avoid native module requirement in tests
jest.mock('expo-linking', () => ({
  createURL: jest.fn((path) => {
    const normalized = String(path ?? '').replace(/^\//, '');
    return `priority-lens://${normalized}`;
  }),
}));

// Mock react-native with proper component implementations
jest.mock('react-native', () => {
  const React = require('react');

  // Create mock components that render their children
  const createMockComponent = (name) => {
    const Component = (props) => {
      return React.createElement(name, props, props.children);
    };
    Component.displayName = name;
    return Component;
  };

  // Mock Animated.Value class
  class MockAnimatedValue {
    constructor(value) {
      this._value = value;
    }
    setValue(value) {
      this._value = value;
    }
    interpolate() {
      return this;
    }
  }

  // Factory function to create mock animations
  function createMockAnimation() {
    return {
      start: function(callback) {
        if (callback) {
          setTimeout(function() { callback({ finished: true }); }, 0);
        }
      },
      stop: function() {},
      reset: function() {},
    };
  }

  return {
    Platform: {
      OS: 'ios',
      select: jest.fn((obj) => obj.ios ?? obj.default),
    },
    StyleSheet: {
      create: (styles) => styles,
      flatten: (style) => style,
    },
    Dimensions: {
      get: jest.fn(() => ({ width: 375, height: 812 })),
    },
    Animated: {
      Value: MockAnimatedValue,
      View: createMockComponent('Animated.View'),
      Text: createMockComponent('Animated.Text'),
      timing: function() { return createMockAnimation(); },
      spring: function() { return createMockAnimation(); },
      loop: function(animation) {
        return {
          start: function(callback) {
            if (callback) setTimeout(function() { callback({ finished: true }); }, 0);
          },
          stop: function() {},
        };
      },
      sequence: function() { return createMockAnimation(); },
      parallel: function() { return createMockAnimation(); },
      delay: function() { return createMockAnimation(); },
    },
    View: createMockComponent('View'),
    Text: createMockComponent('Text'),
    TextInput: createMockComponent('TextInput'),
    TouchableOpacity: (props) => {
      const React = require('react');
      const { onPress, disabled, children, ...rest } = props;
      // Create a press handler that respects disabled prop
      const handlePress = () => {
        if (!disabled && onPress) {
          onPress();
        }
      };
      return React.createElement('TouchableOpacity', {
        ...rest,
        onPress: handlePress,
        disabled,
      }, children);
    },
    KeyboardAvoidingView: createMockComponent('KeyboardAvoidingView'),
    ActivityIndicator: createMockComponent('ActivityIndicator'),
    ScrollView: createMockComponent('ScrollView'),
    SafeAreaView: createMockComponent('SafeAreaView'),
    Image: createMockComponent('Image'),
    Pressable: createMockComponent('Pressable'),
    NativeModules: {},
    Alert: {
      alert: jest.fn(),
    },
    Keyboard: {
      addListener: jest.fn().mockImplementation(() => ({ remove: jest.fn() })),
      dismiss: jest.fn(),
      removeAllListeners: jest.fn(),
    },
  };
});

// Mock react-native-reanimated with inline mock (compatible with newer versions)
jest.mock('react-native-reanimated', () => {
  const React = require('react');

  const createMockComponent = (name) => {
    const Component = (props) => {
      return React.createElement(name, props, props.children);
    };
    Component.displayName = name;
    return Component;
  };

  // Create a shared value mock that allows property assignment
  const createSharedValue = (initial) => {
    let _value = initial;
    return {
      get value() { return _value; },
      set value(v) { _value = v; },
    };
  };

  return {
    default: {
      call: jest.fn(),
      createAnimatedComponent: (component) => component,
      View: createMockComponent('Animated.View'),
      Text: createMockComponent('Animated.Text'),
    },
    View: createMockComponent('Animated.View'),
    Text: createMockComponent('Animated.Text'),
    useSharedValue: jest.fn(createSharedValue),
    useAnimatedStyle: jest.fn(() => ({})),
    useDerivedValue: jest.fn((fn) => ({ value: fn() })),
    useAnimatedProps: jest.fn(() => ({})),
    withTiming: jest.fn((value) => value),
    withSpring: jest.fn((value) => value),
    withRepeat: jest.fn((value) => value),
    withSequence: jest.fn((...values) => values[0]),
    withDelay: jest.fn((_, value) => value),
    Easing: {
      linear: jest.fn(),
      ease: jest.fn(),
      inOut: jest.fn((easing) => easing),
    },
    interpolate: jest.fn((value) => value),
    Extrapolate: {
      CLAMP: 'clamp',
      EXTEND: 'extend',
    },
    runOnJS: jest.fn((fn) => fn),
    runOnUI: jest.fn((fn) => fn),
    cancelAnimation: jest.fn(),
    createAnimatedComponent: (component) => component,
  };
});

// Mock expo-secure-store
jest.mock('expo-secure-store', () => ({
  getItemAsync: jest.fn(),
  setItemAsync: jest.fn(),
  deleteItemAsync: jest.fn(),
}));

// Mock expo-haptics
jest.mock('expo-haptics', () => ({
  impactAsync: jest.fn(),
  notificationAsync: jest.fn(),
  selectionAsync: jest.fn(),
  ImpactFeedbackStyle: {
    Light: 'Light',
    Medium: 'Medium',
    Heavy: 'Heavy',
  },
  NotificationFeedbackType: {
    Success: 'Success',
    Warning: 'Warning',
    Error: 'Error',
  },
}));

// Mock expo-av
jest.mock('expo-av', () => ({
  Audio: {
    setAudioModeAsync: jest.fn(),
    Sound: {
      createAsync: jest.fn(),
    },
  },
}));

// Mock @clerk/clerk-expo
jest.mock('@clerk/clerk-expo', () => ({
  ClerkProvider: ({ children }) => children,
  useAuth: jest.fn(() => ({
    isLoaded: true,
    isSignedIn: false,
    userId: null,
    signOut: jest.fn(),
    getToken: jest.fn(),
  })),
  useUser: jest.fn(() => ({
    isLoaded: true,
    user: null,
  })),
  useSignIn: jest.fn(() => ({
    signIn: null,
    setActive: jest.fn(),
    isLoaded: true,
  })),
  useSignUp: jest.fn(() => ({
    signUp: null,
    setActive: jest.fn(),
    isLoaded: true,
  })),
  useClerk: jest.fn(() => ({
    signOut: jest.fn(),
  })),
}));

// Mock @react-native-google-signin/google-signin
jest.mock('@react-native-google-signin/google-signin', () => ({
  GoogleSignin: {
    configure: jest.fn(),
    hasPlayServices: jest.fn(() => Promise.resolve(true)),
    signIn: jest.fn(),
    signOut: jest.fn(),
    isSignedIn: jest.fn(() => Promise.resolve(false)),
    getCurrentUser: jest.fn(() => Promise.resolve(null)),
    getTokens: jest.fn(),
    revokeAccess: jest.fn(),
  },
  statusCodes: {
    SIGN_IN_CANCELLED: 'SIGN_IN_CANCELLED',
    IN_PROGRESS: 'IN_PROGRESS',
    PLAY_SERVICES_NOT_AVAILABLE: 'PLAY_SERVICES_NOT_AVAILABLE',
  },
}));

// Mock @livekit/react-native
jest.mock('@livekit/react-native', () => ({
  useRoom: jest.fn(() => ({
    room: null,
    connect: jest.fn(),
    disconnect: jest.fn(),
  })),
  useParticipant: jest.fn(() => ({
    isSpeaking: false,
    isMicrophoneEnabled: false,
  })),
  useLocalParticipant: jest.fn(() => ({
    localParticipant: null,
    isMicrophoneEnabled: false,
  })),
  AudioSession: {
    startAudioSession: jest.fn(),
    stopAudioSession: jest.fn(),
    configureAudio: jest.fn(),
  },
  registerGlobals: jest.fn(),
}));

// Mock livekit-client with a MockRoom class for LiveKitContext
jest.mock('livekit-client', () => {
  class MockRoom {
    _connected = false;
    _handlers = new Map();
    localParticipant = {
      setMicrophoneEnabled: jest.fn().mockResolvedValue(undefined),
    };

    get state() {
      return this._connected ? 'connected' : 'disconnected';
    }

    async connect() {
      this._connected = true;
    }

    async disconnect() {
      this._connected = false;
    }

    on(event, handler) {
      this._handlers.set(event, handler);
      return this;
    }

    off(event, handler) {
      this._handlers.delete(event);
      return this;
    }

    async performRpc() {
      return '';
    }
  }

  return {
    Room: MockRoom,
    RoomEvent: {
      Connected: 'connected',
      Disconnected: 'disconnected',
      DataReceived: 'dataReceived',
      ParticipantConnected: 'participantConnected',
      ActiveSpeakersChanged: 'activeSpeakersChanged',
    },
  };
});

// Mock @react-navigation
jest.mock('@react-navigation/native', () => ({
  useNavigation: jest.fn(() => ({
    navigate: jest.fn(),
    goBack: jest.fn(),
    reset: jest.fn(),
  })),
  useRoute: jest.fn(() => ({
    params: {},
  })),
  useFocusEffect: jest.fn(),
  NavigationContainer: ({ children }) => children,
}));

// Silence console warnings during tests (optional - can be removed for debugging)
const originalConsoleWarn = console.warn;
console.warn = (...args) => {
  if (
    typeof args[0] === 'string' &&
    args[0].includes('Animated: `useNativeDriver`')
  ) {
    return;
  }
  originalConsoleWarn(...args);
};

// Silence react-test-renderer deprecation warnings
const originalConsoleError = console.error;
console.error = (...args) => {
  if (
    typeof args[0] === 'string' &&
    args[0].includes('react-test-renderer is deprecated')
  ) {
    return;
  }
  originalConsoleError(...args);
};

// Global fetch mock for API tests
global.fetch = jest.fn(() =>
  Promise.resolve({
    ok: true,
    json: () => Promise.resolve({}),
    text: () => Promise.resolve(''),
    status: 200,
    headers: new Headers(),
  })
);
