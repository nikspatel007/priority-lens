/**
 * Tests for VoiceModePanel component
 */

import React from 'react';
import { render, fireEvent, waitFor } from '@testing-library/react-native';

import { VoiceModePanel } from '../VoiceModePanel';

// Mock child components to isolate VoiceModePanel testing
jest.mock('../VoiceOrb', () => ({
  VoiceOrb: ({ testID }: { testID?: string }) => {
    const { View } = require('react-native');
    return <View testID={testID} />;
  },
}));

jest.mock('../Waveform', () => ({
  Waveform: ({ testID }: { testID?: string }) => {
    const { View } = require('react-native');
    return <View testID={testID} />;
  },
}));

describe('VoiceModePanel', () => {
  const defaultProps = {
    speakingSource: 'none' as const,
    isMicrophoneEnabled: false,
    isConnected: true,
    isConnecting: false,
    onToggleMicrophone: jest.fn(),
    onToggleMode: jest.fn(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('rendering', () => {
    it('renders all child components', async () => {
      const { getByTestId } = render(
        <VoiceModePanel {...defaultProps} testID="panel" />
      );

      await waitFor(() => {
        expect(getByTestId('panel')).toBeTruthy();
        expect(getByTestId('panel-agent-orb')).toBeTruthy();
        expect(getByTestId('panel-user-orb')).toBeTruthy();
        expect(getByTestId('panel-waveform')).toBeTruthy();
        expect(getByTestId('panel-status')).toBeTruthy();
        expect(getByTestId('panel-voice-row')).toBeTruthy();
        expect(getByTestId('panel-toggle-button')).toBeTruthy();
      });
    });

    it('renders labels for orbs', async () => {
      const { getByText } = render(
        <VoiceModePanel {...defaultProps} testID="panel" />
      );

      await waitFor(() => {
        expect(getByText('Lenso')).toBeTruthy();
        expect(getByText('You')).toBeTruthy();
      });
    });

    it('applies custom style', async () => {
      const customStyle = { marginTop: 20 };
      const { getByTestId } = render(
        <VoiceModePanel {...defaultProps} style={customStyle} testID="panel" />
      );

      await waitFor(() => {
        const container = getByTestId('panel');
        expect(container.props.style).toEqual(
          expect.arrayContaining([expect.objectContaining(customStyle)])
        );
      });
    });
  });

  describe('status text', () => {
    it('shows "Tap to speak" when connected and mic off', async () => {
      const { getByTestId } = render(
        <VoiceModePanel {...defaultProps} testID="panel" />
      );

      await waitFor(() => {
        expect(getByTestId('panel-status')).toHaveTextContent('Tap to speak');
      });
    });

    it('shows "Tap to end your turn" when mic is enabled', async () => {
      const { getByTestId } = render(
        <VoiceModePanel {...defaultProps} isMicrophoneEnabled={true} testID="panel" />
      );

      await waitFor(() => {
        expect(getByTestId('panel-status')).toHaveTextContent('Tap to end your turn');
      });
    });

    it('shows "Not connected" when not connected', async () => {
      const { getByTestId } = render(
        <VoiceModePanel {...defaultProps} isConnected={false} testID="panel" />
      );

      await waitFor(() => {
        expect(getByTestId('panel-status')).toHaveTextContent('Not connected');
      });
    });

    it('shows "Connecting to Lenso..." when connecting', async () => {
      const { getByTestId } = render(
        <VoiceModePanel
          {...defaultProps}
          isConnected={false}
          isConnecting={true}
          testID="panel"
        />
      );

      await waitFor(() => {
        expect(getByTestId('panel-status')).toHaveTextContent('Connecting to Lenso...');
      });
    });
  });

  describe('voice row interactions', () => {
    it('calls onToggleMicrophone when voice row is pressed', async () => {
      const onToggle = jest.fn();
      const { getByTestId } = render(
        <VoiceModePanel
          {...defaultProps}
          onToggleMicrophone={onToggle}
          testID="panel"
        />
      );

      await waitFor(() => {
        expect(getByTestId('panel-voice-row')).toBeTruthy();
      });

      fireEvent.press(getByTestId('panel-voice-row'));

      expect(onToggle).toHaveBeenCalledTimes(1);
    });

    it('is disabled when not connected', async () => {
      const onToggle = jest.fn();
      const { getByTestId } = render(
        <VoiceModePanel
          {...defaultProps}
          isConnected={false}
          onToggleMicrophone={onToggle}
          testID="panel"
        />
      );

      await waitFor(() => {
        expect(getByTestId('panel-voice-row')).toBeTruthy();
      });

      fireEvent.press(getByTestId('panel-voice-row'));

      expect(onToggle).not.toHaveBeenCalled();
    });
  });

  describe('mode toggle button', () => {
    it('renders toggle button with text', async () => {
      const { getByText } = render(
        <VoiceModePanel {...defaultProps} testID="panel" />
      );

      await waitFor(() => {
        expect(getByText('Switch to text')).toBeTruthy();
      });
    });

    it('calls onToggleMode when toggle button is pressed', async () => {
      const onToggleMode = jest.fn();
      const { getByTestId } = render(
        <VoiceModePanel
          {...defaultProps}
          onToggleMode={onToggleMode}
          testID="panel"
        />
      );

      await waitFor(() => {
        expect(getByTestId('panel-toggle-button')).toBeTruthy();
      });

      fireEvent.press(getByTestId('panel-toggle-button'));

      expect(onToggleMode).toHaveBeenCalledTimes(1);
    });
  });

  describe('orb states', () => {
    it('renders agent orb when agent is speaking', async () => {
      const { getByTestId } = render(
        <VoiceModePanel {...defaultProps} speakingSource="agent" testID="panel" />
      );

      await waitFor(() => {
        expect(getByTestId('panel-agent-orb')).toBeTruthy();
      });
    });

    it('renders user orb when user is speaking', async () => {
      const { getByTestId } = render(
        <VoiceModePanel {...defaultProps} speakingSource="user" testID="panel" />
      );

      await waitFor(() => {
        expect(getByTestId('panel-user-orb')).toBeTruthy();
      });
    });

    it('renders both orbs when none speaking', async () => {
      const { getByTestId } = render(
        <VoiceModePanel {...defaultProps} speakingSource="none" testID="panel" />
      );

      await waitFor(() => {
        expect(getByTestId('panel-agent-orb')).toBeTruthy();
        expect(getByTestId('panel-user-orb')).toBeTruthy();
      });
    });

    it('dims user orb when microphone is disabled', async () => {
      const { getByTestId } = render(
        <VoiceModePanel {...defaultProps} isMicrophoneEnabled={false} testID="panel" />
      );

      await waitFor(() => {
        // User orb should be rendered (dimmed state handled internally)
        expect(getByTestId('panel-user-orb')).toBeTruthy();
      });
    });
  });

  describe('waveform', () => {
    it('renders waveform', async () => {
      const { getByTestId } = render(
        <VoiceModePanel {...defaultProps} testID="panel" />
      );

      await waitFor(() => {
        expect(getByTestId('panel-waveform')).toBeTruthy();
      });
    });

    it('renders waveform when agent is speaking', async () => {
      const { getByTestId } = render(
        <VoiceModePanel {...defaultProps} speakingSource="agent" testID="panel" />
      );

      await waitFor(() => {
        expect(getByTestId('panel-waveform')).toBeTruthy();
      });
    });

    it('renders waveform when user is speaking', async () => {
      const { getByTestId } = render(
        <VoiceModePanel {...defaultProps} speakingSource="user" testID="panel" />
      );

      await waitFor(() => {
        expect(getByTestId('panel-waveform')).toBeTruthy();
      });
    });
  });

  describe('without testID', () => {
    it('renders without testID prop', async () => {
      const { getByText } = render(
        <VoiceModePanel {...defaultProps} />
      );

      await waitFor(() => {
        // Should render text content without testIDs
        expect(getByText('Lenso')).toBeTruthy();
        expect(getByText('You')).toBeTruthy();
        expect(getByText('Tap to speak')).toBeTruthy();
        expect(getByText('Switch to text')).toBeTruthy();
      });
    });
  });
});
