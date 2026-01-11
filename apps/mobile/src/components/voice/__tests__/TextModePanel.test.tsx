/**
 * Tests for TextModePanel component
 */

import React from 'react';
import { render, fireEvent } from '@testing-library/react-native';

import { TextModePanel } from '../TextModePanel';

describe('TextModePanel', () => {
  const defaultProps = {
    onSendMessage: jest.fn(),
    onToggleMode: jest.fn(),
  };

  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('renders with testID', () => {
    const { getByTestId } = render(<TextModePanel {...defaultProps} testID="panel" />);
    expect(getByTestId('panel')).toBeTruthy();
  });

  it('renders voice toggle button', () => {
    const { getByTestId } = render(<TextModePanel {...defaultProps} testID="panel" />);
    expect(getByTestId('panel-voice-toggle')).toBeTruthy();
  });

  it('renders input container', () => {
    const { getByTestId } = render(<TextModePanel {...defaultProps} testID="panel" />);
    expect(getByTestId('panel-input-container')).toBeTruthy();
  });

  it('renders text input', () => {
    const { getByTestId } = render(<TextModePanel {...defaultProps} testID="panel" />);
    expect(getByTestId('panel-input')).toBeTruthy();
  });

  it('renders send button by default', () => {
    const { getByTestId } = render(<TextModePanel {...defaultProps} testID="panel" />);
    // Send button shows when keyboard is not visible (default state)
    expect(getByTestId('panel-send')).toBeTruthy();
  });

  it('applies custom style', () => {
    const customStyle = { marginTop: 20 };
    const { getByTestId } = render(
      <TextModePanel {...defaultProps} style={customStyle} testID="panel" />
    );
    const container = getByTestId('panel');
    expect(container.props.style).toEqual(
      expect.arrayContaining([expect.objectContaining(customStyle)])
    );
  });

  describe('input behavior', () => {
    it('uses default placeholder', () => {
      const { getByTestId } = render(<TextModePanel {...defaultProps} testID="panel" />);
      expect(getByTestId('panel-input').props.placeholder).toBe('Message Lenso...');
    });

    it('uses custom placeholder', () => {
      const { getByTestId } = render(
        <TextModePanel {...defaultProps} placeholder="Custom placeholder" testID="panel" />
      );
      expect(getByTestId('panel-input').props.placeholder).toBe('Custom placeholder');
    });

    it('updates input value on change', () => {
      const { getByTestId } = render(<TextModePanel {...defaultProps} testID="panel" />);
      const input = getByTestId('panel-input');
      fireEvent.changeText(input, 'Hello');
      expect(input.props.value).toBe('Hello');
    });
  });

  describe('send button', () => {
    it('send button is disabled when input is empty', () => {
      const { getByTestId } = render(<TextModePanel {...defaultProps} testID="panel" />);
      const sendButton = getByTestId('panel-send');
      fireEvent.press(sendButton);
      expect(defaultProps.onSendMessage).not.toHaveBeenCalled();
    });

    it('send button is disabled when input has only whitespace', () => {
      const { getByTestId } = render(<TextModePanel {...defaultProps} testID="panel" />);
      const input = getByTestId('panel-input');
      fireEvent.changeText(input, '   ');
      const sendButton = getByTestId('panel-send');
      fireEvent.press(sendButton);
      expect(defaultProps.onSendMessage).not.toHaveBeenCalled();
    });

    it('calls onSendMessage when send button is pressed with text', () => {
      const onSendMessage = jest.fn();
      const { getByTestId } = render(
        <TextModePanel {...defaultProps} onSendMessage={onSendMessage} testID="panel" />
      );
      const input = getByTestId('panel-input');
      fireEvent.changeText(input, 'Hello');
      const sendButton = getByTestId('panel-send');
      fireEvent.press(sendButton);
      expect(onSendMessage).toHaveBeenCalledWith('Hello');
    });

    it('trims message before sending', () => {
      const onSendMessage = jest.fn();
      const { getByTestId } = render(
        <TextModePanel {...defaultProps} onSendMessage={onSendMessage} testID="panel" />
      );
      const input = getByTestId('panel-input');
      fireEvent.changeText(input, '  Hello World  ');
      const sendButton = getByTestId('panel-send');
      fireEvent.press(sendButton);
      expect(onSendMessage).toHaveBeenCalledWith('Hello World');
    });

    it('clears input after sending', () => {
      const onSendMessage = jest.fn();
      const { getByTestId } = render(
        <TextModePanel {...defaultProps} onSendMessage={onSendMessage} testID="panel" />
      );
      const input = getByTestId('panel-input');
      fireEvent.changeText(input, 'Hello');
      const sendButton = getByTestId('panel-send');
      fireEvent.press(sendButton);
      expect(input.props.value).toBe('');
    });

    it('send button is disabled when isSending is true', () => {
      const onSendMessage = jest.fn();
      const { getByTestId } = render(
        <TextModePanel
          {...defaultProps}
          onSendMessage={onSendMessage}
          isSending={true}
          testID="panel"
        />
      );
      const input = getByTestId('panel-input');
      fireEvent.changeText(input, 'Hello');
      const sendButton = getByTestId('panel-send');
      fireEvent.press(sendButton);
      expect(onSendMessage).not.toHaveBeenCalled();
    });
  });

  describe('voice toggle', () => {
    it('calls onToggleMode when voice toggle is pressed', () => {
      const onToggleMode = jest.fn();
      const { getByTestId } = render(
        <TextModePanel {...defaultProps} onToggleMode={onToggleMode} testID="panel" />
      );
      fireEvent.press(getByTestId('panel-voice-toggle'));
      expect(onToggleMode).toHaveBeenCalledTimes(1);
    });
  });

  describe('submit editing', () => {
    it('calls onSendMessage on submit editing with text', () => {
      const onSendMessage = jest.fn();
      const { getByTestId } = render(
        <TextModePanel {...defaultProps} onSendMessage={onSendMessage} testID="panel" />
      );
      const input = getByTestId('panel-input');
      fireEvent.changeText(input, 'Hello');
      fireEvent(input, 'submitEditing');
      expect(onSendMessage).toHaveBeenCalledWith('Hello');
    });
  });
});
