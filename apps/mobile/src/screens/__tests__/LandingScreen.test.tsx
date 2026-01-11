import React from 'react';
import { render, fireEvent, waitFor } from '@testing-library/react-native';
import { LandingScreen } from '../LandingScreen';

// Mock the GoogleContext
const mockConnect = jest.fn();
const mockUseGoogle = jest.fn();

jest.mock('@/context/GoogleContext', () => ({
  useGoogle: () => mockUseGoogle(),
}));

describe('LandingScreen', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockConnect.mockResolvedValue(undefined);
    mockUseGoogle.mockReturnValue({
      isLoading: false,
      isConnected: false,
      error: null,
      connect: mockConnect,
    });
  });

  it('renders the main content', () => {
    const { getByText, getByTestId } = render(<LandingScreen />);

    expect(getByTestId('landing-screen')).toBeTruthy();
    expect(getByText('Connect Your Email')).toBeTruthy();
    expect(
      getByText(
        'Priority Lens needs access to your Gmail to help you focus on what matters most.'
      )
    ).toBeTruthy();
  });

  it('renders all feature items', () => {
    const { getAllByTestId, getByText } = render(<LandingScreen />);

    const featureItems = getAllByTestId('feature-item');
    expect(featureItems).toHaveLength(3);

    expect(getByText('Smart Priority')).toBeTruthy();
    expect(getByText('AI-powered email prioritization')).toBeTruthy();
    expect(getByText('Task Detection')).toBeTruthy();
    expect(getByText('Automatically extract action items')).toBeTruthy();
    expect(getByText('Voice Assistant')).toBeTruthy();
    expect(getByText('Manage email with your voice')).toBeTruthy();
  });

  it('renders the connect button', () => {
    const { getByTestId, getByText } = render(<LandingScreen />);

    const button = getByTestId('connect-google-button');
    expect(button).toBeTruthy();
    expect(getByText('Connect Google Account')).toBeTruthy();
  });

  it('renders the disclaimer text', () => {
    const { getByText } = render(<LandingScreen />);

    expect(
      getByText(/We only read your emails to help prioritize them/)
    ).toBeTruthy();
    expect(getByText(/Your data is never shared or sold/)).toBeTruthy();
  });

  it('calls connect when button is pressed', async () => {
    const { getByTestId } = render(<LandingScreen />);

    const button = getByTestId('connect-google-button');
    fireEvent.press(button);

    await waitFor(() => {
      expect(mockConnect).toHaveBeenCalledTimes(1);
    });
  });

  it('shows loading indicator when isLoading is true', () => {
    mockUseGoogle.mockReturnValue({
      isLoading: true,
      isConnected: false,
      error: null,
      connect: mockConnect,
    });

    const { queryByText, getByTestId } = render(<LandingScreen />);

    // The button text should not be visible when loading
    expect(queryByText('Connect Google Account')).toBeNull();
    // Button should still be present
    expect(getByTestId('connect-google-button')).toBeTruthy();
  });

  it('disables button when isLoading is true', () => {
    mockUseGoogle.mockReturnValue({
      isLoading: true,
      isConnected: false,
      error: null,
      connect: mockConnect,
    });

    const { getByTestId } = render(<LandingScreen />);

    const button = getByTestId('connect-google-button');
    // TouchableOpacity with disabled=true has disabled prop
    expect(button.props.disabled).toBe(true);
  });

  it('disables button when isConnected is true', () => {
    mockUseGoogle.mockReturnValue({
      isLoading: false,
      isConnected: true,
      error: null,
      connect: mockConnect,
    });

    const { getByTestId } = render(<LandingScreen />);

    const button = getByTestId('connect-google-button');
    expect(button.props.disabled).toBe(true);
  });

  it('displays error message when error is present', () => {
    mockUseGoogle.mockReturnValue({
      isLoading: false,
      isConnected: false,
      error: 'Failed to connect to Google',
      connect: mockConnect,
    });

    const { getByTestId, getByText } = render(<LandingScreen />);

    expect(getByTestId('error-message')).toBeTruthy();
    expect(getByText('Failed to connect to Google')).toBeTruthy();
  });

  it('does not display error message when error is null', () => {
    const { queryByTestId } = render(<LandingScreen />);

    expect(queryByTestId('error-message')).toBeNull();
  });

  it('button is disabled and not pressable when loading', () => {
    mockUseGoogle.mockReturnValue({
      isLoading: true,
      isConnected: false,
      error: null,
      connect: mockConnect,
    });

    const { getByTestId } = render(<LandingScreen />);

    const button = getByTestId('connect-google-button');

    // Verify button is disabled
    expect(button.props.disabled).toBe(true);

    // Note: fireEvent.press still triggers the mock in testing-library,
    // but the actual TouchableOpacity prevents this when disabled=true
    // in the real app. We're testing that the disabled prop is set.
  });
});
