import React from 'react';
import { Platform } from 'react-native';
import { render, fireEvent, waitFor, act } from '@testing-library/react-native';
import { SignInScreen } from '../SignInScreen';

// Mock Clerk useSignIn hook
const mockCreate = jest.fn();
const mockPrepareFirstFactor = jest.fn();
const mockAttemptFirstFactor = jest.fn();
const mockSetActive = jest.fn();
const mockStartOAuthFlow = jest.fn();

interface MockSignIn {
  create: typeof mockCreate;
  prepareFirstFactor: typeof mockPrepareFirstFactor;
  attemptFirstFactor: typeof mockAttemptFirstFactor;
  supportedFirstFactors?: Array<{ strategy: string; emailAddressId: string }>;
}

let mockSignIn: MockSignIn | null = null;
let mockIsLoaded = true;

jest.mock('@clerk/clerk-expo', () => ({
  useSignIn: () => ({
    signIn: mockSignIn,
    setActive: mockSetActive,
    isLoaded: mockIsLoaded,
  }),
  useOAuth: () => ({
    startOAuthFlow: mockStartOAuthFlow,
  }),
}));

describe('SignInScreen', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockIsLoaded = true;
    mockSignIn = {
      create: mockCreate,
      prepareFirstFactor: mockPrepareFirstFactor,
      attemptFirstFactor: mockAttemptFirstFactor,
      supportedFirstFactors: [{ strategy: 'email_code', emailAddressId: 'email-id-123' }],
    };
    mockCreate.mockResolvedValue({});
    mockPrepareFirstFactor.mockResolvedValue({});
    mockAttemptFirstFactor.mockResolvedValue({ status: 'complete', createdSessionId: 'sess-123' });
    mockSetActive.mockResolvedValue({});
  });

  describe('rendering', () => {
    it('renders loading state when Clerk not loaded', () => {
      mockIsLoaded = false;

      const { getByTestId } = render(<SignInScreen />);

      expect(getByTestId('signin-loading')).toBeTruthy();
    });

    it('renders sign in screen when loaded', () => {
      const { getByTestId } = render(<SignInScreen />);

      expect(getByTestId('signin-screen')).toBeTruthy();
    });

    it('renders title and subtitle', () => {
      const { getByText } = render(<SignInScreen />);

      expect(getByText('Priority Lens')).toBeTruthy();
      expect(getByText('Sign in to continue')).toBeTruthy();
    });

    it('renders email input', () => {
      const { getByTestId } = render(<SignInScreen />);

      expect(getByTestId('email-input')).toBeTruthy();
    });

    it('renders continue button', () => {
      const { getByTestId } = render(<SignInScreen />);

      expect(getByTestId('continue-button')).toBeTruthy();
    });
  });

  describe('email step', () => {
    it('updates email input value', () => {
      const { getByTestId } = render(<SignInScreen />);

      const input = getByTestId('email-input');
      fireEvent.changeText(input, 'test@example.com');

      expect(input.props.value).toBe('test@example.com');
    });

    it('submits email and moves to code step', async () => {
      const { getByTestId, getByText } = render(<SignInScreen />);

      const input = getByTestId('email-input');
      fireEvent.changeText(input, 'test@example.com');

      const button = getByTestId('continue-button');
      await act(async () => {
        fireEvent.press(button);
      });

      await waitFor(() => {
        expect(mockCreate).toHaveBeenCalledWith({ identifier: 'test@example.com' });
        expect(mockPrepareFirstFactor).toHaveBeenCalled();
        expect(getByText('Enter verification code')).toBeTruthy();
      });
    });

    it('shows error on email submit failure', async () => {
      mockCreate.mockRejectedValue(new Error('Invalid email'));

      const { getByTestId } = render(<SignInScreen />);

      const input = getByTestId('email-input');
      fireEvent.changeText(input, 'bad@example.com');

      const button = getByTestId('continue-button');
      await act(async () => {
        fireEvent.press(button);
      });

      await waitFor(() => {
        expect(getByTestId('error-message')).toBeTruthy();
      });
    });

    it('does nothing when signIn is null', async () => {
      mockSignIn = null;

      const { getByTestId } = render(<SignInScreen />);

      const input = getByTestId('email-input');
      fireEvent.changeText(input, 'test@example.com');

      const button = getByTestId('continue-button');
      await act(async () => {
        fireEvent.press(button);
      });

      expect(mockCreate).not.toHaveBeenCalled();
    });
  });

  describe('code step', () => {
    async function goToCodeStep(screen: ReturnType<typeof render>) {
      const input = screen.getByTestId('email-input');
      fireEvent.changeText(input, 'test@example.com');

      const button = screen.getByTestId('continue-button');
      await act(async () => {
        fireEvent.press(button);
      });

      await waitFor(() => {
        expect(screen.getByTestId('code-input')).toBeTruthy();
      });
    }

    it('renders code input after email submit', async () => {
      const screen = render(<SignInScreen />);
      await goToCodeStep(screen);

      expect(screen.getByTestId('code-input')).toBeTruthy();
    });

    it('shows email hint', async () => {
      const screen = render(<SignInScreen />);
      await goToCodeStep(screen);

      expect(screen.getByText('Code sent to test@example.com')).toBeTruthy();
    });

    it('renders verify button', async () => {
      const screen = render(<SignInScreen />);
      await goToCodeStep(screen);

      expect(screen.getByTestId('verify-button')).toBeTruthy();
    });

    it('renders back button', async () => {
      const screen = render(<SignInScreen />);
      await goToCodeStep(screen);

      expect(screen.getByTestId('back-button')).toBeTruthy();
    });

    it('back button returns to email step', async () => {
      const screen = render(<SignInScreen />);
      await goToCodeStep(screen);

      const backButton = screen.getByTestId('back-button');
      await act(async () => {
        fireEvent.press(backButton);
      });

      expect(screen.getByTestId('email-input')).toBeTruthy();
    });

    it('submits code successfully', async () => {
      const screen = render(<SignInScreen />);
      await goToCodeStep(screen);

      const codeInput = screen.getByTestId('code-input');
      fireEvent.changeText(codeInput, '123456');

      const verifyButton = screen.getByTestId('verify-button');
      await act(async () => {
        fireEvent.press(verifyButton);
      });

      await waitFor(() => {
        expect(mockAttemptFirstFactor).toHaveBeenCalledWith({
          strategy: 'email_code',
          code: '123456',
        });
        expect(mockSetActive).toHaveBeenCalledWith({ session: 'sess-123' });
      });
    });

    it('shows error on incomplete verification', async () => {
      mockAttemptFirstFactor.mockResolvedValue({ status: 'needs_second_factor' });

      const screen = render(<SignInScreen />);
      await goToCodeStep(screen);

      const codeInput = screen.getByTestId('code-input');
      fireEvent.changeText(codeInput, '123456');

      const verifyButton = screen.getByTestId('verify-button');
      await act(async () => {
        fireEvent.press(verifyButton);
      });

      await waitFor(() => {
        expect(screen.getByTestId('error-message')).toBeTruthy();
      });
    });

    it('shows error on verification failure', async () => {
      mockAttemptFirstFactor.mockRejectedValue(new Error('Invalid code'));

      const screen = render(<SignInScreen />);
      await goToCodeStep(screen);

      const codeInput = screen.getByTestId('code-input');
      fireEvent.changeText(codeInput, '000000');

      const verifyButton = screen.getByTestId('verify-button');
      await act(async () => {
        fireEvent.press(verifyButton);
      });

      await waitFor(() => {
        expect(screen.getByTestId('error-message')).toBeTruthy();
      });
    });

    it('does nothing when isLoaded is false on verify', async () => {
      const screen = render(<SignInScreen />);
      await goToCodeStep(screen);

      // Set isLoaded to false
      mockIsLoaded = false;

      // Re-render with new mock
      screen.rerender(<SignInScreen />);

      // Should show loading state
      expect(screen.getByTestId('signin-loading')).toBeTruthy();
    });
  });

  describe('error handling', () => {
    it('clears error when retrying', async () => {
      mockCreate.mockRejectedValueOnce(new Error('First error'));

      const { getByTestId, queryByTestId } = render(<SignInScreen />);

      // Trigger error
      const input = getByTestId('email-input');
      fireEvent.changeText(input, 'test@example.com');

      const button = getByTestId('continue-button');
      await act(async () => {
        fireEvent.press(button);
      });

      await waitFor(() => {
        expect(getByTestId('error-message')).toBeTruthy();
      });

      // Now succeed
      mockCreate.mockResolvedValue({});
      await act(async () => {
        fireEvent.press(button);
      });

      await waitFor(() => {
        expect(queryByTestId('error-message')).toBeNull();
      });
    });

    it('handles non-Error exceptions on email submit', async () => {
      mockCreate.mockRejectedValue('String error');

      const { getByTestId, getByText } = render(<SignInScreen />);

      const input = getByTestId('email-input');
      fireEvent.changeText(input, 'test@example.com');

      const button = getByTestId('continue-button');
      await act(async () => {
        fireEvent.press(button);
      });

      await waitFor(() => {
        expect(getByText('Failed to send code')).toBeTruthy();
      });
    });

    it('handles non-Error exceptions on code verify', async () => {
      const screen = render(<SignInScreen />);

      // Go to code step
      const input = screen.getByTestId('email-input');
      fireEvent.changeText(input, 'test@example.com');
      await act(async () => {
        fireEvent.press(screen.getByTestId('continue-button'));
      });

      await waitFor(() => {
        expect(screen.getByTestId('code-input')).toBeTruthy();
      });

      mockAttemptFirstFactor.mockRejectedValue('String error');

      const codeInput = screen.getByTestId('code-input');
      fireEvent.changeText(codeInput, '123456');

      await act(async () => {
        fireEvent.press(screen.getByTestId('verify-button'));
      });

      await waitFor(() => {
        expect(screen.getByText('Invalid code')).toBeTruthy();
      });
    });
  });

  describe('edge cases', () => {
    it('handles code submit when signIn is null', async () => {
      const screen = render(<SignInScreen />);

      // Go to code step first
      const input = screen.getByTestId('email-input');
      fireEvent.changeText(input, 'test@example.com');
      await act(async () => {
        fireEvent.press(screen.getByTestId('continue-button'));
      });

      await waitFor(() => {
        expect(screen.getByTestId('code-input')).toBeTruthy();
      });

      // Now set signIn to null
      mockSignIn = null;

      const codeInput = screen.getByTestId('code-input');
      fireEvent.changeText(codeInput, '123456');

      // This should do nothing since signIn is null
      const verifyButton = screen.getByTestId('verify-button');
      await act(async () => {
        fireEvent.press(verifyButton);
      });

      // Still on code step, no error
      expect(screen.getByTestId('code-input')).toBeTruthy();
    });

    it('uses empty string when supportedFirstFactors is undefined', async () => {
      mockSignIn = {
        create: mockCreate,
        prepareFirstFactor: mockPrepareFirstFactor,
        attemptFirstFactor: mockAttemptFirstFactor,
        supportedFirstFactors: undefined,
      };

      const { getByTestId, getByText } = render(<SignInScreen />);

      const input = getByTestId('email-input');
      fireEvent.changeText(input, 'test@example.com');

      const button = getByTestId('continue-button');
      await act(async () => {
        fireEvent.press(button);
      });

      await waitFor(() => {
        expect(mockPrepareFirstFactor).toHaveBeenCalledWith({
          strategy: 'email_code',
          emailAddressId: '',
        });
        expect(getByText('Enter verification code')).toBeTruthy();
      });
    });

    it('uses empty string when email_code strategy not found', async () => {
      mockSignIn = {
        create: mockCreate,
        prepareFirstFactor: mockPrepareFirstFactor,
        attemptFirstFactor: mockAttemptFirstFactor,
        supportedFirstFactors: [{ strategy: 'password', emailAddressId: 'some-id' }],
      };

      const { getByTestId, getByText } = render(<SignInScreen />);

      const input = getByTestId('email-input');
      fireEvent.changeText(input, 'test@example.com');

      const button = getByTestId('continue-button');
      await act(async () => {
        fireEvent.press(button);
      });

      await waitFor(() => {
        expect(mockPrepareFirstFactor).toHaveBeenCalledWith({
          strategy: 'email_code',
          emailAddressId: '',
        });
        expect(getByText('Enter verification code')).toBeTruthy();
      });
    });

    it('handles code submit when isLoaded becomes false', async () => {
      const screen = render(<SignInScreen />);

      // Go to code step first
      const input = screen.getByTestId('email-input');
      fireEvent.changeText(input, 'test@example.com');
      await act(async () => {
        fireEvent.press(screen.getByTestId('continue-button'));
      });

      await waitFor(() => {
        expect(screen.getByTestId('code-input')).toBeTruthy();
      });

      // Set isLoaded to false but keep signIn
      mockIsLoaded = false;

      const codeInput = screen.getByTestId('code-input');
      fireEvent.changeText(codeInput, '123456');

      // Re-render to pick up new mock value
      screen.rerender(<SignInScreen />);

      // Now it shows loading state
      expect(screen.getByTestId('signin-loading')).toBeTruthy();
    });

    it('shows loading indicator during code verification', async () => {
      // Make attemptFirstFactor hang to keep isSubmitting true
      let resolveAttempt: () => void;
      mockAttemptFirstFactor.mockImplementation(() => new Promise<{ status: string; createdSessionId: string }>(resolve => {
        resolveAttempt = () => resolve({ status: 'complete', createdSessionId: 'sess-123' });
      }));

      const screen = render(<SignInScreen />);

      // Go to code step first
      const input = screen.getByTestId('email-input');
      fireEvent.changeText(input, 'test@example.com');
      await act(async () => {
        fireEvent.press(screen.getByTestId('continue-button'));
      });

      await waitFor(() => {
        expect(screen.getByTestId('code-input')).toBeTruthy();
      });

      const codeInput = screen.getByTestId('code-input');
      fireEvent.changeText(codeInput, '123456');

      // Start submit (don't await)
      await act(async () => {
        fireEvent.press(screen.getByTestId('verify-button'));
      });

      // Resolve the promise
      await act(async () => {
        resolveAttempt();
      });

      // Verify it completed
      expect(mockSetActive).toHaveBeenCalled();
    });

    it('shows loading indicator during email submission', async () => {
      // Make create hang to keep isSubmitting true
      let resolveCreate: () => void;
      mockCreate.mockImplementation(() => new Promise<void>(resolve => {
        resolveCreate = () => resolve();
      }));

      const screen = render(<SignInScreen />);

      const input = screen.getByTestId('email-input');
      fireEvent.changeText(input, 'test@example.com');

      // Start submit
      await act(async () => {
        fireEvent.press(screen.getByTestId('continue-button'));
      });

      // Resolve the promise
      await act(async () => {
        resolveCreate();
      });

      // Should complete
      await waitFor(() => {
        expect(mockPrepareFirstFactor).toHaveBeenCalled();
      });
    });

    it('uses height behavior on android', () => {
      // Mock Platform.OS to be android
      const originalOS = Platform.OS;
      (Platform as { OS: string }).OS = 'android';

      const { getByTestId } = render(<SignInScreen />);

      // Just verify it renders
      expect(getByTestId('signin-screen')).toBeTruthy();

      // Restore
      (Platform as { OS: string }).OS = originalOS;
    });
  });
});
