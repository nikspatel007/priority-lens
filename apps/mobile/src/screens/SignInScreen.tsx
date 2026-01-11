import React, { useCallback, useState } from 'react';
import {
  View,
  Text,
  TextInput,
  TouchableOpacity,
  StyleSheet,
  KeyboardAvoidingView,
  Platform,
  ActivityIndicator,
} from 'react-native';
import { useSignIn, useOAuth } from '@clerk/clerk-expo';
import * as Linking from 'expo-linking';
import { colors, typography, spacing, borderRadius } from '@/theme';

type SignInStep = 'email' | 'code';

/**
 * SignInScreen renders the Clerk authentication UI
 *
 * Two-step flow:
 * 1. Enter email
 * 2. Enter verification code
 */
export function SignInScreen(): React.JSX.Element {
  const { signIn, setActive, isLoaded } = useSignIn();
  const { startOAuthFlow: startGoogleOAuthFlow } = useOAuth({
    strategy: 'oauth_google',
  });
  const [step, setStep] = useState<SignInStep>('email');
  const [email, setEmail] = useState('');
  const [code, setCode] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleGoogleSignIn = useCallback(async () => {
    if (!isLoaded) return;

    setIsSubmitting(true);
    setError(null);

    try {
      const redirectUrl = Linking.createURL('oauth-native-callback');
      const { createdSessionId, setActive: setOAuthActive } =
        await startGoogleOAuthFlow({ redirectUrl });

      if (createdSessionId) {
        await setOAuthActive({ session: createdSessionId });
        return;
      }

      setError(
        'Google sign-in did not complete. Please try again or use email.'
      );
    } catch (err) {
      const message =
        err instanceof Error ? err.message : 'Google sign-in failed';
      setError(message);
    } finally {
      setIsSubmitting(false);
    }
  }, [isLoaded, startGoogleOAuthFlow]);

  const handleEmailSubmit = useCallback(async () => {
    if (!isLoaded || !signIn) return;

    setIsSubmitting(true);
    setError(null);

    try {
      const identifier = email.trim().toLowerCase();
      await signIn.create({
        identifier,
      });

      // Prepare email verification
      await signIn.prepareFirstFactor({
        strategy: 'email_code',
        emailAddressId: signIn.supportedFirstFactors?.find(
          (f) => f.strategy === 'email_code'
        )?.emailAddressId ?? '',
      });

      setStep('code');
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to send code';
      setError(message);
    } finally {
      setIsSubmitting(false);
    }
  }, [isLoaded, signIn, email]);

  const handleCodeSubmit = useCallback(async () => {
    if (!isLoaded || !signIn) return;

    setIsSubmitting(true);
    setError(null);

    try {
      const result = await signIn.attemptFirstFactor({
        strategy: 'email_code',
        code,
      });

      if (result.status === 'complete' && result.createdSessionId) {
        await setActive({ session: result.createdSessionId });
      } else {
        setError('Verification failed');
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Invalid code';
      setError(message);
    } finally {
      setIsSubmitting(false);
    }
  }, [isLoaded, signIn, setActive, code]);

  const handleBack = useCallback(() => {
    setStep('email');
    setCode('');
    setError(null);
  }, []);

  if (!isLoaded) {
    return (
      <View style={styles.container} testID="signin-loading">
        <ActivityIndicator size="large" color={colors.primary[500]} />
      </View>
    );
  }

  return (
    <KeyboardAvoidingView
      style={styles.container}
      behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      testID="signin-screen"
    >
      <View style={styles.content}>
        <Text style={styles.title}>Priority Lens</Text>
        <Text style={styles.subtitle}>
          {step === 'email' ? 'Sign in to continue' : 'Enter verification code'}
        </Text>

        {step === 'email' && (
          <TouchableOpacity
            style={styles.googleButton}
            onPress={handleGoogleSignIn}
            disabled={isSubmitting}
            testID="google-button"
          >
            {isSubmitting ? (
              <ActivityIndicator color={colors.text.inverse} size="small" />
            ) : (
              <Text style={styles.googleButtonText}>Continue with Google</Text>
            )}
          </TouchableOpacity>
        )}

        {step === 'email' && <Text style={styles.orText}>or</Text>}

        {step === 'email' ? (
          <>
            <TextInput
              style={styles.input}
              placeholder="Email address"
              placeholderTextColor={colors.text.disabled}
              value={email}
              onChangeText={setEmail}
              keyboardType="email-address"
              autoCapitalize="none"
              autoComplete="email"
              autoCorrect={false}
              testID="email-input"
            />
            <TouchableOpacity
              style={[styles.button, !email && styles.buttonDisabled]}
              onPress={handleEmailSubmit}
              disabled={!email || isSubmitting}
              testID="continue-button"
            >
              {isSubmitting ? (
                <ActivityIndicator color={colors.text.inverse} size="small" />
              ) : (
                <Text style={styles.buttonText}>Continue</Text>
              )}
            </TouchableOpacity>
          </>
        ) : (
          <>
            <Text style={styles.emailHint}>Code sent to {email}</Text>
            <TextInput
              style={styles.input}
              placeholder="Verification code"
              placeholderTextColor={colors.text.disabled}
              value={code}
              onChangeText={setCode}
              keyboardType="number-pad"
              autoComplete="one-time-code"
              testID="code-input"
            />
            <TouchableOpacity
              style={[styles.button, !code && styles.buttonDisabled]}
              onPress={handleCodeSubmit}
              disabled={!code || isSubmitting}
              testID="verify-button"
            >
              {isSubmitting ? (
                <ActivityIndicator color={colors.text.inverse} size="small" />
              ) : (
                <Text style={styles.buttonText}>Verify</Text>
              )}
            </TouchableOpacity>
            <TouchableOpacity
              style={styles.backButton}
              onPress={handleBack}
              testID="back-button"
            >
              <Text style={styles.backButtonText}>Use a different email</Text>
            </TouchableOpacity>
          </>
        )}

        {error && (
          <Text style={styles.errorText} testID="error-message">
            {error}
          </Text>
        )}
      </View>
    </KeyboardAvoidingView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background.primary,
  },
  content: {
    flex: 1,
    justifyContent: 'center',
    paddingHorizontal: spacing[4],
  },
  title: {
    fontFamily: typography.fontFamily.serif,
    fontSize: typography.fontSize['3xl'],
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
    textAlign: 'center',
    marginBottom: spacing[1],
  },
  subtitle: {
    fontFamily: typography.fontFamily.sans,
    fontSize: typography.fontSize.lg,
    color: colors.text.secondary,
    textAlign: 'center',
    marginBottom: spacing[4],
  },
  googleButton: {
    height: 48,
    backgroundColor: colors.gray[900],
    borderRadius: borderRadius.md,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: spacing[3],
  },
  googleButtonText: {
    fontFamily: typography.fontFamily.sans,
    fontSize: typography.fontSize.base,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.inverse,
  },
  orText: {
    fontFamily: typography.fontFamily.sans,
    fontSize: typography.fontSize.sm,
    color: colors.text.secondary,
    textAlign: 'center',
    marginBottom: spacing[3],
  },
  emailHint: {
    fontFamily: typography.fontFamily.sans,
    fontSize: typography.fontSize.sm,
    color: colors.text.secondary,
    textAlign: 'center',
    marginBottom: spacing[2],
  },
  input: {
    height: 48,
    borderWidth: 1,
    borderColor: colors.border.medium,
    borderRadius: borderRadius.md,
    paddingHorizontal: spacing[2],
    fontFamily: typography.fontFamily.sans,
    fontSize: typography.fontSize.base,
    color: colors.text.primary,
    backgroundColor: colors.background.primary,
    marginBottom: spacing[2],
  },
  button: {
    height: 48,
    backgroundColor: colors.primary[500],
    borderRadius: borderRadius.md,
    justifyContent: 'center',
    alignItems: 'center',
    marginTop: spacing[1],
  },
  buttonDisabled: {
    backgroundColor: colors.gray[300],
  },
  buttonText: {
    fontFamily: typography.fontFamily.sans,
    fontSize: typography.fontSize.base,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.inverse,
  },
  backButton: {
    marginTop: spacing[2],
    padding: spacing[1],
  },
  backButtonText: {
    fontFamily: typography.fontFamily.sans,
    fontSize: typography.fontSize.sm,
    color: colors.primary[500],
    textAlign: 'center',
  },
  errorText: {
    fontFamily: typography.fontFamily.sans,
    fontSize: typography.fontSize.sm,
    color: colors.error,
    textAlign: 'center',
    marginTop: spacing[2],
  },
});
