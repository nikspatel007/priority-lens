import React from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  ActivityIndicator,
} from 'react-native';
import { useGoogle } from '@/context/GoogleContext';
import { colors, typography, spacing, borderRadius } from '@/theme';

/**
 * LandingScreen prompts the user to connect their Google account
 *
 * This screen is shown after Clerk authentication but before
 * the user has connected their Google account for Gmail access.
 */
export function LandingScreen(): React.JSX.Element {
  const { isLoading, isConnected, error, connect } = useGoogle();

  const handleConnect = async () => {
    await connect();
  };

  return (
    <View style={styles.container} testID="landing-screen">
      <View style={styles.content}>
        <Text style={styles.title}>Connect Your Email</Text>
        <Text style={styles.subtitle}>
          Priority Lens needs access to your Gmail to help you focus on what
          matters most.
        </Text>

        <View style={styles.features}>
          <FeatureItem
            icon="inbox"
            title="Smart Priority"
            description="AI-powered email prioritization"
          />
          <FeatureItem
            icon="tasks"
            title="Task Detection"
            description="Automatically extract action items"
          />
          <FeatureItem
            icon="voice"
            title="Voice Assistant"
            description="Manage email with your voice"
          />
        </View>

        <TouchableOpacity
          style={[styles.button, isLoading && styles.buttonDisabled]}
          onPress={handleConnect}
          disabled={isLoading || isConnected}
          testID="connect-google-button"
        >
          {isLoading ? (
            <ActivityIndicator color={colors.text.inverse} size="small" />
          ) : (
            <Text style={styles.buttonText}>Connect Google Account</Text>
          )}
        </TouchableOpacity>

        {error && (
          <Text style={styles.errorText} testID="error-message">
            {error}
          </Text>
        )}

        <Text style={styles.disclaimer}>
          We only read your emails to help prioritize them.{'\n'}
          Your data is never shared or sold.
        </Text>
      </View>
    </View>
  );
}

interface FeatureItemProps {
  icon: string;
  title: string;
  description: string;
}

function FeatureItem({ title, description }: FeatureItemProps): React.JSX.Element {
  return (
    <View style={styles.featureItem} testID="feature-item">
      <Text style={styles.featureTitle}>{title}</Text>
      <Text style={styles.featureDescription}>{description}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.backgrounds.primary,
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
    fontSize: typography.fontSize.base,
    color: colors.text.secondary,
    textAlign: 'center',
    marginBottom: spacing[4],
    lineHeight: typography.fontSize.base * typography.lineHeight.relaxed,
  },
  features: {
    marginBottom: spacing[4],
  },
  featureItem: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: spacing[2],
    paddingHorizontal: spacing[2],
  },
  featureTitle: {
    fontFamily: typography.fontFamily.sans,
    fontSize: typography.fontSize.base,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
    marginRight: spacing[1],
  },
  featureDescription: {
    fontFamily: typography.fontFamily.sans,
    fontSize: typography.fontSize.sm,
    color: colors.text.secondary,
    flex: 1,
  },
  button: {
    height: 48,
    backgroundColor: colors.primary[500],
    borderRadius: borderRadius.md,
    justifyContent: 'center',
    alignItems: 'center',
    marginTop: spacing[2],
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
  errorText: {
    fontFamily: typography.fontFamily.sans,
    fontSize: typography.fontSize.sm,
    color: colors.error,
    textAlign: 'center',
    marginTop: spacing[2],
  },
  disclaimer: {
    fontFamily: typography.fontFamily.sans,
    fontSize: typography.fontSize.xs,
    color: colors.text.disabled,
    textAlign: 'center',
    marginTop: spacing[4],
    lineHeight: typography.fontSize.xs * typography.lineHeight.relaxed,
  },
});
