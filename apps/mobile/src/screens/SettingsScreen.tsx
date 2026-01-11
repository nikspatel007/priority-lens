/**
 * SettingsScreen
 *
 * User settings and account management screen.
 * Allows viewing account info and signing out.
 */

import React, { useCallback, useState } from 'react';
import {
  View,
  Text,
  TouchableOpacity,
  ScrollView,
  StyleSheet,
  Alert,
  ActivityIndicator,
  Image,
} from 'react-native';
import { useAuthContext } from '@/context/AuthContext';
import { useGoogle } from '@/context/GoogleContext';
import { colors, typography, spacing, borderRadius } from '@/theme';

export interface SettingsScreenProps {
  /** Callback when back button is pressed */
  onBack?: () => void;
}

/**
 * SettingsScreen displays user account info and settings
 *
 * Features:
 * - User profile display
 * - Google connection status
 * - Sign out functionality
 * - Disconnect Google
 */
export function SettingsScreen({
  onBack,
}: SettingsScreenProps): React.JSX.Element {
  const { user, signOut } = useAuthContext();
  const { isConnected: googleConnected, user: googleUser, disconnect: disconnectGoogle } = useGoogle();
  const [isSigningOut, setIsSigningOut] = useState(false);
  const [isDisconnectingGoogle, setIsDisconnectingGoogle] = useState(false);

  // Handle sign out
  const handleSignOut = useCallback(async () => {
    Alert.alert(
      'Sign Out',
      'Are you sure you want to sign out?',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Sign Out',
          style: 'destructive',
          onPress: async () => {
            setIsSigningOut(true);
            try {
              await signOut();
            } catch {
              Alert.alert('Error', 'Failed to sign out. Please try again.');
            } finally {
              setIsSigningOut(false);
            }
          },
        },
      ]
    );
  }, [signOut]);

  // Handle Google disconnect
  const handleDisconnectGoogle = useCallback(async () => {
    Alert.alert(
      'Disconnect Google',
      'This will remove access to your Gmail and Calendar. You can reconnect later.',
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Disconnect',
          style: 'destructive',
          onPress: async () => {
            setIsDisconnectingGoogle(true);
            try {
              await disconnectGoogle();
            } catch {
              Alert.alert('Error', 'Failed to disconnect Google. Please try again.');
            } finally {
              setIsDisconnectingGoogle(false);
            }
          },
        },
      ]
    );
  }, [disconnectGoogle]);

  return (
    <View style={styles.container} testID="settings-screen">
      {/* Header */}
      <View style={styles.header}>
        {onBack && (
          <TouchableOpacity
            style={styles.backButton}
            onPress={onBack}
            testID="back-button"
          >
            <Text style={styles.backButtonText}>Back</Text>
          </TouchableOpacity>
        )}
        <Text style={styles.headerTitle}>Settings</Text>
        <View style={styles.headerSpacer} />
      </View>

      <ScrollView style={styles.content} testID="settings-scroll">
        {/* User Profile Section */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Account</Text>
          <View style={styles.profileCard} testID="profile-card">
            {user?.imageUrl ? (
              <Image
                source={{ uri: user.imageUrl }}
                style={styles.profileImage}
                testID="profile-image"
              />
            ) : (
              <View style={styles.profileImagePlaceholder} testID="profile-placeholder">
                <Text style={styles.profileImagePlaceholderText}>
                  {user?.firstName?.[0] || user?.email?.[0] || '?'}
                </Text>
              </View>
            )}
            <View style={styles.profileInfo}>
              <Text style={styles.profileName} testID="profile-name">
                {user?.fullName || user?.email || 'Unknown User'}
              </Text>
              <Text style={styles.profileEmail} testID="profile-email">
                {user?.email || ''}
              </Text>
            </View>
          </View>
        </View>

        {/* Google Connection Section */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>Connections</Text>
          <View style={styles.connectionCard} testID="google-connection-card">
            <View style={styles.connectionInfo}>
              <Text style={styles.connectionLabel}>Google Account</Text>
              <Text style={styles.connectionStatus} testID="google-status">
                {googleConnected
                  ? googleUser?.user?.email || 'Connected'
                  : 'Not connected'}
              </Text>
            </View>
            {googleConnected && (
              <TouchableOpacity
                style={styles.disconnectButton}
                onPress={handleDisconnectGoogle}
                disabled={isDisconnectingGoogle}
                testID="disconnect-google-button"
              >
                {isDisconnectingGoogle ? (
                  <ActivityIndicator size="small" color={colors.error} />
                ) : (
                  <Text style={styles.disconnectButtonText}>Disconnect</Text>
                )}
              </TouchableOpacity>
            )}
          </View>
        </View>

        {/* App Info Section */}
        <View style={styles.section}>
          <Text style={styles.sectionTitle}>About</Text>
          <View style={styles.infoCard}>
            <View style={styles.infoRow}>
              <Text style={styles.infoLabel}>Version</Text>
              <Text style={styles.infoValue} testID="app-version">1.0.0</Text>
            </View>
          </View>
        </View>

        {/* Sign Out Button */}
        <View style={styles.signOutSection}>
          <TouchableOpacity
            style={styles.signOutButton}
            onPress={handleSignOut}
            disabled={isSigningOut}
            testID="sign-out-button"
          >
            {isSigningOut ? (
              <ActivityIndicator size="small" color={colors.text.inverse} />
            ) : (
              <Text style={styles.signOutButtonText}>Sign Out</Text>
            )}
          </TouchableOpacity>
        </View>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.backgrounds.secondary,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: spacing[4],
    paddingVertical: spacing[3],
    backgroundColor: colors.backgrounds.primary,
    borderBottomWidth: 1,
    borderBottomColor: colors.border.light,
  },
  backButton: {
    paddingVertical: spacing[1],
    paddingRight: spacing[2],
  },
  backButtonText: {
    fontFamily: typography.fontFamily.sans,
    fontSize: typography.fontSize.base,
    color: colors.primary[500],
  },
  headerTitle: {
    fontFamily: typography.fontFamily.serif,
    fontSize: typography.fontSize.xl,
    fontWeight: typography.fontWeight.bold,
    color: colors.text.primary,
  },
  headerSpacer: {
    width: 50,
  },
  content: {
    flex: 1,
  },
  section: {
    marginTop: spacing[4],
    paddingHorizontal: spacing[4],
  },
  sectionTitle: {
    fontFamily: typography.fontFamily.sans,
    fontSize: typography.fontSize.sm,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.secondary,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
    marginBottom: spacing[2],
  },
  profileCard: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: colors.backgrounds.primary,
    padding: spacing[4],
    borderRadius: borderRadius.lg,
  },
  profileImage: {
    width: 56,
    height: 56,
    borderRadius: 28,
  },
  profileImagePlaceholder: {
    width: 56,
    height: 56,
    borderRadius: 28,
    backgroundColor: colors.primary[500],
    justifyContent: 'center',
    alignItems: 'center',
  },
  profileImagePlaceholderText: {
    fontFamily: typography.fontFamily.sans,
    fontSize: typography.fontSize.xl,
    fontWeight: typography.fontWeight.bold,
    color: colors.text.inverse,
    textTransform: 'uppercase',
  },
  profileInfo: {
    marginLeft: spacing[3],
    flex: 1,
  },
  profileName: {
    fontFamily: typography.fontFamily.sans,
    fontSize: typography.fontSize.lg,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.primary,
  },
  profileEmail: {
    fontFamily: typography.fontFamily.sans,
    fontSize: typography.fontSize.sm,
    color: colors.text.secondary,
    marginTop: spacing[1],
  },
  connectionCard: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: colors.backgrounds.primary,
    padding: spacing[4],
    borderRadius: borderRadius.lg,
  },
  connectionInfo: {
    flex: 1,
  },
  connectionLabel: {
    fontFamily: typography.fontFamily.sans,
    fontSize: typography.fontSize.base,
    fontWeight: typography.fontWeight.medium,
    color: colors.text.primary,
  },
  connectionStatus: {
    fontFamily: typography.fontFamily.sans,
    fontSize: typography.fontSize.sm,
    color: colors.text.secondary,
    marginTop: spacing[1],
  },
  disconnectButton: {
    paddingVertical: spacing[2],
    paddingHorizontal: spacing[3],
  },
  disconnectButtonText: {
    fontFamily: typography.fontFamily.sans,
    fontSize: typography.fontSize.sm,
    color: colors.error,
  },
  infoCard: {
    backgroundColor: colors.backgrounds.primary,
    padding: spacing[4],
    borderRadius: borderRadius.lg,
  },
  infoRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  infoLabel: {
    fontFamily: typography.fontFamily.sans,
    fontSize: typography.fontSize.base,
    color: colors.text.primary,
  },
  infoValue: {
    fontFamily: typography.fontFamily.sans,
    fontSize: typography.fontSize.base,
    color: colors.text.secondary,
  },
  signOutSection: {
    padding: spacing[4],
    marginTop: spacing[4],
    marginBottom: spacing[8],
  },
  signOutButton: {
    backgroundColor: colors.error,
    padding: spacing[4],
    borderRadius: borderRadius.lg,
    alignItems: 'center',
  },
  signOutButtonText: {
    fontFamily: typography.fontFamily.sans,
    fontSize: typography.fontSize.base,
    fontWeight: typography.fontWeight.semibold,
    color: colors.text.inverse,
  },
});
