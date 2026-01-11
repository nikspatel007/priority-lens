/**
 * SDUI Button Component
 *
 * Interactive button with variants.
 */

import React from 'react';
import {
  TouchableOpacity,
  Text,
  StyleSheet,
  ActivityIndicator,
} from 'react-native';
import { ButtonProps } from '../types';
import { colors } from '../../theme';

const variantStyles = {
  primary: {
    container: {
      backgroundColor: colors.primary[500],
    },
    text: {
      color: '#FFFFFF',
    },
  },
  secondary: {
    container: {
      backgroundColor: colors.gray[100],
    },
    text: {
      color: colors.text.primary,
    },
  },
  outline: {
    container: {
      backgroundColor: 'transparent',
      borderWidth: 1,
      borderColor: colors.gray[200],
    },
    text: {
      color: colors.text.primary,
    },
  },
  ghost: {
    container: {
      backgroundColor: 'transparent',
    },
    text: {
      color: colors.primary[500],
    },
  },
  destructive: {
    container: {
      backgroundColor: colors.error,
    },
    text: {
      color: '#FFFFFF',
    },
  },
};

const sizeStyles = {
  sm: {
    paddingVertical: 8,
    paddingHorizontal: 12,
    fontSize: 14,
  },
  md: {
    paddingVertical: 12,
    paddingHorizontal: 16,
    fontSize: 16,
  },
  lg: {
    paddingVertical: 16,
    paddingHorizontal: 24,
    fontSize: 18,
  },
};

interface SDUIButtonProps extends ButtonProps {
  onPress?: () => void;
}

export function SDUIButton({
  label,
  variant = 'primary',
  size = 'md',
  loading = false,
  disabled = false,
  onPress,
}: SDUIButtonProps) {
  const variantStyle = variantStyles[variant];
  const sizeStyle = sizeStyles[size];

  return (
    <TouchableOpacity
      style={[
        styles.container,
        variantStyle.container,
        {
          paddingVertical: sizeStyle.paddingVertical,
          paddingHorizontal: sizeStyle.paddingHorizontal,
        },
        disabled && styles.disabled,
      ]}
      onPress={onPress}
      disabled={disabled || loading}
      activeOpacity={0.7}
      testID="sdui-button"
    >
      {loading ? (
        <ActivityIndicator size="small" color={variantStyle.text.color} />
      ) : (
        <Text
          style={[
            styles.label,
            variantStyle.text,
            { fontSize: sizeStyle.fontSize },
          ]}
        >
          {label}
        </Text>
      )}
    </TouchableOpacity>
  );
}

const styles = StyleSheet.create({
  container: {
    borderRadius: 8,
    alignItems: 'center',
    justifyContent: 'center',
    flexDirection: 'row',
  },
  label: {
    fontWeight: '600',
  },
  disabled: {
    opacity: 0.5,
  },
});
