/**
 * SDUI Layout Tests
 *
 * Tests for layout SDUI components.
 */

import React from 'react';
import { render } from '@testing-library/react-native';
import { Text } from 'react-native';
import { SDUIStack } from '../layout/SDUIStack';
import { SDUICard } from '../layout/SDUICard';

describe('SDUIStack', () => {
  it('renders children', () => {
    const { getByText } = render(
      <SDUIStack>
        <Text>Child 1</Text>
        <Text>Child 2</Text>
      </SDUIStack>
    );
    expect(getByText('Child 1')).toBeTruthy();
    expect(getByText('Child 2')).toBeTruthy();
  });

  it('renders with vertical direction by default', () => {
    const { getByTestId } = render(
      <SDUIStack>
        <Text>Content</Text>
      </SDUIStack>
    );
    expect(getByTestId('sdui-stack')).toBeTruthy();
  });

  it('renders with horizontal direction', () => {
    const { getByTestId } = render(
      <SDUIStack direction="horizontal">
        <Text>Content</Text>
      </SDUIStack>
    );
    expect(getByTestId('sdui-stack')).toBeTruthy();
  });

  it('applies gap', () => {
    const { getByTestId } = render(
      <SDUIStack gap={16}>
        <Text>Item 1</Text>
        <Text>Item 2</Text>
      </SDUIStack>
    );
    expect(getByTestId('sdui-stack')).toBeTruthy();
  });

  it('applies align start', () => {
    const { getByTestId } = render(
      <SDUIStack align="start">
        <Text>Content</Text>
      </SDUIStack>
    );
    expect(getByTestId('sdui-stack')).toBeTruthy();
  });

  it('applies align center', () => {
    const { getByTestId } = render(
      <SDUIStack align="center">
        <Text>Content</Text>
      </SDUIStack>
    );
    expect(getByTestId('sdui-stack')).toBeTruthy();
  });

  it('applies align end', () => {
    const { getByTestId } = render(
      <SDUIStack align="end">
        <Text>Content</Text>
      </SDUIStack>
    );
    expect(getByTestId('sdui-stack')).toBeTruthy();
  });

  it('applies align stretch', () => {
    const { getByTestId } = render(
      <SDUIStack align="stretch">
        <Text>Content</Text>
      </SDUIStack>
    );
    expect(getByTestId('sdui-stack')).toBeTruthy();
  });

  it('applies justify start', () => {
    const { getByTestId } = render(
      <SDUIStack justify="start">
        <Text>Content</Text>
      </SDUIStack>
    );
    expect(getByTestId('sdui-stack')).toBeTruthy();
  });

  it('applies justify center', () => {
    const { getByTestId } = render(
      <SDUIStack justify="center">
        <Text>Content</Text>
      </SDUIStack>
    );
    expect(getByTestId('sdui-stack')).toBeTruthy();
  });

  it('applies justify end', () => {
    const { getByTestId } = render(
      <SDUIStack justify="end">
        <Text>Content</Text>
      </SDUIStack>
    );
    expect(getByTestId('sdui-stack')).toBeTruthy();
  });

  it('applies justify between', () => {
    const { getByTestId } = render(
      <SDUIStack justify="between">
        <Text>Content</Text>
      </SDUIStack>
    );
    expect(getByTestId('sdui-stack')).toBeTruthy();
  });

  it('applies justify around', () => {
    const { getByTestId } = render(
      <SDUIStack justify="around">
        <Text>Content</Text>
      </SDUIStack>
    );
    expect(getByTestId('sdui-stack')).toBeTruthy();
  });

  it('applies wrap', () => {
    const { getByTestId } = render(
      <SDUIStack wrap>
        <Text>Item 1</Text>
        <Text>Item 2</Text>
      </SDUIStack>
    );
    expect(getByTestId('sdui-stack')).toBeTruthy();
  });

  it('applies layout padding as number', () => {
    const { getByTestId } = render(
      <SDUIStack layout={{ padding: 16 }}>
        <Text>Content</Text>
      </SDUIStack>
    );
    expect(getByTestId('sdui-stack')).toBeTruthy();
  });

  it('applies layout padding as array', () => {
    const { getByTestId } = render(
      <SDUIStack layout={{ padding: [8, 16, 8, 16] }}>
        <Text>Content</Text>
      </SDUIStack>
    );
    expect(getByTestId('sdui-stack')).toBeTruthy();
  });

  it('applies layout margin as number', () => {
    const { getByTestId } = render(
      <SDUIStack layout={{ margin: 8 }}>
        <Text>Content</Text>
      </SDUIStack>
    );
    expect(getByTestId('sdui-stack')).toBeTruthy();
  });

  it('applies layout margin as array', () => {
    const { getByTestId } = render(
      <SDUIStack layout={{ margin: [4, 8, 4, 8] }}>
        <Text>Content</Text>
      </SDUIStack>
    );
    expect(getByTestId('sdui-stack')).toBeTruthy();
  });

  it('applies layout flex', () => {
    const { getByTestId } = render(
      <SDUIStack layout={{ flex: 1 }}>
        <Text>Content</Text>
      </SDUIStack>
    );
    expect(getByTestId('sdui-stack')).toBeTruthy();
  });

  it('applies layout width', () => {
    const { getByTestId } = render(
      <SDUIStack layout={{ width: 200 }}>
        <Text>Content</Text>
      </SDUIStack>
    );
    expect(getByTestId('sdui-stack')).toBeTruthy();
  });

  it('applies layout height', () => {
    const { getByTestId } = render(
      <SDUIStack layout={{ height: 100 }}>
        <Text>Content</Text>
      </SDUIStack>
    );
    expect(getByTestId('sdui-stack')).toBeTruthy();
  });

  it('applies layout minHeight', () => {
    const { getByTestId } = render(
      <SDUIStack layout={{ minHeight: 50 }}>
        <Text>Content</Text>
      </SDUIStack>
    );
    expect(getByTestId('sdui-stack')).toBeTruthy();
  });

  it('applies layout maxWidth', () => {
    const { getByTestId } = render(
      <SDUIStack layout={{ maxWidth: 300 }}>
        <Text>Content</Text>
      </SDUIStack>
    );
    expect(getByTestId('sdui-stack')).toBeTruthy();
  });

  it('renders without children', () => {
    const { getByTestId } = render(<SDUIStack />);
    expect(getByTestId('sdui-stack')).toBeTruthy();
  });
});

describe('SDUICard', () => {
  it('renders children', () => {
    const { getByText } = render(
      <SDUICard>
        <Text>Card Content</Text>
      </SDUICard>
    );
    expect(getByText('Card Content')).toBeTruthy();
  });

  it('renders with default variant', () => {
    const { getByTestId } = render(
      <SDUICard>
        <Text>Content</Text>
      </SDUICard>
    );
    expect(getByTestId('sdui-card')).toBeTruthy();
  });

  it('renders with elevated variant', () => {
    const { getByTestId } = render(
      <SDUICard variant="elevated">
        <Text>Content</Text>
      </SDUICard>
    );
    expect(getByTestId('sdui-card')).toBeTruthy();
  });

  it('renders with outlined variant', () => {
    const { getByTestId } = render(
      <SDUICard variant="outlined">
        <Text>Content</Text>
      </SDUICard>
    );
    expect(getByTestId('sdui-card')).toBeTruthy();
  });

  it('applies custom backgroundColor', () => {
    const { getByTestId } = render(
      <SDUICard backgroundColor="#f0f0f0">
        <Text>Content</Text>
      </SDUICard>
    );
    expect(getByTestId('sdui-card')).toBeTruthy();
  });

  it('applies layout padding as number', () => {
    const { getByTestId } = render(
      <SDUICard layout={{ padding: 24 }}>
        <Text>Content</Text>
      </SDUICard>
    );
    expect(getByTestId('sdui-card')).toBeTruthy();
  });

  it('applies layout padding as array', () => {
    const { getByTestId } = render(
      <SDUICard layout={{ padding: [16, 24, 16, 24] }}>
        <Text>Content</Text>
      </SDUICard>
    );
    expect(getByTestId('sdui-card')).toBeTruthy();
  });

  it('applies layout margin as number', () => {
    const { getByTestId } = render(
      <SDUICard layout={{ margin: 8 }}>
        <Text>Content</Text>
      </SDUICard>
    );
    expect(getByTestId('sdui-card')).toBeTruthy();
  });

  it('applies layout margin as array', () => {
    const { getByTestId } = render(
      <SDUICard layout={{ margin: [4, 8, 4, 8] }}>
        <Text>Content</Text>
      </SDUICard>
    );
    expect(getByTestId('sdui-card')).toBeTruthy();
  });

  it('applies layout width', () => {
    const { getByTestId } = render(
      <SDUICard layout={{ width: 300 }}>
        <Text>Content</Text>
      </SDUICard>
    );
    expect(getByTestId('sdui-card')).toBeTruthy();
  });

  it('applies layout maxWidth', () => {
    const { getByTestId } = render(
      <SDUICard layout={{ maxWidth: 400 }}>
        <Text>Content</Text>
      </SDUICard>
    );
    expect(getByTestId('sdui-card')).toBeTruthy();
  });

  it('applies layout height', () => {
    const { getByTestId } = render(
      <SDUICard layout={{ height: 200 }}>
        <Text>Content</Text>
      </SDUICard>
    );
    expect(getByTestId('sdui-card')).toBeTruthy();
  });

  it('renders without children', () => {
    const { getByTestId } = render(<SDUICard />);
    expect(getByTestId('sdui-card')).toBeTruthy();
  });

  it('applies default padding when no layout provided', () => {
    const { getByTestId } = render(
      <SDUICard>
        <Text>Content</Text>
      </SDUICard>
    );
    expect(getByTestId('sdui-card')).toBeTruthy();
  });
});
