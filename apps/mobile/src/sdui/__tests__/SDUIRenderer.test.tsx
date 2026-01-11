/**
 * SDUIRenderer Tests
 *
 * Tests for the main SDUI renderer component.
 */

import React from 'react';
import { render, fireEvent } from '@testing-library/react-native';
import { SDUIRenderer, SDUIBlockList } from '../SDUIRenderer';
import { UIBlock, UIAction } from '../types';

describe('SDUIRenderer', () => {
  describe('primitive rendering', () => {
    it('renders text block', () => {
      const block: UIBlock = {
        id: '1',
        type: 'text',
        props: { value: 'Hello World' },
      };
      const { getByText } = render(<SDUIRenderer block={block} />);
      expect(getByText('Hello World')).toBeTruthy();
    });

    it('renders text block with variant', () => {
      const block: UIBlock = {
        id: '1',
        type: 'text',
        props: { value: 'Title Text', variant: 'title' },
      };
      const { getByText } = render(<SDUIRenderer block={block} />);
      expect(getByText('Title Text')).toBeTruthy();
    });

    it('renders avatar block with image', () => {
      const block: UIBlock = {
        id: '1',
        type: 'avatar',
        props: { name: 'John Doe', src: 'https://example.com/avatar.jpg' },
      };
      const { getByTestId } = render(<SDUIRenderer block={block} />);
      expect(getByTestId('sdui-avatar-image')).toBeTruthy();
    });

    it('renders avatar block with fallback', () => {
      const block: UIBlock = {
        id: '1',
        type: 'avatar',
        props: { name: 'John Doe' },
      };
      const { getByTestId } = render(<SDUIRenderer block={block} />);
      expect(getByTestId('sdui-avatar-fallback')).toBeTruthy();
    });

    it('renders badge block', () => {
      const block: UIBlock = {
        id: '1',
        type: 'badge',
        props: { value: 'New', variant: 'success' },
      };
      const { getByText } = render(<SDUIRenderer block={block} />);
      expect(getByText('New')).toBeTruthy();
    });

    it('renders button block', () => {
      const block: UIBlock = {
        id: '1',
        type: 'button',
        props: { label: 'Click Me' },
      };
      const { getByText } = render(<SDUIRenderer block={block} />);
      expect(getByText('Click Me')).toBeTruthy();
    });

    it('renders spacer block', () => {
      const block: UIBlock = {
        id: '1',
        type: 'spacer',
        props: { size: 24 },
      };
      const { getByTestId } = render(<SDUIRenderer block={block} />);
      expect(getByTestId('sdui-spacer')).toBeTruthy();
    });

    it('renders spacer with default size', () => {
      const block: UIBlock = {
        id: '1',
        type: 'spacer',
        props: {},
      };
      const { getByTestId } = render(<SDUIRenderer block={block} />);
      expect(getByTestId('sdui-spacer')).toBeTruthy();
    });

    it('renders divider block', () => {
      const block: UIBlock = {
        id: '1',
        type: 'divider',
        props: { thickness: 2, color: '#000' },
      };
      const { getByTestId } = render(<SDUIRenderer block={block} />);
      expect(getByTestId('sdui-divider')).toBeTruthy();
    });

    it('renders divider with default props', () => {
      const block: UIBlock = {
        id: '1',
        type: 'divider',
        props: {},
      };
      const { getByTestId } = render(<SDUIRenderer block={block} />);
      expect(getByTestId('sdui-divider')).toBeTruthy();
    });
  });

  describe('layout rendering', () => {
    it('renders stack with children', () => {
      const block: UIBlock = {
        id: '1',
        type: 'stack',
        props: { direction: 'vertical', gap: 8 },
        children: [
          { id: '2', type: 'text', props: { value: 'Child 1' } },
          { id: '3', type: 'text', props: { value: 'Child 2' } },
        ],
      };
      const { getByText } = render(<SDUIRenderer block={block} />);
      expect(getByText('Child 1')).toBeTruthy();
      expect(getByText('Child 2')).toBeTruthy();
    });

    it('renders card with children', () => {
      const block: UIBlock = {
        id: '1',
        type: 'card',
        props: { variant: 'elevated' },
        children: [{ id: '2', type: 'text', props: { value: 'Card Content' } }],
      };
      const { getByText, getByTestId } = render(<SDUIRenderer block={block} />);
      expect(getByTestId('sdui-card')).toBeTruthy();
      expect(getByText('Card Content')).toBeTruthy();
    });

    it('renders box with children', () => {
      const block: UIBlock = {
        id: '1',
        type: 'box',
        props: { backgroundColor: '#f0f0f0', borderRadius: 8 },
        layout: { padding: 16 },
        children: [{ id: '2', type: 'text', props: { value: 'Box Content' } }],
      };
      const { getByText, getByTestId } = render(<SDUIRenderer block={block} />);
      expect(getByTestId('sdui-box')).toBeTruthy();
      expect(getByText('Box Content')).toBeTruthy();
    });

    it('applies layout props to stack', () => {
      const block: UIBlock = {
        id: '1',
        type: 'stack',
        props: { direction: 'horizontal' },
        layout: { padding: 16, margin: 8 },
      };
      const { getByTestId } = render(<SDUIRenderer block={block} />);
      expect(getByTestId('sdui-stack')).toBeTruthy();
    });
  });

  describe('composite rendering', () => {
    it('renders personCard', () => {
      const block: UIBlock = {
        id: '1',
        type: 'personCard',
        props: { name: 'John Doe', title: 'Developer', email: 'john@example.com' },
      };
      const { getByText } = render(<SDUIRenderer block={block} />);
      expect(getByText('John Doe')).toBeTruthy();
      expect(getByText('Developer')).toBeTruthy();
      expect(getByText('john@example.com')).toBeTruthy();
    });

    it('renders invoiceCard', () => {
      const block: UIBlock = {
        id: '1',
        type: 'invoiceCard',
        props: {
          vendor: 'Acme Inc',
          amount: '$100.00',
          dueDate: '2024-01-15',
          status: 'pending',
        },
      };
      const { getByText } = render(<SDUIRenderer block={block} />);
      expect(getByText('Acme Inc')).toBeTruthy();
      expect(getByText('$100.00')).toBeTruthy();
      expect(getByText('Pending')).toBeTruthy();
    });

    it('renders actionItem', () => {
      const block: UIBlock = {
        id: '1',
        type: 'actionItem',
        props: { text: 'Complete task', checked: false },
      };
      const { getByText } = render(<SDUIRenderer block={block} />);
      expect(getByText('Complete task')).toBeTruthy();
    });

    it('renders header with title and subtitle', () => {
      const block: UIBlock = {
        id: '1',
        type: 'header',
        props: { title: 'Page Title', subtitle: 'Subtitle text' },
      };
      const { getByText } = render(<SDUIRenderer block={block} />);
      expect(getByText('Page Title')).toBeTruthy();
      expect(getByText('Subtitle text')).toBeTruthy();
    });

    it('renders header without subtitle', () => {
      const block: UIBlock = {
        id: '1',
        type: 'header',
        props: { title: 'Page Title' },
      };
      const { getByText, queryByText } = render(<SDUIRenderer block={block} />);
      expect(getByText('Page Title')).toBeTruthy();
    });

    it('renders header with children', () => {
      const block: UIBlock = {
        id: '1',
        type: 'header',
        props: { title: 'Page Title' },
        children: [
          { id: '2', type: 'button', props: { label: 'Action' } },
        ],
      };
      const { getByText } = render(<SDUIRenderer block={block} />);
      expect(getByText('Page Title')).toBeTruthy();
      expect(getByText('Action')).toBeTruthy();
    });

    it('renders listItem with title and subtitle', () => {
      const block: UIBlock = {
        id: '1',
        type: 'listItem',
        props: { title: 'List Title', subtitle: 'List subtitle' },
      };
      const { getByText } = render(<SDUIRenderer block={block} />);
      expect(getByText('List Title')).toBeTruthy();
      expect(getByText('List subtitle')).toBeTruthy();
    });

    it('renders listItem with leading block', () => {
      const block: UIBlock = {
        id: '1',
        type: 'listItem',
        props: {
          title: 'List Title',
          leading: { id: '2', type: 'avatar', props: { name: 'John' } },
        },
      };
      const { getByText, getByTestId } = render(<SDUIRenderer block={block} />);
      expect(getByText('List Title')).toBeTruthy();
      expect(getByTestId('sdui-avatar-fallback')).toBeTruthy();
    });

    it('renders listItem with trailing block', () => {
      const block: UIBlock = {
        id: '1',
        type: 'listItem',
        props: {
          title: 'List Title',
          trailing: { id: '2', type: 'badge', props: { value: 'New' } },
        },
      };
      const { getByText } = render(<SDUIRenderer block={block} />);
      expect(getByText('List Title')).toBeTruthy();
      expect(getByText('New')).toBeTruthy();
    });

    it('renders viewToggle', () => {
      const block: UIBlock = {
        id: '1',
        type: 'viewToggle',
        props: { mode: 'cards' },
      };
      const { getByTestId } = render(<SDUIRenderer block={block} />);
      expect(getByTestId('sdui-view-toggle')).toBeTruthy();
    });

    it('renders viewToggle with default mode when mode is undefined', () => {
      const block: UIBlock = {
        id: '1',
        type: 'viewToggle',
        props: {},
      };
      const { getByTestId } = render(<SDUIRenderer block={block} />);
      // Should render with default 'cards' mode
      expect(getByTestId('sdui-view-toggle')).toBeTruthy();
    });
  });

  describe('action handling', () => {
    it('calls onAction when button is pressed', () => {
      const onAction = jest.fn();
      const action: UIAction = { trigger: 'press', type: 'test.action' };
      const block: UIBlock = {
        id: '1',
        type: 'button',
        props: { label: 'Click Me' },
        actions: [action],
      };
      const { getByText } = render(
        <SDUIRenderer block={block} onAction={onAction} />
      );
      fireEvent.press(getByText('Click Me'));
      expect(onAction).toHaveBeenCalledWith(action);
    });

    it('calls onAction with payload', () => {
      const onAction = jest.fn();
      const action: UIAction = {
        trigger: 'press',
        type: 'task.complete',
        payload: { taskId: '123' },
      };
      const block: UIBlock = {
        id: '1',
        type: 'button',
        props: { label: 'Complete' },
        actions: [action],
      };
      const { getByText } = render(
        <SDUIRenderer block={block} onAction={onAction} />
      );
      fireEvent.press(getByText('Complete'));
      expect(onAction).toHaveBeenCalledWith(action);
    });

    it('does not call onAction when no actions defined', () => {
      const onAction = jest.fn();
      const block: UIBlock = {
        id: '1',
        type: 'button',
        props: { label: 'No Action' },
      };
      const { getByText } = render(
        <SDUIRenderer block={block} onAction={onAction} />
      );
      fireEvent.press(getByText('No Action'));
      expect(onAction).not.toHaveBeenCalled();
    });

    it('does not call action when onAction is not provided but actions exist', () => {
      const action: UIAction = { trigger: 'press', type: 'test.action' };
      const block: UIBlock = {
        id: '1',
        type: 'button',
        props: { label: 'Has Action' },
        actions: [action],
      };
      // Render without onAction prop - should not throw
      const { getByText } = render(<SDUIRenderer block={block} />);
      fireEvent.press(getByText('Has Action'));
      // No error thrown is the test
    });

    it('calls onAction for actionItem toggle', () => {
      const onAction = jest.fn();
      const action: UIAction = { trigger: 'press', type: 'item.toggle' };
      const block: UIBlock = {
        id: '1',
        type: 'actionItem',
        props: { text: 'Toggle me', checked: false },
        actions: [action],
      };
      const { getByTestId } = render(
        <SDUIRenderer block={block} onAction={onAction} />
      );
      fireEvent.press(getByTestId('sdui-action-item'));
      expect(onAction).toHaveBeenCalledWith(action);
    });

    it('calls onAction for viewToggle change', () => {
      const onAction = jest.fn();
      const block: UIBlock = {
        id: '1',
        type: 'viewToggle',
        props: { mode: 'cards' },
      };
      const { getByTestId } = render(
        <SDUIRenderer block={block} onAction={onAction} />
      );
      fireEvent.press(getByTestId('sdui-view-toggle-list'));
      expect(onAction).toHaveBeenCalledWith({
        trigger: 'change',
        type: 'viewMode.change',
        payload: { mode: 'list' },
      });
    });

    it('does not throw when viewToggle changes without onAction', () => {
      const block: UIBlock = {
        id: '1',
        type: 'viewToggle',
        props: { mode: 'cards' },
      };
      // Render without onAction prop
      const { getByTestId } = render(<SDUIRenderer block={block} />);
      // Pressing should not throw even without onAction
      fireEvent.press(getByTestId('sdui-view-toggle-list'));
      // No error thrown is the test
    });
  });

  describe('unknown type handling', () => {
    it('renders fallback for unknown type in dev', () => {
      const block: UIBlock = {
        id: '1',
        type: 'unknownComponent',
        props: {},
      };
      const { getByTestId, getByText } = render(<SDUIRenderer block={block} />);
      expect(getByTestId('sdui-unknown')).toBeTruthy();
      expect(getByText('Unknown component: unknownComponent')).toBeTruthy();
    });

    it('renders nothing for unknown type in production', () => {
      const originalDev = (global as Record<string, unknown>).__DEV__;
      (global as Record<string, unknown>).__DEV__ = false;

      const block: UIBlock = {
        id: '1',
        type: 'unknownComponent',
        props: {},
      };
      const { queryByTestId } = render(<SDUIRenderer block={block} />);
      expect(queryByTestId('sdui-unknown')).toBeNull();

      (global as Record<string, unknown>).__DEV__ = originalDev;
    });
  });

  describe('nested children', () => {
    it('renders deeply nested children', () => {
      const block: UIBlock = {
        id: '1',
        type: 'stack',
        children: [
          {
            id: '2',
            type: 'card',
            children: [
              {
                id: '3',
                type: 'stack',
                children: [
                  { id: '4', type: 'text', props: { value: 'Deep Text' } },
                ],
              },
            ],
          },
        ],
      };
      const { getByText } = render(<SDUIRenderer block={block} />);
      expect(getByText('Deep Text')).toBeTruthy();
    });

    it('renders stack without children', () => {
      const block: UIBlock = {
        id: '1',
        type: 'stack',
        props: { direction: 'vertical' },
      };
      const { getByTestId } = render(<SDUIRenderer block={block} />);
      expect(getByTestId('sdui-stack')).toBeTruthy();
    });
  });
});

describe('SDUIBlockList', () => {
  it('renders multiple blocks', () => {
    const blocks: UIBlock[] = [
      { id: '1', type: 'text', props: { value: 'Block 1' } },
      { id: '2', type: 'text', props: { value: 'Block 2' } },
      { id: '3', type: 'text', props: { value: 'Block 3' } },
    ];
    const { getByText } = render(<SDUIBlockList blocks={blocks} />);
    expect(getByText('Block 1')).toBeTruthy();
    expect(getByText('Block 2')).toBeTruthy();
    expect(getByText('Block 3')).toBeTruthy();
  });

  it('renders empty list', () => {
    const result = render(<SDUIBlockList blocks={[]} />);
    expect(result).toBeTruthy();
  });

  it('passes onAction to all blocks', () => {
    const onAction = jest.fn();
    const blocks: UIBlock[] = [
      {
        id: '1',
        type: 'button',
        props: { label: 'Button 1' },
        actions: [{ trigger: 'press', type: 'action1' }],
      },
      {
        id: '2',
        type: 'button',
        props: { label: 'Button 2' },
        actions: [{ trigger: 'press', type: 'action2' }],
      },
    ];
    const { getByText } = render(
      <SDUIBlockList blocks={blocks} onAction={onAction} />
    );

    fireEvent.press(getByText('Button 1'));
    expect(onAction).toHaveBeenCalledWith({ trigger: 'press', type: 'action1' });

    fireEvent.press(getByText('Button 2'));
    expect(onAction).toHaveBeenCalledWith({ trigger: 'press', type: 'action2' });
  });
});
