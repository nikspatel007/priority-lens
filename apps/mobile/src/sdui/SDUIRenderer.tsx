/**
 * SDUI Renderer
 *
 * Main entry point for rendering SDUI schemas.
 * Takes a UIBlock and recursively renders components.
 */

import React, { useCallback } from 'react';
import { View, Text } from 'react-native';
import { UIBlock, UIAction } from './types';

// Primitives
import { SDUIText } from './primitives/SDUIText';
import { SDUIAvatar } from './primitives/SDUIAvatar';
import { SDUIButton } from './primitives/SDUIButton';
import { SDUIBadge } from './primitives/SDUIBadge';
import { SDUIViewToggle } from './primitives/SDUIViewToggle';

// Layout
import { SDUIStack } from './layout/SDUIStack';
import { SDUICard } from './layout/SDUICard';

// Composites
import { SDUIPersonCard } from './composites/SDUIPersonCard';
import { SDUIInvoiceCard } from './composites/SDUIInvoiceCard';
import { SDUIActionItem } from './composites/SDUIActionItem';

interface SDUIRendererProps {
  block: UIBlock;
  onAction?: (action: UIAction) => void;
}

export function SDUIRenderer({ block, onAction }: SDUIRendererProps) {
  // Handle action triggers
  const handlePress = useCallback(() => {
    const pressAction = block.actions?.find((a) => a.trigger === 'press');
    if (pressAction && onAction) {
      onAction(pressAction);
    }
  }, [block.actions, onAction]);

  // Recursively render children
  const renderChildren = useCallback(() => {
    if (!block.children) return null;
    return block.children.map((child) => (
      <SDUIRenderer key={child.id} block={child} onAction={onAction} />
    ));
  }, [block.children, onAction]);

  // Component mapping
  switch (block.type) {
    // ===========================================
    // Primitives
    // ===========================================
    case 'text':
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      return <SDUIText {...(block.props as any)} />;

    case 'avatar':
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      return <SDUIAvatar {...(block.props as any)} />;

    case 'badge':
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      return <SDUIBadge {...(block.props as any)} />;

    case 'button':
      return (
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        <SDUIButton
          {...(block.props as any)}
          onPress={block.actions?.length ? handlePress : undefined}
        />
      );

    case 'spacer':
      return (
        <View
          style={{ height: (block.props?.size as number) || 16 }}
          testID="sdui-spacer"
        />
      );

    case 'divider':
      return (
        <View
          style={{
            height: (block.props?.thickness as number) || 1,
            backgroundColor: (block.props?.color as string) || '#E5E7EB',
            marginVertical: 8,
          }}
          testID="sdui-divider"
        />
      );

    // ===========================================
    // Layout
    // ===========================================
    case 'stack':
      return (
        <SDUIStack {...(block.props as Record<string, unknown>)} layout={block.layout}>
          {renderChildren()}
        </SDUIStack>
      );

    case 'card':
      return (
        <SDUICard {...(block.props as Record<string, unknown>)} layout={block.layout}>
          {renderChildren()}
        </SDUICard>
      );

    case 'box':
      return (
        <View
          style={{
            backgroundColor: block.props?.backgroundColor as string,
            borderRadius: block.props?.borderRadius as number,
            padding: block.layout?.padding as number,
          }}
          testID="sdui-box"
        >
          {renderChildren()}
        </View>
      );

    // ===========================================
    // Composites
    // ===========================================
    case 'personCard':
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      return <SDUIPersonCard {...(block.props as any)} />;

    case 'invoiceCard':
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      return <SDUIInvoiceCard {...(block.props as any)} />;

    case 'actionItem':
      return (
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        <SDUIActionItem
          {...(block.props as any)}
          onToggle={block.actions?.length ? handlePress : undefined}
        />
      );

    case 'header': {
      const subtitle = block.props?.subtitle as string | undefined;
      const title = block.props?.title as string;
      return (
        <SDUIStack direction="vertical" gap={4} layout={block.layout}>
          {subtitle ? <SDUIText value={subtitle} variant="caption" /> : null}
          <SDUIText value={title} variant="title" />
          {block.children && (
            <SDUIStack direction="horizontal" gap={8}>
              {block.children.map((child) => (
                <SDUIRenderer key={child.id} block={child} onAction={onAction} />
              ))}
            </SDUIStack>
          )}
        </SDUIStack>
      );
    }

    case 'listItem': {
      const leadingBlock = block.props?.leading as UIBlock | undefined;
      const trailingBlock = block.props?.trailing as UIBlock | undefined;
      const listTitle = block.props?.title as string;
      const listSubtitle = block.props?.subtitle as string | undefined;
      return (
        <SDUIStack
          direction="horizontal"
          gap={12}
          align="center"
          layout={{ padding: [12, 0, 12, 0] }}
        >
          {leadingBlock && (
            <SDUIRenderer block={leadingBlock} onAction={onAction} />
          )}
          <SDUIStack direction="vertical" gap={2} layout={{ flex: 1 }}>
            <SDUIText value={listTitle} variant="label" />
            {listSubtitle ? (
              <SDUIText value={listSubtitle} variant="caption" />
            ) : null}
          </SDUIStack>
          {trailingBlock && (
            <SDUIRenderer block={trailingBlock} onAction={onAction} />
          )}
        </SDUIStack>
      );
    }

    case 'viewToggle':
      return (
        <SDUIViewToggle
          mode={(block.props?.mode as 'cards' | 'list') || 'cards'}
          onChange={(mode) => {
            if (onAction) {
              onAction({
                trigger: 'change',
                type: 'viewMode.change',
                payload: { mode },
              });
            }
          }}
        />
      );

    // ===========================================
    // Unknown
    // ===========================================
    default:
      if (__DEV__) {
        return (
          <View
            style={{
              padding: 8,
              backgroundColor: '#FEE2E2',
              borderRadius: 4,
            }}
            testID="sdui-unknown"
          >
            <Text style={{ color: '#991B1B', fontSize: 12 }}>
              Unknown component: {block.type}
            </Text>
          </View>
        );
      }
      return null;
  }
}

/**
 * Render a list of UI blocks
 */
export function SDUIBlockList({
  blocks,
  onAction,
}: {
  blocks: UIBlock[];
  onAction?: (action: UIAction) => void;
}) {
  return (
    <>
      {blocks.map((block) => (
        <SDUIRenderer key={block.id} block={block} onAction={onAction} />
      ))}
    </>
  );
}
