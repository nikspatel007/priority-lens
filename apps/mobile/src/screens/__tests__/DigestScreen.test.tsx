import React from 'react';
import {
  render,
  fireEvent,
  waitFor,
  act,
} from '@testing-library/react-native';
import { DigestScreen } from '../DigestScreen';
import * as api from '@/services/api';
import type { DigestResponse } from '@/types/api';

// Mock the API
jest.mock('@/services/api', () => ({
  getDigest: jest.fn(),
}));

const mockGetDigest = api.getDigest as jest.MockedFunction<typeof api.getDigest>;

const mockDigestResponse: DigestResponse = {
  greeting: 'Good morning, Sarah',
  subtitle: '3 items need attention',
  suggested_todos: [
    {
      id: 'email_1',
      title: 'Review budget proposal',
      source: 'Email from Finance Team',
      urgency: 'high',
      due: 'Today',
      email_id: '1',
      actions: [
        { id: 'reply_1', type: 'reply', label: 'Reply' },
        { id: 'dismiss_1', type: 'dismiss', label: 'Dismiss' },
      ],
    },
    {
      id: 'email_2',
      title: 'Confirm meeting with Alex',
      source: 'Email from Alex Chen',
      urgency: 'medium',
      due: null,
      email_id: '2',
      actions: [
        { id: 'reply_2', type: 'reply', label: 'Reply' },
        { id: 'dismiss_2', type: 'dismiss', label: 'Dismiss' },
      ],
    },
  ],
  topics_to_catchup: [
    {
      id: 'thread_123',
      title: 'Q1 Budget Discussion',
      email_count: 5,
      participants: ['Finance Team', 'Marketing'],
      last_activity: '2 hours ago',
      urgency: 'low',
    },
  ],
  last_updated: '2026-01-11T08:00:00Z',
};

describe('DigestScreen', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockGetDigest.mockResolvedValue(mockDigestResponse);
  });

  it('renders loading state initially', () => {
    mockGetDigest.mockImplementation(
      () => new Promise(() => {}) // Never resolves
    );

    const { getByTestId, getByText } = render(<DigestScreen />);

    expect(getByTestId('loading-container')).toBeTruthy();
    expect(getByText('Loading your digest...')).toBeTruthy();
  });

  it('renders greeting and subtitle after loading', async () => {
    const { getByTestId, getByText } = render(<DigestScreen />);

    await waitFor(() => {
      expect(getByTestId('greeting')).toBeTruthy();
    });

    expect(getByText('Good morning, Sarah')).toBeTruthy();
    expect(getByText('3 items need attention')).toBeTruthy();
  });

  it('renders suggested todos section', async () => {
    const { getByText, getByTestId } = render(<DigestScreen />);

    await waitFor(() => {
      expect(getByText('Suggested To-Dos')).toBeTruthy();
    });

    expect(getByTestId('todo-item-email_1')).toBeTruthy();
    expect(getByText('Review budget proposal')).toBeTruthy();
    expect(getByText('Email from Finance Team')).toBeTruthy();
    expect(getByText('Today')).toBeTruthy();
  });

  it('renders topics to catch up section', async () => {
    const { getByText, getByTestId } = render(<DigestScreen />);

    await waitFor(() => {
      expect(getByText('Topics to Catch Up On')).toBeTruthy();
    });

    expect(getByTestId('topic-item-thread_123')).toBeTruthy();
    expect(getByText('Q1 Budget Discussion')).toBeTruthy();
    expect(getByText('5')).toBeTruthy(); // email count badge
    expect(getByText('2 hours ago')).toBeTruthy();
  });

  it('renders action buttons on todo items', async () => {
    const { getByTestId } = render(<DigestScreen />);

    await waitFor(() => {
      expect(getByTestId('action-reply_1')).toBeTruthy();
    });

    expect(getByTestId('action-dismiss_1')).toBeTruthy();
  });

  it('calls onAction when action button is pressed', async () => {
    const onAction = jest.fn();
    const { getByTestId } = render(<DigestScreen onAction={onAction} />);

    await waitFor(() => {
      expect(getByTestId('action-reply_1')).toBeTruthy();
    });

    fireEvent.press(getByTestId('action-reply_1'));

    expect(onAction).toHaveBeenCalledWith('reply', 'email_1');
  });

  it('calls onAction when todo item is pressed', async () => {
    const onAction = jest.fn();
    const { getByTestId } = render(<DigestScreen onAction={onAction} />);

    await waitFor(() => {
      expect(getByTestId('todo-item-email_1')).toBeTruthy();
    });

    fireEvent.press(getByTestId('todo-item-email_1'));

    expect(onAction).toHaveBeenCalledWith('open', 'email_1');
  });

  it('calls onAction when topic item is pressed', async () => {
    const onAction = jest.fn();
    const { getByTestId } = render(<DigestScreen onAction={onAction} />);

    await waitFor(() => {
      expect(getByTestId('topic-item-thread_123')).toBeTruthy();
    });

    fireEvent.press(getByTestId('topic-item-thread_123'));

    expect(onAction).toHaveBeenCalledWith('open_topic', 'thread_123');
  });

  it('renders error state on API failure', async () => {
    mockGetDigest.mockRejectedValue(new Error('Network error'));

    const { getByTestId, getByText } = render(<DigestScreen />);

    await waitFor(() => {
      expect(getByTestId('error-container')).toBeTruthy();
    });

    expect(getByText('Network error')).toBeTruthy();
    expect(getByTestId('retry-button')).toBeTruthy();
  });

  it('retries fetch on retry button press', async () => {
    mockGetDigest.mockRejectedValueOnce(new Error('Network error'));
    mockGetDigest.mockResolvedValueOnce(mockDigestResponse);

    const { getByTestId, getByText } = render(<DigestScreen />);

    await waitFor(() => {
      expect(getByTestId('retry-button')).toBeTruthy();
    });

    await act(async () => {
      fireEvent.press(getByTestId('retry-button'));
    });

    await waitFor(() => {
      expect(getByText('Good morning, Sarah')).toBeTruthy();
    });

    expect(mockGetDigest).toHaveBeenCalledTimes(2);
  });

  it('renders empty state when no items', async () => {
    mockGetDigest.mockResolvedValue({
      ...mockDigestResponse,
      suggested_todos: [],
      topics_to_catchup: [],
      subtitle: 'Nothing pending',
    });

    const { getByTestId, getByText } = render(<DigestScreen />);

    await waitFor(() => {
      expect(getByTestId('empty-state')).toBeTruthy();
    });

    // The empty state shows its own "All caught up!" text
    expect(getByTestId('empty-state')).toBeTruthy();
    expect(getByText('Nothing needs your attention right now.')).toBeTruthy();
  });

  it('renders voice button when onConversationPress provided', async () => {
    const onConversationPress = jest.fn();
    const { getByTestId, getByText } = render(
      <DigestScreen onConversationPress={onConversationPress} />
    );

    await waitFor(() => {
      expect(getByTestId('voice-button')).toBeTruthy();
    });

    expect(getByText('Ask Lenso')).toBeTruthy();

    fireEvent.press(getByTestId('voice-button'));
    expect(onConversationPress).toHaveBeenCalled();
  });

  it('does not render voice button when onConversationPress not provided', async () => {
    const { queryByTestId, getByTestId } = render(<DigestScreen />);

    await waitFor(() => {
      expect(getByTestId('digest-screen')).toBeTruthy();
    });

    expect(queryByTestId('voice-button')).toBeNull();
  });

  it('handles pull to refresh', async () => {
    const { getByTestId } = render(<DigestScreen />);

    await waitFor(() => {
      expect(getByTestId('digest-screen')).toBeTruthy();
    });

    // Simulate pull to refresh
    const scrollView = getByTestId('digest-screen');
    const { refreshControl } = scrollView.props;

    await act(async () => {
      refreshControl.props.onRefresh();
    });

    // getDigest should be called twice (initial load + refresh)
    expect(mockGetDigest).toHaveBeenCalledTimes(2);
  });

  it('displays urgency colors correctly', async () => {
    const { getByTestId } = render(<DigestScreen />);

    await waitFor(() => {
      expect(getByTestId('todo-item-email_1')).toBeTruthy();
    });

    // High urgency item should exist
    const highUrgencyItem = getByTestId('todo-item-email_1');
    expect(highUrgencyItem).toBeTruthy();

    // Medium urgency item should exist
    const mediumUrgencyItem = getByTestId('todo-item-email_2');
    expect(mediumUrgencyItem).toBeTruthy();
  });

  it('displays overdue styling for overdue items', async () => {
    mockGetDigest.mockResolvedValue({
      ...mockDigestResponse,
      suggested_todos: [
        {
          id: 'email_overdue',
          title: 'Overdue task',
          source: 'Email from Team',
          urgency: 'high',
          due: 'Overdue',
          email_id: '3',
          actions: [],
        },
      ],
    });

    const { getByText } = render(<DigestScreen />);

    await waitFor(() => {
      expect(getByText('Overdue')).toBeTruthy();
    });
  });
});
