/**
 * SDUI Composites Tests
 *
 * Tests for composite SDUI components.
 */

import React from 'react';
import { render, fireEvent } from '@testing-library/react-native';
import { SDUIPersonCard } from '../composites/SDUIPersonCard';
import { SDUIInvoiceCard } from '../composites/SDUIInvoiceCard';
import { SDUIActionItem } from '../composites/SDUIActionItem';

describe('SDUIPersonCard', () => {
  it('renders name', () => {
    const { getByText } = render(<SDUIPersonCard name="John Doe" />);
    expect(getByText('John Doe')).toBeTruthy();
  });

  it('renders title when provided', () => {
    const { getByText } = render(
      <SDUIPersonCard name="John Doe" title="Software Engineer" />
    );
    expect(getByText('John Doe')).toBeTruthy();
    expect(getByText('Software Engineer')).toBeTruthy();
  });

  it('renders email when provided', () => {
    const { getByText } = render(
      <SDUIPersonCard name="John Doe" email="john@example.com" />
    );
    expect(getByText('john@example.com')).toBeTruthy();
  });

  it('renders all fields', () => {
    const { getByText } = render(
      <SDUIPersonCard
        name="John Doe"
        title="Software Engineer"
        email="john@example.com"
      />
    );
    expect(getByText('John Doe')).toBeTruthy();
    expect(getByText('Software Engineer')).toBeTruthy();
    expect(getByText('john@example.com')).toBeTruthy();
  });

  it('renders compact variant', () => {
    const { getByText, queryByText } = render(
      <SDUIPersonCard
        name="John Doe"
        title="Engineer"
        email="john@example.com"
        compact
      />
    );
    expect(getByText('John Doe')).toBeTruthy();
    expect(getByText('Engineer')).toBeTruthy();
    // Email should not show in compact mode
    expect(queryByText('john@example.com')).toBeNull();
  });

  it('renders compact without title', () => {
    const { getByText, queryByText } = render(
      <SDUIPersonCard name="John Doe" compact />
    );
    expect(getByText('John Doe')).toBeTruthy();
  });

  it('renders with avatar', () => {
    const { getByTestId } = render(
      <SDUIPersonCard name="John Doe" avatar="https://example.com/avatar.jpg" />
    );
    expect(getByTestId('sdui-avatar-image')).toBeTruthy();
  });

  it('renders fallback avatar when no src', () => {
    const { getByTestId } = render(<SDUIPersonCard name="John Doe" />);
    expect(getByTestId('sdui-avatar-fallback')).toBeTruthy();
  });

  it('renders compact with avatar', () => {
    const { getByTestId } = render(
      <SDUIPersonCard
        name="Jane Smith"
        avatar="https://example.com/jane.jpg"
        compact
      />
    );
    expect(getByTestId('sdui-avatar-image')).toBeTruthy();
  });
});

describe('SDUIInvoiceCard', () => {
  it('renders vendor name', () => {
    const { getByText } = render(
      <SDUIInvoiceCard vendor="Acme Inc" amount="$100.00" />
    );
    expect(getByText('Acme Inc')).toBeTruthy();
  });

  it('renders amount', () => {
    const { getByText } = render(
      <SDUIInvoiceCard vendor="Acme Inc" amount="$100.00" />
    );
    expect(getByText('$100.00')).toBeTruthy();
  });

  it('renders description when provided', () => {
    const { getByText } = render(
      <SDUIInvoiceCard
        vendor="Acme Inc"
        amount="$100.00"
        description="Monthly subscription"
      />
    );
    expect(getByText('Monthly subscription')).toBeTruthy();
  });

  it('renders due date when provided', () => {
    const { getByText } = render(
      <SDUIInvoiceCard
        vendor="Acme Inc"
        amount="$100.00"
        dueDate="Jan 15, 2024"
      />
    );
    expect(getByText('Jan 15, 2024')).toBeTruthy();
    expect(getByText('Due')).toBeTruthy();
  });

  it('renders pending status', () => {
    const { getByText } = render(
      <SDUIInvoiceCard vendor="Acme Inc" amount="$100.00" status="pending" />
    );
    expect(getByText('Pending')).toBeTruthy();
  });

  it('renders paid status', () => {
    const { getByText } = render(
      <SDUIInvoiceCard vendor="Acme Inc" amount="$100.00" status="paid" />
    );
    expect(getByText('Paid')).toBeTruthy();
  });

  it('renders overdue status', () => {
    const { getByText } = render(
      <SDUIInvoiceCard vendor="Acme Inc" amount="$100.00" status="overdue" />
    );
    expect(getByText('Overdue')).toBeTruthy();
  });

  it('renders default pending status when not specified', () => {
    const { getByText } = render(
      <SDUIInvoiceCard vendor="Acme Inc" amount="$100.00" />
    );
    expect(getByText('Pending')).toBeTruthy();
  });

  it('renders complete invoice card', () => {
    const { getByText } = render(
      <SDUIInvoiceCard
        vendor="Acme Corporation"
        description="Annual license renewal"
        amount="$1,200.00"
        dueDate="February 28, 2024"
        status="pending"
      />
    );
    expect(getByText('Acme Corporation')).toBeTruthy();
    expect(getByText('Annual license renewal')).toBeTruthy();
    expect(getByText('$1,200.00')).toBeTruthy();
    expect(getByText('February 28, 2024')).toBeTruthy();
    expect(getByText('Pending')).toBeTruthy();
    expect(getByText('Amount')).toBeTruthy();
    expect(getByText('Due')).toBeTruthy();
  });

  it('renders without due date', () => {
    const { getByText, queryByText } = render(
      <SDUIInvoiceCard vendor="Acme Inc" amount="$100.00" />
    );
    expect(getByText('$100.00')).toBeTruthy();
    expect(queryByText('Due')).toBeNull();
  });
});

describe('SDUIActionItem', () => {
  it('renders text', () => {
    const { getByText } = render(
      <SDUIActionItem text="Complete task" checked={false} />
    );
    expect(getByText('Complete task')).toBeTruthy();
  });

  it('renders unchecked state', () => {
    const { getByTestId } = render(
      <SDUIActionItem text="Task" checked={false} />
    );
    expect(getByTestId('sdui-action-item')).toBeTruthy();
  });

  it('renders checked state', () => {
    const { getByTestId } = render(
      <SDUIActionItem text="Task" checked={true} />
    );
    expect(getByTestId('sdui-action-item')).toBeTruthy();
  });

  it('calls onToggle when pressed', () => {
    const onToggle = jest.fn();
    const { getByTestId } = render(
      <SDUIActionItem text="Task" checked={false} onToggle={onToggle} />
    );
    fireEvent.press(getByTestId('sdui-action-item'));
    expect(onToggle).toHaveBeenCalled();
  });

  it('renders assignee avatar when provided', () => {
    const { getByTestId } = render(
      <SDUIActionItem text="Task" checked={false} assignee="John Doe" />
    );
    expect(getByTestId('sdui-avatar-fallback')).toBeTruthy();
  });

  it('renders without assignee', () => {
    const { queryByTestId } = render(
      <SDUIActionItem text="Task" checked={false} />
    );
    expect(queryByTestId('sdui-avatar-fallback')).toBeNull();
  });

  it('handles press without onToggle', () => {
    const { getByTestId } = render(
      <SDUIActionItem text="Task" checked={false} />
    );
    // Should not throw
    fireEvent.press(getByTestId('sdui-action-item'));
  });

  it('toggles from unchecked to checked', () => {
    const onToggle = jest.fn();
    const { getByTestId, rerender } = render(
      <SDUIActionItem text="Task" checked={false} onToggle={onToggle} />
    );

    fireEvent.press(getByTestId('sdui-action-item'));
    expect(onToggle).toHaveBeenCalledTimes(1);

    rerender(<SDUIActionItem text="Task" checked={true} onToggle={onToggle} />);
    fireEvent.press(getByTestId('sdui-action-item'));
    expect(onToggle).toHaveBeenCalledTimes(2);
  });

  it('renders with different text styles based on checked state', () => {
    const { getByText, rerender } = render(
      <SDUIActionItem text="My Task" checked={false} />
    );
    expect(getByText('My Task')).toBeTruthy();

    rerender(<SDUIActionItem text="My Task" checked={true} />);
    expect(getByText('My Task')).toBeTruthy();
  });
});
