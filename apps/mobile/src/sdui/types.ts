/**
 * SDUI Type Definitions
 *
 * Server-Driven UI schema types for Priority Lens.
 * These types are shared between server and client.
 */

// ===========================================
// Layout Types
// ===========================================

export interface GridLayout {
  columns?: number | string;
  rows?: number | string;
  gap?: number;
  areas?: string[];
}

export interface LayoutProps {
  grid?: GridLayout;
  gridArea?: string;
  span?: { col?: number; row?: number };
  padding?: number | [number, number, number, number];
  margin?: number | [number, number, number, number];
  flex?: number;
  align?: 'start' | 'center' | 'end' | 'stretch';
  justify?: 'start' | 'center' | 'end' | 'between' | 'around';
  width?: number | string;
  height?: number | string;
  minHeight?: number;
  maxWidth?: number;
}

// ===========================================
// Action Types
// ===========================================

export type ActionTrigger = 'press' | 'longPress' | 'change' | 'submit';

export interface UIAction {
  id?: string;
  trigger: ActionTrigger;
  type: string;
  payload?: Record<string, unknown>;
  navigate?: string;
  haptic?: 'light' | 'medium' | 'heavy';
}

// ===========================================
// Base UIBlock
// ===========================================

export interface UIBlock {
  id: string;
  type: string;
  props?: Record<string, unknown>;
  layout?: LayoutProps;
  children?: UIBlock[];
  actions?: UIAction[];
}

// ===========================================
// Primitive Components
// ===========================================

export interface TextProps {
  value: string;
  variant?: 'body' | 'heading' | 'title' | 'caption' | 'label';
  color?: string;
  weight?: 'normal' | 'medium' | 'semibold' | 'bold';
  align?: 'left' | 'center' | 'right';
  numberOfLines?: number;
}

export interface IconProps {
  name: string;
  size?: number;
  color?: string;
}

export interface ImageProps {
  src: string;
  alt?: string;
  aspectRatio?: number;
  fit?: 'cover' | 'contain' | 'fill';
  borderRadius?: number;
}

export interface SpacerProps {
  size: number;
}

export interface DividerProps {
  color?: string;
  thickness?: number;
  direction?: 'horizontal' | 'vertical';
}

export interface BadgeProps {
  value: string | number;
  variant?: 'default' | 'success' | 'warning' | 'error' | 'info';
}

export interface AvatarProps {
  src?: string;
  name: string;
  size?: number;
  fallback?: string;
}

export interface ProgressProps {
  value: number;
  max?: number;
  variant?: 'bar' | 'circle';
  color?: string;
}

// ===========================================
// Input Components
// ===========================================

export interface ButtonProps {
  label: string;
  variant?: 'primary' | 'secondary' | 'outline' | 'ghost' | 'destructive';
  icon?: string;
  iconPosition?: 'left' | 'right';
  loading?: boolean;
  disabled?: boolean;
  size?: 'sm' | 'md' | 'lg';
}

export interface IconButtonProps {
  icon: string;
  variant?: 'primary' | 'secondary' | 'outline' | 'ghost';
  size?: number;
}

export interface TextInputProps {
  placeholder?: string;
  value?: string;
  multiline?: boolean;
  numberOfLines?: number;
  keyboard?: 'default' | 'email' | 'numeric' | 'phone' | 'url';
  secureTextEntry?: boolean;
}

export interface CheckboxProps {
  checked: boolean;
  label?: string;
}

export interface SwitchProps {
  value: boolean;
  label?: string;
}

export interface SelectProps {
  options: Array<{ label: string; value: string }>;
  value?: string;
  placeholder?: string;
}

export interface DatePickerProps {
  value?: string;
  mode?: 'date' | 'time' | 'datetime';
  minDate?: string;
  maxDate?: string;
}

// ===========================================
// Layout Components
// ===========================================

export interface BoxProps {
  backgroundColor?: string;
  borderRadius?: number;
  borderWidth?: number;
  borderColor?: string;
}

export interface StackProps {
  direction?: 'horizontal' | 'vertical';
  gap?: number;
  align?: 'start' | 'center' | 'end' | 'stretch';
  justify?: 'start' | 'center' | 'end' | 'between' | 'around';
  wrap?: boolean;
}

export interface GridProps {
  columns?: number | string;
  rows?: number | string;
  gap?: number;
  areas?: string[];
}

export interface ScrollProps {
  direction?: 'horizontal' | 'vertical';
  showIndicator?: boolean;
}

export interface CardProps {
  variant?: 'default' | 'elevated' | 'outlined';
  backgroundColor?: string;
}

// ===========================================
// Composite Components
// ===========================================

export interface ListItemProps {
  title: string;
  subtitle?: string;
  leading?: UIBlock;
  trailing?: UIBlock;
  divider?: boolean;
}

export interface PersonCardProps {
  name: string;
  title?: string;
  email?: string;
  avatar?: string;
  compact?: boolean;
}

export interface TaskCardProps {
  title: string;
  status?: 'pending' | 'in_progress' | 'completed';
  priority?: 'high' | 'medium' | 'low';
  dueDate?: string;
  assignee?: string;
}

export interface InvoiceCardProps {
  vendor: string;
  description?: string;
  amount: string;
  dueDate?: string;
  status?: 'pending' | 'paid' | 'overdue';
}

export interface CalendarEventProps {
  title: string;
  start: string;
  end: string;
  location?: string;
  attendees?: string[];
  color?: string;
}

export interface AttachmentProps {
  filename: string;
  type: 'pdf' | 'image' | 'doc' | 'sheet' | 'other';
  size?: string;
  url?: string;
  thumbnail?: string;
}

export interface ActionItemProps {
  text: string;
  checked: boolean;
  assignee?: string;
}

export interface MetricProps {
  label: string;
  value: string | number;
  change?: number;
  trend?: 'up' | 'down' | 'neutral';
  unit?: string;
}

export interface TimelineProps {
  items: Array<{
    title: string;
    subtitle?: string;
    date: string;
    status?: 'completed' | 'current' | 'upcoming';
  }>;
}

export interface HeaderProps {
  title: string;
  subtitle?: string;
  actions?: UIBlock[];
}

// ===========================================
// Rich Card Components (Interactive)
// ===========================================

export interface RichAvatarProps {
  imageUrl?: string;
  initials: string;
  colors?: [string, string];
  size?: 'sm' | 'md' | 'lg';
}

export interface RichPersonProps {
  id?: string;
  name: string;
  email?: string;
  title?: string;
  avatar: RichAvatarProps;
}

export interface RichAttachmentProps {
  id: string;
  name: string;
  type:
    | 'pdf'
    | 'spreadsheet'
    | 'document'
    | 'image'
    | 'video'
    | 'audio'
    | 'archive'
    | 'other';
  size?: string;
  previewUrl?: string;
  downloadUrl?: string;
  updatedAt?: string;
  updatedDisplay?: string;
}

export interface QuickReplyOption {
  label: string;
  value: string;
}

export type EmailType =
  | 'fyi'
  | 'needs_reply'
  | 'has_deadline'
  | 'document_shared'
  | 'meeting_request'
  | 'payment_request'
  | 'approval_needed'
  | 'follow_up'
  | 'newsletter';

export interface EmailCardProps {
  emailId: string;
  threadId?: string;
  sender: RichPersonProps;
  subject: string;
  snippet: string;
  fullBody?: string;
  isUrgent?: boolean;
  isImportant?: boolean;
  isUnread?: boolean;
  receivedAt?: string;
  attachments?: RichAttachmentProps[];
  quickReplies?: QuickReplyOption[];
  emailType?: EmailType;
  summary?: string;
  requiresResponse?: boolean;
  hasReplied?: boolean;
  threadCount?: number;
  deadline?: string;
  confidence?: number;
}

export interface MeetingAttendee extends RichPersonProps {
  status: 'accepted' | 'declined' | 'tentative' | 'pending';
  isOrganizer?: boolean;
}

export interface MeetingLocation {
  type: 'zoom' | 'meet' | 'teams' | 'phone' | 'physical' | 'other';
  value: string;
  display: string;
}

export interface MeetingCardProps {
  eventId: string;
  title: string;
  startTime: string;
  endTime: string;
  location?: MeetingLocation;
  isAllDay?: boolean;
  attendees: MeetingAttendee[];
  myStatus: 'accepted' | 'declined' | 'tentative' | 'pending';
  organizer?: string;
  description?: string;
}

export interface VendorProps {
  name: string;
  logo?: {
    type: 'image' | 'text';
    value: string;
    background?: string;
    color?: string;
  };
}

export interface AmountProps {
  value: number;
  currency: string;
  display: string;
}

export interface PaymentCardProps {
  invoiceId: string;
  vendor: VendorProps;
  description?: string;
  amount: AmountProps;
  dueDate: string;
  dueDateDisplay?: string;
  daysUntilDue?: number;
  status: 'pending' | 'paid' | 'overdue';
  invoice?: RichAttachmentProps;
  recurring?: {
    isRecurring: boolean;
    frequency?: 'weekly' | 'monthly' | 'yearly';
  };
}

export interface RichBadgeProps {
  label: string;
  variant: 'error' | 'warning' | 'success' | 'info' | 'neutral';
  pulse?: boolean;
}

export interface DocumentCardProps {
  threadId: string;
  subject: string;
  messageCount?: number;
  participantCount?: number;
  badge?: RichBadgeProps;
  documents: RichAttachmentProps[];
  summary?: string;
  summarySource?: 'ai' | 'user';
}

export interface TimeSlotProps {
  id: string;
  date: string;
  dateDisplay: string;
  startTime: string;
  endTime: string;
  timeDisplay: string;
  availability: {
    status: 'all_available' | 'conflicts' | 'unavailable';
    display: string;
    conflicts: number;
  };
}

export interface SchedulingCardProps {
  title: string;
  attendeeCount: number;
  duration: number;
  timeSlots: TimeSlotProps[];
  selectedSlotId?: string;
}

export interface AlertDetailsProps {
  type: 'text' | 'code';
  value: string;
}

export interface AlertCardProps {
  alertType: 'error' | 'warning' | 'info' | 'expiring';
  title: string;
  subtitle?: string;
  badge?: RichBadgeProps;
  details?: AlertDetailsProps;
  primaryAction?: {
    label: string;
    url?: string;
  };
  secondaryAction?: {
    label: string;
    url?: string;
  };
}

// ===========================================
// Execution Mode Components
// ===========================================

export interface TaskCarouselItem {
  id: string;
  type: 'invoice' | 'email' | 'calendar' | 'task' | 'person';
  label?: string;
  title: string;
  subtitle?: string;
  detail?: string;
  amount?: string;
  dueDate?: string;
  actionLabel?: string;
  metadata?: {
    urgent?: boolean;
    important?: boolean;
    location?: string;
    attendees?: string[];
    isAllDay?: boolean;
    priority?: 'high' | 'medium' | 'low';
    assignee?: string;
    status?: string;
    source?: string;
    avatar?: string;
    email?: string;
    phone?: string;
    company?: string;
    relationship?: string;
  };
}

export interface TaskCarouselProps {
  tasks: TaskCarouselItem[];
  currentIndex?: number;
}

export interface ExecutionModeProps {
  taskLabel?: string;
  taskName: string;
  tasks: TaskCarouselItem[];
  question: string;
  primaryAction: {
    label: string;
    actionType: string;
    payload?: Record<string, unknown>;
  };
  secondaryAction?: {
    label: string;
    actionType: string;
    payload?: Record<string, unknown>;
  };
}

export interface ProgressDotsProps {
  total: number;
  current: number;
}

export type ViewMode = 'cards' | 'list';

export interface ViewToggleProps {
  mode: ViewMode;
  onChange: (mode: ViewMode) => void;
}

export interface TaskListProps {
  tasks: TaskCarouselItem[];
  selectedTaskId?: string | null;
  onTaskSelect?: (task: TaskCarouselItem) => void;
  onTaskAction?: (task: TaskCarouselItem) => void;
  onTaskDone?: (task: TaskCarouselItem) => void;
  onTaskDefer?: (task: TaskCarouselItem) => void;
  urgencyMap?: Record<string, 'critical' | 'high' | 'medium' | 'low'>;
  timestampMap?: Record<string, string>;
}

// ===========================================
// Event Types (for streaming)
// ===========================================

export type UIEventType =
  | { type: 'ui.block.add'; block: UIBlock }
  | { type: 'ui.block.update'; id: string; changes: Partial<UIBlock> }
  | { type: 'ui.block.remove'; id: string }
  | { type: 'ui.clear' }
  | { type: 'ui.action.result'; actionId: string; result: unknown };

// ===========================================
// Component Registry Type
// ===========================================

export type ComponentType =
  // Primitives
  | 'text'
  | 'icon'
  | 'image'
  | 'spacer'
  | 'divider'
  | 'badge'
  | 'avatar'
  | 'progress'
  // Inputs
  | 'button'
  | 'iconButton'
  | 'textInput'
  | 'checkbox'
  | 'switch'
  | 'select'
  | 'datePicker'
  // Layout
  | 'box'
  | 'stack'
  | 'grid'
  | 'scroll'
  | 'card'
  // Composite
  | 'listItem'
  | 'personCard'
  | 'taskCard'
  | 'invoiceCard'
  | 'calendarEvent'
  | 'attachment'
  | 'actionItem'
  | 'metric'
  | 'timeline'
  | 'header'
  // Execution Mode
  | 'taskCarousel'
  | 'taskList'
  | 'taskListItem'
  | 'viewToggle'
  | 'executionMode'
  | 'progressDots'
  // Rich Interactive Cards
  | 'emailCard'
  | 'meetingCard'
  | 'paymentCard'
  | 'documentCard'
  | 'schedulingCard'
  | 'alertCard';
