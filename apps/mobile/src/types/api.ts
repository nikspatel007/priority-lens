/**
 * API Types for Priority Lens Mobile
 *
 * These types mirror the Priority Lens backend Pydantic schemas.
 */

// ============================================================================
// Common Types
// ============================================================================

export type TaskStatus = 'pending' | 'in_progress' | 'completed' | 'dismissed';
export type TaskType =
  | 'review'
  | 'send'
  | 'schedule'
  | 'decision'
  | 'research'
  | 'create'
  | 'follow_up'
  | 'other';
export type TaskComplexity =
  | 'trivial'
  | 'quick'
  | 'medium'
  | 'substantial'
  | 'unknown';
export type SessionMode = 'text' | 'voice';
export type SessionStatus = 'active' | 'ended';
export type EventActor = 'system' | 'user' | 'agent';

// ============================================================================
// Inbox Types
// ============================================================================

export interface EmailSummary {
  id: number;
  message_id: string;
  thread_id: string | null;
  subject: string | null;
  from_email: string | null;
  from_name: string | null;
  date_parsed: string | null;
  body_preview: string | null;
  is_sent: boolean;
  action: string | null;
  has_attachments: boolean;
  labels: string[] | null;
}

export interface PriorityContext {
  email_id: number;
  sender_email: string | null;
  sender_importance: number | null;
  sender_reply_rate: number | null;
  thread_length: number | null;
  is_business_hours: boolean | null;
  age_hours: number | null;
  people_score: number | null;
  temporal_score: number | null;
  relationship_score: number | null;
  overall_priority: number | null;
}

export interface PriorityEmail {
  email: EmailSummary;
  priority_rank: number;
  priority_score: number;
  context: PriorityContext | null;
  task_count: number;
  project_name: string | null;
}

export interface PriorityInboxResponse {
  emails: PriorityEmail[];
  total: number;
  limit: number;
  offset: number;
  has_more: boolean;
  pending_tasks: number;
  urgent_count: number;
  from_real_people_count: number;
}

export interface InboxStats {
  total_emails: number;
  unread_count: number;
  pending_tasks: number;
  urgent_emails: number;
  from_real_people: number;
  avg_priority_score: number | null;
  oldest_unanswered_hours: number | null;
}

// ============================================================================
// Task Types
// ============================================================================

export interface TaskResponse {
  id: number;
  task_id: string;
  email_id: number | null;
  project_id: number | null;
  description: string;
  task_type: string | null;
  complexity: string | null;
  deadline: string | null;
  deadline_text: string | null;
  urgency_score: number | null;
  status: string;
  assigned_to: string | null;
  assigned_by: string | null;
  is_assigned_to_user: boolean;
  extraction_method: string | null;
  completed_at: string | null;
  created_at: string;
  user_id: string | null;
}

export interface TaskDetailResponse extends TaskResponse {
  email_subject: string | null;
  email_from: string | null;
  email_date: string | null;
  source_text: string | null;
  project_name: string | null;
}

export interface TaskListResponse {
  tasks: TaskResponse[];
  total: number;
  limit: number;
  offset: number;
  has_more: boolean;
}

export interface TaskQueryOptions {
  status?: TaskStatus | TaskStatus[];
  project_id?: number;
  limit?: number;
  offset?: number;
  include_dismissed?: boolean;
}

// ============================================================================
// Project Types
// ============================================================================

export interface ProjectResponse {
  id: number;
  name: string;
  project_type: string | null;
  owner_email: string | null;
  participants: string[] | null;
  is_active: boolean;
  priority: number | null;
  email_count: number;
  last_activity: string | null;
  detected_from: string | null;
  confidence: number | null;
  user_id: string | null;
  created_at: string;
}

export interface ProjectDetailResponse extends ProjectResponse {
  description: string | null;
  keywords: string[] | null;
  start_date: string | null;
  due_date: string | null;
  completed_at: string | null;
  cluster_id: number | null;
  related_email_count: number;
}

export interface ProjectListResponse {
  projects: ProjectResponse[];
  total: number;
  limit: number;
  offset: number;
  has_more: boolean;
}

// ============================================================================
// Thread/Session Types
// ============================================================================

export interface ThreadCreate {
  title?: string | null;
  metadata?: Record<string, unknown>;
}

export interface ThreadResponse {
  id: string;
  org_id: string;
  user_id: string;
  title: string | null;
  metadata: Record<string, unknown>;
  created_at: string;
  updated_at: string;
}

export interface ThreadListResponse {
  threads: ThreadResponse[];
  total: number;
}

export interface SessionCreate {
  mode?: SessionMode;
  livekit_room?: string | null;
  metadata?: Record<string, unknown>;
}

export interface SessionResponse {
  id: string;
  thread_id: string;
  org_id: string;
  mode: string;
  status: string;
  livekit_room: string | null;
  metadata: Record<string, unknown>;
  started_at: string;
  ended_at: string | null;
}

export interface SessionListResponse {
  sessions: SessionResponse[];
  total: number;
}

// ============================================================================
// Event Types
// ============================================================================

export type EventType =
  | 'turn.user.open'
  | 'turn.user.close'
  | 'turn.agent.open'
  | 'turn.agent.close'
  | 'ui.text.submit'
  | 'stt.final'
  | 'assistant.text.delta'
  | 'assistant.text.final'
  | 'tool.call'
  | 'tool.result'
  | 'ui.block'
  | 'ui.clear'
  | 'action.result';

export interface EventResponse {
  event_id: string;
  thread_id: string;
  org_id: string;
  seq: number;
  ts: number;
  actor: string;
  type: string;
  payload: Record<string, unknown>;
  correlation_id: string | null;
  session_id: string | null;
  user_id: string | null;
}

export interface EventListResponse {
  events: EventResponse[];
  next_seq: number;
  has_more: boolean;
}

// ============================================================================
// Turn Types
// ============================================================================

export interface TextInput {
  type: 'text';
  text: string;
}

export interface VoiceInput {
  type: 'voice';
  transcript: string;
  confidence?: number;
  duration_ms?: number;
}

export type TurnInput = TextInput | VoiceInput;

export interface TurnCreate {
  session_id: string;
  input: TurnInput;
}

export interface TurnResponse {
  correlation_id: string;
  accepted: boolean;
  thread_id: string;
  session_id: string;
  seq: number;
}

// ============================================================================
// LiveKit Types
// ============================================================================

export interface LiveKitTokenRequest {
  thread_id: string;
  session_id: string;
  participant_name?: string;
  ttl_seconds?: number;
}

export interface LiveKitTokenResponse {
  token: string;
  room_name: string;
  livekit_url: string;
  expires_in: number;
}

// ============================================================================
// Action Types
// ============================================================================

export interface ActionRequest {
  id: string;
  type: string;
  thread_id: string;
  session_id?: string | null;
  payload?: Record<string, unknown>;
}

export interface ActionResponse {
  ok: boolean;
  action_id: string;
  status: 'success' | 'failure' | 'pending';
  message: string;
  data?: Record<string, unknown> | null;
  error?: string | null;
}

// ============================================================================
// Sync Types
// ============================================================================

export type SyncStatus = 'pending' | 'syncing' | 'completed' | 'failed';

export interface SyncStatusResponse {
  status: SyncStatus;
  emails_synced: number;
  total_emails: number | null;
  progress: number;
  error: string | null;
}

export interface CompleteConnectionResponse {
  status: string;
  sync_started: boolean;
}

// ============================================================================
// Digest Types
// ============================================================================

export type UrgencyLevel = 'high' | 'medium' | 'low';

export interface DigestAction {
  id: string;
  type: string;
  label: string;
  endpoint?: string | null;
  params?: Record<string, unknown>;
}

export interface DigestTodoItem {
  id: string;
  title: string;
  source: string;
  urgency: UrgencyLevel;
  due?: string | null;
  context?: string | null;
  email_id?: string | null;
  actions: DigestAction[];
}

export interface DigestTopicItem {
  id: string;
  title: string;
  email_count: number;
  participants: string[];
  last_activity: string;
  summary?: string | null;
  urgency: UrgencyLevel;
}

export interface DigestResponse {
  greeting: string;
  subtitle: string;
  suggested_todos: DigestTodoItem[];
  topics_to_catchup: DigestTopicItem[];
  last_updated: string;
  user_preferences?: Record<string, unknown>;
}

// ============================================================================
// SDUI Types
// ============================================================================

export interface UIAction {
  trigger: 'press' | 'longPress' | 'swipe';
  type: string;
  payload?: Record<string, unknown>;
}

export interface LayoutProps {
  padding?: number;
  margin?: number;
  gap?: number;
  flex?: number;
  alignItems?: 'flex-start' | 'center' | 'flex-end' | 'stretch';
  justifyContent?:
    | 'flex-start'
    | 'center'
    | 'flex-end'
    | 'space-between'
    | 'space-around';
}

export interface UIBlock {
  id: string;
  type: string;
  props?: Record<string, unknown>;
  layout?: LayoutProps;
  children?: UIBlock[];
  actions?: UIAction[];
}

// ============================================================================
// API Error Types
// ============================================================================

export interface APIErrorResponse {
  detail: string;
}

export class APIError extends Error {
  readonly status: number;
  readonly detail?: string;

  constructor(status: number, message: string, detail?: string) {
    super(message);
    this.name = 'APIError';
    this.status = status;
    this.detail = detail;
  }
}
