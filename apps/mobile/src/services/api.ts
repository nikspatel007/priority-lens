/**
 * Priority Lens API Service
 *
 * Client for interacting with the Priority Lens backend API.
 * Handles authentication, request formatting, and error handling.
 */

import type {
  PriorityInboxResponse,
  InboxStats,
  TaskResponse,
  TaskDetailResponse,
  TaskListResponse,
  TaskQueryOptions,
  ProjectResponse,
  ProjectDetailResponse,
  ProjectListResponse,
  ThreadCreate,
  ThreadResponse,
  ThreadListResponse,
  SessionCreate,
  SessionResponse,
  SessionListResponse,
  EventListResponse,
  TurnCreate,
  TurnResponse,
  LiveKitTokenRequest,
  LiveKitTokenResponse,
  ActionRequest,
  ActionResponse,
  SyncStatusResponse,
  CompleteConnectionResponse,
  APIError,
} from '@/types/api';

// ============================================================================
// Configuration
// ============================================================================

const API_BASE_URL =
  process.env['EXPO_PUBLIC_API_BASE_URL'] || 'http://localhost:8000';

// Auth token getter - set by AuthContext
let authTokenGetter: (() => Promise<string | null>) | null = null;

/**
 * Set the auth token getter function
 * Called by AuthContext to provide JWT retrieval
 */
export function setAuthTokenGetter(
  getter: (() => Promise<string | null>) | null
): void {
  authTokenGetter = getter;
}

// ============================================================================
// Request Helpers
// ============================================================================

/**
 * Get authentication headers for API requests
 */
async function getAuthHeaders(): Promise<Record<string, string>> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
  };

  if (authTokenGetter) {
    const token = await authTokenGetter();
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
      console.log('Auth token added to request');
    } else {
      console.warn('Auth token getter returned null');
    }
  } else {
    console.warn('No auth token getter set');
  }

  return headers;
}

/**
 * Build URL with query parameters
 */
function buildUrl(
  path: string,
  params?: Record<string, string | number | boolean>
): string {
  const url = new URL(path, API_BASE_URL);
  if (params) {
    Object.entries(params).forEach(([key, value]) => {
      url.searchParams.append(key, String(value));
    });
  }
  return url.toString();
}

/**
 * Handle API response and throw on errors
 */
async function handleResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail: string | undefined;
    let errorBody: Record<string, unknown> | undefined;
    try {
      errorBody = await response.json();
      detail = (errorBody?.detail || errorBody?.message) as string | undefined;
    } catch {
      // Ignore JSON parse errors
    }
    console.error('API error:', response.status, response.url, errorBody);
    const error = new Error(detail || `HTTP ${response.status}`) as Error & {
      status?: number;
      detail?: string;
    };
    error.status = response.status;
    error.detail = detail;
    throw error as APIError;
  }

  // Handle 204 No Content
  if (response.status === 204) {
    return undefined as T;
  }

  return response.json();
}

/**
 * Make an authenticated GET request
 */
async function get<T>(
  path: string,
  params?: Record<string, string | number | boolean | undefined>
): Promise<T> {
  const headers = await getAuthHeaders();
  // Filter out undefined values
  const filteredParams = params
    ? Object.fromEntries(
        Object.entries(params).filter(([, v]) => v !== undefined)
      ) as Record<string, string | number | boolean>
    : undefined;
  const url = buildUrl(path, filteredParams);
  const response = await fetch(url, { method: 'GET', headers });
  return handleResponse<T>(response);
}

/**
 * Make an authenticated POST request
 */
async function post<T>(path: string, body?: unknown): Promise<T> {
  const headers = await getAuthHeaders();
  const url = buildUrl(path);
  const response = await fetch(url, {
    method: 'POST',
    headers,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  return handleResponse<T>(response);
}

/**
 * Make an authenticated PUT request
 */
async function put<T>(path: string, body: unknown): Promise<T> {
  const headers = await getAuthHeaders();
  const url = buildUrl(path);
  const response = await fetch(url, {
    method: 'PUT',
    headers,
    body: JSON.stringify(body),
  });
  return handleResponse<T>(response);
}

/**
 * Make an authenticated DELETE request
 */
async function del<T>(path: string): Promise<T> {
  const headers = await getAuthHeaders();
  const url = buildUrl(path);
  const response = await fetch(url, { method: 'DELETE', headers });
  return handleResponse<T>(response);
}

// ============================================================================
// Inbox API
// ============================================================================

/**
 * Get priority inbox with ranked emails
 */
export async function getInbox(
  limit: number = 20,
  offset: number = 0
): Promise<PriorityInboxResponse> {
  return get<PriorityInboxResponse>('/api/v1/inbox', { limit, offset });
}

/**
 * Get inbox statistics
 */
export async function getInboxStats(): Promise<InboxStats> {
  return get<InboxStats>('/api/v1/inbox/stats');
}

// ============================================================================
// Task API
// ============================================================================

/**
 * Get list of tasks with optional filtering
 */
export async function getTasks(
  options?: TaskQueryOptions
): Promise<TaskListResponse> {
  const params: Record<string, string | number | boolean> = {};

  if (options?.status) {
    params['status'] = Array.isArray(options.status)
      ? options.status.join(',')
      : options.status;
  }
  if (options?.project_id !== undefined) {
    params['project_id'] = options.project_id;
  }
  if (options?.limit !== undefined) {
    params['limit'] = options.limit;
  }
  if (options?.offset !== undefined) {
    params['offset'] = options.offset;
  }
  if (options?.include_dismissed !== undefined) {
    params['include_dismissed'] = options.include_dismissed;
  }

  return get<TaskListResponse>('/api/v1/tasks', params);
}

/**
 * Get a single task by ID
 */
export async function getTask(taskId: number): Promise<TaskDetailResponse> {
  return get<TaskDetailResponse>(`/api/v1/tasks/${taskId}`);
}

/**
 * Complete a task
 */
export async function completeTask(taskId: number): Promise<TaskResponse> {
  return put<TaskResponse>(`/api/v1/tasks/${taskId}/status`, {
    status: 'completed',
  });
}

/**
 * Dismiss a task
 */
export async function dismissTask(taskId: number): Promise<TaskResponse> {
  return put<TaskResponse>(`/api/v1/tasks/${taskId}/status`, {
    status: 'dismissed',
  });
}

// ============================================================================
// Project API
// ============================================================================

/**
 * Get list of projects
 */
export async function getProjects(
  limit: number = 50,
  offset: number = 0
): Promise<ProjectListResponse> {
  return get<ProjectListResponse>('/api/v1/projects', { limit, offset });
}

/**
 * Get a single project by ID
 */
export async function getProject(
  projectId: number
): Promise<ProjectDetailResponse> {
  return get<ProjectDetailResponse>(`/api/v1/projects/${projectId}`);
}

// ============================================================================
// Thread API
// ============================================================================

/**
 * Create a new conversation thread
 */
export async function createThread(
  data?: ThreadCreate
): Promise<ThreadResponse> {
  return post<ThreadResponse>('/api/v1/threads', data);
}

/**
 * Get a thread by ID
 */
export async function getThread(threadId: string): Promise<ThreadResponse> {
  return get<ThreadResponse>(`/api/v1/threads/${threadId}`);
}

/**
 * List threads for the current user
 */
export async function listThreads(
  limit: number = 20,
  offset: number = 0
): Promise<ThreadListResponse> {
  return get<ThreadListResponse>('/api/v1/threads', { limit, offset });
}

/**
 * Delete a thread
 */
export async function deleteThread(threadId: string): Promise<void> {
  return del<void>(`/api/v1/threads/${threadId}`);
}

// ============================================================================
// Session API
// ============================================================================

/**
 * Create a new session within a thread
 */
export async function createSession(
  threadId: string,
  data?: SessionCreate
): Promise<SessionResponse> {
  return post<SessionResponse>(
    `/api/v1/threads/${threadId}/sessions`,
    data || { mode: 'text' }
  );
}

/**
 * Get a session by ID
 */
export async function getSession(
  threadId: string,
  sessionId: string
): Promise<SessionResponse> {
  return get<SessionResponse>(
    `/api/v1/threads/${threadId}/sessions/${sessionId}`
  );
}

/**
 * List sessions for a thread
 */
export async function listSessions(
  threadId: string
): Promise<SessionListResponse> {
  return get<SessionListResponse>(`/api/v1/threads/${threadId}/sessions`);
}

/**
 * Close a session
 */
export async function closeSession(
  threadId: string,
  sessionId: string
): Promise<SessionResponse> {
  return put<SessionResponse>(
    `/api/v1/threads/${threadId}/sessions/${sessionId}`,
    { status: 'ended' }
  );
}

// ============================================================================
// Turn API
// ============================================================================

/**
 * Submit a turn to the conversation
 */
export async function submitTurn(
  threadId: string,
  data: TurnCreate
): Promise<TurnResponse> {
  return post<TurnResponse>(`/api/v1/threads/${threadId}/turns`, data);
}

// ============================================================================
// Event API
// ============================================================================

/**
 * Get events for a thread after a given sequence number
 */
export async function getEvents(
  threadId: string,
  afterSeq?: number
): Promise<EventListResponse> {
  const params: Record<string, number> = {};
  if (afterSeq !== undefined) {
    params['after_seq'] = afterSeq;
  }
  return get<EventListResponse>(`/api/v1/threads/${threadId}/events`, params);
}

// ============================================================================
// LiveKit API
// ============================================================================

/**
 * Get a LiveKit token for voice sessions
 */
export async function getLiveKitToken(
  data: LiveKitTokenRequest
): Promise<LiveKitTokenResponse> {
  return post<LiveKitTokenResponse>('/api/v1/livekit/token', data);
}

// ============================================================================
// Action API
// ============================================================================

/**
 * Execute an SDUI action
 */
export async function executeAction(
  request: ActionRequest
): Promise<ActionResponse> {
  return post<ActionResponse>('/api/v1/actions', request);
}

/**
 * Get list of available action types
 */
export async function getActionTypes(): Promise<{ types: string[] }> {
  return get<{ types: string[] }>('/api/v1/actions/types');
}

// ============================================================================
// Connection API
// ============================================================================

/**
 * Connection status response from backend
 */
export interface ConnectionStatusResponse {
  provider: string;
  state: 'not_connected' | 'connected' | 'error' | 'expired';
  is_connected: boolean;
  email: string | null;
  connected_at: string | null;
  last_sync: string | null;
  error: string | null;
}

/**
 * Check if Gmail is connected on the backend
 * This verifies the backend has valid OAuth tokens
 */
export async function getGmailConnectionStatus(): Promise<ConnectionStatusResponse> {
  return get<ConnectionStatusResponse>('/api/v1/connections/gmail');
}

// ============================================================================
// Sync API
// ============================================================================

/**
 * Get sync status for Gmail connection
 * Used by mobile app to poll sync progress
 */
export async function getSyncStatus(): Promise<SyncStatusResponse> {
  return get<SyncStatusResponse>('/api/v1/connections/gmail/sync/status');
}

/**
 * Complete Google connection by sending server auth code to backend
 * The backend exchanges this code for access/refresh tokens
 * This triggers initial sync for first-time users
 */
export async function completeGoogleConnection(
  serverAuthCode: string
): Promise<CompleteConnectionResponse> {
  // mobile=true tells backend this is a mobile serverAuthCode (no redirect_uri needed)
  return post<CompleteConnectionResponse>(
    `/api/v1/connections/gmail/callback?code=${encodeURIComponent(serverAuthCode)}&mobile=true`
  );
}
