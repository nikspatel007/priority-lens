/**
 * MSW Request Handlers for Priority Lens Mobile
 *
 * Mock API responses for testing.
 */

import { http, HttpResponse } from 'msw';
import type {
  PriorityInboxResponse,
  InboxStats,
  TaskListResponse,
  ProjectListResponse,
  ThreadResponse,
  SessionResponse,
  EventListResponse,
  LiveKitTokenResponse,
  ActionResponse,
  GoogleConnection,
} from '../types/api';

const API_BASE_URL = process.env['EXPO_PUBLIC_API_BASE_URL'] || 'http://localhost:8000';

// ============================================================
// Mock Data
// ============================================================

export const mockInboxResponse: PriorityInboxResponse = {
  emails: [
    {
      id: 1,
      messageId: 'msg_001',
      threadId: 'thread_001',
      subject: 'Q1 Planning Meeting - Action Required',
      sender: 'John Smith',
      senderEmail: 'john@example.com',
      receivedAt: new Date().toISOString(),
      snippet: 'Please review the attached agenda and confirm your attendance...',
      labels: ['INBOX', 'IMPORTANT'],
      priorityScore: 0.95,
      priorityRank: 1,
      isFromRealPerson: true,
      hasActionableContent: true,
      suggestedAction: 'reply',
      aiClassification: {
        category: 'meeting_request',
        confidence: 0.92,
      },
    },
    {
      id: 2,
      messageId: 'msg_002',
      threadId: 'thread_002',
      subject: 'Project Update: Phase 2 Complete',
      sender: 'Jane Doe',
      senderEmail: 'jane@example.com',
      receivedAt: new Date(Date.now() - 3600000).toISOString(),
      snippet: 'Great news! We have completed Phase 2 ahead of schedule...',
      labels: ['INBOX'],
      priorityScore: 0.75,
      priorityRank: 2,
      isFromRealPerson: true,
      hasActionableContent: false,
    },
  ],
  total: 2,
  limit: 10,
  offset: 0,
  hasMore: false,
  pendingTasks: 3,
  urgentCount: 1,
  fromRealPeopleCount: 2,
};

export const mockInboxStats: InboxStats = {
  totalEmails: 150,
  unreadCount: 12,
  urgentCount: 3,
  pendingTasks: 8,
  fromRealPeopleCount: 45,
};

export const mockTasksResponse: TaskListResponse = {
  tasks: [
    {
      id: 1,
      title: 'Review Q1 Planning Agenda',
      description: 'Review and provide feedback on the Q1 planning agenda',
      status: 'pending',
      priority: 'urgent',
      dueDate: new Date(Date.now() + 86400000).toISOString(),
      sourceEmailId: 1,
      sourceEmailSubject: 'Q1 Planning Meeting - Action Required',
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    },
    {
      id: 2,
      title: 'Prepare budget report',
      status: 'in_progress',
      priority: 'high',
      projectId: 1,
      projectName: 'Q1 Planning',
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    },
  ],
  total: 2,
  limit: 10,
  offset: 0,
  hasMore: false,
};

export const mockProjectsResponse: ProjectListResponse = {
  projects: [
    {
      id: 1,
      name: 'Q1 Planning',
      description: 'Q1 2024 planning and strategy',
      color: '#2196F3',
      taskCount: 5,
      completedTaskCount: 2,
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
    },
  ],
  total: 1,
};

// ============================================================
// Request Handlers
// ============================================================

export const handlers = [
  // Gmail Connection
  http.get(`${API_BASE_URL}/api/v1/connections/gmail`, () => {
    return HttpResponse.json<GoogleConnection>({
      isConnected: true,
      email: 'user@example.com',
      connectedAt: new Date().toISOString(),
    });
  }),

  // Inbox
  http.get(`${API_BASE_URL}/api/v1/inbox`, () => {
    return HttpResponse.json(mockInboxResponse);
  }),

  http.get(`${API_BASE_URL}/api/v1/inbox/stats`, () => {
    return HttpResponse.json(mockInboxStats);
  }),

  // Tasks
  http.get(`${API_BASE_URL}/api/v1/tasks`, () => {
    return HttpResponse.json(mockTasksResponse);
  }),

  http.get(`${API_BASE_URL}/api/v1/tasks/:taskId`, ({ params }) => {
    const task = mockTasksResponse.tasks.find(
      (t) => t.id === Number(params['taskId'])
    );
    if (!task) {
      return new HttpResponse(null, { status: 404 });
    }
    return HttpResponse.json(task);
  }),

  http.post(`${API_BASE_URL}/api/v1/tasks/:taskId/complete`, () => {
    return new HttpResponse(null, { status: 204 });
  }),

  http.post(`${API_BASE_URL}/api/v1/tasks/:taskId/dismiss`, () => {
    return new HttpResponse(null, { status: 204 });
  }),

  // Projects
  http.get(`${API_BASE_URL}/api/v1/projects`, () => {
    return HttpResponse.json(mockProjectsResponse);
  }),

  // Threads
  http.post(`${API_BASE_URL}/api/v1/threads`, () => {
    return HttpResponse.json<ThreadResponse>({
      thread: {
        id: 'thread-uuid-123',
        userId: 'user-uuid-123',
        orgId: 'org-uuid-123',
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
      },
    });
  }),

  // Sessions
  http.post(`${API_BASE_URL}/api/v1/threads/:threadId/sessions`, () => {
    return HttpResponse.json<SessionResponse>({
      session: {
        id: 'session-uuid-123',
        threadId: 'thread-uuid-123',
        mode: 'voice',
        createdAt: new Date().toISOString(),
      },
    });
  }),

  // Turns
  http.post(`${API_BASE_URL}/api/v1/threads/:threadId/turns`, () => {
    return HttpResponse.json({
      turnId: 'turn-uuid-123',
      status: 'accepted',
    });
  }),

  // Events
  http.get(`${API_BASE_URL}/api/v1/threads/:threadId/events`, () => {
    return HttpResponse.json<EventListResponse>({
      events: [],
      lastSeq: 0,
    });
  }),

  // LiveKit Token
  http.post(`${API_BASE_URL}/api/v1/livekit/token`, () => {
    return HttpResponse.json<LiveKitTokenResponse>({
      token: 'mock-livekit-token',
      url: 'wss://livekit.example.com',
      roomName: 'room-123',
    });
  }),

  // Actions
  http.post(`${API_BASE_URL}/api/v1/actions`, () => {
    return HttpResponse.json<ActionResponse>({
      success: true,
      message: 'Action executed successfully',
    });
  }),
];
