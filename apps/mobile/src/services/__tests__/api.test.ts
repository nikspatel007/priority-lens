/**
 * Tests for Priority Lens API Service
 */

import {
  setAuthTokenGetter,
  getInbox,
  getInboxStats,
  getTasks,
  getTask,
  completeTask,
  dismissTask,
  getProjects,
  getProject,
  createThread,
  getThread,
  listThreads,
  deleteThread,
  createSession,
  getSession,
  listSessions,
  closeSession,
  submitTurn,
  getEvents,
  getLiveKitToken,
  executeAction,
  getActionTypes,
  getSyncStatus,
  completeGoogleConnection,
} from '../api';

// Mock fetch globally
const mockFetch = jest.fn();
global.fetch = mockFetch;

describe('API Service', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    setAuthTokenGetter(null);
  });

  describe('Authentication', () => {
    it('makes requests without auth header when no token getter', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () => Promise.resolve({ emails: [], total: 0, limit: 20, offset: 0, has_more: false, pending_tasks: 0, urgent_count: 0, from_real_people_count: 0 }),
      });

      await getInbox();

      expect(mockFetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          headers: {
            'Content-Type': 'application/json',
          },
        })
      );
    });

    it('makes requests with auth header when token getter is set', async () => {
      setAuthTokenGetter(() => Promise.resolve('test-token'));

      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () => Promise.resolve({ emails: [], total: 0, limit: 20, offset: 0, has_more: false, pending_tasks: 0, urgent_count: 0, from_real_people_count: 0 }),
      });

      await getInbox();

      expect(mockFetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          headers: {
            'Content-Type': 'application/json',
            'Authorization': 'Bearer test-token',
          },
        })
      );
    });

    it('makes requests without auth header when token getter returns null', async () => {
      setAuthTokenGetter(() => Promise.resolve(null));

      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () => Promise.resolve({ emails: [], total: 0, limit: 20, offset: 0, has_more: false, pending_tasks: 0, urgent_count: 0, from_real_people_count: 0 }),
      });

      await getInbox();

      expect(mockFetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          headers: {
            'Content-Type': 'application/json',
          },
        })
      );
    });
  });

  describe('URL Building', () => {
    it('skips undefined query params', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () => Promise.resolve({
          tasks: [],
          total: 0,
          limit: 20,
          offset: 0,
          has_more: false,
        }),
      });

      // Only pass status, leave other options undefined
      await getTasks({ status: 'pending' });

      const url = mockFetch.mock.calls[0][0];
      expect(url).toContain('status=pending');
      expect(url).not.toContain('project_id');
      expect(url).not.toContain('limit');
      expect(url).not.toContain('offset');
      expect(url).not.toContain('include_dismissed');
    });
  });

  describe('POST/PUT without body', () => {
    it('POST sends undefined body when data not provided', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () => Promise.resolve({
          id: 'thread-123',
          org_id: 'org-123',
          user_id: 'user-123',
          title: null,
          metadata: {},
          created_at: '2024-01-01T00:00:00Z',
          updated_at: '2024-01-01T00:00:00Z',
        }),
      });

      await createThread();

      // createThread called without args sends undefined body
      expect(mockFetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          method: 'POST',
          body: undefined,
        })
      );
    });

    it('POST sends JSON body when data is provided', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () => Promise.resolve({
          id: 'thread-123',
          org_id: 'org-123',
          user_id: 'user-123',
          title: 'My Thread',
          metadata: {},
          created_at: '2024-01-01T00:00:00Z',
          updated_at: '2024-01-01T00:00:00Z',
        }),
      });

      await createThread({ title: 'My Thread' });

      expect(mockFetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({ title: 'My Thread' }),
        })
      );
    });
  });

  describe('Error Handling', () => {
    it('throws error with status on non-OK response', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 404,
        json: () => Promise.resolve({ detail: 'Not found' }),
      });

      await expect(getInbox()).rejects.toMatchObject({
        message: 'Not found',
        status: 404,
        detail: 'Not found',
      });
    });

    it('throws error with HTTP status when JSON parse fails', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
        json: () => Promise.reject(new Error('Invalid JSON')),
      });

      await expect(getInbox()).rejects.toMatchObject({
        message: 'HTTP 500',
        status: 500,
      });
    });

    it('handles error response with message field', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 400,
        json: () => Promise.resolve({ message: 'Bad request' }),
      });

      await expect(getInbox()).rejects.toMatchObject({
        message: 'Bad request',
        status: 400,
      });
    });
  });

  describe('Inbox API', () => {
    it('getInbox calls correct endpoint with defaults', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () => Promise.resolve({
          emails: [],
          total: 0,
          limit: 20,
          offset: 0,
          has_more: false,
          pending_tasks: 0,
          urgent_count: 0,
          from_real_people_count: 0,
        }),
      });

      const result = await getInbox();

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/v1/inbox?limit=20&offset=0'),
        expect.any(Object)
      );
      expect(result.emails).toEqual([]);
    });

    it('getInbox passes custom limit and offset', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () => Promise.resolve({
          emails: [],
          total: 100,
          limit: 10,
          offset: 20,
          has_more: true,
          pending_tasks: 5,
          urgent_count: 3,
          from_real_people_count: 50,
        }),
      });

      await getInbox(10, 20);

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/v1/inbox?limit=10&offset=20'),
        expect.any(Object)
      );
    });

    it('getInboxStats calls correct endpoint', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () => Promise.resolve({
          total_emails: 100,
          unread_count: 10,
          pending_tasks: 5,
          urgent_emails: 3,
          from_real_people: 50,
          avg_priority_score: 0.75,
          oldest_unanswered_hours: 24,
        }),
      });

      const result = await getInboxStats();

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/v1/inbox/stats'),
        expect.any(Object)
      );
      expect(result.total_emails).toBe(100);
    });
  });

  describe('Task API', () => {
    it('getTasks calls correct endpoint with no options', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () => Promise.resolve({
          tasks: [],
          total: 0,
          limit: 20,
          offset: 0,
          has_more: false,
        }),
      });

      await getTasks();

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/v1/tasks'),
        expect.any(Object)
      );
    });

    it('getTasks passes filter options', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () => Promise.resolve({
          tasks: [],
          total: 0,
          limit: 10,
          offset: 5,
          has_more: false,
        }),
      });

      await getTasks({
        status: 'pending',
        project_id: 123,
        limit: 10,
        offset: 5,
        include_dismissed: true,
      });

      const url = mockFetch.mock.calls[0][0];
      expect(url).toContain('status=pending');
      expect(url).toContain('project_id=123');
      expect(url).toContain('limit=10');
      expect(url).toContain('offset=5');
      expect(url).toContain('include_dismissed=true');
    });

    it('getTasks handles array of statuses', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () => Promise.resolve({
          tasks: [],
          total: 0,
          limit: 20,
          offset: 0,
          has_more: false,
        }),
      });

      await getTasks({ status: ['pending', 'in_progress'] });

      const url = mockFetch.mock.calls[0][0];
      expect(url).toContain('status=pending%2Cin_progress');
    });

    it('getTask calls correct endpoint', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () => Promise.resolve({
          id: 1,
          task_id: 'task-123',
          description: 'Test task',
          status: 'pending',
          created_at: '2024-01-01T00:00:00Z',
        }),
      });

      const result = await getTask(1);

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/v1/tasks/1'),
        expect.any(Object)
      );
      expect(result.id).toBe(1);
    });

    it('completeTask calls correct endpoint', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () => Promise.resolve({
          id: 1,
          task_id: 'task-123',
          description: 'Test task',
          status: 'completed',
          created_at: '2024-01-01T00:00:00Z',
        }),
      });

      const result = await completeTask(1);

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/v1/tasks/1/status'),
        expect.objectContaining({
          method: 'PUT',
          body: JSON.stringify({ status: 'completed' }),
        })
      );
      expect(result.status).toBe('completed');
    });

    it('dismissTask calls correct endpoint', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () => Promise.resolve({
          id: 1,
          task_id: 'task-123',
          description: 'Test task',
          status: 'dismissed',
          created_at: '2024-01-01T00:00:00Z',
        }),
      });

      const result = await dismissTask(1);

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/v1/tasks/1/status'),
        expect.objectContaining({
          method: 'PUT',
          body: JSON.stringify({ status: 'dismissed' }),
        })
      );
      expect(result.status).toBe('dismissed');
    });
  });

  describe('Project API', () => {
    it('getProjects calls correct endpoint with defaults', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () => Promise.resolve({
          projects: [],
          total: 0,
          limit: 50,
          offset: 0,
          has_more: false,
        }),
      });

      await getProjects();

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/v1/projects?limit=50&offset=0'),
        expect.any(Object)
      );
    });

    it('getProject calls correct endpoint', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () => Promise.resolve({
          id: 1,
          name: 'Test Project',
          is_active: true,
          email_count: 10,
          created_at: '2024-01-01T00:00:00Z',
        }),
      });

      const result = await getProject(1);

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/v1/projects/1'),
        expect.any(Object)
      );
      expect(result.id).toBe(1);
    });
  });

  describe('Thread API', () => {
    it('createThread calls correct endpoint', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () => Promise.resolve({
          id: 'thread-123',
          org_id: 'org-123',
          user_id: 'user-123',
          title: null,
          metadata: {},
          created_at: '2024-01-01T00:00:00Z',
          updated_at: '2024-01-01T00:00:00Z',
        }),
      });

      const result = await createThread();

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/v1/threads'),
        expect.objectContaining({
          method: 'POST',
          body: undefined,
        })
      );
      expect(result.id).toBe('thread-123');
    });

    it('createThread passes data', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () => Promise.resolve({
          id: 'thread-123',
          org_id: 'org-123',
          user_id: 'user-123',
          title: 'My Thread',
          metadata: { key: 'value' },
          created_at: '2024-01-01T00:00:00Z',
          updated_at: '2024-01-01T00:00:00Z',
        }),
      });

      await createThread({ title: 'My Thread', metadata: { key: 'value' } });

      expect(mockFetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          body: JSON.stringify({ title: 'My Thread', metadata: { key: 'value' } }),
        })
      );
    });

    it('getThread calls correct endpoint', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () => Promise.resolve({
          id: 'thread-123',
          org_id: 'org-123',
          user_id: 'user-123',
          title: null,
          metadata: {},
          created_at: '2024-01-01T00:00:00Z',
          updated_at: '2024-01-01T00:00:00Z',
        }),
      });

      await getThread('thread-123');

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/v1/threads/thread-123'),
        expect.objectContaining({ method: 'GET' })
      );
    });

    it('listThreads calls correct endpoint', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () => Promise.resolve({
          threads: [],
          total: 0,
        }),
      });

      await listThreads();

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/v1/threads?limit=20&offset=0'),
        expect.any(Object)
      );
    });

    it('deleteThread calls correct endpoint and handles 204 response', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 204,
      });

      const result = await deleteThread('thread-123');

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/v1/threads/thread-123'),
        expect.objectContaining({ method: 'DELETE' })
      );
      expect(result).toBeUndefined();
    });
  });

  describe('Session API', () => {
    it('createSession calls correct endpoint', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () => Promise.resolve({
          id: 'session-123',
          thread_id: 'thread-123',
          org_id: 'org-123',
          mode: 'text',
          status: 'active',
          livekit_room: null,
          metadata: {},
          started_at: '2024-01-01T00:00:00Z',
          ended_at: null,
        }),
      });

      const result = await createSession('thread-123');

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/v1/threads/thread-123/sessions'),
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({ mode: 'text' }),
        })
      );
      expect(result.id).toBe('session-123');
    });

    it('createSession with custom data', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () => Promise.resolve({
          id: 'session-123',
          thread_id: 'thread-123',
          org_id: 'org-123',
          mode: 'voice',
          status: 'active',
          livekit_room: 'room-123',
          metadata: { key: 'value' },
          started_at: '2024-01-01T00:00:00Z',
          ended_at: null,
        }),
      });

      await createSession('thread-123', {
        mode: 'voice',
        livekit_room: 'room-123',
        metadata: { key: 'value' },
      });

      expect(mockFetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          body: JSON.stringify({
            mode: 'voice',
            livekit_room: 'room-123',
            metadata: { key: 'value' },
          }),
        })
      );
    });

    it('getSession calls correct endpoint', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () => Promise.resolve({
          id: 'session-123',
          thread_id: 'thread-123',
          org_id: 'org-123',
          mode: 'text',
          status: 'active',
          livekit_room: null,
          metadata: {},
          started_at: '2024-01-01T00:00:00Z',
          ended_at: null,
        }),
      });

      await getSession('thread-123', 'session-123');

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/v1/threads/thread-123/sessions/session-123'),
        expect.any(Object)
      );
    });

    it('listSessions calls correct endpoint', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () => Promise.resolve({
          sessions: [],
          total: 0,
        }),
      });

      await listSessions('thread-123');

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/v1/threads/thread-123/sessions'),
        expect.any(Object)
      );
    });

    it('closeSession calls correct endpoint', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () => Promise.resolve({
          id: 'session-123',
          thread_id: 'thread-123',
          org_id: 'org-123',
          mode: 'text',
          status: 'ended',
          livekit_room: null,
          metadata: {},
          started_at: '2024-01-01T00:00:00Z',
          ended_at: '2024-01-01T01:00:00Z',
        }),
      });

      const result = await closeSession('thread-123', 'session-123');

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/v1/threads/thread-123/sessions/session-123'),
        expect.objectContaining({
          method: 'PUT',
          body: JSON.stringify({ status: 'ended' }),
        })
      );
      expect(result.status).toBe('ended');
    });
  });

  describe('Turn API', () => {
    it('submitTurn calls correct endpoint', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () => Promise.resolve({
          correlation_id: 'corr-123',
          accepted: true,
          thread_id: 'thread-123',
          session_id: 'session-123',
          seq: 1,
        }),
      });

      const result = await submitTurn('thread-123', {
        session_id: 'session-123',
        input: { type: 'text', text: 'Hello!' },
      });

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/v1/threads/thread-123/turns'),
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({
            session_id: 'session-123',
            input: { type: 'text', text: 'Hello!' },
          }),
        })
      );
      expect(result.accepted).toBe(true);
    });
  });

  describe('Event API', () => {
    it('getEvents calls correct endpoint', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () => Promise.resolve({
          events: [],
          next_seq: 0,
          has_more: false,
        }),
      });

      await getEvents('thread-123');

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/v1/threads/thread-123/events'),
        expect.any(Object)
      );
    });

    it('getEvents passes after_seq parameter', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () => Promise.resolve({
          events: [],
          next_seq: 10,
          has_more: false,
        }),
      });

      await getEvents('thread-123', 5);

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('after_seq=5'),
        expect.any(Object)
      );
    });
  });

  describe('LiveKit API', () => {
    it('getLiveKitToken calls correct endpoint', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () => Promise.resolve({
          token: 'jwt-token',
          room_name: 'room-123',
          livekit_url: 'wss://livekit.example.com',
          expires_in: 120,
        }),
      });

      const result = await getLiveKitToken({
        thread_id: 'thread-123',
        session_id: 'session-123',
      });

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/v1/livekit/token'),
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({
            thread_id: 'thread-123',
            session_id: 'session-123',
          }),
        })
      );
      expect(result.token).toBe('jwt-token');
    });
  });

  describe('Action API', () => {
    it('executeAction calls correct endpoint', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () => Promise.resolve({
          ok: true,
          action_id: 'action-123',
          status: 'success',
          message: 'Action completed',
          data: null,
          error: null,
        }),
      });

      const result = await executeAction({
        id: 'action-123',
        type: 'complete',
        thread_id: 'thread-123',
        payload: { task_id: 1 },
      });

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/v1/actions'),
        expect.objectContaining({
          method: 'POST',
          body: JSON.stringify({
            id: 'action-123',
            type: 'complete',
            thread_id: 'thread-123',
            payload: { task_id: 1 },
          }),
        })
      );
      expect(result.ok).toBe(true);
    });

    it('getActionTypes calls correct endpoint', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () => Promise.resolve({
          types: ['archive', 'complete', 'dismiss', 'snooze'],
        }),
      });

      const result = await getActionTypes();

      expect(mockFetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/v1/actions/types'),
        expect.any(Object)
      );
      expect(result.types).toContain('archive');
    });
  });

});

describe('Sync API', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    setAuthTokenGetter(null);
  });

  describe('getSyncStatus', () => {
    it('fetches sync status from the correct endpoint', async () => {
      const mockSyncStatus = {
        status: 'syncing',
        emails_synced: 500,
        total_emails: 1000,
        progress: 0.5,
        error: null,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () => Promise.resolve(mockSyncStatus),
      });

      const result = await getSyncStatus();

      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/connections/gmail/sync/status',
        expect.objectContaining({
          method: 'GET',
          headers: expect.objectContaining({
            'Content-Type': 'application/json',
          }),
        })
      );
      expect(result).toEqual(mockSyncStatus);
    });

    it('returns completed status', async () => {
      const mockSyncStatus = {
        status: 'completed',
        emails_synced: 1000,
        total_emails: 1000,
        progress: 1.0,
        error: null,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () => Promise.resolve(mockSyncStatus),
      });

      const result = await getSyncStatus();

      expect(result.status).toBe('completed');
      expect(result.progress).toBe(1.0);
    });

    it('returns failed status with error', async () => {
      const mockSyncStatus = {
        status: 'failed',
        emails_synced: 200,
        total_emails: null,
        progress: 0.2,
        error: 'Network error',
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () => Promise.resolve(mockSyncStatus),
      });

      const result = await getSyncStatus();

      expect(result.status).toBe('failed');
      expect(result.error).toBe('Network error');
    });
  });

  describe('completeGoogleConnection', () => {
    it('sends server auth code to the correct endpoint', async () => {
      const mockResponse = {
        status: 'connected',
        sync_started: true,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () => Promise.resolve(mockResponse),
      });

      const result = await completeGoogleConnection('server-auth-code-123');

      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/connections/gmail/callback?code=server-auth-code-123&mobile=true',
        expect.objectContaining({
          method: 'POST',
          headers: expect.objectContaining({
            'Content-Type': 'application/json',
          }),
        })
      );
      expect(result).toEqual(mockResponse);
    });

    it('URL encodes the server auth code', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () => Promise.resolve({ status: 'connected', sync_started: true }),
      });

      await completeGoogleConnection('code/with+special&chars');

      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/connections/gmail/callback?code=code%2Fwith%2Bspecial%26chars&mobile=true',
        expect.any(Object)
      );
    });

    it('indicates when sync was started', async () => {
      const mockResponse = {
        status: 'connected',
        sync_started: true,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () => Promise.resolve(mockResponse),
      });

      const result = await completeGoogleConnection('auth-code');

      expect(result.sync_started).toBe(true);
    });

    it('indicates when sync was not started (returning user)', async () => {
      const mockResponse = {
        status: 'connected',
        sync_started: false,
      };

      mockFetch.mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: () => Promise.resolve(mockResponse),
      });

      const result = await completeGoogleConnection('auth-code');

      expect(result.sync_started).toBe(false);
    });
  });
});

describe('API Types', () => {
  it('exports setAuthTokenGetter function', () => {
    expect(typeof setAuthTokenGetter).toBe('function');
  });

  it('exports all API functions', () => {
    expect(typeof getInbox).toBe('function');
    expect(typeof getInboxStats).toBe('function');
    expect(typeof getTasks).toBe('function');
    expect(typeof getTask).toBe('function');
    expect(typeof completeTask).toBe('function');
    expect(typeof dismissTask).toBe('function');
    expect(typeof getProjects).toBe('function');
    expect(typeof getProject).toBe('function');
    expect(typeof createThread).toBe('function');
    expect(typeof getThread).toBe('function');
    expect(typeof listThreads).toBe('function');
    expect(typeof deleteThread).toBe('function');
    expect(typeof createSession).toBe('function');
    expect(typeof getSession).toBe('function');
    expect(typeof listSessions).toBe('function');
    expect(typeof closeSession).toBe('function');
    expect(typeof submitTurn).toBe('function');
    expect(typeof getEvents).toBe('function');
    expect(typeof getLiveKitToken).toBe('function');
    expect(typeof executeAction).toBe('function');
    expect(typeof getActionTypes).toBe('function');
    expect(typeof getSyncStatus).toBe('function');
    expect(typeof completeGoogleConnection).toBe('function');
  });
});
