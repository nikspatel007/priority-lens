/**
 * Tests for API Types
 */

import { APIError } from '../api';

describe('API Types', () => {
  describe('APIError', () => {
    it('creates error with status and message', () => {
      const error = new APIError(404, 'Not found');

      expect(error.status).toBe(404);
      expect(error.message).toBe('Not found');
      expect(error.detail).toBeUndefined();
      expect(error.name).toBe('APIError');
    });

    it('creates error with status, message, and detail', () => {
      const error = new APIError(400, 'Bad request', 'Invalid email format');

      expect(error.status).toBe(400);
      expect(error.message).toBe('Bad request');
      expect(error.detail).toBe('Invalid email format');
    });

    it('is an instance of Error', () => {
      const error = new APIError(500, 'Server error');

      expect(error).toBeInstanceOf(Error);
      expect(error).toBeInstanceOf(APIError);
    });

    it('has correct name property', () => {
      const error = new APIError(401, 'Unauthorized');

      expect(error.name).toBe('APIError');
    });

    it('can be thrown and caught', () => {
      const throwError = () => {
        throw new APIError(403, 'Forbidden', 'Access denied');
      };

      expect(throwError).toThrow(APIError);
      expect(throwError).toThrow('Forbidden');
    });

    it('works with try/catch', () => {
      try {
        throw new APIError(422, 'Validation failed', 'Email is required');
      } catch (error) {
        if (error instanceof APIError) {
          expect(error.status).toBe(422);
          expect(error.message).toBe('Validation failed');
          expect(error.detail).toBe('Email is required');
        }
      }
    });
  });
});

// Type-level tests (these validate at compile time)
describe('Type Definitions', () => {
  it('exports task status type', () => {
    const status: import('../api').TaskStatus = 'pending';
    expect(['pending', 'in_progress', 'completed', 'dismissed']).toContain(status);
  });

  it('exports task type', () => {
    const taskType: import('../api').TaskType = 'review';
    expect(['review', 'send', 'schedule', 'decision', 'research', 'create', 'follow_up', 'other']).toContain(taskType);
  });

  it('exports task complexity type', () => {
    const complexity: import('../api').TaskComplexity = 'medium';
    expect(['trivial', 'quick', 'medium', 'substantial', 'unknown']).toContain(complexity);
  });

  it('exports session mode type', () => {
    const mode: import('../api').SessionMode = 'voice';
    expect(['text', 'voice']).toContain(mode);
  });

  it('exports session status type', () => {
    const status: import('../api').SessionStatus = 'active';
    expect(['active', 'ended']).toContain(status);
  });

  it('exports event actor type', () => {
    const actor: import('../api').EventActor = 'user';
    expect(['system', 'user', 'agent']).toContain(actor);
  });

  it('exports event type', () => {
    const eventType: import('../api').EventType = 'ui.block';
    const validTypes = [
      'turn.user.open',
      'turn.user.close',
      'turn.agent.open',
      'turn.agent.close',
      'ui.text.submit',
      'stt.final',
      'assistant.text.delta',
      'assistant.text.final',
      'tool.call',
      'tool.result',
      'ui.block',
      'ui.clear',
      'action.result',
    ];
    expect(validTypes).toContain(eventType);
  });
});

// Interface tests (validate structure at runtime)
describe('Interface Structures', () => {
  it('EmailSummary has required fields', () => {
    const email: import('../api').EmailSummary = {
      id: 1,
      message_id: 'msg-123',
      thread_id: 'thread-123',
      subject: 'Test Subject',
      from_email: 'test@example.com',
      from_name: 'Test User',
      date_parsed: '2024-01-01T00:00:00Z',
      body_preview: 'Preview text...',
      is_sent: false,
      action: null,
      has_attachments: false,
      labels: ['INBOX'],
    };

    expect(email.id).toBe(1);
    expect(email.message_id).toBe('msg-123');
  });

  it('TaskResponse has required fields', () => {
    const task: import('../api').TaskResponse = {
      id: 1,
      task_id: 'task-123',
      email_id: null,
      project_id: null,
      description: 'Test task',
      task_type: 'review',
      complexity: 'medium',
      deadline: null,
      deadline_text: null,
      urgency_score: 0.5,
      status: 'pending',
      assigned_to: null,
      assigned_by: null,
      is_assigned_to_user: false,
      extraction_method: null,
      completed_at: null,
      created_at: '2024-01-01T00:00:00Z',
      user_id: null,
    };

    expect(task.id).toBe(1);
    expect(task.description).toBe('Test task');
  });

  it('ThreadResponse has required fields', () => {
    const thread: import('../api').ThreadResponse = {
      id: 'thread-123',
      org_id: 'org-123',
      user_id: 'user-123',
      title: 'Test Thread',
      metadata: {},
      created_at: '2024-01-01T00:00:00Z',
      updated_at: '2024-01-01T00:00:00Z',
    };

    expect(thread.id).toBe('thread-123');
    expect(thread.metadata).toEqual({});
  });

  it('SessionResponse has required fields', () => {
    const session: import('../api').SessionResponse = {
      id: 'session-123',
      thread_id: 'thread-123',
      org_id: 'org-123',
      mode: 'text',
      status: 'active',
      livekit_room: null,
      metadata: {},
      started_at: '2024-01-01T00:00:00Z',
      ended_at: null,
    };

    expect(session.id).toBe('session-123');
    expect(session.mode).toBe('text');
  });

  it('TurnInput can be text or voice', () => {
    const textInput: import('../api').TurnInput = {
      type: 'text',
      text: 'Hello!',
    };

    const voiceInput: import('../api').TurnInput = {
      type: 'voice',
      transcript: 'Hello!',
      confidence: 0.95,
      duration_ms: 1500,
    };

    expect(textInput.type).toBe('text');
    expect(voiceInput.type).toBe('voice');
  });

  it('ActionRequest has required fields', () => {
    const action: import('../api').ActionRequest = {
      id: 'action-123',
      type: 'complete',
      thread_id: 'thread-123',
      session_id: 'session-123',
      payload: { task_id: 1 },
    };

    expect(action.id).toBe('action-123');
    expect(action.type).toBe('complete');
  });

  it('ActionResponse has required fields', () => {
    const response: import('../api').ActionResponse = {
      ok: true,
      action_id: 'action-123',
      status: 'success',
      message: 'Action completed successfully',
      data: null,
      error: null,
    };

    expect(response.ok).toBe(true);
    expect(response.status).toBe('success');
  });

  it('LiveKitTokenResponse has required fields', () => {
    const tokenResponse: import('../api').LiveKitTokenResponse = {
      token: 'jwt-token',
      room_name: 'room-123',
      livekit_url: 'wss://livekit.example.com',
      expires_in: 120,
    };

    expect(tokenResponse.token).toBe('jwt-token');
    expect(tokenResponse.room_name).toBe('room-123');
  });

  it('UIBlock has required fields', () => {
    const block: import('../api').UIBlock = {
      id: 'block-123',
      type: 'text',
      props: { value: 'Hello' },
      layout: { padding: 8 },
      children: [],
      actions: [{ trigger: 'press', type: 'test.action' }],
    };

    expect(block.id).toBe('block-123');
    expect(block.type).toBe('text');
    expect(block.props).toEqual({ value: 'Hello' });
  });

  it('UIAction has required fields', () => {
    const action: import('../api').UIAction = {
      trigger: 'press',
      type: 'navigate',
      payload: { screen: 'Settings' },
    };

    expect(action.trigger).toBe('press');
    expect(action.type).toBe('navigate');
  });

  it('LayoutProps has optional fields', () => {
    const layout: import('../api').LayoutProps = {
      padding: 16,
      margin: 8,
      gap: 4,
      flex: 1,
      alignItems: 'center',
      justifyContent: 'space-between',
    };

    expect(layout.padding).toBe(16);
    expect(layout.alignItems).toBe('center');
  });

  it('SyncStatusResponse has required fields', () => {
    const sync: import('../api').SyncStatusResponse = {
      status: 'syncing',
      emails_synced: 500,
      total_emails: 1000,
      progress: 0.5,
      error: null,
    };

    expect(sync.status).toBe('syncing');
    expect(sync.progress).toBe(0.5);
  });
});
