# Phase 6: Proactive AI Assistant - Full Mobile App Experience

## Overview

Phase 6 transforms Priority Lens from a reactive email management tool into a **proactive AI assistant** that:
- Anticipates user needs before they ask
- Surfaces what's important at the right time
- Combines Voice AI + SDUI for the optimal interaction experience
- Learns user preferences and adapts over time

**Goal**: Create an AI assistant that knows what is important to the user, proactively brings it up, and helps take care of it through natural voice conversation and beautiful Server-Driven UI.

## Research & 2026 Design Principles

Based on comprehensive research of 2026 UI/UX trends and AI assistant best practices:

### UI/UX Trends (2026)
- **AI-Powered Personalization**: Apps must adapt, predict intent, guide actions, reduce user effort
- **Minimalist Design**: Clean, clutter-free reduces cognitive load by 20%
- **Glassmorphism**: Frosted glass effects with layered depth
- **Dark Mode**: Expected as standard, reduces eye strain
- **Passwordless Auth**: Biometrics/passkeys as default
- **Voice-First**: Natural dialogue replacing menus and buttons

### Voice AI Best Practices
- **Agentic UX**: AI takes initiative to complete tasks autonomously
- **Clarity over personality**: 72% prefer functional tone over casual
- **Trust & Control**: Transparency in what AI does on user's behalf
- **Context-Awareness**: Understand on-device context, operate apps by voice

### SDUI Patterns
- **Template-Based Rendering**: JSON-based UI contracts
- **Real-Time Personalization**: Server generates UI based on user data
- **Cross-Platform Consistency**: One UI logic, multiple platforms
- **Offline Caching**: Handle poor connectivity gracefully

### Proactive AI Principles
- **Predictive Models**: Learn behavior and preferences
- **Relevance Filter**: Only interrupt when valuable
- **Smart Notifications**: Context-aware, timing-sensitive
- **Autonomous Actions**: Execute with user permission

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Priority Lens Mobile App                              │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────────────────┐   │
│  │  Proactive    │  │   Voice AI    │  │   Adaptive SDUI           │   │
│  │  Engine       │  │   Assistant   │  │   Renderer                │   │
│  │  (Push/Pull)  │  │   (LiveKit)   │  │   (Cards/Lists/Actions)   │   │
│  └───────────────┘  └───────────────┘  └───────────────────────────┘   │
│           │                  │                      │                   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │              Unified Context & State Management                  │   │
│  │  (User Preferences, Learning Engine, Action Queue)              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│           │                  │                      │                   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │              API Layer (REST + WebSocket + LiveKit)             │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         Priority Lens Backend                            │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────────────────┐   │
│  │  ML Priority  │  │  LangGraph    │  │   SDUI Generator          │   │
│  │  Engine       │  │  Agent        │  │   (Dynamic Cards)         │   │
│  └───────────────┘  └───────────────┘  └───────────────────────────┘   │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────────────────┐   │
│  │  Proactive    │  │  User         │  │   Action                  │   │
│  │  Scheduler    │  │  Preferences  │  │   Executor                │   │
│  └───────────────┘  └───────────────┘  └───────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Iteration 1: Smart Digest View

### Goal
Create the "AI Inbox" experience with two key sections: "Suggested To-Dos" and "Topics to Catch Up On" - similar to Gmail's 2026 AI Inbox but tailored for our priority-based approach.

### User Journey
1. User opens app in the morning
2. App shows personalized digest instead of raw email list
3. "Good morning, Sarah. Here's what needs your attention today:"
4. First section shows actionable items with clear next steps
5. Second section groups related emails by topic
6. User can tap to expand or ask voice assistant about any item

### Implementation

#### Backend Changes
```python
# src/priority_lens/api/routes/digest.py
@router.get("/digest")
async def get_smart_digest(
    user: CurrentUser,
    session: SessionDep,
) -> DigestResponse:
    """Generate personalized daily digest."""
    return DigestResponse(
        greeting=generate_greeting(user),
        suggested_todos=await get_actionable_items(user),
        topics_to_catchup=await get_topic_clusters(user),
        last_updated=datetime.utcnow(),
    )
```

#### Mobile Changes
```typescript
// src/screens/DigestScreen.tsx
interface DigestSection {
  id: string;
  title: string;
  items: DigestItem[];
  expandable: boolean;
}

interface DigestItem {
  id: string;
  type: 'todo' | 'topic';
  title: string;
  summary: string;
  urgency: 'high' | 'medium' | 'low';
  actions: UIAction[];
  relatedCount?: number;
}
```

#### SDUI Schema
```json
{
  "type": "digest_view",
  "props": {
    "greeting": "Good morning, Sarah",
    "subtitle": "3 items need attention today"
  },
  "children": [
    {
      "type": "section",
      "props": { "title": "Suggested To-Dos", "icon": "checkbox" },
      "children": [
        {
          "type": "todo_card",
          "props": {
            "title": "Confirm Friday meeting with Alex",
            "source": "Email from Alex Chen",
            "urgency": "high",
            "due": "Today"
          },
          "actions": [
            { "type": "confirm_meeting", "label": "Confirm" },
            { "type": "reschedule", "label": "Reschedule" }
          ]
        }
      ]
    },
    {
      "type": "section",
      "props": { "title": "Topics to Catch Up On", "icon": "folder" },
      "children": [
        {
          "type": "topic_card",
          "props": {
            "title": "Q1 Budget Discussion",
            "email_count": 5,
            "participants": ["Finance Team", "Marketing"],
            "last_activity": "2 hours ago"
          }
        }
      ]
    }
  ]
}
```

### Verification Criteria
- [ ] Digest loads within 2 seconds on app open
- [ ] Greeting is time-appropriate (morning/afternoon/evening)
- [ ] To-dos are ranked by urgency and actionability
- [ ] Topics are clustered by semantic similarity
- [ ] User can expand/collapse sections
- [ ] Voice assistant can read digest aloud

### Test Cases
```typescript
describe('DigestScreen', () => {
  it('displays personalized greeting based on time of day', async () => {
    jest.useFakeTimers().setSystemTime(new Date('2026-01-11T08:00:00'));
    const { getByText } = render(<DigestScreen />);
    await waitFor(() => {
      expect(getByText(/Good morning/)).toBeTruthy();
    });
  });

  it('shows suggested todos section with action buttons', async () => {
    const { getByTestId, getAllByTestId } = render(<DigestScreen />);
    await waitFor(() => {
      expect(getByTestId('todos-section')).toBeTruthy();
      expect(getAllByTestId(/todo-card-/).length).toBeGreaterThan(0);
    });
  });

  it('groups emails by topic in catch-up section', async () => {
    const { getByTestId, getByText } = render(<DigestScreen />);
    await waitFor(() => {
      expect(getByTestId('topics-section')).toBeTruthy();
      expect(getByText(/Q1 Budget/)).toBeTruthy();
    });
  });

  it('voice assistant can narrate digest', async () => {
    const { getByTestId } = render(<DigestScreen />);
    fireEvent.press(getByTestId('voice-read-digest'));
    await waitFor(() => {
      expect(mockLiveKit.sendMessage).toHaveBeenCalledWith(
        expect.objectContaining({ action: 'read_digest' })
      );
    });
  });
});
```

### Success Metrics
- User engagement: 70%+ of users check digest daily
- Time to action: Reduce time to first action by 40%
- Task completion: 50%+ of suggested to-dos completed same day

---

## Iteration 2: Proactive Notification System

### Goal
Implement intelligent push notifications that surface important items at the right time, using relevance filtering to avoid notification fatigue.

### User Journey
1. User receives push notification: "Meeting with John in 30 min - you have 2 prep emails to review"
2. Tapping notification opens relevant context
3. Voice assistant offers to summarize prep materials
4. User can take action directly from notification (iOS Live Activities / Android Widgets)

### Implementation

#### Backend: Proactive Scheduler
```python
# src/priority_lens/services/proactive_scheduler.py
class ProactiveScheduler:
    """Schedules proactive notifications based on user context."""

    async def evaluate_notifications(
        self,
        user_id: UUID,
        context: UserContext,
    ) -> list[ProactiveNotification]:
        """Evaluate what notifications should be sent now."""
        candidates = await self._get_candidates(user_id)

        # Apply relevance filter
        relevant = [
            n for n in candidates
            if self._passes_relevance_threshold(n, context)
        ]

        # Apply importance threshold
        important = [
            n for n in relevant
            if n.importance_score >= context.interrupt_threshold
        ]

        # Respect quiet hours and user preferences
        return self._filter_by_preferences(important, context)

    def _passes_relevance_threshold(
        self,
        notification: ProactiveNotification,
        context: UserContext,
    ) -> bool:
        """Check if notification is relevant to current context."""
        # Upcoming meetings: relevant 30 min before
        if notification.type == 'meeting_prep':
            return notification.meeting_time - timedelta(minutes=30) <= context.now

        # Urgent emails: always relevant during work hours
        if notification.type == 'urgent_email':
            return context.is_work_hours

        # Digest: relevant at user's preferred time
        if notification.type == 'daily_digest':
            return context.now.hour == context.preferred_digest_hour

        return True
```

#### Mobile: Push Notification Handler
```typescript
// src/services/notifications.ts
interface ProactiveNotification {
  id: string;
  type: 'meeting_prep' | 'urgent_email' | 'deadline' | 'follow_up';
  title: string;
  body: string;
  data: {
    thread_id?: string;
    email_ids?: string[];
    meeting_id?: string;
    actions: NotificationAction[];
  };
  priority: 'high' | 'default' | 'low';
  scheduled_for?: Date;
}

export async function handleNotificationReceived(
  notification: ProactiveNotification
): Promise<void> {
  // Track for learning
  await analytics.track('notification_received', {
    type: notification.type,
    priority: notification.priority,
  });

  // Show with appropriate priority
  if (notification.priority === 'high') {
    await showHighPriorityNotification(notification);
  } else {
    await showStandardNotification(notification);
  }
}
```

### Verification Criteria
- [ ] Notifications arrive at contextually appropriate times
- [ ] High-priority items use prominent notification style
- [ ] User can configure quiet hours
- [ ] Notification actions work without opening app
- [ ] Tapping notification opens correct context
- [ ] Voice assistant can be invoked from notification

### Test Cases
```typescript
describe('ProactiveNotifications', () => {
  it('sends meeting prep notification 30 min before', async () => {
    const meeting = { time: new Date('2026-01-11T14:00:00') };
    jest.useFakeTimers().setSystemTime(new Date('2026-01-11T13:30:00'));

    const notifications = await scheduler.evaluate(userId, context);
    expect(notifications).toContainEqual(
      expect.objectContaining({
        type: 'meeting_prep',
        title: expect.stringContaining('Meeting'),
      })
    );
  });

  it('respects quiet hours preference', async () => {
    const context = { ...defaultContext, is_quiet_hours: true };
    const notifications = await scheduler.evaluate(userId, context);
    expect(notifications.filter(n => n.priority !== 'high')).toHaveLength(0);
  });

  it('notification action opens correct screen', async () => {
    const notification = {
      type: 'urgent_email',
      data: { email_ids: ['email-123'] }
    };
    await handleNotificationTap(notification);
    expect(navigation.navigate).toHaveBeenCalledWith('EmailDetail', {
      emailId: 'email-123'
    });
  });
});
```

### Success Metrics
- Notification open rate: 40%+ (vs industry avg 5-10%)
- Action completion from notification: 30%+
- User-reported satisfaction with timing: 4.5/5
- Notification opt-out rate: <10%

---

## Iteration 3: Adaptive Email Cards

### Goal
Create a flexible SDUI card system that adapts based on email type, urgency, user preferences, and context. Cards should feel like they were designed specifically for each email.

### User Journey
1. User views inbox
2. Urgent email from boss shows prominent red card with quick actions
3. Newsletter shows compact card with "read later" option
4. Meeting request shows calendar-integrated card with availability
5. Thread with many replies shows conversation summary card

### Implementation

#### SDUI Card Types
```typescript
// src/sdui/types.ts
type EmailCardVariant =
  | 'urgent_action'      // Red accent, prominent actions
  | 'meeting_request'    // Calendar integration
  | 'conversation'       // Thread summary
  | 'newsletter'         // Compact, read-later
  | 'transactional'      // Receipt, confirmation
  | 'personal'           // From contacts
  | 'default';           // Standard email

interface AdaptiveEmailCard {
  id: string;
  type: 'email_card';
  variant: EmailCardVariant;
  props: {
    from: ContactInfo;
    subject: string;
    preview: string;
    timestamp: string;
    priority_score: number;
    labels: string[];
    attachments?: AttachmentInfo[];
    thread_count?: number;
    meeting_info?: MeetingInfo;
  };
  layout: {
    size: 'compact' | 'standard' | 'expanded';
    accent_color?: string;
    show_avatar: boolean;
    show_preview: boolean;
  };
  actions: UIAction[];
}
```

#### Backend: Smart Card Generator
```python
# src/priority_lens/sdui/card_generator.py
class EmailCardGenerator:
    """Generates adaptive email cards based on content analysis."""

    def generate_card(
        self,
        email: Email,
        user_prefs: UserPreferences,
        context: ViewContext,
    ) -> UIBlock:
        variant = self._determine_variant(email)
        layout = self._get_layout(variant, user_prefs, context)
        actions = self._get_actions(variant, email)

        return UIBlock(
            type='email_card',
            id=f'email-{email.id}',
            props={
                'from': self._format_sender(email),
                'subject': email.subject,
                'preview': self._get_smart_preview(email, variant),
                'timestamp': self._format_time(email.received_at),
                'priority_score': email.priority_score,
                'labels': email.labels,
                'variant': variant,
            },
            layout=layout,
            actions=actions,
        )

    def _determine_variant(self, email: Email) -> str:
        if email.priority_score > 0.9 and email.requires_action:
            return 'urgent_action'
        if email.has_calendar_event:
            return 'meeting_request'
        if email.thread_count > 3:
            return 'conversation'
        if email.is_newsletter:
            return 'newsletter'
        if email.is_transactional:
            return 'transactional'
        if email.from_contact:
            return 'personal'
        return 'default'
```

### Verification Criteria
- [ ] Card variant matches email type accurately (>95%)
- [ ] Urgent emails are visually distinct
- [ ] Meeting cards show calendar availability
- [ ] Conversation cards show participant avatars
- [ ] Newsletter cards are compact
- [ ] User can customize card preferences

### Test Cases
```typescript
describe('AdaptiveEmailCards', () => {
  it('renders urgent_action variant for high priority actionable emails', () => {
    const email = { priority_score: 0.95, requires_action: true };
    const card = generateCard(email);
    expect(card.variant).toBe('urgent_action');
    expect(card.layout.accent_color).toBe('#EF4444'); // Red
  });

  it('renders meeting_request variant with calendar data', () => {
    const email = { has_calendar_event: true, meeting_info: mockMeeting };
    const card = generateCard(email);
    expect(card.variant).toBe('meeting_request');
    expect(card.props.meeting_info).toBeDefined();
  });

  it('renders conversation variant for long threads', () => {
    const email = { thread_count: 5, participants: ['A', 'B', 'C'] };
    const card = generateCard(email);
    expect(card.variant).toBe('conversation');
    expect(card.props.thread_count).toBe(5);
  });

  it('respects user preference for compact cards', () => {
    const prefs = { default_card_size: 'compact' };
    const card = generateCard(email, prefs);
    expect(card.layout.size).toBe('compact');
  });
});
```

### Success Metrics
- Card type accuracy: 95%+
- User finds relevant action: <2 taps
- Scan time per email: Reduced 30%
- User customization rate: 40%+

---

## Iteration 4: Voice Conversation Improvements

### Goal
Enhance voice interaction to feel like a natural conversation with a capable assistant. Implement proper turn-taking, context retention, and multi-turn task completion.

### User Journey
1. User: "What's my most urgent email today?"
2. Assistant: "You have an email from Sarah about the Q1 report deadline. It's due tomorrow and she's asking for your approval. Would you like me to read it or help you respond?"
3. User: "Read the key points"
4. Assistant: "Sarah says... [summary]. She needs your sign-off before EOD. Should I draft an approval response?"
5. User: "Yes, approve it and thank her"
6. Assistant: "I've drafted: 'Hi Sarah, Approved! Thanks for the thorough work.' Would you like me to send it or would you like to edit first?"

### Implementation

#### Backend: Conversation Context
```python
# src/priority_lens/agent/context.py
@dataclass
class ConversationContext:
    """Maintains context across conversation turns."""

    thread_id: UUID
    session_id: UUID
    user_id: UUID

    # Current focus
    active_email: Email | None = None
    active_task: Task | None = None
    active_topic: str | None = None

    # Conversation state
    pending_action: PendingAction | None = None
    confirmation_needed: bool = False

    # Memory
    mentioned_emails: list[UUID] = field(default_factory=list)
    mentioned_people: list[str] = field(default_factory=list)
    recent_actions: list[ActionLog] = field(default_factory=list)

    def update_focus(self, email: Email = None, task: Task = None):
        """Update what we're currently discussing."""
        if email:
            self.active_email = email
            self.mentioned_emails.append(email.id)
        if task:
            self.active_task = task

    def get_context_prompt(self) -> str:
        """Generate context summary for LLM."""
        parts = []
        if self.active_email:
            parts.append(f"Currently discussing email: {self.active_email.subject}")
        if self.pending_action:
            parts.append(f"Awaiting confirmation for: {self.pending_action.description}")
        return "\n".join(parts)
```

#### Agent: Multi-turn Handler
```python
# src/priority_lens/agent/handlers.py
class ConversationHandler:
    """Handles multi-turn conversations with context."""

    async def handle_turn(
        self,
        user_input: str,
        context: ConversationContext,
    ) -> AgentResponse:
        # Check for confirmation response
        if context.confirmation_needed:
            return await self._handle_confirmation(user_input, context)

        # Parse intent with context
        intent = await self._parse_intent(user_input, context)

        # Handle based on intent
        if intent.type == 'query':
            return await self._handle_query(intent, context)
        elif intent.type == 'action':
            return await self._handle_action(intent, context)
        elif intent.type == 'follow_up':
            return await self._handle_follow_up(intent, context)

    async def _handle_action(
        self,
        intent: Intent,
        context: ConversationContext,
    ) -> AgentResponse:
        # Prepare action
        action = await self._prepare_action(intent, context)

        # Set pending and ask for confirmation
        context.pending_action = action
        context.confirmation_needed = True

        return AgentResponse(
            text=f"I'll {action.description}. {action.confirmation_prompt}",
            requires_response=True,
            sdui_blocks=action.preview_blocks,
        )
```

### Verification Criteria
- [ ] Context persists across turns within session
- [ ] Pronouns resolve correctly ("read it", "send that")
- [ ] Follow-up questions work without repeating context
- [ ] Confirmations are clear and actionable
- [ ] User can cancel pending actions
- [ ] Conversation feels natural (user testing score >4/5)

### Test Cases
```typescript
describe('VoiceConversation', () => {
  it('maintains email context across turns', async () => {
    await agent.process("What's my urgent email from Sarah?");
    const response = await agent.process("Read it");

    expect(response.context.active_email.from).toContain('Sarah');
    expect(response.text).toContain('Sarah says');
  });

  it('asks for confirmation before actions', async () => {
    const response = await agent.process("Send a reply saying I approve");

    expect(response.requires_response).toBe(true);
    expect(response.context.confirmation_needed).toBe(true);
    expect(response.text).toContain('Would you like me to send');
  });

  it('handles confirmation affirmative', async () => {
    await agent.process("Draft a reply to Sarah");
    const response = await agent.process("Yes, send it");

    expect(response.action_taken).toBe('email_sent');
    expect(response.context.confirmation_needed).toBe(false);
  });

  it('handles confirmation negative', async () => {
    await agent.process("Delete this email");
    const response = await agent.process("No, wait");

    expect(response.action_taken).toBeNull();
    expect(response.text).toContain('Okay, I won\'t');
  });
});
```

### Success Metrics
- Multi-turn completion rate: 80%+
- Context retention accuracy: 95%+
- Confirmation success rate: 90%+
- User satisfaction with natural feel: 4.5/5

---

## Iteration 5: Voice Action Execution

### Goal
Enable the voice assistant to execute real actions on behalf of the user - archive emails, schedule meetings, create tasks, send replies, and more.

### User Journey
1. User: "Archive all newsletters from this week"
2. Assistant: "I found 12 newsletters from this week. Should I archive all of them?"
3. User: "Yes"
4. Assistant: [Shows SDUI with archiving progress] "Done! I've archived 12 newsletters. I can also unsubscribe you from any of these if you'd like."

### Implementation

#### Backend: Action Executor
```python
# src/priority_lens/services/action_executor.py
class ActionExecutor:
    """Executes actions on behalf of users."""

    SUPPORTED_ACTIONS = {
        'archive': ArchiveAction,
        'delete': DeleteAction,
        'reply': ReplyAction,
        'forward': ForwardAction,
        'schedule_meeting': ScheduleMeetingAction,
        'create_task': CreateTaskAction,
        'snooze': SnoozeAction,
        'label': LabelAction,
        'unsubscribe': UnsubscribeAction,
    }

    async def execute(
        self,
        action_type: str,
        params: ActionParams,
        user_id: UUID,
        session_id: UUID,
    ) -> ActionResult:
        action_class = self.SUPPORTED_ACTIONS.get(action_type)
        if not action_class:
            raise UnsupportedActionError(action_type)

        action = action_class(params)

        # Validate permissions
        await self._validate_permissions(action, user_id)

        # Execute with audit logging
        async with self._audit_context(action, user_id, session_id):
            result = await action.execute()

        # Generate confirmation SDUI
        confirmation_blocks = self._generate_confirmation(action, result)

        return ActionResult(
            success=True,
            action_type=action_type,
            affected_count=result.affected_count,
            sdui_blocks=confirmation_blocks,
            undo_available=action.supports_undo,
        )
```

#### Mobile: Action Progress UI
```typescript
// src/sdui/components/ActionProgress.tsx
interface ActionProgressProps {
  action: string;
  total: number;
  completed: number;
  status: 'pending' | 'in_progress' | 'completed' | 'failed';
  undoAvailable: boolean;
}

export function ActionProgress({
  action,
  total,
  completed,
  status,
  undoAvailable,
}: ActionProgressProps): React.JSX.Element {
  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <ActionIcon action={action} />
        <Text style={styles.title}>{getActionTitle(action)}</Text>
      </View>

      <ProgressBar progress={completed / total} />

      <Text style={styles.status}>
        {completed} of {total} {action}
      </Text>

      {status === 'completed' && undoAvailable && (
        <Pressable style={styles.undoButton} onPress={onUndo}>
          <Text>Undo</Text>
        </Pressable>
      )}
    </View>
  );
}
```

### Verification Criteria
- [ ] All supported actions execute correctly
- [ ] Bulk actions show progress
- [ ] Failed actions provide clear error messages
- [ ] Undo is available for destructive actions
- [ ] Actions are audit-logged
- [ ] Rate limiting prevents abuse

### Test Cases
```typescript
describe('VoiceActionExecution', () => {
  it('archives multiple emails with confirmation', async () => {
    const response = await agent.process("Archive all newsletters");
    expect(response.pending_action.type).toBe('archive');
    expect(response.pending_action.count).toBe(12);

    const result = await agent.process("Yes");
    expect(result.action_result.success).toBe(true);
    expect(result.action_result.affected_count).toBe(12);
  });

  it('shows progress for bulk actions', async () => {
    await agent.process("Delete all spam");
    await agent.process("Confirm");

    const blocks = await getSDUIBlocks();
    const progressBlock = blocks.find(b => b.type === 'action_progress');
    expect(progressBlock.props.status).toBe('completed');
  });

  it('provides undo for destructive actions', async () => {
    await agent.process("Archive this email");
    await agent.process("Yes");

    const result = await agent.process("Undo that");
    expect(result.action_result.type).toBe('unarchive');
    expect(result.text).toContain('restored');
  });

  it('handles action failures gracefully', async () => {
    mockGmailApi.archive.mockRejectedValue(new Error('Rate limited'));

    const result = await agent.process("Archive all emails");
    await agent.process("Yes");

    expect(result.action_result.success).toBe(false);
    expect(result.text).toContain('couldn\'t archive');
  });
});
```

### Success Metrics
- Action success rate: 99%+
- Undo usage rate: <5% (indicates good confirmation UX)
- User trust score: 4.5/5
- Actions per session: 3+ for power users

---

## Iteration 6: Voice Feedback & Haptics

### Goal
Provide rich audio and haptic feedback for voice interactions, making the experience feel responsive and polished. Include earcon sounds, spoken confirmations, and tactile feedback.

### User Journey
1. User activates voice with long-press
2. [Haptic: Medium impact] + [Sound: Soft chime]
3. User speaks command
4. [Sound: Processing tone] while AI thinks
5. AI responds with speech
6. [Haptic: Success/Error pattern] + [Sound: Completion tone]
7. SDUI updates on screen

### Implementation

#### Mobile: Haptic Feedback System
```typescript
// src/services/haptics.ts
import * as Haptics from 'expo-haptics';

export const HapticPatterns = {
  voiceActivated: async () => {
    await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Medium);
  },

  voiceDeactivated: async () => {
    await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
  },

  actionSuccess: async () => {
    await Haptics.notificationAsync(Haptics.NotificationFeedbackType.Success);
  },

  actionError: async () => {
    await Haptics.notificationAsync(Haptics.NotificationFeedbackType.Error);
  },

  thinking: async () => {
    // Subtle pulse while processing
    for (let i = 0; i < 3; i++) {
      await Haptics.impactAsync(Haptics.ImpactFeedbackStyle.Light);
      await sleep(200);
    }
  },

  agentSpeaking: async () => {
    // Very subtle feedback when agent starts speaking
    await Haptics.selectionAsync();
  },
};
```

#### Mobile: Audio Feedback System
```typescript
// src/services/audio.ts
import { Audio } from 'expo-av';

export class AudioFeedback {
  private sounds: Map<string, Audio.Sound> = new Map();

  async initialize(): Promise<void> {
    const soundFiles = {
      voiceStart: require('@/assets/sounds/voice-start.wav'),
      voiceEnd: require('@/assets/sounds/voice-end.wav'),
      thinking: require('@/assets/sounds/thinking.wav'),
      success: require('@/assets/sounds/success.wav'),
      error: require('@/assets/sounds/error.wav'),
    };

    for (const [name, file] of Object.entries(soundFiles)) {
      const { sound } = await Audio.Sound.createAsync(file);
      this.sounds.set(name, sound);
    }
  }

  async play(name: string): Promise<void> {
    const sound = this.sounds.get(name);
    if (sound) {
      await sound.replayAsync();
    }
  }
}
```

### Verification Criteria
- [ ] Haptics fire at correct moments
- [ ] Audio cues are distinct and not annoying
- [ ] Feedback respects system settings (silent mode)
- [ ] Timing is synchronized with visual feedback
- [ ] Users can disable haptics/sounds independently
- [ ] Accessibility: VoiceOver users get appropriate feedback

### Test Cases
```typescript
describe('VoiceFeedback', () => {
  it('triggers haptic on voice activation', async () => {
    await activateVoice();
    expect(Haptics.impactAsync).toHaveBeenCalledWith(
      Haptics.ImpactFeedbackStyle.Medium
    );
  });

  it('plays thinking sound while processing', async () => {
    await activateVoice();
    await speakCommand("What's my schedule?");

    expect(audioFeedback.play).toHaveBeenCalledWith('thinking');
  });

  it('triggers success haptic on action completion', async () => {
    await completeAction('archive');
    expect(Haptics.notificationAsync).toHaveBeenCalledWith(
      Haptics.NotificationFeedbackType.Success
    );
  });

  it('respects silent mode setting', async () => {
    mockSettings.silentMode = true;
    await activateVoice();

    expect(audioFeedback.play).not.toHaveBeenCalled();
    expect(Haptics.impactAsync).toHaveBeenCalled(); // Haptics still work
  });
});
```

### Success Metrics
- User perception of responsiveness: 4.5/5
- Audio feedback helpfulness: 4/5
- Haptic feedback satisfaction: 4.5/5
- Accessibility compliance: 100%

---

## Iteration 7: Glassmorphism Visual Design

### Goal
Implement modern 2026 visual design with glassmorphism, layered depth, and refined animations. Create a premium, polished visual experience.

### Design System

#### Color Palette
```typescript
// src/theme/colors.ts
export const colors = {
  // Primary brand colors
  primary: {
    50: '#F0F9FF',
    100: '#E0F2FE',
    500: '#0EA5E9',
    600: '#0284C7',
    900: '#0C4A6E',
  },

  // Glass effects
  glass: {
    light: 'rgba(255, 255, 255, 0.7)',
    dark: 'rgba(0, 0, 0, 0.3)',
    border: 'rgba(255, 255, 255, 0.2)',
    blur: 20,
  },

  // Semantic colors
  urgency: {
    high: '#EF4444',
    medium: '#F59E0B',
    low: '#10B981',
  },

  // Dark mode
  dark: {
    background: '#0F172A',
    surface: '#1E293B',
    elevated: '#334155',
  },
};
```

#### Glass Card Component
```typescript
// src/components/ui/GlassCard.tsx
import { BlurView } from 'expo-blur';

interface GlassCardProps {
  children: React.ReactNode;
  intensity?: number;
  tint?: 'light' | 'dark';
  style?: ViewStyle;
}

export function GlassCard({
  children,
  intensity = 50,
  tint = 'light',
  style,
}: GlassCardProps): React.JSX.Element {
  return (
    <BlurView
      intensity={intensity}
      tint={tint}
      style={[styles.card, style]}
    >
      <View style={styles.inner}>
        {children}
      </View>
    </BlurView>
  );
}

const styles = StyleSheet.create({
  card: {
    borderRadius: 20,
    overflow: 'hidden',
    borderWidth: 1,
    borderColor: 'rgba(255, 255, 255, 0.2)',
  },
  inner: {
    padding: spacing[4],
  },
});
```

#### Animations
```typescript
// src/animations/transitions.ts
import Animated, {
  withSpring,
  withTiming,
  interpolate,
  Easing,
} from 'react-native-reanimated';

export const springConfig = {
  damping: 15,
  stiffness: 150,
  mass: 1,
};

export const cardEnterAnimation = (index: number) => {
  'worklet';
  return {
    initialValues: {
      opacity: 0,
      transform: [{ translateY: 20 }],
    },
    animations: {
      opacity: withTiming(1, { duration: 300 }),
      transform: [
        {
          translateY: withSpring(0, {
            ...springConfig,
            delay: index * 50,
          }),
        },
      ],
    },
  };
};
```

### Verification Criteria
- [ ] Glassmorphism renders correctly on iOS and Android
- [ ] Dark mode is fully implemented
- [ ] Animations are 60fps on mid-range devices
- [ ] Color contrast meets WCAG AA standards
- [ ] Visual hierarchy is clear
- [ ] Design feels premium and modern

### Test Cases
```typescript
describe('VisualDesign', () => {
  it('applies glass effect with correct blur', () => {
    const { getByTestId } = render(<GlassCard testID="glass-card" />);
    const card = getByTestId('glass-card');
    expect(card.props.intensity).toBe(50);
  });

  it('switches to dark mode colors', () => {
    mockColorScheme.mockReturnValue('dark');
    const { getByTestId } = render(<ThemedView testID="themed" />);
    expect(getByTestId('themed').props.style.backgroundColor)
      .toBe(colors.dark.background);
  });

  it('animations complete within performance budget', async () => {
    const start = performance.now();
    await animateCardEntry();
    const duration = performance.now() - start;
    expect(duration).toBeLessThan(500);
  });
});
```

### Success Metrics
- Visual appeal rating: 4.5/5
- Animation smoothness: 60fps on 90%+ of devices
- Dark mode preference: 60% of users
- Brand recognition: Strong

---

## Iteration 8: Time-Based Contextual UI

### Goal
Adapt the UI based on time of day, user's schedule, and current context. Morning shows digest, afternoon shows meeting prep, evening shows wrap-up.

### User Journey

#### Morning (6 AM - 10 AM)
- Greeting: "Good morning, Sarah"
- Focus: Daily digest, overnight emails
- Voice: Calm, informative tone
- UI: Bright, energizing colors

#### Work Hours (10 AM - 6 PM)
- Focus: Active tasks, urgent items
- Meeting prep appears 30 min before
- Quick actions prominent
- UI: Productive, efficient layout

#### Evening (6 PM - 10 PM)
- Greeting: "Winding down, Sarah"
- Focus: End-of-day summary, tomorrow prep
- Suggest deferring non-urgent items
- UI: Relaxed, warmer tones

### Implementation

```typescript
// src/context/TimeContext.tsx
interface TimeContext {
  timeOfDay: 'morning' | 'work' | 'evening' | 'night';
  isWorkHours: boolean;
  nextMeeting: Meeting | null;
  minutesToNextMeeting: number | null;
  suggestedFocus: 'digest' | 'active' | 'prep' | 'wrapup';
}

export function useTimeContext(): TimeContext {
  const [context, setContext] = useState<TimeContext>(getTimeContext());

  useEffect(() => {
    const interval = setInterval(() => {
      setContext(getTimeContext());
    }, 60000); // Update every minute

    return () => clearInterval(interval);
  }, []);

  return context;
}

function getTimeContext(): TimeContext {
  const hour = new Date().getHours();
  const nextMeeting = getNextMeeting();
  const minutesToMeeting = nextMeeting
    ? differenceInMinutes(nextMeeting.start, new Date())
    : null;

  let timeOfDay: TimeContext['timeOfDay'];
  let suggestedFocus: TimeContext['suggestedFocus'];

  if (hour >= 6 && hour < 10) {
    timeOfDay = 'morning';
    suggestedFocus = 'digest';
  } else if (hour >= 10 && hour < 18) {
    timeOfDay = 'work';
    suggestedFocus = minutesToMeeting && minutesToMeeting < 30 ? 'prep' : 'active';
  } else if (hour >= 18 && hour < 22) {
    timeOfDay = 'evening';
    suggestedFocus = 'wrapup';
  } else {
    timeOfDay = 'night';
    suggestedFocus = 'digest'; // Next day prep
  }

  return {
    timeOfDay,
    isWorkHours: hour >= 9 && hour < 18,
    nextMeeting,
    minutesToNextMeeting: minutesToMeeting,
    suggestedFocus,
  };
}
```

### Verification Criteria
- [ ] UI adapts based on time of day
- [ ] Meeting prep appears at right time
- [ ] Evening mode suggests deferring items
- [ ] Color temperature shifts appropriately
- [ ] Voice tone matches time context
- [ ] User can override time-based defaults

### Test Cases
```typescript
describe('TimeBasedUI', () => {
  it('shows digest in morning', () => {
    jest.useFakeTimers().setSystemTime(new Date('2026-01-11T08:00:00'));
    const { getByTestId } = render(<HomeScreen />);
    expect(getByTestId('digest-section')).toBeTruthy();
    expect(getByTestId('greeting')).toHaveTextContent('Good morning');
  });

  it('shows meeting prep 30 min before meeting', () => {
    jest.useFakeTimers().setSystemTime(new Date('2026-01-11T13:30:00'));
    mockMeetings = [{ start: new Date('2026-01-11T14:00:00') }];

    const { getByTestId } = render(<HomeScreen />);
    expect(getByTestId('meeting-prep')).toBeTruthy();
  });

  it('suggests deferring in evening', () => {
    jest.useFakeTimers().setSystemTime(new Date('2026-01-11T20:00:00'));
    const { getByText } = render(<EmailCard email={nonUrgentEmail} />);
    expect(getByText('Defer to tomorrow')).toBeTruthy();
  });
});
```

### Success Metrics
- Contextual relevance rating: 4.5/5
- Meeting prep usage: 70%+ of meetings
- Evening deferral acceptance: 40%+
- Time-based engagement: +20% retention

---

## Iteration 9: User Preference Learning

### Goal
Build a learning system that observes user behavior and adapts the experience over time. Learn email handling patterns, priority preferences, and action tendencies.

### Learning Signals
- Which emails user opens first
- How quickly user responds to different senders
- Which notifications user engages with
- What actions user takes on email types
- Time patterns for email checking
- Voice command patterns

### Implementation

#### Backend: Preference Learner
```python
# src/priority_lens/ml/preference_learner.py
class PreferenceLearner:
    """Learns user preferences from behavior."""

    async def record_signal(
        self,
        user_id: UUID,
        signal: BehaviorSignal,
    ) -> None:
        """Record a user behavior signal."""
        await self.signal_store.append(user_id, signal)

        # Update real-time preferences if significant
        if signal.is_significant:
            await self._update_preferences(user_id, signal)

    async def get_preferences(
        self,
        user_id: UUID,
    ) -> UserPreferences:
        """Get learned preferences for user."""
        signals = await self.signal_store.get_recent(user_id, days=30)

        return UserPreferences(
            vip_senders=self._extract_vips(signals),
            email_patterns=self._analyze_patterns(signals),
            preferred_action_times=self._get_action_times(signals),
            notification_preferences=self._get_notification_prefs(signals),
            voice_patterns=self._get_voice_patterns(signals),
        )

    def _extract_vips(
        self,
        signals: list[BehaviorSignal],
    ) -> list[VIPSender]:
        """Identify VIP senders from behavior."""
        sender_stats = defaultdict(SenderStats)

        for signal in signals:
            if signal.type == 'email_opened':
                sender = signal.data['from']
                sender_stats[sender].open_count += 1
                sender_stats[sender].avg_open_time = (
                    self._update_avg(
                        sender_stats[sender].avg_open_time,
                        signal.data['time_to_open'],
                    )
                )

        # Rank by engagement
        ranked = sorted(
            sender_stats.items(),
            key=lambda x: (x[1].open_count, -x[1].avg_open_time),
            reverse=True,
        )

        return [
            VIPSender(email=sender, score=stats.engagement_score)
            for sender, stats in ranked[:20]
        ]
```

### Verification Criteria
- [ ] VIP senders identified within 1 week of use
- [ ] Priority predictions improve over time
- [ ] Action suggestions match user patterns
- [ ] Preferences are explainable to user
- [ ] User can override learned preferences
- [ ] Privacy: Learning is on-device when possible

### Test Cases
```typescript
describe('PreferenceLearning', () => {
  it('identifies VIP senders from open patterns', async () => {
    // Simulate user always opening boss's emails quickly
    for (let i = 0; i < 10; i++) {
      await recordSignal({
        type: 'email_opened',
        from: 'boss@company.com',
        time_to_open: 60, // 1 minute
      });
    }

    const prefs = await getPreferences(userId);
    expect(prefs.vip_senders[0].email).toBe('boss@company.com');
  });

  it('learns preferred notification times', async () => {
    // User always opens digest at 8 AM
    for (let i = 0; i < 7; i++) {
      await recordSignal({
        type: 'notification_opened',
        notification_type: 'digest',
        opened_at: new Date(`2026-01-0${i+1}T08:15:00`),
      });
    }

    const prefs = await getPreferences(userId);
    expect(prefs.preferred_digest_time).toBe('08:00');
  });

  it('allows user to override learned preferences', async () => {
    await setUserOverride(userId, {
      vip_senders: ['important@example.com'],
    });

    const prefs = await getPreferences(userId);
    expect(prefs.vip_senders[0].email).toBe('important@example.com');
    expect(prefs.vip_senders[0].is_manual).toBe(true);
  });
});
```

### Success Metrics
- Preference accuracy: 85%+ after 2 weeks
- User override rate: <10% (indicates good learning)
- Priority prediction improvement: +30% over baseline
- User trust in AI: 4.5/5

---

## Iteration 10: Predictive Actions

### Goal
Anticipate what the user needs and offer proactive suggestions. "You usually reply to Sarah within an hour - would you like me to draft a response?"

### User Journey
1. User receives email from boss
2. AI notices: Boss emails are always high priority, user replies within 30 min
3. AI proactively: "Email from John (your boss) about Q1 targets. Based on past patterns, you usually respond quickly. Want me to help draft a reply?"
4. User: "Yes, remind me of what we discussed last time"
5. AI: "In your last email thread about targets, you mentioned... Here's a draft reply incorporating that context."

### Implementation

#### Backend: Action Predictor
```python
# src/priority_lens/ml/action_predictor.py
class ActionPredictor:
    """Predicts likely user actions based on patterns."""

    async def predict_actions(
        self,
        email: Email,
        user_prefs: UserPreferences,
        context: UserContext,
    ) -> list[PredictedAction]:
        """Predict what actions user is likely to take."""
        predictions = []

        # Check sender patterns
        sender_pattern = user_prefs.get_sender_pattern(email.from_email)
        if sender_pattern:
            if sender_pattern.avg_reply_time < timedelta(hours=1):
                predictions.append(PredictedAction(
                    action='reply',
                    confidence=0.85,
                    reason=f"You usually reply to {email.from_name} within an hour",
                    suggested_timing=sender_pattern.avg_reply_time,
                ))

        # Check email type patterns
        if email.is_meeting_request:
            predictions.append(PredictedAction(
                action='check_calendar',
                confidence=0.9,
                reason="Meeting request - checking your availability",
            ))

        # Check time-based patterns
        if context.time_of_day == 'morning' and email.is_newsletter:
            predictions.append(PredictedAction(
                action='read_later',
                confidence=0.7,
                reason="You usually save newsletters for later",
            ))

        return sorted(predictions, key=lambda p: -p.confidence)
```

#### Agent: Proactive Suggestions
```python
# src/priority_lens/agent/proactive.py
class ProactiveSuggestionAgent:
    """Generates proactive suggestions for users."""

    async def generate_suggestion(
        self,
        email: Email,
        predictions: list[PredictedAction],
        user_prefs: UserPreferences,
    ) -> ProactiveSuggestion | None:
        """Generate a proactive suggestion if appropriate."""
        if not predictions:
            return None

        top_prediction = predictions[0]
        if top_prediction.confidence < 0.7:
            return None

        # Generate natural language suggestion
        if top_prediction.action == 'reply':
            # Get context from past conversations
            past_context = await self._get_conversation_context(
                email.from_email,
                user_prefs.user_id,
            )

            return ProactiveSuggestion(
                type='reply_draft',
                message=self._generate_suggestion_text(top_prediction, past_context),
                actions=[
                    SuggestedAction('draft_reply', 'Help me draft a reply'),
                    SuggestedAction('remind_later', 'Remind me in 30 min'),
                    SuggestedAction('dismiss', 'I\'ll handle it'),
                ],
                context=past_context,
            )

        return None
```

### Verification Criteria
- [ ] Predictions are accurate (>70% confidence threshold)
- [ ] Suggestions feel helpful, not intrusive
- [ ] Context from past conversations is relevant
- [ ] User can dismiss suggestions easily
- [ ] Learning improves prediction accuracy
- [ ] Suggestions respect quiet hours

### Test Cases
```typescript
describe('PredictiveActions', () => {
  it('suggests quick reply for VIP sender', async () => {
    const email = { from: 'boss@company.com', subject: 'Q1 Targets' };
    mockPreferences.vip_senders = [{ email: 'boss@company.com', avg_reply_time: 30 }];

    const suggestion = await getSuggestion(email);
    expect(suggestion.type).toBe('reply_draft');
    expect(suggestion.message).toContain('usually respond quickly');
  });

  it('includes past conversation context', async () => {
    const email = { from: 'client@example.com' };
    mockPastEmails = [
      { subject: 'Project Update', snippet: 'We discussed the timeline...' }
    ];

    const suggestion = await getSuggestion(email);
    expect(suggestion.context).toContain('timeline');
  });

  it('respects quiet hours for non-urgent', async () => {
    jest.useFakeTimers().setSystemTime(new Date('2026-01-11T22:00:00'));
    const email = { from: 'newsletter@example.com' };

    const suggestion = await getSuggestion(email);
    expect(suggestion).toBeNull();
  });
});
```

### Success Metrics
- Prediction accuracy: 75%+
- Suggestion acceptance rate: 40%+
- Time saved per suggestion: 2+ minutes
- User perception of helpfulness: 4/5

---

## Iteration 11: Smart Categorization & Clustering

### Goal
Automatically organize emails into smart categories and clusters. Group related conversations, detect project threads, and surface patterns.

### Categories & Clusters

#### Auto-Generated Categories
- **Action Required**: Emails needing response/action
- **Awaiting Response**: Emails you sent waiting for reply
- **Meeting Related**: Calendar invites, meeting notes
- **Project: [Name]**: Auto-detected project threads
- **Newsletters**: Subscriptions and updates
- **Transactional**: Receipts, confirmations

#### Smart Clusters
- Group by conversation thread
- Group by project/topic
- Group by sender organization
- Group by urgency level

### Implementation

#### Backend: Smart Categorizer
```python
# src/priority_lens/ml/categorizer.py
class SmartCategorizer:
    """Categorizes and clusters emails intelligently."""

    BUILTIN_CATEGORIES = [
        'action_required',
        'awaiting_response',
        'meeting_related',
        'newsletters',
        'transactional',
    ]

    async def categorize(
        self,
        emails: list[Email],
        user_id: UUID,
    ) -> CategorizedInbox:
        """Categorize emails into smart categories."""
        categories = {cat: [] for cat in self.BUILTIN_CATEGORIES}
        projects = defaultdict(list)

        for email in emails:
            # Determine primary category
            category = self._get_primary_category(email)
            categories[category].append(email)

            # Detect project affiliation
            project = await self._detect_project(email, user_id)
            if project:
                projects[project].append(email)

        # Generate dynamic project categories
        for project_name, project_emails in projects.items():
            if len(project_emails) >= 3:  # Minimum for a project
                categories[f'project_{project_name}'] = project_emails

        return CategorizedInbox(
            categories=categories,
            projects=list(projects.keys()),
        )

    async def _detect_project(
        self,
        email: Email,
        user_id: UUID,
    ) -> str | None:
        """Detect if email belongs to a project."""
        # Check subject line patterns
        project_patterns = [
            r'\[([^\]]+)\]',  # [Project Name]
            r'RE: ([^-]+) -',  # RE: Project Name -
        ]

        for pattern in project_patterns:
            match = re.search(pattern, email.subject)
            if match:
                return match.group(1).strip()

        # Use embedding similarity to existing projects
        existing_projects = await self._get_user_projects(user_id)
        for project in existing_projects:
            if await self._is_similar(email, project):
                return project.name

        return None
```

#### Mobile: Category Navigation
```typescript
// src/components/inbox/CategoryTabs.tsx
interface CategoryTabsProps {
  categories: Category[];
  activeCategory: string;
  onCategoryChange: (category: string) => void;
}

export function CategoryTabs({
  categories,
  activeCategory,
  onCategoryChange,
}: CategoryTabsProps): React.JSX.Element {
  return (
    <ScrollView
      horizontal
      showsHorizontalScrollIndicator={false}
      style={styles.container}
    >
      {categories.map((category) => (
        <Pressable
          key={category.id}
          style={[
            styles.tab,
            activeCategory === category.id && styles.activeTab,
          ]}
          onPress={() => onCategoryChange(category.id)}
        >
          <CategoryIcon category={category.id} />
          <Text style={styles.tabText}>{category.name}</Text>
          <Badge count={category.unreadCount} />
        </Pressable>
      ))}
    </ScrollView>
  );
}
```

### Verification Criteria
- [ ] Categories are accurate (>90%)
- [ ] Projects are auto-detected reliably
- [ ] User can create custom categories
- [ ] Category counts update in real-time
- [ ] Voice can navigate by category
- [ ] Clusters group related emails correctly

### Test Cases
```typescript
describe('SmartCategorization', () => {
  it('categorizes actionable emails correctly', async () => {
    const email = { subject: 'Please review and approve', has_question: true };
    const result = await categorize([email]);
    expect(result.categories.action_required).toContain(email);
  });

  it('detects project from subject pattern', async () => {
    const email = { subject: '[Website Redesign] New mockups ready' };
    const result = await categorize([email]);
    expect(result.projects).toContain('Website Redesign');
  });

  it('groups related emails in cluster', async () => {
    const emails = [
      { subject: 'Q1 Budget Review', from: 'finance@company.com' },
      { subject: 'RE: Q1 Budget Review', from: 'cfo@company.com' },
      { subject: 'Updated Q1 Budget', from: 'finance@company.com' },
    ];
    const clusters = await clusterEmails(emails);
    expect(clusters[0].emails.length).toBe(3);
  });
});
```

### Success Metrics
- Category accuracy: 90%+
- Project detection rate: 80%+
- User creates custom category: 30%+
- Navigation time: -40%

---

## Iteration 12: Email Summarization

### Goal
Generate intelligent summaries for emails, threads, and topics. Enable "brief me" functionality for quick understanding.

### Summary Types
- **Single Email**: Key points in 2-3 sentences
- **Thread Summary**: Conversation arc with decisions made
- **Topic Summary**: Cross-thread overview of a topic
- **Daily Digest Summary**: Spoken overview of day's highlights

### Implementation

#### Backend: Summarization Service
```python
# src/priority_lens/services/summarizer.py
class SummarizationService:
    """Generates intelligent email summaries."""

    async def summarize_email(
        self,
        email: Email,
        style: str = 'brief',
    ) -> Summary:
        """Generate summary for a single email."""
        prompt = self._build_email_prompt(email, style)

        summary = await self.llm.generate(prompt)

        return Summary(
            text=summary,
            key_points=await self._extract_key_points(summary),
            action_items=await self._extract_actions(email),
            sentiment=await self._analyze_sentiment(email),
        )

    async def summarize_thread(
        self,
        thread: list[Email],
    ) -> ThreadSummary:
        """Generate summary for an email thread."""
        prompt = f"""
        Summarize this email thread:

        {self._format_thread(thread)}

        Include:
        1. Main topic and purpose
        2. Key decisions made
        3. Current status / next steps
        4. Any unresolved questions

        Keep it conversational for voice reading.
        """

        summary = await self.llm.generate(prompt)

        return ThreadSummary(
            text=summary,
            participants=list(set(e.from_email for e in thread)),
            email_count=len(thread),
            decisions=await self._extract_decisions(thread),
            next_steps=await self._extract_next_steps(thread),
        )

    async def summarize_topic(
        self,
        topic: str,
        emails: list[Email],
    ) -> TopicSummary:
        """Generate summary across multiple threads for a topic."""
        prompt = f"""
        Summarize everything about "{topic}" from these emails:

        {self._format_emails(emails)}

        Provide:
        1. Overall status of {topic}
        2. Key stakeholders and their positions
        3. Timeline of events
        4. Current blockers or next steps
        """

        return await self.llm.generate(prompt)
```

#### Voice: Brief Me Command
```python
# src/priority_lens/agent/handlers/brief.py
class BriefMeHandler:
    """Handles 'brief me' voice commands."""

    async def handle(
        self,
        request: BriefRequest,
        context: ConversationContext,
    ) -> AgentResponse:
        if request.target == 'today':
            return await self._brief_today(context)
        elif request.target == 'thread':
            return await self._brief_thread(context.active_email)
        elif request.target == 'topic':
            return await self._brief_topic(request.topic, context)

    async def _brief_today(
        self,
        context: ConversationContext,
    ) -> AgentResponse:
        digest = await self.digest_service.get_digest(context.user_id)
        summary = await self.summarizer.summarize_digest(digest)

        return AgentResponse(
            text=summary.spoken_text,
            sdui_blocks=[
                self._create_summary_card(summary),
                self._create_action_items_list(summary.action_items),
            ],
        )
```

### Verification Criteria
- [ ] Single email summaries are accurate and concise
- [ ] Thread summaries capture key decisions
- [ ] Topic summaries span multiple threads
- [ ] Summaries are optimized for voice reading
- [ ] User can ask follow-up questions
- [ ] Summaries generate in <3 seconds

### Test Cases
```typescript
describe('Summarization', () => {
  it('generates concise email summary', async () => {
    const email = { body: longEmailContent };
    const summary = await summarize(email);

    expect(summary.text.length).toBeLessThan(500);
    expect(summary.key_points.length).toBeGreaterThan(0);
  });

  it('extracts action items from thread', async () => {
    const thread = [
      { body: 'Can you send the report by Friday?' },
      { body: 'Sure, I\'ll have it ready.' },
    ];
    const summary = await summarizeThread(thread);

    expect(summary.action_items).toContain('Send report by Friday');
  });

  it('voice reads summary naturally', async () => {
    const response = await agent.process('Brief me on my day');

    expect(response.text).not.toContain('[');
    expect(response.text).not.toContain('```');
    expect(response.text.length).toBeLessThan(1000);
  });
});
```

### Success Metrics
- Summary accuracy: 90%+
- User comprehension: 95%+
- Time saved per thread: 3+ minutes
- "Brief me" usage: 60% of active users

---

## Iteration 13: Offline Support & Caching

### Goal
Enable full functionality offline with intelligent caching. SDUI components cache, actions queue, and sync when back online.

### Offline Capabilities
- View cached emails and digest
- Draft replies (queue for sending)
- Mark items for action (sync later)
- Voice assistant with cached responses
- SDUI renders from cache

### Implementation

#### Mobile: Offline Cache Manager
```typescript
// src/services/offlineCache.ts
class OfflineCacheManager {
  private db: SQLite.SQLiteDatabase;
  private syncQueue: ActionQueue;

  async cacheEmails(emails: Email[]): Promise<void> {
    const batch = emails.map(email => ({
      sql: `INSERT OR REPLACE INTO emails
            (id, data, cached_at) VALUES (?, ?, ?)`,
      args: [email.id, JSON.stringify(email), Date.now()],
    }));

    await this.db.execBatch(batch);
  }

  async cacheSDUI(screenId: string, blocks: UIBlock[]): Promise<void> {
    await this.db.runAsync(
      `INSERT OR REPLACE INTO sdui_cache
       (screen_id, blocks, cached_at) VALUES (?, ?, ?)`,
      [screenId, JSON.stringify(blocks), Date.now()]
    );
  }

  async getCachedSDUI(screenId: string): Promise<UIBlock[] | null> {
    const result = await this.db.getFirstAsync(
      `SELECT blocks FROM sdui_cache WHERE screen_id = ?`,
      [screenId]
    );
    return result ? JSON.parse(result.blocks) : null;
  }

  async queueAction(action: PendingAction): Promise<void> {
    await this.syncQueue.enqueue(action);
  }

  async syncPendingActions(): Promise<SyncResult> {
    const pending = await this.syncQueue.getAll();
    const results = [];

    for (const action of pending) {
      try {
        await this.executeAction(action);
        await this.syncQueue.remove(action.id);
        results.push({ id: action.id, success: true });
      } catch (error) {
        results.push({ id: action.id, success: false, error });
      }
    }

    return { synced: results.filter(r => r.success).length };
  }
}
```

#### Network State Hook
```typescript
// src/hooks/useNetworkState.ts
export function useNetworkState() {
  const [isOnline, setIsOnline] = useState(true);
  const [pendingActions, setPendingActions] = useState(0);

  useEffect(() => {
    const unsubscribe = NetInfo.addEventListener((state) => {
      const wasOffline = !isOnline;
      setIsOnline(state.isConnected ?? false);

      // Sync when coming back online
      if (state.isConnected && wasOffline) {
        syncPendingActions();
      }
    });

    return unsubscribe;
  }, [isOnline]);

  return { isOnline, pendingActions };
}
```

### Verification Criteria
- [ ] App launches offline with cached data
- [ ] SDUI renders from cache
- [ ] Actions queue and sync when online
- [ ] User sees offline indicator
- [ ] Cache expires appropriately
- [ ] Conflict resolution works correctly

### Test Cases
```typescript
describe('OfflineSupport', () => {
  it('loads cached emails when offline', async () => {
    await cacheManager.cacheEmails(mockEmails);
    mockNetInfo.isConnected = false;

    const { getByTestId } = render(<InboxScreen />);
    await waitFor(() => {
      expect(getByTestId('email-list')).toBeTruthy();
      expect(getByTestId('offline-indicator')).toBeTruthy();
    });
  });

  it('queues actions when offline', async () => {
    mockNetInfo.isConnected = false;
    await archiveEmail('email-123');

    const pending = await cacheManager.getPendingActions();
    expect(pending).toContainEqual(
      expect.objectContaining({ type: 'archive', emailId: 'email-123' })
    );
  });

  it('syncs queued actions when online', async () => {
    await cacheManager.queueAction({ type: 'archive', emailId: 'email-123' });
    mockNetInfo.isConnected = true;

    const result = await cacheManager.syncPendingActions();
    expect(result.synced).toBe(1);
    expect(mockApi.archive).toHaveBeenCalledWith('email-123');
  });
});
```

### Success Metrics
- Offline usability: 90% of features work
- Sync success rate: 99%+
- Cache hit rate: 80%+
- User frustration (offline): <2/5

---

## Iteration 14: Accessibility & Inclusivity

### Goal
Ensure the app is fully accessible to users with disabilities. Support screen readers, voice-only mode, high contrast, and motor accessibility.

### Accessibility Features
- **VoiceOver/TalkBack**: Full screen reader support
- **Voice-Only Mode**: Complete functionality via voice
- **High Contrast**: WCAG AAA compliant option
- **Reduced Motion**: Respect system preferences
- **Large Text**: Dynamic type support
- **Switch Control**: Full keyboard/switch navigation

### Implementation

#### Accessibility Context
```typescript
// src/context/AccessibilityContext.tsx
interface AccessibilitySettings {
  screenReaderEnabled: boolean;
  voiceOnlyMode: boolean;
  highContrast: boolean;
  reducedMotion: boolean;
  preferredTextSize: number;
}

export function AccessibilityProvider({ children }: Props) {
  const [settings, setSettings] = useState<AccessibilitySettings>({
    screenReaderEnabled: false,
    voiceOnlyMode: false,
    highContrast: false,
    reducedMotion: false,
    preferredTextSize: 1,
  });

  useEffect(() => {
    // Listen to system accessibility settings
    AccessibilityInfo.isScreenReaderEnabled().then((enabled) => {
      setSettings(s => ({ ...s, screenReaderEnabled: enabled }));
    });

    const subscription = AccessibilityInfo.addEventListener(
      'screenReaderChanged',
      (enabled) => {
        setSettings(s => ({ ...s, screenReaderEnabled: enabled }));
      }
    );

    return () => subscription.remove();
  }, []);

  return (
    <AccessibilityContext.Provider value={{ settings, setSettings }}>
      {children}
    </AccessibilityContext.Provider>
  );
}
```

#### Accessible Components
```typescript
// src/components/ui/AccessibleCard.tsx
interface AccessibleCardProps {
  title: string;
  subtitle?: string;
  actions: UIAction[];
  children: React.ReactNode;
}

export function AccessibleCard({
  title,
  subtitle,
  actions,
  children,
}: AccessibleCardProps): React.JSX.Element {
  const { settings } = useAccessibility();

  // Generate accessibility label for screen readers
  const accessibilityLabel = [
    title,
    subtitle,
    `${actions.length} actions available`,
  ].filter(Boolean).join('. ');

  return (
    <View
      accessible={true}
      accessibilityLabel={accessibilityLabel}
      accessibilityRole="button"
      accessibilityActions={actions.map(a => ({
        name: a.type,
        label: a.label,
      }))}
      onAccessibilityAction={(event) => {
        const action = actions.find(a => a.type === event.nativeEvent.actionName);
        if (action) action.onPress();
      }}
      style={[
        styles.card,
        settings.highContrast && styles.highContrastCard,
      ]}
    >
      <Text
        style={[
          styles.title,
          { fontSize: 16 * settings.preferredTextSize },
        ]}
      >
        {title}
      </Text>
      {children}
    </View>
  );
}
```

### Verification Criteria
- [ ] VoiceOver/TalkBack navigates all elements
- [ ] All actions accessible via voice
- [ ] Color contrast meets WCAG AAA (7:1)
- [ ] Animations respect reduced motion
- [ ] Dynamic type scales correctly
- [ ] Focus order is logical

### Test Cases
```typescript
describe('Accessibility', () => {
  it('all interactive elements have accessibility labels', () => {
    const { getAllByRole } = render(<InboxScreen />);
    const buttons = getAllByRole('button');

    buttons.forEach((button) => {
      expect(button.props.accessibilityLabel).toBeTruthy();
    });
  });

  it('cards announce correct information to screen reader', () => {
    const { getByTestId } = render(
      <EmailCard email={mockEmail} />
    );
    const card = getByTestId('email-card');

    expect(card.props.accessibilityLabel).toContain(mockEmail.subject);
    expect(card.props.accessibilityLabel).toContain(mockEmail.from);
  });

  it('high contrast mode applies correct colors', () => {
    mockAccessibility.highContrast = true;
    const { getByTestId } = render(<EmailCard email={mockEmail} />);

    expect(getByTestId('email-card').props.style).toMatchObject({
      borderWidth: 2,
      borderColor: '#000000',
    });
  });

  it('respects reduced motion preference', () => {
    mockAccessibility.reducedMotion = true;
    const { getByTestId } = render(<AnimatedCard />);

    expect(getByTestId('animated-card').props.style.transform).toBeUndefined();
  });
});
```

### Success Metrics
- VoiceOver usability: 100%
- WCAG AAA compliance: 100%
- Accessibility audit pass: 100%
- User with disabilities satisfaction: 4.5/5

---

## Iteration 15: Performance Optimization & Launch Ready

### Goal
Optimize performance for production launch. Achieve fast startup, smooth scrolling, efficient battery usage, and minimal bundle size.

### Performance Targets
- **App Launch**: <2 seconds cold start
- **Time to Interactive**: <3 seconds
- **Frame Rate**: 60fps scrolling
- **Memory**: <200MB active
- **Battery**: <5% per hour active use
- **Bundle Size**: <50MB

### Implementation

#### Performance Monitoring
```typescript
// src/services/performance.ts
class PerformanceMonitor {
  private metrics: Map<string, number[]> = new Map();

  async measureStartup(): Promise<StartupMetrics> {
    const start = performance.now();

    // Measure key milestones
    const milestones = {
      nativeBoot: await this.measureNativeBoot(),
      jsBundle: await this.measureJSBundle(),
      firstRender: await this.measureFirstRender(),
      interactive: await this.measureInteractive(),
    };

    return {
      total: performance.now() - start,
      ...milestones,
    };
  }

  trackFrameRate(screenName: string): () => void {
    const frameMetrics: number[] = [];
    let lastFrameTime = performance.now();

    const frameCallback = () => {
      const now = performance.now();
      const frameDuration = now - lastFrameTime;
      frameMetrics.push(1000 / frameDuration); // FPS
      lastFrameTime = now;
      requestAnimationFrame(frameCallback);
    };

    requestAnimationFrame(frameCallback);

    return () => {
      const avgFps = frameMetrics.reduce((a, b) => a + b, 0) / frameMetrics.length;
      this.reportMetric(`${screenName}_fps`, avgFps);
    };
  }
}
```

#### Optimization Strategies
```typescript
// List virtualization
import { FlashList } from '@shopify/flash-list';

// Image optimization
import { Image } from 'expo-image';

// Memoization
const MemoizedEmailCard = React.memo(EmailCard, (prev, next) => {
  return prev.email.id === next.email.id &&
         prev.email.read === next.email.read;
});

// Lazy loading
const SettingsScreen = React.lazy(() => import('./screens/SettingsScreen'));

// Bundle splitting
// metro.config.js
module.exports = {
  transformer: {
    getTransformOptions: async () => ({
      transform: {
        experimentalImportSupport: false,
        inlineRequires: true,
      },
    }),
  },
};
```

#### Battery Optimization
```typescript
// src/services/battery.ts
class BatteryOptimizer {
  async optimizeForBattery(): Promise<void> {
    // Reduce location updates
    await Location.stopLocationUpdatesAsync();

    // Reduce sync frequency
    BackgroundFetch.setMinimumIntervalAsync(15 * 60); // 15 min

    // Disable non-essential animations
    LayoutAnimation.setEnabled(false);

    // Reduce network polling
    this.pollInterval = 60000; // 1 min instead of 15s
  }

  async optimizeForPerformance(): Promise<void> {
    // Restore full functionality
    await Location.startLocationUpdatesAsync();
    BackgroundFetch.setMinimumIntervalAsync(5 * 60);
    LayoutAnimation.setEnabled(true);
    this.pollInterval = 15000;
  }
}
```

### Verification Criteria
- [ ] Cold start <2 seconds
- [ ] 60fps scrolling on iPhone 12 and Pixel 6
- [ ] Memory stays under 200MB
- [ ] Battery drain <5%/hour
- [ ] Bundle size <50MB
- [ ] No memory leaks in 1-hour session

### Test Cases
```typescript
describe('Performance', () => {
  it('app launches within 2 seconds', async () => {
    const metrics = await measureStartup();
    expect(metrics.total).toBeLessThan(2000);
  });

  it('inbox scrolls at 60fps', async () => {
    const { getByTestId } = render(<InboxScreen />);
    const list = getByTestId('email-list');

    const stopTracking = performanceMonitor.trackFrameRate('inbox');

    // Simulate fast scroll
    for (let i = 0; i < 100; i++) {
      fireEvent.scroll(list, { nativeEvent: { contentOffset: { y: i * 100 } } });
    }

    const avgFps = stopTracking();
    expect(avgFps).toBeGreaterThan(55);
  });

  it('memory stays under limit during extended use', async () => {
    const initialMemory = await getMemoryUsage();

    // Simulate 30 minutes of use
    for (let i = 0; i < 30; i++) {
      await navigateToScreen('Inbox');
      await navigateToScreen('Digest');
      await navigateToScreen('Settings');
    }

    const finalMemory = await getMemoryUsage();
    expect(finalMemory).toBeLessThan(200 * 1024 * 1024);
  });
});
```

### Launch Checklist
- [ ] All 15 iterations complete and tested
- [ ] 100% test coverage on critical paths
- [ ] Performance metrics meet targets
- [ ] Accessibility audit passed
- [ ] Security audit passed
- [ ] App Store/Play Store assets ready
- [ ] Privacy policy and terms updated
- [ ] Analytics and crash reporting configured
- [ ] Beta testing complete with 100+ users
- [ ] Support documentation ready

### Success Metrics
- App Store rating: 4.5+
- Crash-free rate: 99.9%
- DAU/MAU ratio: 40%+
- User retention (D7): 50%+
- NPS score: 50+

---

## Summary

Phase 6 transforms Priority Lens into a truly proactive AI assistant through 15 carefully planned iterations:

| # | Iteration | Key Deliverable |
|---|-----------|-----------------|
| 1 | Smart Digest View | AI Inbox with to-dos and topics |
| 2 | Proactive Notifications | Smart, contextual push alerts |
| 3 | Adaptive Email Cards | Dynamic cards for email types |
| 4 | Voice Conversation | Multi-turn natural dialogue |
| 5 | Voice Action Execution | Complete tasks via voice |
| 6 | Voice Feedback & Haptics | Rich audio/tactile responses |
| 7 | Glassmorphism Design | Modern 2026 visual polish |
| 8 | Time-Based UI | Context-aware interface |
| 9 | Preference Learning | Adaptive AI from behavior |
| 10 | Predictive Actions | Anticipate user needs |
| 11 | Smart Categorization | Auto-organize emails |
| 12 | Email Summarization | Brief me functionality |
| 13 | Offline Support | Full offline capability |
| 14 | Accessibility | Inclusive for all users |
| 15 | Performance & Launch | Production-ready polish |

**Total Estimated Effort**: 15 iterations, each 2-3 days = ~6-8 weeks

**Ultimate Outcome**: An AI assistant that knows what's important to you, proactively brings it up at the right time, and helps you take care of it through natural voice conversation and beautiful adaptive UI.

---

## References

### 2026 UI/UX Trends
- [Key Mobile App UI/UX Design Trends for 2026](https://www.elinext.com/services/ui-ux-design/trends/key-mobile-app-ui-ux-design-trends/)
- [10 UX Design Shifts You Can't Ignore in 2026](https://uxdesign.cc/10-ux-design-shifts-you-cant-ignore-in-2026-8f0da1c6741d)
- [Best UI Design Practices for Mobile Apps in 2026](https://uidesignz.com/blogs/mobile-ui-design-best-practices)

### Voice AI & Conversational UI
- [Voice UI Design: Crafting User Experiences for Voice Assistants](https://medium.com/design-bootcamp/voice-ui-design-crafting-user-experiences-for-voice-assistants-2beec5284bea)
- [Conversational AI Assistant Design: 7 UX/UI Best Practices](https://www.willowtreeapps.com/insights/willowtrees-7-ux-ui-rules-for-designing-a-conversational-ai-assistant)
- [The AI User Interface of the Future = Voice](https://www.theneurondaily.com/p/deep-dive-the-ai-user-interface-of-the-future-voice)

### SDUI Patterns
- [Server-Driven UI Basics - Apollo GraphQL](https://www.apollographql.com/docs/graphos/schema-design/guides/sdui/basics)
- [Server-Driven UI Design Patterns: A Professional Guide](https://devcookies.medium.com/server-driven-ui-design-patterns-a-professional-guide-with-examples-a536c8f9965f)
- [How Server Driven UI Changed Mobile Development](https://www.verygood.ventures/blog/how-server-driven-ui-sdui-changed-the-way-i-think-about-mobile-development)

### Proactive AI
- [Proactive AI Agents: Anticipating Needs Before You Do](https://www.hey-steve.com/insights/proactive-ai-agents-anticipating-needs-before-you-do)
- [10 Best AI Personal Assistants in 2026](https://www.dume.ai/blog/10-ai-personal-assistants-youll-need-in-2026)
- [Gmail's AI Inbox - Biggest Update in 20 Years](https://www.tomsguide.com/ai/gmails-biggest-update-in-20-years-5-ai-features-that-could-change-email-forever)
