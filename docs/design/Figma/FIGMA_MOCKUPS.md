# Priority Lens Figma Mockups

> Low-fidelity wireframes defining the visual direction and user flows for Priority Lens.

---

## Overview

These wireframes establish the core screens, components, and user journeys for Priority Lens. They represent lo-fi concepts that guide implementation while leaving room for visual refinement.

---

## 1. Authentication Flow
**File:** `01_authentication_flow.png`

### Screens Defined

| Screen | Purpose |
|--------|---------|
| Splash | Priority Lens logo + tagline + "Sign in with Google" button |
| Google OAuth | System modal requesting access permissions |
| 2-Factor Setup | "Secure your account with 2-factor" + phone number input |
| Email Verification | 6-digit code entry for verification |
| Voice Assistant Intro | "Hi I'm Cortex, our voice-assisted agent..." |
| Voice Mode Choice | "Let's go" vs "I'd rather go it alone" options |
| Microphone Permission | System permission request with context |
| Name Entry | "For starters, what should I call you?" |

### Key Implementation Notes
- Google OAuth is primary authentication method
- 2FA via phone/email is secondary security layer
- Voice assistant (Cortex/Lenso) introduces itself before onboarding
- User can opt-out of voice mode ("I'd rather go it alone")
- Microphone permission requested with user context, not cold

---

## 2. Voice Onboarding Flow
**File:** `02_voice_onboarding_flow.png`

### Complete Onboarding Sequence

1. **Introduction**
   - "Hi I'm Cortex, you can talk to me and my job is too...we're going to..."

2. **Name Capture**
   - "For starters, what should I call you?"
   - User responds: "Ishmael" (or their name)

3. **Context Gathering** (while scanning)
   - "Hey Nik, great to meet you."
   - "I'm looking through XYZ. While I do that, let's get to know each other a bit more."

4. **Personality Questions**
   - "One thing I'd like to know is what's the most important thing you have to get done today?"
   - "How do you like to communicate with people?"
   - "How do you like people to communicate with you?"
   - "When are you at your best to focus?"

5. **Confirmation Screens**
   - "Based on what you have told us, here's the type of personality we believe you are."
   - "Would you agree?" (Yes/Almost there.../No buttons)

6. **Topic Prioritization**
   - "Based on looking at your calendar, emails, meetings, and communications, these are the folks you communicate with the most?"
   - "Are these the most important topics for you?"
   - "Would you agree?" confirmation

7. **Processing State**
   - "Scanning emails...done."
   - "Scanning meetings...done."
   - "Scanning contacts...done."
   - "Preparing your experience now"

8. **Completion**
   - "The dashboard is ready for you."
   - "Take me there" button

### Visual Elements
- **Pulsating blue orb** - AI speaking/thinking indicator
- **Waveform** - User speaking indicator ("Listening...")
- **Progress bar** - Multi-step progress at top
- **Back button** - Navigation to previous question

---

## 3. UI Components & Task Cards
**File:** `03_ui_components_task_cards.png`

### Mode Navigation
- **Execution** | **Prep** | **Eagle** - Three-mode pill selector
- Positioned at top of screen
- Keyboard toggle icon for text mode

### Task Card Component

```
┌─────────────────────────────────┐
│  Your Task                      │
│  {Task Name}                    │
│                                 │
│  ┌─────────┐  Dunder Mifflin    │
│  │  📄     │  A giant ream of   │
│  │         │  paper             │
│  └─────────┘                    │
│              Due Wednesday      │
│              $999               │
│                                 │
│  "What would you like to do?"   │
│                                 │
│  [Pay Now]  [Deal with it later]│
└─────────────────────────────────┘
```

### AI Conversation Pattern
Shows the back-and-forth between AI and user:
- AI: "Here's what I'm seeing. Did I get it right?"
- AI: "Here's what I can do for you. Thoughts?"
- AI: "Here's an example of how I'd start out taking care of this. Thoughts?"
- AI: "I'm on it. Anything else you'd like to share before I begin?"

### People List Component
```
┌──────────────────┐
│ Bob              │
│ {Title}          │
│ {email}          │
├──────────────────┤
│ Bob              │
│ {Title}          │
│ {email}          │
└──────────────────┘
```

### Invoice Cards (Multiple States)
- Small: Company name, due date, amount
- Medium: With date highlight "Due Wednesday"
- Large: Full details with invoice number

### Additional Components
- Calendar widget placeholder
- Time/Clock component
- List component
- Project Duration indicator
- Goal tracker
- Contact card (John Smith, Motivational Speaker)

### Voice Mode Elements
- Blue dot (AI ready)
- Waveform visualization
- Pink/magenta dot (recording active)
- Keyboard toggle icon

---

## 4. Navigation & Input Patterns
**File:** `04_navigation_input_patterns.png`

### Top Navigation Variations

**Variation 1:** Pill selector with keyboard toggle
```
[ Execution | Prep | Eagle ]  ⌨️
```

**Variation 2:** Icon-based mode selector + text input
```
[✓][⊞][✳]  [________________⬆️]  🎤
```

**Variation 3:** Centered pill with voice
```
      [ Execution | Prep | Eagle ]  🎤
[________________⬆️]
```

**Variation 4:** Top pill, bottom input
```
      [ Execution | Prep | Eagle ]  🎤
      [________________⬆️]
```

### Bottom Navigation Bar

**Standard Layout:**
```
☰  [✓ ⊞ ✳]  ❓ 👤
```

**Components:**
- Hamburger menu (☰) - Side drawer access
- Mode icons (✓ ⊞ ✳) - Quick mode switching
- Help (❓) - Support/FAQ
- Profile (👤) - User settings

### Voice Mode Interface
```
[________________⬆️]  🎤

🔵 · · | · · · | · · 🟣  ⌨️
```
- Blue dot: AI indicator
- Waveform: Audio visualization
- Pink dot: Recording indicator
- Keyboard icon: Switch to text mode

---

## 5. Typography & Color System
**File:** `05_typography_color_system.png`

### Typography Scale

| Variant | Font | Weight | Size | Line Height |
|---------|------|--------|------|-------------|
| h1 | Inter | Bold | 36px | 133% |
| h2 | Inter | Bold | 30px | 133% |
| h3 | Inter | Semi Bold | 24px | 150% |
| h4 | Inter | Semi Bold | 20px | 155% |
| title-lg | Inter | Semi Bold | 18px | 166% |
| title-md | Inter | Semi Bold | 16px | 150% |
| title-sm | Inter | Semi Bold | 14px | 142% |
| body-lg | Inter | Regular | 18px | 155% |
| body-md | Inter | Regular | 16px | 150% |
| body-sm | Inter | Regular | 14px | 142% |
| body-xs | Inter | Semi Bold | 12px | 166% |

### Font Family Specifications

**Sans Serif** (Labels, body content, small text):
```css
font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
             "Helvetica Neue", Arial, "Noto Sans", "Liberation Sans",
             system-ui, sans-serif;
```

**Serif** (AI transcription, headlines, titles):
```css
font-family: ui-serif, "New York", "Times New Roman", Times, Georgia,
             Cambria, "Liberation Serif", "Noto Serif", serif;
```

### Color System

**Primary Hue:** 212 (Deep Navy Blue)
**Secondary Hue:** 188 (Teal)

Color gradient from dark navy to light blue-gray is shown, with consistent hue application across the scale.

---

## 6. Information Architecture
**File:** `06_information_architecture.png`

### User Journey Flow

```
Sign In (Gmail/Outlook)
        ↓
   OTP Secondary Auth
        ↓
  "Welcome to Cortex..."
        ↓
    First Question:
  "What should I call you?"
        ↓
   Personality Questions
        ↓
  "I'm looking through XYZ.
   While I do that, let's get
   to know you a bit more."
```

### Action Classification System

Every incoming item is classified as:

| Classification | Description |
|----------------|-------------|
| I need to reply to an email | User action required - email response |
| Someone else needs to reply to an email | Follow-up tracking - awaiting response |
| I need to take an action other than replying | User action required - non-email |
| Someone else needs to take an action | Follow-up tracking - delegated task |

### Core Modes Defined

**Execution Mode:**
- "These are the 7 things on your plate"
- "Here's what needs to happen next"
- "Here's what I can do for you"

**Prep Mode:**
- Preparing for upcoming work
- Meeting briefs
- Context gathering

**Eagle View:**
- "Here are your projects"
- "Here's the status of each project"
- Strategic overview

### Scenario: Meeting Scheduling
```
"I'm on the hook to schedule a meeting."
  → I can set up the meeting with you.
  → 30 minute meeting, agenda, participants
  → Coordinate times and schedule a meeting.
```

### vNow Concept
- PL Agents talk to each other
- Agent-to-agent communication for complex workflows

### Information Grouping
- Person
- Subject
- Task

---

## Implementation Priority

Based on these mockups, the build order should be:

### Phase 1: Core Authentication
1. Splash screen with Google Sign-in
2. OAuth integration
3. 2FA setup flow
4. Voice mode choice screen

### Phase 2: Voice Onboarding
1. Voice assistant introduction
2. Name capture
3. Personality questions flow
4. Progress indicators
5. Completion/dashboard transition

### Phase 3: Execution Mode
1. Mode navigation (Execution/Prep/Eagle)
2. Task card component
3. Action buttons (Pay Now / Deal with it later)
4. Voice/text input toggle
5. Bottom navigation bar

### Phase 4: Polish
1. People list components
2. Invoice/document cards
3. Calendar widgets
4. Full typography implementation
5. Color system application

---

## Design Decisions Captured

1. **Voice-first, text-fallback** - Every screen supports both modes
2. **Three modes** - Execution, Prep, Eagle are the core navigation
3. **Action classification** - All items tagged as "I need to act" or "Someone else needs to act"
4. **Conversational AI** - Back-and-forth confirmation pattern before taking action
5. **Minimal chrome** - Navigation is subtle, content is primary
6. **Progressive disclosure** - Information revealed through conversation

---

*Generated from Figma wireframes - December 2024*
