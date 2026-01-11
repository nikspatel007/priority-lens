# Priority Lens Design Constitution

> The immutable principles that guide every design decision.

---

## Core Philosophy

### The Fundamental Truth
> "The hardest thing about any of this is not actually doing the task, or putting the task in the system. It's actually remembering that I have to put a task in a system. That's the thing that I want to solve for."

We are not building a task manager. We are building **a system that removes the cognitive load of remembering**.

### Proactive, Not Reactive
> "Ultimately, the goal is not that UI is the only way for people to interact. We want more proactive than reactive. If the user needs to rely on UI, that's reactive."

The system comes to the user. The user doesn't come to the system.

---

## Design Principles

### 1. Work First, Interface Second
The primary thing is **work**, not conversation. Screen real estate belongs to getting things done.

- Chat interfaces are not our identity
- Every pixel should serve the task at hand
- We show work, not messages

### 2. AI as Equal Partner
> "In an ideal world, it would be on equal footing."

- AI is not a sidebar or afterthought
- AI is not the decision maker - the person is
- AI is the recommender, the human decides
- Position AI prominently, not tucked away

### 3. Minimal Chrome
> "The navigation is no longer prominent... it's no longer stealing your attention."

- Navigation exists but doesn't dominate
- No more than 5 items in bottom nav
- Utility items (profile, settings) go in drawer
- This is not social media - no infinite scroll

### 4. Voice-First Design
> "I would rather talk to a computer all day to get my stuff done."
> "The way I speak is completely different from the way I type."

- Voice mode is primary, text is fallback
- Design for speaking speed, not typing speed
- Support the toggle between voice and text gracefully

### 5. Perception of Privacy & Trust
> "Designing for perception of privacy."

- Security should be felt, not just implemented
- Trust is built through transparency
- Show what AI is doing, not just the results

---

## Visual Language

### Typography

| Use Case | Font Type | Examples |
|----------|-----------|----------|
| AI-generated content | **Serif** (Georgia) | Transcription, briefs, summaries, recommendations |
| Headlines & Titles | **Serif** (Georgia) | H1, H2, large display text |
| Body & UI | **Sans Serif** (System) | Labels, buttons, navigation, user input |
| Small text | **Sans Serif** (System) | Captions, metadata, timestamps |

**Why:** Serif for AI creates distinction and gravitas. Sans-serif for UI keeps things clean and functional.

### Color Philosophy

**Approach:** Build in grayscale first, add color intentionally.

> "The best use of color is few and far between. You don't want to blow everything up with color."

- **Primary:** Blue (Hue 212) - Trust, professionalism
- **Grayscale:** Cool-toned (not pure gray)
- **Semantic:** Use color for meaning, not decoration
- **Consistency:** Stick to one hue family

### Icons

- **System:** MGC Icon System Pro
- **Style:** Cute Filled (approachable, extensive)
- **Avoid:** Material UI, Font Awesome (too ubiquitous, no personality)

### Spacing & Layout

- Precision in design, simplicity in appearance
- "Don't make me think"
- Nothing ostentatious, no unnecessary embellishment

---

## Interaction Patterns

### The Three Modes

| Mode | Purpose | Entry Point |
|------|---------|-------------|
| **Execution** | Get stuff done NOW | Default landing after onboarding |
| **Prep** | Prepare for upcoming work | When execution queue is empty |
| **Eagle View** | System overview | User-initiated |

### Execution Mode Framework

Every task follows this structure:

1. **"Here's what I'm seeing"** - Present the task with context
2. **"Did I get that right?"** - Confirm understanding
3. **"Here's what I can do"** - Present action options
4. **"Here's how I'd start"** - Show example (optional)
5. **"Anything else before I begin?"** - Final confirmation

### Action Classification

All incoming items get tagged:
- **Action Required** - Someone needs to do something
  - I need to act
  - Someone else needs to act (follow-up)
- **FYI** - Just need to know about it

### Data Collection Through Use

> "When we show these components, give them actions: approve, reject, yes, no, change."

Every interaction is a data point:
- Track approvals vs rejections
- Learn user preferences over time
- Offer automation after trust is built
- "Last 8 out of 10 times you approved - want us to go auto?"

---

## Component Guidelines

### Generative UI Components

Build reusable components for common scenarios:
- **Time/Calendar** - Date selection, availability
- **People list** - Name, position, email
- **Document preview** - Attachments, files
- **Action buttons** - Approve, reject, modify
- **Email preview** - Draft messages
- **Task card** - Container for all elements

### Voice Interface Elements

- Two distinct waveforms: User speaking vs Lenso speaking
- Pulsating orb indicates active state
- Transcript at bottom (collapsible)
- Keyboard icon for text mode toggle

### Navigation

**Mobile:**
- Bottom navigation (max 5 items)
- Side drawer for utilities
- No left sidebar

**Desktop:**
- Left navigation (can be collapsed)
- AI prominent, not sidebar

---

## Anti-Patterns (What We Don't Do)

### Never
- Make AI feel like an afterthought
- Require users to remember to check the app
- Design primarily for chat/conversation
- Use time estimates ("this will take 5 minutes")
- Create modal interruptions like ChatGPT's voice mode
- Use generic/ubiquitous icon sets
- Mix warm and cool color hues

### Avoid
- Infinite scroll
- Complex navigation hierarchies
- Feature bloat for v1
- Over-engineering for hypothetical scale
- Building compliance features before product-market fit

---

## The Test

Before shipping any design, ask:

1. **Does this reduce cognitive load?** If the user has to remember something, we failed.
2. **Is this proactive?** Does it come to the user, or wait for them?
3. **Is AI an equal partner?** Not hidden, not dominant.
4. **Does work come first?** Screen space for tasks, not chrome.
5. **Would someone use this by the fire?** Simple enough for holiday mode.

---

## Amendments

This constitution can be amended when:
- User research contradicts a principle
- A principle blocks meaningful progress
- The team unanimously agrees to change

All amendments must be documented with rationale.

---

*Established December 2024 by Nik Patel & Josh Smith*
