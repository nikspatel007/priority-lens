# Execution Flow & Navigation Meeting

**Date:** December 21, 2024 (Saturday, ~15 minutes)
**Attendees:** Nik Patel, Josh Smith
**Topic:** Execution Mode UX, Navigation Concepts, Generative UI, Task Components

---

## Context

Quick Saturday sync focused on execution mode interface. Both had limited time (Josh still recovering, Nik heading out for family time). Josh warned: "If Bo ever told you about me, I tend to have ideas that are way off the deep end of what people are expecting."

---

## Naming Discussion

**Execution Mode alternatives explored:**
- "Get It Done Mode"
- "GID Mode"
- "Get'er Done"

Settled on keeping "Execution Mode" for now.

---

## Navigation Philosophy

### Key Principle
> "What if by default they get into an experience where there's no left-hand navigation?"
> "The navigation is no longer prominent... it's no longer stealing your attention."

### Design Decision
- Navigation exists but is minimized
- On mobile: Bottom nav only, no left sidebar
- On desktop: Navigation can exist but isn't prominent
- Focus is 100% on the current task

---

## Voice Interface Layout

### Nik's Proposal
Move voice mode indicators to minimize screen real estate:

```
┌─────────────────────────┐
│                         │
│    [TASK CONTENT]       │
│    Dynamic components   │
│    Context, info        │
│                         │
├─────────────────────────┤
│  ◀── Task  ──▶         │  (task navigation)
├─────────────────────────┤
│ [User] ━━━━━ [Lenso]   │  (voice indicators)
│       Execution Mode    │
└─────────────────────────┘
```

**Waveform Behavior:**
- When user is talking: User side highlights + waves
- When Lenso is talking: Lenso side highlights + waves
- Pulsating indicates active state

### Benefit
> "That opens up a significant amount of screen asset for us, both desktop but mainly on the mobile, where we could show information."

---

## Generative UI Concept

### The Inspiration
Nik showed "Multi-stack Conversation Generative UI" concept:

> "As you are streaming the data, generally it's just text. But there comes a time where you are trying to do something and it's not just text, you're showing the actual behavior."

### How It Works
1. AI determines type of information to display
2. Triggers appropriate component
3. Component renders dynamically on the fly

### Example Components
- **Weather component** - "What's the weather in Tokyo?"
- **Calendar component** - Time and date selection
- **List component** - People, options, items
- **Document component** - Meeting notes, attachments
- **Time clock component** - Scheduling
- **Email preview component** - Draft emails

### Why This Works for Priority Lens
> "There are certain things that we know are going to be common: What times do I need to pick? What my calendar looks like, what are my meeting notes, who are the people involved, meeting briefs."

---

## Task Flow Framework

Josh outlined a 4-step conversational framework:

### Step 1: "Here's what I'm seeing"
- Present the task and all pieces included
- Show relevant components (calendar, time, people list)
- End with: **"Did I get that right?"**

**Component Example:**
> "When they show a list of people, always show: name, position, email. Just design that component."

### Step 2: "Here's what I can do for you"
- Present what AI can handle
- Show generated content (agenda, draft email)
- **"I can coordinate finding the right time for you."**

### Step 3: Example preview (optional)
- **"Here's an example of how I'd start taking care of this."**
- Show first email draft or initial action
- Not every single step - just the first piece to build comfort

### Step 4: Final confirmation
- **"Great! Anything else you'd like to share before I begin?"**
- Last opportunity to add context or modify
- Then move on to next task

---

## Action Buttons & Data Collection

### Nik's Key Insight
> "One of the things we should build into this is when we show these components, give it action: approve, reject, yes, no, change."

### Why This Matters
1. Every approval = data point that AI is working well
2. Track what's working, what's not
3. Build evaluation metrics
4. **Future automation:** "Last 8 out of 10 times you approved, do you want us to go auto?"

### Trust Building
> Josh: "I have been approaching concepting based on the target of folks who already have high trust in AI."
> "People who have low trust want to see and approve over time before the AI says 'do you want us to take care of it for you?'"

---

## Component Design Approach

### Philosophy
> "Don't provide visuals for every single thing, just what are the most common ones."

### Priority Components to Build
1. **Time/Calendar** - Date selection, availability
2. **People list** - Name, position, email format
3. **Document/Attachment** - Preview of files
4. **Action buttons** - Approve, reject, modify
5. **Email preview** - Draft messages
6. **Task card** - Container for all elements

---

## Weekend Goals

Nik's plan:
1. Build Gmail/Calendar integration flow
2. Ingest emails and calendar data
3. Display: "What do I do with this ingestion?"

Josh to continue:
1. Refine execution mode mockups
2. Create more concrete task examples

---

## Key Quotes

On screen real estate:
> "The whole need for [the voice element] to exist goes away, which opens up a canvas for us to show the things we want to show, because that's where we should be paying attention."

On transcription:
> Josh: "Maybe on mobile you don't need transcription?"
> Nik: "If we do need transcription, we are taking bottom half, and then we have top half available."

On trust building:
> "Even when it comes up, even if they say 'yeah, this looks good, send it,' we'll mark it as approved button clicked, which gives us the data we need behind the scene."

---

## Action Items

1. **Tomorrow (Sunday):** Walk through concrete email examples
2. **This weekend:** Nik builds Gmail/Calendar integration
3. **Ongoing:** Define most common component types
4. **Future:** Auto-rule creation based on approval patterns
