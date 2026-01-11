# Deep Dive Session

**Date:** Early December 2024 (Friday - extended session ~75+ minutes)
**Attendees:** Nik Patel, Josh Smith
**Topic:** Product Structure, UX Framework, AI Interaction Patterns

---

## The Core Problem Statement

**This is the most important quote from all meetings:**

> "The hardest thing about any of this is not actually doing the task, or putting the task in the system. If I can put a task, then obviously it will remind me and everything. That's not the hard part. It's actually remembering that I have to put a task in a system. That's the thing that I want to solve for."

This is the fundamental problem Priority Lens aims to solve - not task management, but **the cognitive load of remembering to track things**.

---

## Heads Up App - Testing Ground

Nik built a "Heads Up" app over Thanksgiving weekend as a light version of Priority Lens:

### Features Built
1. **Double-tap gesture** to open voice transcription
2. **Voice input** with LLM processing ("remind Anki to do something tomorrow at 9")
3. **Family/group concept** for shared reminders
4. **Gmail integration** for calendar/email awareness
5. **Copy-paste task creation** - paste anything and it converts to task
6. **Share sheet integration** (though "too many steps")
7. **Siri shortcuts** - "Hey Siri, ask heads up to do something"
8. **Action button support** for iPhone 15+

### Key Discovery
The double-tap gesture is becoming **habit-forming**:
> "Anki's saying something, tap, tap. And it's becoming my habit."

### Why This Matters
> "In pretty much 95% cases... anytime I have used this software, including Jira for development teams, the issue has never been that we can't get the task done, or we don't know what to do. It's how do I get that in there so I know what my collective priorities look like?"

---

## Mobile-First Decision

**App over browser is required:**
> "From a technical perspective, not optimal. Voice mode and some other things that trying to do from browser is, from a technical perspective, not optimal."

### Navigation Structure Decided
1. **Bottom navigation** - 5 items max, reflects left sidebar from desktop
2. **Side drawer** - For utility items (profile, settings, help)
3. **No infinite scroll** - "This is not social media"

### Chrome Design Philosophy
Josh recommended keeping things simple:
> "Design a slick bottom navigation... represent that in no more than 5 with an option to open up."

---

## Information Architecture

### Terminology Discussion

| Term | Meaning | Decision |
|------|---------|----------|
| **Projects** | Structured work with tasks | Use this - more universal |
| **Topics** | Cross-sectional tagging | Keep as horizontal thread |
| **Outcomes** | Sophisticated, OKR-style | Appeals to specific market |
| **Strategies** | Long-term (2-4 quarters) | Too amorphous |

**Decision:** Use "Projects" because "Outcomes is sophisticated... I wish I lived in a world where everybody understood outcomes."

### Data Cube Concept
> "People, tasks, projects, outcomes... think of that like a data cube, and this is like a different way of slicing and dicing and showing that data with a different entry point."

### Home Page Purpose
- Priority of tasks across all projects
- Priority of delegated tasks across projects
- Surfacing what needs attention NOW

---

## Two Modes of User Interaction

### 1. Decision Maker Mode
- AI surfaces recommendations
- User says yes/no
- Quick decisions, minimal friction
- **AI is recommender, user is decision maker**

### 2. Collaborator Mode
- Working together toward a decision
- "Decision's not known yet, so we need to work together"
- User is driving, AI is companion
- Research, analysis, exploration

> "AI is not the decision maker, person is. AI is the recommender."

---

## AI Positioning - Left vs Right

### The Fovea Insight
> "Humans have an area of focus that's actually the size of a quarter, everything else is peripheral... Anything you put outside of that is initially out of sight, out of mind."

### The Debate
- **Right side (common pattern):** Copilot, Slack threads, etc.
- **Left side (Josh's recommendation):** For AI-centric tools where AI is central

### Key Insight
> "Is it just a browser extension sitting over on the right and sliding in, or is it more central to the work?"

**Nik's answer:** "More the latter... In an ideal world, I think would be great is if it was on equal footing."

### Conclusion
> "A lot of AI-centric tools right now, where AI is prominent, of course, it's on the left."

Priority Lens should put AI on equal footing, not as an afterthought.

---

## Creative Process Framework

Josh introduced a creative thinking framework that most tools ignore:

### The Phases
```
1. DIVERGENCE / BRAINSTORMING
   - Discovering without judgment
   - Playing around
   - Gathering information
   - Looking at information in different ways

2. PLANNING
   - Prioritization factors
   - Narrowing down approach

3. DECISION
   - Creates its own outcomes
   - Feeds back into new divergence
```

### The Problem
> "Jira, most tools... most tools don't design for [divergence] at all. Figma is not built for this. FigJam kind of is."

**Why this matters for Priority Lens:** AI can help with the divergence phase - gathering ideas, doing research - work that used to take hours.

---

## Proactive vs Reactive System

**Critical design philosophy:**

> "Ultimately, the goal is not that UI is the only way for people to interact. We wanted more proactive than reactive. If user needs to rely on UI, that's reactive. They have to come to my system. We want it the other way around."

The system should:
1. Proactively surface what needs attention
2. Bring users to UI only when needed
3. Show them exactly what they need to get work done

---

## Delegation Concept

### What It Means
- Tasks I delegate to others
- Tasks others delegate to me
- Tracking "is it moving forward?"

### The FYI Pattern
> "What needs my attention and I need to act on. And what is FYI for me?"

FYI items:
- Delegated tasks - just need to know they're moving
- Quick decisions - "Approve, approve, reject, reject"
- AI handling routine actions

---

## AI Parameters & Personalization

Nik proposed a concept of tunable AI:

### Task-Level Tuning
- Creativity scale
- Risk tolerance
- Mode selection (researcher, analyst, etc.)

### System-Level Memory
- Company context
- Personal preferences
- Historical patterns
- Like ChatGPT's personalization settings

> "We could put AI in a certain mode... We need a simpler way to do that, where they're not spending time on tuning the AI."

---

## Voice Mode as Primary

**Strong preference for voice:**

> "Voice mode is primary. I've been getting more and more into voice mode. Even for my coding stuff, I'm just talking to code now. Talking to Claude Code than typing. Cause then I can be unfiltered, raw. Versus when I'm starting to type, I'm rethinking myself."

### Voice Interview for Project Creation
Instead of filling out business canvases:
> "If it was like, alright, let's take an interview, let's talk about your project, and they are just giving us raw thoughts, we can start to synthesize that and create a canvas for them on the fly."

---

## People Management

### The Merge Problem
- Same person across multiple emails
- Different names/nicknames (Anki = Ankita = spouse)
- Context-aware identification ("Dan" in work vs personal context)

### AI Recommendations
- Surface people you didn't manually add
- Recommend merging duplicates
- "There's 3 Josh Smiths - do you want to combine them?"

---

## Context Preservation

### The Challenge
When interacting with lists and objects, users shouldn't lose their place:
> "Thinking about interactions that you can introduce to keep them from feeling like they're being taken away."

**Not everything needs a modal** - consider in-context editing.

---

## Tool Discussion

- **Zoom paid version** - Has built-in transcription, sends only to host
- **Otter** - Had issues with auto-sending to all participants
- **Google Tasks, Microsoft, Jira, Asana** - All have tasks, but the entry problem remains
- **Claude Code** - Nik uses voice mode for coding

---

## Shift in Engagement Model

Josh proposed a change in how they work together:

**Before:** Mid-fidelity Figma mockups
**After:** Brainstorming conversations with lo-fi/wireframe output

> "This is much less about what I had done, which was mid-fi, page to page, but more about having conversations like this and brainstorming. Coming to some conclusions about what things should look like and how they should interact."

**Cadence:** 2-3 times per week, mornings or evenings

---

## Key Quotes

On the value proposition:
> "It's like a light version of Priority Lens, right? Just because the hardest thing about any of this is not actually doing the task."

On habit formation:
> "Just because it's a really cool idea doesn't mean it will slide in easily to people's everyday use."

On design simplicity:
> "Coding's getting easier now... The human behavior part is always gonna be really difficult."

On Figma's future:
> "Figma's already tanking. Because they're realizing that the way we design needs to change. We can't think about it as page to page to page."

---

## Action Items

1. **Nik:** Continue building and dogfooding Heads Up
2. **Josh:** Lower fidelity of designs, focus on UX structure
3. **Both:** Work through the structures that make sense as a user
4. **Both:** Define how AI interacts within the system

---

## Meeting Scheduling

- Mornings: 9:30-11 work well (after daycare drop-off)
- Evenings: Generally flexible
- Weekends: Much easier for both
- Next week: Both traveling, Thursday return

---

## Key Takeaway

The meeting fundamentally shifted from "design pretty screens" to "design the right structures and interactions." This is about understanding how users think and how AI can truly serve them, not about pixel-perfect mockups.
