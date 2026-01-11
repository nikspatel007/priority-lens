# Holiday Follow-up Session

**Date:** December 22, 2024 (Sunday)
**Attendees:** Nik Patel, Josh Smith
**Duration:** ~30 minutes
**Topic:** Companion Mode Concept, Voice/Text Toggle, Frontend Progress

---

## Context

Josh is in "holiday mode" - literally sitting by the fire, mostly checked out for the holidays. Quick sync to keep momentum going.

### Availability
- Josh: Monday, Tuesday, Friday next week
- Not available Wednesday/Thursday
- Nik continuing to build during quiet holiday period

---

## Progress Update

### Frontend-Design Skill for Claude Code
> "I have Claude building something right now... I built a frontend-design skill that basically enforces - well, not enforces, but guides the typography and font and everything."

- Using GlueStack UI as discussed
- Attempting to replicate Google AI Studio prototype
- Applying Priority Lens colors to the design

### Current Status
Nik didn't get as far as hoped over weekend:
> "I didn't go as far as I wanted to over the weekend. Just some other stuff came up."

---

## Previous Decisions Confirmed

Josh recapped where they left off:
- Execution mode concept is "as minimal as possible"
- Still need Prep and Eagle views
- "Much more beyond that" to explore

---

## New Concept: Companion Mode

Nik introduced a new idea he'd been thinking about:

### What Is Companion Mode?
> "You're working on something together. We are sharing the same context, working on the same thing, looking at same files, I'm working on something, and I'm asking you to do something."

**Key characteristic:** Shared context between human and AI

### Connection to Canvas Mode
> "You know how we had talked about the canvas mode? Where we could create a canvas and AI's doing something, human's doing something, they're collaborating together."

### Open Questions
- How does it differ from voice mode?
- What instances would companion mode be helpful?
- How does it work on a small screen?
- Is it just "GPT but designed differently"?

Josh's concern:
> "Is it just me dropping files in and typing some stuff while the companion puts some ideas in? Is it suddenly just GPT but designed differently?"

Nik's vision:
> "The experience, probably in my mind, is we are sharing the same context, we are working on the same thing."

**Decision:** Park this for now, focus on Prep and Eagle modes first.

---

## Voice Mode vs Text Mode Toggle

### The Problem
What if voice isn't available?
> "What if voice mode is not available? I'm in a silent room, or working in a library, or it's not something I can loudly talk about. It's a sensitive matter of some kind."

### Priority Lens Difference from ChatGPT
> "ChatGPT has these things where everything being said is also being written down in text... That's easy because it's a chat interface."

> "For us, that's not the primary thing. For us, the primary thing is work, so our screen asset is very limited."

### Toggle Design Options
Josh sketched several concepts:

**Option 1: Inline Toggle**
- Keyboard icon next to voice indicators
- Clicking switches to text input
- Text box appears inline

**Option 2: Expand Above**
- Text input expands above navigation
- Keeps navigation visible

**Option 3: Expand Below**
- Text input appears below navigation
- Nav stays "always present" across modes

### ChatGPT Comparison
> Josh: "In ChatGPT, when you go to voice mode, it's secondary, and you close it to leave it. I don't prefer ChatGPT's method where you click it and it puts you in an experience almost like a modal."

> "To me, it's like a toggle. Do you transition into typing only, or do you transition into talking?"

---

## Technical Considerations

### Mobile vs Desktop
> "This is ideal for mobile. For desktop, you can move it up there."

### Keyboard Conflict
Nik raised concern about bottom navigation:
> "The menu at the bottom, which is also clickable, could get confusing for someone to click that and all of a sudden lose the thing you were doing."

---

## Key Decisions

1. **Companion mode** - Interesting concept, needs more thought
2. **Voice/Text toggle** - Needed, should feel like a toggle not a modal
3. **Frontend skill** - Working, guiding Claude Code development
4. **Focus order** - Prep and Eagle modes before Companion mode

---

## Action Items

1. **Nik:** Continue building frontend with Claude Code
2. **Nik:** Think through companion mode examples
3. **Josh:** Design Prep and Eagle mode concepts
4. **Both:** Revisit voice/text toggle design
5. **Both:** Meet Monday/Tuesday to continue
