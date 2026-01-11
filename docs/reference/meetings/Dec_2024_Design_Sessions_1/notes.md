# Design Session 1

**Date:** December 2024 (Midweek)
**Attendees:** Nik Patel, Josh Smith, Bo Motlagh (joined mid-meeting)
**Duration:** ~45 minutes
**Topic:** Competitor Analysis, Execution Mode Flow, Architecture Discussion

---

## Context

Josh still recovering from illness ("brain fog for over a week"). Bo Motlagh joined partway through the meeting after Nik invited him to contribute to the architecture discussion.

---

## Competitor Analysis

### Unnamed Competitor "Daily Brief" Product
Josh analyzed a competitor's approach:

**Structure:**
1. "Top of Mind" section
2. Action + time estimate + more details + action button
3. FYI section
4. "On Your Calendar" section

**Issues Identified:**
- Time estimates untrustworthy: "Things that used to take a month takes hours, things that used to take hours takes months"
- Reads like "strings of text" not "objects"
- Actions come after time (harder to read)
- Deals mixed into FYI section

**Priority Lens Differentiation:**
> "You're going well beyond this - it's coordinate scheduling, not 'add to calendar.'"

> Josh: "I was initially assuming what you were building, that these were objects that could be enhanced and have more complexity to them over time, and be connected to larger initiatives."

---

## Four Modes Confirmed

1. **Execution Mode** - "Get stuff done" - ONE ITEM AT A TIME
2. **Prep Mode** - What's coming up, briefs for upcoming work
3. **Eagle View** - What's going on across everything
4. **AI's Work** - "What is AI doing for you"

### Mode 4: AI's Work View
> "A to-do task list that AI itself is tracking and finishing... everything else is more for you and your teams and delegation."

Bo's reaction: "Sounds like a stand-up with me and Casey"

---

## Market Positioning

> Nik: "Now we are not even talking about emails or text, we are talking about work. And we are talking about a piece or unit of work that needs to get done, or that you need to know about."

> "It puts us in a different category. We are still in AI productivity category, but it's easier for us to not be compared against another email client."

---

## Architecture Discussion: Sandboxed Agents

### The Concept
Nik explained the technical approach:

1. Create tools for agents
2. Give agents a sandbox to execute
3. Context preserved for human conversation (not bloated by MCP tools)
4. Concurrent execution of multiple agents

**Why Sandbox?**
> "If I'm giving MCP into my system context, my system context gets bloated up pretty quickly, and I start to lose the quality of those results."

### Bo's Clarification
> Bo: "So you're basically... you want to spawn agents for tasks. You don't want a centralized system just doing everything for you."

> Nik: "Well, to the user, that's what it would look like. Behind the scene, it's gonna be different."

**Benefits:**
- Concurrency
- Different agents for different context types
- Isolated execution environments
- Ability to limit concurrent tasks for pricing tiers

---

## Execution Mode Flow

### The Complete Flow
1. **Onboarding completes** → Land on Execution Mode
2. **Show Action Required items** (one at a time)
3. **For each item:**
   - Here's the context
   - Here's what I recommend
   - Here's who's involved
   - "Do you want me to take care of it?"
   - User says yes → Move to next
4. **After Actions complete** → Show FYI items
5. **FYI section** → Links naturally to Prep Mode

### Example Flows

**Follow-up needed:**
> "This one needed a follow-up. No one has followed up, do you want me to? Yes, please go ahead. Done."

**Meeting scheduling:**
> "You need to schedule a meeting, you had said you would. Here are your options. I checked your times, I recommend 30 minutes meeting, here's the agenda. I can send 3 times out. Do you want to do that? Yes, now done with it."

**Complex task (like building a deck):**
> "Put a deck together for an important sales meeting tomorrow. You're not gonna finish that in 2 seconds. We'll block your calendar for tomorrow, it's gonna take 30 minutes, do you need more? Is there anything you want me to prep on before you get into it?"

---

## Action Required vs FYI Tagging

System should tag all incoming items:
- **Action Required** - Needs someone to do something
  - I need to take action
  - Someone else needs to take action (follow-up)
- **FYI** - Just need to know about it

> "Anything that has action required, identify who's supposed to take an action. Do I need to follow up? Or do I need to take an action myself as a user?"

---

## Trust Building & Review Toggle

### The Debate
> Josh: "I've worked for very wealthy people... they don't want to do anything. To be pampered and to not have to click would be nice."

> Josh: "You're going a step beyond 'write an email and let me review before you send' to 'just take care of the whole thing for me.' I think it's provocative, but I can't make a determination for you about whether the market is ready."

### Bo's Real-World Example
> "Google employees are doing this right now. The Google rep in our area - it's very obvious, but it's just good enough where you're like, I mean, that works. It's definitely clear when he's not writing those emails."

### Solution: Toggle Concept
> Bo: "Can this just be a toggle? If the toggle is switched on, it never lets it go without an official go."

> Nik: "It's the 'dangerously skip permissions flag' on Claude Code."

**Implementation options:**
1. Global toggle: "Review everything before sending"
2. Per-task choice: "Take care of it" vs "I want to review it"
3. Learning approach: Show drafts initially, learn user's style, then offer auto mode

---

## Personalization Through Use

### Learning Style Through Execution
> Nik: "Here's what I would type. Does this look good? Oh yeah, here's the changes I made. Okay, it seems like your style is this, is that correct? Yeah, do you want me to take care of it from here on?"

> "Which is onboarding without onboarding - it's execution plus personalization together."

### User Editable
Bo raised: "Is there a place they can edit that?"
- Like ChatGPT's narrative/personalization settings
- Need ability to change over time ("make me less rigid")
- "Stop telling everyone to ___ off"

---

## Draft Handling

Discussion on where unsent items live:

**Options:**
1. Save to email drafts (what Fixer/Superhuman do)
2. Keep in Priority Lens queue

> Nik: "We could just convert that into a draft and save it there and pull it from there."

**Fixer Lesson:**
> Bo: "Frank bailed on Fixer - he just couldn't wrap his head around its model. His messages kept disappearing because it kept reorganizing everything, and he hated it."

---

## Go-to-Market Pricing Idea

> Nik: "For $20, you only get 3 concurrent actions. If you have more than 3 rules, it's not happening right away."

> Bo: "To be fair, for 20 bucks, you could say you don't actually get any agent spawning. The centralized system can handle it."

---

## Key Quotes

On differentiation through UX:
> "If we can nail down the UX, then we know we are working towards building that UX. That's the part that's going to differentiate us from a stickiness perspective."

> "Trying not to compromise on the UX, and then figure out tech stuff after."

On landing experience:
> Josh: "The idea of hitting them up with execution mode immediately is to immediately demonstrate that value to them."

On trust in AI:
> Bo: "Google employees are doing this right now... there's gotta be a button somewhere where he's just basically saying 'handle that for me' and it does."

---

## Action Items

1. **Nik:** Send Bo the Google AI Studio link again
2. **Team:** Continue refining execution mode flow
3. **Team:** Decide on draft handling approach
4. **Team:** Define toggle/permissions model for auto-send
5. **Josh:** Design execution mode screens with object-based (not text-based) approach
