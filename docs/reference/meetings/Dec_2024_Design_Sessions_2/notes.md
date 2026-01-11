# Design Session 2

**Date:** December 2024 (Midweek)
**Attendees:** Nik Patel, Josh Smith
**Duration:** ~45 minutes
**Topic:** Voice UI Review, Post-Onboarding UX, Color System, Logo Analysis

---

## Context

Josh still recovering from illness - went to doctor, confirmed bacterial infection, now on antibiotics. Nik had been up late (2 AM) building the Google AI Studio prototype.

Recording meeting for Claude Code processing:
> "I get transcript and I'm fitting transcript straight into Claude Code to be able to figure out decisions that we are making when we go through requirements gathering."

---

## Lenso Voice UI Prototype Review

### Live Demo
Josh tested the Lenso prototype directly in the meeting:
- Introduced himself as "Josh" (spelled N-I-K joke about Nik's spelling)
- Prototype asked communication preferences
- Asked for top 3 priorities

### How It Was Built
> "For what it's worth, I didn't write a single line of code. This was 2 in the morning... it's literally this prompt."

Built using Google AI Studio with prompts, then iterative fixes.

### What's Working
- Waveform visualization active and responsive
- Two distinct waveforms: user speaking vs Lenso speaking
- Conversation transcript at bottom "gives a good feeling"
- Onboarding flows in ~20 seconds naturally
- Multi-language support: "I asked it to talk in Gujarati and Hindi... it does start talking in that language just fine"

### Issues Identified

**AI Interruption:**
> Josh: "It was totally interrupting me while I was talking... that kills me."
> "I would rather just be like, when you're done, say 'I'm done' or tap the I'm done button."

**Transcription Issues:**
- Name spellings: "Nick" always goes to NICK, Anki spelled as "Ankit"
- Accent affects transcription display
- Gemini thinks user is speaking different language sometimes

**Solution for Names:**
> "Once they login for real, I'm gonna get the profile name from their user profile... I have captured your name as that, is that what you want me to call you?"

### Voice-First Preference

> Josh: "I like it, and I prefer it over typing out. I would rather talk to a computer all day to get my stuff done."

> Nik: "I can think faster, I can talk faster than I can type. For a productivity app, that's worth it."

> Josh: "The way I speak is completely different from the way I type. Some people say I'm like a different person - much more serious when I type."

### Future Vision
Josh described his ideal:
> "I look forward to the day when my computer can just say 'hey, you got an email from this vendor asking you for an update' and I can just be like 'okay, can you just connect with Hannah and ask her to send me XYZ, let me know when I get it, I'd like to review it.'"

> "That takes less than a minute for me to do, but for me to coordinate all of that on the computer takes much time."

---

## Brain-Computer Interface Tangent

Both expressed interest in BCI technology:

> Nik: "I'm a big proponent of brain-computer interfaces. It's very infancy right now."

**Current landscape:**
- Neuralink: Invasive (requires skull surgery)
- Meta FAIR: Uses fMRI to map thoughts while typing
- Meta wristband: Gesture detection via muscle movement
- Consumer EEG devices already exist

Nik's experience:
> "I had built this software 4 years ago using EEG... you would think 'go right, go left, go up, go down' and then it would capture that and start moving."

---

## Post-Onboarding UX Priority

### Strategic Decision
UX before marketing:
> "If we can get 5 to 10 people to be addicted to the product, marketing will get much, much easier. We know what we are doing, we know what people like and don't like."

> "Then it's like talking the reality, not making up stuff."

### Urgency
> "Speed matters, just because I'm seeing other companies starting to get into it. At least we want to put ourselves out there."

### What Needs to Happen
1. Build full working app with Google login
2. Capture and process emails
3. Voice assistant for directing actions
4. Convert emails to to-do list with basic actions

> "That will put us ahead of anyone that I have seen so far."

---

## Color Palette Analysis

### Current State
Josh analyzed the brand colors and found issues:

**Hue Values:**
- Deep Navy: 212
- Secondary color: 188
- Another blue: 178
- Teal: 174

> Josh: "These are all over the place."

### The Problem
> "These aren't very distinct from one another."

Different hue values mean colors don't harmonize:
- Warm blues and cool blues don't mix well
- Affects all colors thereafter

### Cool vs Warm Grayscale Discussion
> "Most grayscales are not fully gray - they have either cool or warm tones."

Examples:
- ChatGPT: Cool grayscale
- Slack: More cool
- Claude: "Way on the warm end"

### Typography: Inter Font
> "Inter is the new Helvetica - the most neutral font choice."

---

## Logo Design Analysis

### Current Logo Intent (Nik's Explanation)
1. **Human-AI collaboration** - Two eyes working together
2. **Radar element** - Left eye has radar, "looking out for you"
3. **Cyborg feeling** - "It's a system after all"
4. **Companion** - "Feel like you're talking to a natural thing, not a terminator"
5. **App icon ready** - Can serve as both logo and icon

Created in Canva AI after multiple iterations.

### Josh's Observations
Colors are inconsistent between logo elements:
- Both navies are 212 hue (consistent)
- But other colors vary widely

---

## Review of Previous Meeting Notes

Josh recapped Friday's discussion:
- Rules vs Agents (one simple, one complex - could be same to user)
- Value articulation in marketing/branding
- Daily/weekly summary feature
- 5-minute start/end of week conversations
- Managing multiple emails and calendars
- Future marketplace for customer-built agents
- Design principles: perception of privacy, product speaks for itself

---

## Team Coordination

### Paula's Work
- Building website with "day in life using Priority Lens" content
- Working on messaging (not visual branding)
- Brand guidelines PDF is from Paula

### Josh's Approach Options
> "Dialing it up: I can get into the weeds - details in color, spacing, type scale, psychology of things."
> "Dialing it down: just tell me what to do."

Nik's preference:
> "I'm more leaning towards the former... still in favor of time, I'd lean on you to say 'go with this, we'll be okay.'"

---

## Key Quotes

On prototyping speed:
> Josh: "Holy moly, the future is here."

On design iteration:
> Nik: "Sooner we get to working product, the better."

On color expertise:
> Nik: "The only thing I can say is when I look at it, what I feel about it. I don't have the technical expertise to say this makes sense or not."

---

## Key Decisions

1. **Post-onboarding UX is top priority** - nail this before marketing
2. **Speed matters** - competitors entering market
3. **Goal: 5-10 addicted users** - then marketing becomes easier
4. **Voice-first confirmed** - both prefer talking over typing
5. **Color palette needs consolidation** - hues too scattered

---

## Action Items

1. **Nik:** Build working app with Google login
2. **Josh:** Consolidate color palette to consistent hue values
3. **Josh:** Refine logo with proper mathematical structure
4. **Both:** Define post-onboarding UX flow
5. **Nik:** Share Paula's "day in life" content with Josh when ready
