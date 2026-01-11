# Branding & Design System Meeting

**Date:** December 20, 2024 (Friday)
**Attendees:** Nik Patel, Josh Smith
**Duration:** ~60 minutes
**Topic:** Logo, Typography, Colors, Icon System, UI Framework

---

## Context

Nik is heading into a building weekend and wants concrete design decisions so he can implement them while coding. The goal is to get:
1. Logo direction
2. Colors to use
3. Font decisions
4. UI framework selection
5. Execution view screen finalized

Personal notes: Josh will be wrapping presents; Nik has a white elephant party with friends.

---

## Logo Exploration

### Josh's Explorations
Josh presented multiple logo variants he'd been playing with:

**Color Consolidation:**
- Took only 2 variant hues: 212 and 188
- 212 from Deep Navy
- Stratified across different tonal ranges
- "Cleans up the look and feel" by removing inconsistent hues

**Logo Variants Explored:**
1. **P shape** - Priority Lens → PL, but too similar to Evernote
2. **Yin-yang concept** - Balance, human-AI duality
3. **Comma shape** - Simplified down, "reduce the complexity"
4. **Double quote** - Looks like P&L, gets yin-yang in there
5. **Top quadrant only** - Just the runner section in a circle

### Preferred Directions
Nik liked:
- Last four variants
- First one
- Double quote P&L concept

**Design Intent:**
> "Somehow we want to bring that human-machine systems feeling in it."

### Logo Pitfalls to Avoid
Josh warned about unintended imagery in negative space:
- Bird shapes appearing
- Squid-like forms
- "Once you see it, you can't unsee it"

**Example:** Slack logo is "four ducks kissing each other's butts" - biggest designer complaint.

---

## Typography Decisions

### Type Scale Selection
Josh presented multiple type scales:

**Recommended (Simple):**
- H1, H2, H3, H4
- Title Large/Medium/Small
- Body Large/Medium/Small
- Labels

**Rejected (MUI-style):**
- Too many variants: caption, overline, subtitle, etc.
- "When you're at a startup, that's not something we should be caring about"

### Font Pairing Decision

**Serif Font** (Georgia family, system serifs)
Use for:
- AI transcription text
- Display headlines
- Titles (larger font sizes)
- Longer content that AI generates

**Sans Serif Font** (System fonts: SF Pro, Segoe UI, Roboto)
Use for:
- Labels
- Body/paragraph content
- Small font-size content
- Most UI elements

**Why This Pairing:**
> "That's becoming the trend in more conversational UIs - the AI is represented with serif."

**Font Recommendation:**
Use system fonts for sans-serif to feel native:
```css
font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
```

For serif:
```css
font-family: Georgia, 'Times New Roman', serif;
```

### Type Specifications
- Font weight: 100-900 scale (400 regular, 500-600 semi-bold, 700 bold)
- Line height: Specified per size
- Letter spacing: 0 for most
- Body medium: 16px (standard)
- Body large: 18px
- Body small: 12px

---

## Color Palette Approach

### Key Principles

**Limit Hues:**
> "Try and narrow it down to two colors. If you're gonna use all these variations of greens, just narrow it down to this one green, this one blue across those tonal ranges."

**Cool Tone Consistency:**
- Higher hue numbers = cooler (toward purple)
- Lower hue numbers = warmer
- Stick to consistent hue family

**Grayscale First:**
> "Feel free to build it fully in Grayscale. We can add back in color where color would be valuable."
> "The best use of color is few and far between. You don't want to blow everything up with color."

**Cool Grayscale:**
- Not pure gray (that's "too pure")
- Add slight cool tone that complements blue primary
- Matches the cool primary/secondary colors

---

## Icon System Selection

### Selected: MGC Icon System Pro

Josh shared his license with Nik.

**Recommended Styles:**
- **Cute Light** - Josh's go-to, extensive library
- **Cute Filled** - Slightly more prominent
- Light versions for approachability

**Why NOT Material UI or Font Awesome:**
> "They're so ubiquitous these days, they lack any sense of personality."

**Icon Style Guide:**
- Light = more approachable and soft
- Sharp = more professional, less approachable
- Two-tone = option for emphasis

---

## UI Framework Decision

### Options Evaluated

All three use NativeWind under the hood:

1. **Native Wind UI** - "Too familiar to the OS experience" - Rejected
2. **React Native Reusables** - "Very direct and clean, like GitHub" - heads-down coding feel
3. **GlueStack UI** - "More approachable" - **SELECTED**

### GlueStack UI

**Decision:** Use free version
**Pro version ($200):** Just templates, not needed for custom UI

**What it includes:**
- Pre-built components
- Type scale
- Color tokens (with grayscale)

**Why GlueStack:**
> "GlueStack is actually a bit more approachable."

---

## Three Modes Finalized

1. **Execution Mode** - Get it done
2. **Prep Mode** - Prepare for work
3. **Eagle View** - System overview

Icons to be selected for each from MGC Icon System.

---

## Implementation Plan

### Build Approach
1. Start with GlueStack UI base
2. Build screens in grayscale first
3. Add color intentionally where needed
4. Use system serif for AI content
5. Use system sans-serif for UI elements

### TestFlight Setup
Nik will set up TestFlight and invite Josh to test the app on his iPhone.

---

## Design System Summary for Claude Code Skill

```
TYPOGRAPHY:
- Serif: Georgia (AI transcription, headlines, titles)
- Sans Serif: System fonts (body, labels, UI elements)
- Scale: H1-H4, Title (lg/md/sm), Body (lg/md/sm), Labels

COLORS:
- Primary: Blue hue 212
- Secondary: Teal hue 188
- Grayscale: Cool-toned (not pure gray)
- Approach: Mostly grayscale, color used intentionally

ICONS:
- System: MGC Icon System Pro
- Style: Cute Light or Cute Filled
- Avoid: Material UI, Font Awesome (too common)

UI FRAMEWORK:
- GlueStack UI (React Native)
- Built on NativeWind
- Free version sufficient
```

---

## Quotes

On logo design:
> "This is just me playing around with how do I get as close as possible to what you have while setting some rules around it."

On color usage:
> "To be really intentional about where you place the color, and where you choose the color to go."

On font choices:
> "Intern is the new Helvetica... everyone and their mom uses it, so it's vanilla."

On icons:
> "I use MGC on a lot of my projects, only because it's the most extensive I've ever found. They've got something for everything."

---

## Action Items

1. **Nik:** Build over the weekend using these design decisions
2. **Nik:** Set up GlueStack UI in project
3. **Nik:** Create Claude Code frontend-design skill with guidelines
4. **Nik:** Set up TestFlight and invite Josh
5. **Josh:** Continue exploring logo variations
6. **Josh:** Potentially refine color palette
