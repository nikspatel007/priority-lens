# Product Decisions

Track key product decisions with context and rationale.

## Format

```
### [YYYY-MM-DD] Decision Title

**Context:** Why this decision was needed
**Decision:** What was decided
**Rationale:** Why this choice was made
**Alternatives Considered:** Other options that were rejected
**Impact:** What this affects
```

---

## Decisions

### [2024-12-22] Typography System - Serif vs Sans Serif

**Context:** Needed to establish font usage rules for the app
**Decision:** Use Serif for AI transcription/headlines, Sans Serif for body/labels
**Rationale:** Serif fonts convey authority and trust for AI-generated content; Sans Serif is more readable for user input and UI labels
**Alternatives Considered:** Single font family throughout
**Impact:** All UI components, design system

### [2024-12-22] Voice Assistant Visual Design

**Context:** Initial prototype had complex orb with multiple visual layers creating "two orb" appearance
**Decision:** Simplified to single clean orb with AI icon from MGC Icon System
**Rationale:** Aligns with Priority Lens design principles - clean, uncluttered, professional
**Alternatives Considered:** Animated liquid blobs, waveform visualization
**Impact:** VoiceOrb component, brand perception
