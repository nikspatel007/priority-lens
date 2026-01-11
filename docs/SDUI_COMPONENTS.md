# SDUI Component System

Server-Driven UI component architecture for Priority Lens.

## Design Principles

1. **Primitives First** - Base components are atomic, composable
2. **Grid-Based Layout** - CSS Grid-like system for responsive positioning
3. **Type-Safe Schema** - JSON schema validated on both server and client
4. **Action-Driven** - Components emit actions, server handles logic

---

## Schema Structure

```typescript
interface UIBlock {
  id: string;                    // Unique identifier
  type: string;                  // Component type
  props?: Record<string, any>;   // Component-specific props
  layout?: LayoutProps;          // Grid/positioning
  children?: UIBlock[];          // Nested components
  actions?: ActionProps[];       // Interactive behaviors
}

interface LayoutProps {
  grid?: {
    columns?: number | string;   // "1fr 2fr" or 3
    rows?: number | string;
    gap?: number;
    areas?: string[];            // ["header header", "sidebar main"]
  };
  gridArea?: string;             // "header" or "1 / 1 / 2 / 3"
  span?: { col?: number; row?: number };
  padding?: number | number[];   // [top, right, bottom, left]
  margin?: number | number[];
  flex?: number;
  align?: 'start' | 'center' | 'end' | 'stretch';
  justify?: 'start' | 'center' | 'end' | 'between' | 'around';
  width?: number | string;       // 100 or "50%"
  height?: number | string;
  minHeight?: number;
  maxWidth?: number;
}

interface ActionProps {
  trigger: 'press' | 'longPress' | 'change' | 'submit';
  type: string;                  // Action type identifier
  payload?: Record<string, any>; // Data to send
  navigate?: string;             // Screen navigation
  haptic?: 'light' | 'medium' | 'heavy';
}
```

---

## Component Hierarchy

### Level 1: Primitives (Atoms)

These are the building blocks. Cannot be decomposed further.

| Component | Props | Description |
|-----------|-------|-------------|
| `text` | `value`, `variant`, `color`, `weight`, `align` | Typography |
| `icon` | `name`, `size`, `color` | SF Symbols / Material icons |
| `image` | `src`, `alt`, `aspectRatio`, `fit` | Images |
| `spacer` | `size` | Empty space |
| `divider` | `color`, `thickness` | Horizontal/vertical line |
| `badge` | `value`, `variant` | Status indicators |
| `avatar` | `src`, `name`, `size`, `fallback` | User avatars |
| `progress` | `value`, `max`, `variant` | Progress indicators |

### Level 2: Inputs (Interactive Atoms)

| Component | Props | Description |
|-----------|-------|-------------|
| `button` | `label`, `variant`, `icon`, `loading`, `disabled` | Tap actions |
| `iconButton` | `icon`, `variant`, `size` | Icon-only buttons |
| `textInput` | `placeholder`, `value`, `multiline`, `keyboard` | Text entry |
| `checkbox` | `checked`, `label` | Boolean toggle |
| `switch` | `value`, `label` | On/off toggle |
| `select` | `options`, `value`, `placeholder` | Dropdown |
| `slider` | `value`, `min`, `max`, `step` | Range input |
| `datePicker` | `value`, `mode`, `minDate`, `maxDate` | Date selection |

### Level 3: Layout (Containers)

| Component | Props | Description |
|-----------|-------|-------------|
| `box` | `layout` | Generic container with layout |
| `stack` | `direction`, `gap`, `align` | Flex stack (vstack/hstack) |
| `grid` | `columns`, `rows`, `gap`, `areas` | CSS Grid container |
| `scroll` | `direction`, `showIndicator` | Scrollable area |
| `card` | `variant`, `elevated` | Styled container |
| `modal` | `visible`, `title` | Overlay container |
| `sheet` | `visible`, `snapPoints` | Bottom sheet |

### Level 4: Composite (Molecules)

Built from primitives and layouts.

| Component | Props | Description |
|-----------|-------|-------------|
| `listItem` | `title`, `subtitle`, `leading`, `trailing` | List row |
| `personCard` | `name`, `title`, `email`, `avatar` | Contact display |
| `taskCard` | `title`, `status`, `priority`, `dueDate` | Task display |
| `invoiceCard` | `vendor`, `amount`, `dueDate`, `status` | Invoice display |
| `calendarEvent` | `title`, `start`, `end`, `location` | Event display |
| `attachment` | `filename`, `type`, `size`, `url` | File attachment |
| `actionItem` | `text`, `checked`, `assignee` | Checklist item |
| `metric` | `label`, `value`, `change`, `trend` | KPI display |
| `timeline` | `items` | Vertical timeline |
| `header` | `title`, `subtitle`, `actions` | Section header |

### Level 5: Templates (Organisms)

Pre-built layouts for common task types.

| Template | Use Case |
|----------|----------|
| `paymentView` | Invoice payment flow |
| `meetingPrep` | Meeting preparation |
| `emailReply` | Email response |
| `taskDetail` | Generic task detail |
| `peopleList` | Contact list |
| `scheduleView` | Calendar/schedule |
| `summaryView` | AI-generated summary |

---

## Layout Examples

### Grid Layout

```json
{
  "type": "grid",
  "layout": {
    "grid": {
      "columns": "1fr 2fr",
      "rows": "auto 1fr auto",
      "gap": 16,
      "areas": [
        "avatar header",
        "avatar content",
        "footer footer"
      ]
    }
  },
  "children": [
    {
      "type": "avatar",
      "layout": { "gridArea": "avatar" },
      "props": { "src": "...", "size": 64 }
    },
    {
      "type": "stack",
      "layout": { "gridArea": "header" },
      "props": { "direction": "vertical", "gap": 4 },
      "children": [
        { "type": "text", "props": { "value": "John Doe", "variant": "heading" } },
        { "type": "text", "props": { "value": "CEO, Acme Inc", "variant": "caption" } }
      ]
    },
    {
      "type": "text",
      "layout": { "gridArea": "content" },
      "props": { "value": "Meeting notes..." }
    },
    {
      "type": "stack",
      "layout": { "gridArea": "footer" },
      "props": { "direction": "horizontal", "gap": 8, "justify": "end" },
      "children": [
        { "type": "button", "props": { "label": "Decline", "variant": "outline" } },
        { "type": "button", "props": { "label": "Accept", "variant": "primary" } }
      ]
    }
  ]
}
```

### Responsive Stack

```json
{
  "type": "stack",
  "props": { "direction": "vertical", "gap": 16 },
  "layout": { "padding": [16, 16, 16, 16] },
  "children": [
    {
      "type": "header",
      "props": { "title": "Your Task", "subtitle": "Pay Invoice" }
    },
    {
      "type": "invoiceCard",
      "props": {
        "vendor": "Dunder Mifflin",
        "description": "A giant ream of paper",
        "amount": "$999",
        "dueDate": "Wednesday",
        "status": "pending"
      }
    },
    {
      "type": "stack",
      "props": { "direction": "horizontal", "gap": 12, "justify": "center" },
      "children": [
        {
          "type": "button",
          "props": { "label": "Pay Now", "variant": "primary" },
          "actions": [{ "trigger": "press", "type": "pay_invoice", "payload": { "id": "inv_123" } }]
        },
        {
          "type": "button",
          "props": { "label": "Deal with it later", "variant": "outline" },
          "actions": [{ "trigger": "press", "type": "snooze_task" }]
        }
      ]
    }
  ]
}
```

---

## Task Type → UI Mapping

| Task Type | Primary Components | Layout |
|-----------|-------------------|--------|
| `payment` | `invoiceCard`, `button` | Stack with action footer |
| `meeting_prep` | `calendarEvent`, `personCard[]`, `actionItem[]` | Grid with sidebar |
| `email_reply` | `personCard`, `text`, `textInput` | Stack |
| `document_review` | `attachment`, `actionItem[]` | Tabs + list |
| `follow_up` | `timeline`, `personCard`, `button` | Stack |
| `deadline` | `metric`, `progress`, `actionItem[]` | Grid with stats |

---

## Action Types

Actions are emitted when users interact with components.

| Action Type | Payload | Description |
|-------------|---------|-------------|
| `navigate` | `{ screen, params }` | Navigate to screen |
| `submit_form` | `{ formData }` | Submit form data |
| `pay_invoice` | `{ invoiceId }` | Process payment |
| `snooze_task` | `{ taskId, duration }` | Delay task |
| `complete_task` | `{ taskId }` | Mark complete |
| `open_url` | `{ url }` | Open external link |
| `call` | `{ phoneNumber }` | Initiate call |
| `email` | `{ to, subject }` | Compose email |
| `share` | `{ content }` | Share sheet |
| `voice_command` | `{ command }` | Send to Lenso |

---

## Streaming Protocol

UI blocks are streamed as canonical events over LiveKit data channel.

```typescript
// Event types
type UIEvent =
  | { type: 'ui.block.add', block: UIBlock }
  | { type: 'ui.block.update', id: string, changes: Partial<UIBlock> }
  | { type: 'ui.block.remove', id: string }
  | { type: 'ui.clear' }
  | { type: 'ui.action.result', actionId: string, result: any }
```

---

## Implementation Checklist

### Backend (Python/LangGraph)
- [ ] Define UIBlock Pydantic models
- [ ] Create `generate_ui` tool for agent
- [ ] Task type → template mapping
- [ ] Stream UI events via LiveKit
- [ ] Handle action events from client

### Frontend (React Native)
- [ ] Create primitive components
- [ ] Create layout components (Grid, Stack)
- [ ] Create composite components
- [ ] Build SDUIRenderer
- [ ] Action handler system
- [ ] Animation/transitions

### Schema
- [ ] JSON Schema validation
- [ ] TypeScript types (shared)
- [ ] Zod schemas for runtime validation
