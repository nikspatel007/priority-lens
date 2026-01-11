# Phase 5: Priority Lens Mobile App - Iteration Specifications

## Overview

**Approach**: PORT the pl-app-react-native frontend to work with the Priority Lens backend.

We are **not** building from scratch. The pl-app-react-native has:
- Comprehensive design system (colors, typography, spacing)
- 35+ SDUI components
- Voice UI with LiveKit
- Auth with Clerk + Google OAuth
- ~11,874 lines of production code

What pl-app-react-native **lacks**:
- Tests (0% coverage currently)
- Integration with Priority Lens backend APIs

**Our goal**: Port the frontend + add 100% test coverage + connect to Priority Lens APIs.

**Location**: `apps/mobile/` (monorepo with priority-lens backend)

---

## Testing Strategy

### Framework Stack

| Layer | Tool | Purpose |
|-------|------|---------|
| Unit Tests | Jest + React Native Testing Library | Component & logic testing |
| Integration Tests | Jest + MSW (Mock Service Worker) | API integration testing |
| E2E Tests | Detox | Full user flow testing |
| Coverage | Jest --coverage | 100% required |

### Test File Conventions
```
src/
├── context/
│   ├── AuthContext.tsx
│   └── __tests__/
│       └── AuthContext.test.tsx
├── sdui/
│   ├── SDUIRenderer.tsx
│   └── __tests__/
│       └── SDUIRenderer.test.tsx
├── services/
│   ├── api.ts
│   └── __tests__/
│       └── api.test.ts
```

### Coverage Requirements
- **Statements**: 100%
- **Branches**: 100%
- **Functions**: 100%
- **Lines**: 100%

### Testing Dependencies
```json
{
  "devDependencies": {
    "jest": "^29.7.0",
    "jest-expo": "~52.0.0",
    "@testing-library/react-native": "^12.4.0",
    "@testing-library/jest-native": "^5.4.3",
    "msw": "^2.0.0",
    "detox": "^20.0.0",
    "@types/jest": "^29.5.0"
  }
}
```

---

## Iteration 1: Project Setup & Testing Infrastructure ✅ COMPLETE

### Story
As a developer, I need a properly configured Expo project with testing infrastructure so that I can port the pl-app frontend with 100% test coverage.

### Success Criteria
- [x] Expo project initialized in `apps/mobile/`
- [x] All production dependencies installed (from pl-app)
- [x] Jest + React Native Testing Library configured
- [x] MSW configured for API mocking
- [ ] Detox configured for E2E tests (deferred to Iteration 8)
- [x] Coverage threshold set to 100%
- [ ] CI workflow for tests (pending)
- [x] TypeScript strict mode with path aliases
- [x] Design system ported from pl-app

### Deliverables

1. **Copy from pl-app-react-native**
   - `src/theme/index.ts` - Full design system
   - `app.json` - Expo config with plugins
   - `babel.config.js` - Reanimated plugin
   - `tsconfig.json` - Strict TypeScript

2. **Jest Configuration** (`jest.config.js`)
   ```javascript
   module.exports = {
     preset: 'jest-expo',
     setupFilesAfterEnv: [
       '@testing-library/jest-native/extend-expect',
       './jest.setup.js'
     ],
     transformIgnorePatterns: [
       'node_modules/(?!((jest-)?react-native|@react-native(-community)?)|expo(nent)?|@expo(nent)?/.*|@expo-google-fonts/.*|react-navigation|@react-navigation/.*|@unimodules/.*|unimodules|sentry-expo|native-base|react-native-svg|@clerk/.*|@livekit/.*)'
     ],
     collectCoverage: true,
     collectCoverageFrom: [
       'src/**/*.{ts,tsx}',
       '!src/**/*.d.ts',
       '!src/**/__tests__/**'
     ],
     coverageThreshold: {
       global: {
         statements: 100,
         branches: 100,
         functions: 100,
         lines: 100
       }
     },
     moduleNameMapper: {
       '^@/(.*)$': '<rootDir>/src/$1'
     }
   };
   ```

3. **Jest Setup** (`jest.setup.js`)
   ```javascript
   import '@testing-library/jest-native/extend-expect';
   import { server } from './src/mocks/server';

   beforeAll(() => server.listen());
   afterEach(() => server.resetHandlers());
   afterAll(() => server.close());

   // Mock react-native-reanimated
   jest.mock('react-native-reanimated', () => require('react-native-reanimated/mock'));

   // Mock expo-secure-store
   jest.mock('expo-secure-store', () => ({
     getItemAsync: jest.fn(),
     setItemAsync: jest.fn(),
     deleteItemAsync: jest.fn(),
   }));
   ```

4. **MSW Setup** (`src/mocks/handlers.ts`)
   ```typescript
   import { http, HttpResponse } from 'msw';

   const API_URL = 'http://localhost:8000';

   export const handlers = [
     http.get(`${API_URL}/api/v1/inbox`, () => {
       return HttpResponse.json({
         emails: [],
         total: 0,
         limit: 20,
         offset: 0,
         has_more: false,
       });
     }),
     // ... more handlers
   ];
   ```

5. **Detox Configuration** (`detox.config.js`)
   ```javascript
   module.exports = {
     testRunner: {
       args: { $0: 'jest', config: 'e2e/jest.config.js' },
       jest: { setupTimeout: 120000 }
     },
     apps: {
       'ios.debug': {
         type: 'ios.app',
         binaryPath: 'ios/build/Build/Products/Debug-iphonesimulator/PriorityLens.app',
         build: 'xcodebuild -workspace ios/PriorityLens.xcworkspace -scheme PriorityLens -configuration Debug -sdk iphonesimulator -derivedDataPath ios/build'
       }
     },
     devices: {
       simulator: { type: 'ios.simulator', device: { type: 'iPhone 15' } }
     },
     configurations: {
       'ios.sim.debug': { device: 'simulator', app: 'ios.debug' }
     }
   };
   ```

6. **GitHub Actions** (`.github/workflows/mobile-tests.yml`)
   ```yaml
   name: Mobile Tests
   on: [push, pull_request]
   jobs:
     test:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v4
         - uses: actions/setup-node@v4
           with: { node-version: '20' }
         - run: cd apps/mobile && npm ci
         - run: cd apps/mobile && npm run typecheck
         - run: cd apps/mobile && npm run lint
         - run: cd apps/mobile && npm run test -- --coverage
         - name: Check 100% coverage
           run: |
             cd apps/mobile
             coverage=$(cat coverage/coverage-summary.json | jq '.total.lines.pct')
             if [ "$coverage" != "100" ]; then
               echo "Coverage is $coverage%, required 100%"
               exit 1
             fi
   ```

### Test Cases (Iteration 1)

| ID | Test | Type | Expected Result |
|----|------|------|-----------------|
| 1.1 | `npm install` succeeds | Manual | All deps installed |
| 1.2 | `npm run typecheck` passes | CI | No TypeScript errors |
| 1.3 | `npm run lint` passes | CI | No ESLint errors |
| 1.4 | `npm run test` runs | CI | Jest executes |
| 1.5 | Jest finds test files | CI | At least 1 test discovered |
| 1.6 | MSW mocks API calls | Unit | Mock responses returned |
| 1.7 | Coverage report generates | CI | coverage/ directory created |
| 1.8 | `npx expo start` works | Manual | Dev server starts |
| 1.9 | Theme file loads | Unit | Colors, spacing exported |
| 1.10 | Path aliases work | Unit | `@/theme` imports correctly |

### Verification
```bash
# 1. Run tests
npm run test

# 2. Check coverage
npm run test -- --coverage
cat coverage/coverage-summary.json

# 3. Verify Expo works
npx expo start --ios
```

---

## Iteration 2: Authentication - Clerk (with Tests) ✅ COMPLETE

### Story
As a user, I need to sign in with Clerk so that my identity is verified and I can access my personalized data.

### Success Criteria
- [x] AuthContext ported from pl-app
- [x] SignInScreen ported from pl-app
- [x] 100% test coverage for AuthContext
- [x] 100% test coverage for SignInScreen
- [x] JWT tokens accessible for API calls
- [x] Auth state persists across app restarts

### Deliverables

1. **Port from pl-app**
   - `src/context/AuthContext.tsx`
   - `src/screens/SignInScreen.tsx`

2. **AuthContext Tests** (`src/context/__tests__/AuthContext.test.tsx`)
   ```typescript
   import { render, waitFor } from '@testing-library/react-native';
   import { AuthProvider, useAuthContext } from '../AuthContext';

   // Mock Clerk
   jest.mock('@clerk/clerk-expo', () => ({
     useAuth: jest.fn(),
     useUser: jest.fn(),
     ClerkProvider: ({ children }) => children,
     ClerkLoaded: ({ children }) => children,
   }));

   describe('AuthContext', () => {
     it('provides initial loading state', () => {
       const { useAuth } = require('@clerk/clerk-expo');
       useAuth.mockReturnValue({ isLoaded: false, isSignedIn: false });

       const TestComponent = () => {
         const { isLoading } = useAuthContext();
         return <Text testID="loading">{isLoading.toString()}</Text>;
       };

       const { getByTestId } = render(
         <AuthProvider><TestComponent /></AuthProvider>
       );

       expect(getByTestId('loading')).toHaveTextContent('true');
     });

     it('provides user when signed in', async () => { /* ... */ });
     it('provides null user when signed out', async () => { /* ... */ });
     it('signOut clears user state', async () => { /* ... */ });
     it('getToken returns JWT string', async () => { /* ... */ });
     it('sets auth token getter for API', async () => { /* ... */ });
   });
   ```

3. **SignInScreen Tests** (`src/screens/__tests__/SignInScreen.test.tsx`)
   ```typescript
   import { render, fireEvent } from '@testing-library/react-native';
   import { SignInScreen } from '../SignInScreen';

   jest.mock('@clerk/clerk-expo', () => ({
     SignIn: ({ children }) => <View testID="clerk-signin">{children}</View>,
   }));

   describe('SignInScreen', () => {
     it('renders Clerk SignIn component', () => {
       const { getByTestId } = render(<SignInScreen />);
       expect(getByTestId('clerk-signin')).toBeTruthy();
     });

     it('applies correct styling', () => { /* ... */ });
     it('handles keyboard correctly', () => { /* ... */ });
   });
   ```

### Test Cases (Iteration 2)

| ID | Test | Type | Expected Result |
|----|------|------|-----------------|
| 2.1 | AuthContext provides isLoading initially | Unit | isLoading = true |
| 2.2 | AuthContext provides isSignedIn after load | Unit | isSignedIn matches Clerk |
| 2.3 | AuthContext provides user object | Unit | user has id, email |
| 2.4 | AuthContext.signOut() works | Unit | isSignedIn becomes false |
| 2.5 | AuthContext.getToken() returns JWT | Unit | Returns string token |
| 2.6 | SignInScreen renders Clerk UI | Unit | Clerk component visible |
| 2.7 | SignInScreen handles errors | Unit | Error state displays |
| 2.8 | Token getter set on auth load | Unit | setAuthTokenGetter called |
| 2.9 | Auth persists across restart | Integration | Clerk restores session |
| 2.10 | Navigation guards work | Integration | Redirect when not signed in |

---

## Iteration 3: Authentication - Google OAuth (with Tests) ✅ COMPLETE

### Story
As a user, I need to connect my Google account so that Priority Lens can access my Gmail and Calendar data.

### Success Criteria
- [x] GoogleContext ported from pl-app
- [x] googleAuth service ported from pl-app
- [x] LandingScreen ported from pl-app
- [x] 100% test coverage for GoogleContext
- [x] 100% test coverage for googleAuth
- [x] Backend token sync working
- [x] SyncProgressScreen for initial sync flow (bonus)

### Deliverables

1. **Port from pl-app**
   - `src/context/GoogleContext.tsx`
   - `src/services/googleAuth.ts`
   - `src/screens/LandingScreen.tsx`

2. **GoogleContext Tests** (`src/context/__tests__/GoogleContext.test.tsx`)
   ```typescript
   import { render, act, waitFor } from '@testing-library/react-native';
   import { GoogleProvider, useGoogle } from '../GoogleContext';
   import * as googleAuth from '@/services/googleAuth';

   jest.mock('@/services/googleAuth');
   jest.mock('@/services/api');

   describe('GoogleContext', () => {
     beforeEach(() => {
       jest.clearAllMocks();
     });

     it('initializes with loading state', () => { /* ... */ });
     it('attempts silent sign-in on mount', async () => { /* ... */ });
     it('updates state after successful connect', async () => { /* ... */ });
     it('syncs token to backend after connect', async () => { /* ... */ });
     it('handles connection errors gracefully', async () => { /* ... */ });
     it('disconnect clears state', async () => { /* ... */ });
     it('refreshToken returns new token', async () => { /* ... */ });
   });
   ```

3. **googleAuth Tests** (`src/services/__tests__/googleAuth.test.ts`)
   ```typescript
   import { GoogleSignin } from '@react-native-google-signin/google-signin';
   import {
     configureGoogleSignIn,
     signInWithGoogle,
     silentSignIn,
     signOutGoogle,
   } from '../googleAuth';

   jest.mock('@react-native-google-signin/google-signin');

   describe('googleAuth', () => {
     describe('configureGoogleSignIn', () => {
       it('configures with correct client IDs', () => { /* ... */ });
       it('includes correct scopes', () => { /* ... */ });
       it('enables offline access', () => { /* ... */ });
     });

     describe('signInWithGoogle', () => {
       it('returns success with user and tokens', async () => { /* ... */ });
       it('handles cancelled sign-in', async () => { /* ... */ });
       it('handles play services error', async () => { /* ... */ });
     });

     describe('silentSignIn', () => {
       it('restores previous session', async () => { /* ... */ });
       it('returns error when no session', async () => { /* ... */ });
     });
   });
   ```

### Test Cases (Iteration 3)

| ID | Test | Type | Expected Result |
|----|------|------|-----------------|
| 3.1 | GoogleContext provides loading state | Unit | isLoading initially true |
| 3.2 | Silent sign-in attempted on mount | Unit | silentSignIn() called |
| 3.3 | connect() calls signInWithGoogle | Unit | Google dialog would open |
| 3.4 | Successful connect updates state | Unit | isConnected = true |
| 3.5 | Token synced to backend | Unit | storeGoogleToken() called |
| 3.6 | Connection error handled | Unit | Error state set |
| 3.7 | disconnect() clears state | Unit | isConnected = false |
| 3.8 | LandingScreen renders button | Unit | "Connect Google" visible |
| 3.9 | Button triggers connect() | Unit | onPress calls connect |
| 3.10 | Scopes include gmail.readonly | Unit | GOOGLE_SCOPES correct |

---

## Iteration 4: API Service Layer (with Tests) ✅ COMPLETE

### Story
As a developer, I need a typed API client so that the app can communicate with the Priority Lens backend.

### Success Criteria
- [x] API service created matching Priority Lens endpoints
- [x] MSW handlers for all endpoints
- [x] 100% test coverage for api.ts
- [x] Types match Priority Lens Pydantic schemas
- [x] Error handling tested

### Deliverables

1. **Create** (adapted from pl-app patterns)
   - `src/services/api.ts` - Full API client
   - `src/types/api.ts` - Type definitions

2. **API Tests** (`src/services/__tests__/api.test.ts`)
   ```typescript
   import { http, HttpResponse } from 'msw';
   import { server } from '@/mocks/server';
   import {
     getInbox,
     getTasks,
     createThread,
     getLiveKitToken,
     executeAction,
   } from '../api';

   describe('API Service', () => {
     describe('getInbox', () => {
       it('fetches priority inbox', async () => {
         const inbox = await getInbox();
         expect(inbox.emails).toBeInstanceOf(Array);
       });

       it('includes auth header', async () => {
         let capturedHeaders: Headers;
         server.use(
           http.get('*/api/v1/inbox', ({ request }) => {
             capturedHeaders = request.headers;
             return HttpResponse.json({ emails: [] });
           })
         );

         await getInbox();
         expect(capturedHeaders.get('Authorization')).toMatch(/^Bearer /);
       });

       it('handles pagination params', async () => { /* ... */ });
     });

     describe('getTasks', () => {
       it('fetches task list', async () => { /* ... */ });
       it('filters by status', async () => { /* ... */ });
     });

     describe('createThread', () => {
       it('creates thread and returns response', async () => { /* ... */ });
     });

     describe('error handling', () => {
       it('throws AuthError on 401', async () => { /* ... */ });
       it('throws NotFoundError on 404', async () => { /* ... */ });
       it('throws NetworkError on fetch failure', async () => { /* ... */ });
     });
   });
   ```

3. **MSW Handlers** (`src/mocks/handlers.ts`)
   ```typescript
   import { http, HttpResponse } from 'msw';

   const API = 'http://localhost:8000/api/v1';

   export const handlers = [
     // Inbox
     http.get(`${API}/inbox`, () => HttpResponse.json(mockInboxResponse)),
     http.get(`${API}/inbox/stats`, () => HttpResponse.json(mockInboxStats)),

     // Tasks
     http.get(`${API}/tasks`, () => HttpResponse.json(mockTasksResponse)),
     http.get(`${API}/tasks/:id`, ({ params }) => HttpResponse.json(mockTask(params.id))),
     http.post(`${API}/tasks/:id/complete`, () => HttpResponse.json({ ok: true })),

     // Threads
     http.post(`${API}/threads`, () => HttpResponse.json(mockThreadResponse)),
     http.get(`${API}/threads/:id/events`, () => HttpResponse.json(mockEventsResponse)),

     // LiveKit
     http.post(`${API}/livekit/token`, () => HttpResponse.json(mockLiveKitToken)),

     // Actions
     http.post(`${API}/actions`, () => HttpResponse.json(mockActionResponse)),
   ];
   ```

### Test Cases (Iteration 4)

| ID | Test | Type | Expected Result |
|----|------|------|-----------------|
| 4.1 | getInbox() returns typed response | Unit | PriorityInboxResponse |
| 4.2 | Auth header included | Unit | Bearer token present |
| 4.3 | Pagination params sent | Unit | limit/offset in query |
| 4.4 | getTasks() filters by status | Unit | status param sent |
| 4.5 | createThread() returns UUID | Unit | ThreadResponse with id |
| 4.6 | 401 throws AuthError | Unit | Error with statusCode 401 |
| 4.7 | 404 throws NotFoundError | Unit | Error with statusCode 404 |
| 4.8 | Network failure handled | Unit | NetworkError thrown |
| 4.9 | completeTask() succeeds | Unit | No error thrown |
| 4.10 | executeAction() sends payload | Unit | Correct request body |

---

## Iteration 5: LiveKit Voice Integration (with Tests)

### Story
As a user, I need to have voice conversations with the AI assistant.

### Success Criteria
- [ ] LiveKitContext ported from pl-app
- [ ] Voice UI components ported (VoiceOrb, Waveform, VoiceModePanel)
- [ ] 100% test coverage for LiveKitContext
- [ ] 100% test coverage for voice components
- [ ] Canonical event handling tested

### Deliverables

1. **Port from pl-app**
   - `src/context/LiveKitContext.tsx`
   - `src/components/voice/VoiceOrb.tsx`
   - `src/components/voice/Waveform.tsx`
   - `src/components/voice/VoiceModePanel.tsx`

2. **LiveKitContext Tests** (`src/context/__tests__/LiveKitContext.test.tsx`)
   ```typescript
   import { render, act } from '@testing-library/react-native';
   import { LiveKitProvider, useLiveKit } from '../LiveKitContext';
   import { Room } from 'livekit-client';

   jest.mock('livekit-client');
   jest.mock('@livekit/react-native');

   describe('LiveKitContext', () => {
     describe('connection', () => {
       it('starts disconnected', () => { /* ... */ });
       it('connect() joins room with token', async () => { /* ... */ });
       it('disconnect() leaves room', async () => { /* ... */ });
     });

     describe('microphone', () => {
       it('toggleMicrophone() enables/disables mic', async () => { /* ... */ });
       it('sends end_turn RPC when mic disabled', async () => { /* ... */ });
     });

     describe('event handling', () => {
       it('parses ui.block events', async () => { /* ... */ });
       it('accumulates sduiBlocks', async () => { /* ... */ });
       it('clears on ui.clear event', async () => { /* ... */ });
       it('tracks lastSeq', async () => { /* ... */ });
     });

     describe('speaking indicator', () => {
       it('detects agent speaking', async () => { /* ... */ });
       it('detects user speaking', async () => { /* ... */ });
     });
   });
   ```

3. **Voice Component Tests**
   ```typescript
   // VoiceOrb.test.tsx
   describe('VoiceOrb', () => {
     it('renders with agent variant', () => { /* ... */ });
     it('renders with user variant', () => { /* ... */ });
     it('animates when speaking', () => { /* ... */ });
     it('applies correct colors from theme', () => { /* ... */ });
   });

   // Waveform.test.tsx
   describe('Waveform', () => {
     it('renders bars', () => { /* ... */ });
     it('animates based on audio level', () => { /* ... */ });
   });

   // VoiceModePanel.test.tsx
   describe('VoiceModePanel', () => {
     it('renders orbs and controls', () => { /* ... */ });
     it('mic toggle calls toggleMicrophone', () => { /* ... */ });
     it('shows correct speaking state', () => { /* ... */ });
   });
   ```

### Test Cases (Iteration 5)

| ID | Test | Type | Expected Result |
|----|------|------|-----------------|
| 5.1 | Starts disconnected | Unit | isConnected = false |
| 5.2 | connect() joins room | Unit | Room.connect called |
| 5.3 | disconnect() leaves room | Unit | Room.disconnect called |
| 5.4 | toggleMicrophone() works | Unit | State toggles |
| 5.5 | end_turn RPC sent on mic off | Unit | performRpc called |
| 5.6 | ui.block event parsed | Unit | Block added to array |
| 5.7 | ui.clear clears blocks | Unit | sduiBlocks = [] |
| 5.8 | lastSeq increments | Unit | Matches event seq |
| 5.9 | VoiceOrb renders | Unit | Component visible |
| 5.10 | Waveform animates | Unit | Bars have animation |

---

## Iteration 6: SDUI System (with Tests)

### Story
As a user, I need to see dynamic UI generated by the AI agent.

### Success Criteria
- [ ] SDUI types ported from pl-app
- [ ] SDUIRenderer ported from pl-app
- [ ] All 35+ components ported
- [ ] 100% test coverage for SDUIRenderer
- [ ] 100% test coverage for each component
- [ ] Action handling tested

### Deliverables

1. **Port from pl-app**
   - `src/sdui/types.ts`
   - `src/sdui/SDUIRenderer.tsx`
   - `src/sdui/primitives/` (all files)
   - `src/sdui/composites/` (all files)

2. **SDUIRenderer Tests** (`src/sdui/__tests__/SDUIRenderer.test.tsx`)
   ```typescript
   import { render, fireEvent } from '@testing-library/react-native';
   import { SDUIRenderer } from '../SDUIRenderer';

   describe('SDUIRenderer', () => {
     describe('primitive rendering', () => {
       it('renders text block', () => {
         const block = { id: '1', type: 'text', props: { value: 'Hello' } };
         const { getByText } = render(<SDUIRenderer block={block} />);
         expect(getByText('Hello')).toBeTruthy();
       });

       it('renders button block', () => { /* ... */ });
       it('renders spacer block', () => { /* ... */ });
       it('renders divider block', () => { /* ... */ });
       it('renders avatar block', () => { /* ... */ });
       it('renders badge block', () => { /* ... */ });
     });

     describe('layout rendering', () => {
       it('renders stack with children', () => { /* ... */ });
       it('renders card with children', () => { /* ... */ });
       it('applies layout props', () => { /* ... */ });
     });

     describe('composite rendering', () => {
       it('renders emailCard', () => { /* ... */ });
       it('renders taskCard', () => { /* ... */ });
       it('renders meetingCard', () => { /* ... */ });
       // ... all 35+ components
     });

     describe('action handling', () => {
       it('calls onAction on press', () => {
         const onAction = jest.fn();
         const block = {
           id: '1',
           type: 'button',
           props: { label: 'Click' },
           actions: [{ trigger: 'press', type: 'test.action' }]
         };
         const { getByText } = render(<SDUIRenderer block={block} onAction={onAction} />);
         fireEvent.press(getByText('Click'));
         expect(onAction).toHaveBeenCalledWith({ trigger: 'press', type: 'test.action' });
       });
     });

     describe('unknown type', () => {
       it('renders fallback in dev', () => { /* ... */ });
       it('renders nothing in prod', () => { /* ... */ });
     });
   });
   ```

3. **Component Tests** (one for each)
   ```typescript
   // SDUIEmailCard.test.tsx
   describe('SDUIEmailCard', () => {
     it('renders sender info', () => { /* ... */ });
     it('renders subject', () => { /* ... */ });
     it('renders snippet', () => { /* ... */ });
     it('shows urgent badge when urgent', () => { /* ... */ });
     it('renders quick reply buttons', () => { /* ... */ });
     it('calls onAction for quick reply', () => { /* ... */ });
     it('expands on tap', () => { /* ... */ });
   });
   ```

### Test Cases (Iteration 6)

| ID | Test | Type | Expected Result |
|----|------|------|-----------------|
| 6.1 | Renders text block | Unit | Text visible |
| 6.2 | Renders button block | Unit | Button visible |
| 6.3 | Renders nested children | Unit | All levels render |
| 6.4 | Applies layout padding | Unit | Correct spacing |
| 6.5 | Renders emailCard | Unit | Email UI shows |
| 6.6 | Renders taskCard | Unit | Task UI shows |
| 6.7 | Renders meetingCard | Unit | Meeting UI shows |
| 6.8 | Action callback fires | Unit | onAction called |
| 6.9 | Unknown type fallback | Unit | Warning in dev |
| 6.10 | All 35+ components render | Unit | No crashes |

---

## Iteration 7: Main Screens (with Tests)

### Story
As a user, I need screens for the complete app experience.

### Success Criteria
- [ ] All screens ported from pl-app
- [ ] Navigation ported from pl-app
- [ ] 100% test coverage for each screen
- [ ] 100% test coverage for navigation
- [ ] Integration with all contexts

### Deliverables

1. **Port from pl-app**
   - `src/screens/SplashScreen.tsx`
   - `src/screens/SignInScreen.tsx`
   - `src/screens/LandingScreen.tsx`
   - `src/screens/ConversationScreen.tsx`
   - `src/screens/SettingsScreen.tsx`
   - `src/navigation/RootNavigator.tsx`

2. **Screen Tests**
   ```typescript
   // ConversationScreen.test.tsx
   describe('ConversationScreen', () => {
     it('renders header with mode toggle', () => { /* ... */ });
     it('renders SDUI blocks from context', () => { /* ... */ });
     it('toggles between voice and text mode', () => { /* ... */ });
     it('shows voice panel in voice mode', () => { /* ... */ });
     it('shows text input in text mode', () => { /* ... */ });
     it('navigates to settings', () => { /* ... */ });
   });

   // RootNavigator.test.tsx
   describe('RootNavigator', () => {
     it('shows SignIn when not authenticated', () => { /* ... */ });
     it('shows Landing when Google not connected', () => { /* ... */ });
     it('shows Conversation when fully authenticated', () => { /* ... */ });
   });
   ```

### Test Cases (Iteration 7)

| ID | Test | Type | Expected Result |
|----|------|------|-----------------|
| 7.1 | SplashScreen shows logo | Unit | Logo visible |
| 7.2 | SignInScreen renders Clerk | Unit | Clerk UI visible |
| 7.3 | LandingScreen shows button | Unit | Connect button visible |
| 7.4 | ConversationScreen renders | Unit | Main UI visible |
| 7.5 | Mode toggle works | Unit | Voice/text switches |
| 7.6 | SettingsScreen shows user | Unit | Email visible |
| 7.7 | Nav: unauthenticated → SignIn | Integration | Correct routing |
| 7.8 | Nav: no Google → Landing | Integration | Correct routing |
| 7.9 | Nav: authenticated → Main | Integration | Correct routing |
| 7.10 | Sign out returns to SignIn | Integration | Navigation works |

---

## Iteration 8: E2E Tests & Polish

### Story
As a user, I need a polished, fully-tested app.

### Success Criteria
- [ ] Detox E2E tests for critical flows
- [ ] Error boundaries implemented and tested
- [ ] Loading states implemented and tested
- [ ] Animations smooth
- [ ] 100% coverage maintained
- [ ] All E2E tests pass

### Deliverables

1. **E2E Tests** (`e2e/`)
   ```typescript
   // auth.e2e.ts
   describe('Authentication Flow', () => {
     beforeEach(async () => {
       await device.reloadReactNative();
     });

     it('shows sign in screen initially', async () => {
       await expect(element(by.id('signin-screen'))).toBeVisible();
     });

     it('completes full auth flow', async () => {
       // Sign in with Clerk
       await element(by.id('email-input')).typeText('test@example.com');
       await element(by.id('continue-button')).tap();
       // ... complete flow
       await expect(element(by.id('conversation-screen'))).toBeVisible();
     });
   });

   // voice.e2e.ts
   describe('Voice Conversation', () => {
     it('connects to LiveKit room', async () => { /* ... */ });
     it('toggles microphone', async () => { /* ... */ });
     it('receives agent response', async () => { /* ... */ });
   });
   ```

2. **Error Handling Tests**
   ```typescript
   describe('ErrorBoundary', () => {
     it('catches render errors', () => { /* ... */ });
     it('displays fallback UI', () => { /* ... */ });
     it('reports error to logging', () => { /* ... */ });
   });
   ```

3. **Loading State Tests**
   ```typescript
   describe('Loading States', () => {
     it('InboxSkeleton renders correctly', () => { /* ... */ });
     it('TaskSkeleton renders correctly', () => { /* ... */ });
     it('shows skeleton during API load', () => { /* ... */ });
   });
   ```

### Test Cases (Iteration 8)

| ID | Test | Type | Expected Result |
|----|------|------|-----------------|
| 8.1 | E2E: Full auth flow | E2E | User reaches main screen |
| 8.2 | E2E: Voice conversation | E2E | Agent responds |
| 8.3 | E2E: Complete task | E2E | Task marked done |
| 8.4 | ErrorBoundary catches crash | Unit | Fallback shows |
| 8.5 | InboxSkeleton renders | Unit | Skeleton visible |
| 8.6 | Pull-to-refresh works | Unit | Data reloads |
| 8.7 | Haptics fire | Unit | Vibration triggered |
| 8.8 | Animations run | Unit | No crashes |
| 8.9 | 100% unit coverage | CI | Coverage = 100% |
| 8.10 | All E2E tests pass | CI | Green build |

---

## Summary

| Iteration | Name | Unit Tests | Integration Tests | E2E Tests |
|-----------|------|------------|-------------------|-----------|
| 1 | Setup & Testing | 10 | - | - |
| 2 | Auth - Clerk | 10 | 2 | - |
| 3 | Auth - Google | 10 | 2 | - |
| 4 | API Service | 10 | 5 | - |
| 5 | LiveKit Voice | 10 | 2 | - |
| 6 | SDUI System | 35+ | - | - |
| 7 | Screens | 10 | 5 | - |
| 8 | E2E & Polish | 10 | - | 10 |
| **Total** | | **105+** | **16** | **10** |

**Total Test Cases**: 131+

---

## Definition of Done

An iteration is complete when:
1. All code ported from pl-app-react-native
2. All success criteria checked off
3. **100% test coverage** (statements, branches, functions, lines)
4. All unit tests pass
5. All integration tests pass
6. Code committed to `apps/mobile/`
7. No TypeScript errors
8. No ESLint errors
9. CI pipeline green

---

## Files to Port from pl-app-react-native

```
FROM: pl-app-react-native/apps/mobile/
TO:   priority-lens/apps/mobile/

COPY DIRECTLY:
├── src/theme/index.ts           # Design system (colors, typography)
├── src/sdui/                    # All SDUI components (35+ files)
├── src/components/voice/        # Voice UI components
├── src/components/header/       # Header components
├── app.json                     # Expo config (modify bundle ID)
├── babel.config.js              # Babel config
└── tsconfig.json                # TypeScript config

ADAPT (change API calls):
├── src/context/AuthContext.tsx  # Keep, works with Clerk
├── src/context/GoogleContext.tsx # Change backend sync endpoints
├── src/context/LiveKitContext.tsx # Keep, minimal changes
├── src/services/api.ts          # REWRITE for Priority Lens endpoints
├── src/services/googleAuth.ts   # Keep, native Google SDK
├── src/screens/                 # Keep UI, update data fetching
└── src/navigation/              # Keep navigation logic

CREATE NEW:
├── jest.config.js               # Jest configuration
├── jest.setup.js                # Test setup with MSW
├── detox.config.js              # E2E configuration
├── src/mocks/                   # MSW handlers
├── src/**/__tests__/            # All test files
└── e2e/                         # Detox E2E tests
```

---

## Environment Variables

```bash
# .env
EXPO_PUBLIC_CLERK_PUBLISHABLE_KEY=pk_test_...
EXPO_PUBLIC_GOOGLE_WEB_CLIENT_ID=xxx.apps.googleusercontent.com
EXPO_PUBLIC_GOOGLE_IOS_CLIENT_ID=xxx.apps.googleusercontent.com
EXPO_PUBLIC_API_BASE_URL=http://localhost:8000
EXPO_PUBLIC_LIVEKIT_URL=wss://...
```
