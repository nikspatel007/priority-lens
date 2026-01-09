# Clerk Authentication Setup Guide

This guide walks through setting up Clerk for user authentication in rl-emails.

## Prerequisites

- Clerk account (free tier available)
- rl-emails project installed locally
- Node.js (for Clerk frontend integration, if applicable)

---

## Step 1: Create Clerk Account and Application

1. Go to [Clerk Dashboard](https://dashboard.clerk.com/)

2. Sign up or log in to your account

3. Click **Create application**

4. Enter application details:
   - **Application name**: `rl-emails` (or your preferred name)
   - **Sign-in options**: Select your preferred methods:
     - Email address (recommended)
     - Google OAuth (optional)
     - GitHub OAuth (optional)

5. Click **Create application**

6. You'll be taken to your application dashboard

---

## Step 2: Get API Keys

1. In your Clerk application dashboard, go to **API Keys**

2. You'll see two types of keys:

   **Publishable Key** (for frontend):
   ```
   pk_test_xxx...
   ```

   **Secret Key** (for backend):
   ```
   sk_test_xxx...
   ```

3. Copy both keys - you'll need them for configuration

4. **Important**: Never expose the Secret Key in frontend code!

---

## Step 3: Configure JWT Settings

1. Go to **JWT Templates** in the Clerk dashboard

2. Click **New template**

3. Select **Blank** template

4. Configure the template:
   ```
   Name: rl-emails-api

   Claims:
   {
     "sub": "{{user.id}}",
     "email": "{{user.primary_email_address}}",
     "first_name": "{{user.first_name}}",
     "last_name": "{{user.last_name}}",
     "image_url": "{{user.image_url}}"
   }

   Token lifetime: 3600 (1 hour)
   ```

5. Click **Save**

6. Note the **Issuer** URL (e.g., `https://your-app.clerk.accounts.dev`)

---

## Step 4: Get JWKS URL

The JWKS (JSON Web Key Set) URL is used to verify JWT signatures.

1. Your JWKS URL follows this pattern:
   ```
   https://your-app.clerk.accounts.dev/.well-known/jwks.json
   ```

2. Or construct from your issuer:
   ```
   {issuer}/.well-known/jwks.json
   ```

3. Verify by visiting the URL in your browser - you should see a JSON response with keys

---

## Step 5: Configure rl-emails

### Environment Variables

Add to your `.env` file:

```bash
# Clerk Authentication
CLERK_SECRET_KEY=sk_test_xxx...
CLERK_ISSUER=https://your-app.clerk.accounts.dev
CLERK_JWKS_URL=https://your-app.clerk.accounts.dev/.well-known/jwks.json

# Optional: Audience for token validation
CLERK_AUDIENCE=

# Optional: Token leeway in seconds (default: 30)
CLERK_TOKEN_LEEWAY=30

# Optional: API keys for service-to-service auth (comma-separated)
CLERK_API_KEYS=
```

### Configuration Class

The configuration is loaded automatically via `ClerkConfig`:

```python
from rl_emails.api.auth.config import ClerkConfig, get_clerk_config

config = get_clerk_config()
print(f"Configured: {config.is_configured}")
print(f"Issuer: {config.issuer}")
```

---

## Step 6: Verify Setup

### Test JWT Validation

```python
from rl_emails.api.auth.clerk import ClerkJWTValidator
from rl_emails.api.auth.config import get_clerk_config

config = get_clerk_config()
validator = ClerkJWTValidator(config)

# Validate a token (from Clerk frontend)
try:
    user = validator.validate_token(token)
    print(f"User ID: {user.id}")
    print(f"Email: {user.email}")
except Exception as e:
    print(f"Validation failed: {e}")
```

### Test API Endpoint

```bash
# Start the API server
uvicorn rl_emails.api.main:app --reload

# Test health endpoint (no auth required)
curl http://localhost:8000/health

# Test authenticated endpoint
curl -H "Authorization: Bearer YOUR_JWT_TOKEN" \
     http://localhost:8000/connections
```

---

## Authentication Flow Diagram

```
┌─────────────┐     ┌─────────────────┐     ┌──────────────────┐
│   Frontend  │     │   rl-emails     │     │   Clerk          │
│   (React)   │     │   Backend       │     │   (Auth Server)  │
└─────────────┘     └─────────────────┘     └──────────────────┘
       │                    │                        │
       │  1. User signs in via Clerk                 │
       │─────────────────────────────────────────────▶
       │                    │                        │
       │  2. Clerk returns session + JWT             │
       │◀────────────────────────────────────────────│
       │                    │                        │
       │  3. API request with JWT                    │
       │───────────────────▶│                        │
       │                    │                        │
       │                    │  4. Fetch JWKS         │
       │                    │───────────────────────▶│
       │                    │                        │
       │                    │  5. Public keys        │
       │                    │◀───────────────────────│
       │                    │                        │
       │                    │  6. Verify JWT locally │
       │                    │                        │
       │  7. API response   │                        │
       │◀───────────────────│                        │
```

---

## API Authentication

### Using Dependencies

```python
from fastapi import APIRouter
from rl_emails.api.auth.dependencies import CurrentUser

router = APIRouter()

@router.get("/me")
async def get_current_user(user: CurrentUser):
    """Get current authenticated user."""
    return {
        "id": user.id,
        "email": user.email,
        "name": user.full_name,
    }
```

### Optional Authentication

```python
from rl_emails.api.auth.dependencies import CurrentUserOptional

@router.get("/public")
async def public_endpoint(user: CurrentUserOptional):
    """Works with or without authentication."""
    if user:
        return {"message": f"Hello, {user.first_name}!"}
    return {"message": "Hello, anonymous!"}
```

### API Key Authentication

For service-to-service communication:

```python
from rl_emails.api.auth.dependencies import CurrentUserOrApiKey

@router.get("/internal")
async def internal_endpoint(user: CurrentUserOrApiKey):
    """Accepts JWT or API key."""
    return {"user_id": user.id}
```

Configure API keys in `.env`:
```bash
CLERK_API_KEYS=key1,key2,key3
```

Call with API key:
```bash
curl -H "X-API-Key: key1" http://localhost:8000/internal
```

---

## User Data Structure

The `ClerkUser` dataclass contains:

```python
@dataclass(frozen=True)
class ClerkUser:
    id: str                          # Clerk user ID (e.g., "user_xxx")
    email: str | None = None         # Primary email
    first_name: str | None = None    # First name
    last_name: str | None = None     # Last name
    image_url: str | None = None     # Profile image URL
    metadata: dict = field(...)      # Public/private metadata

    @property
    def full_name(self) -> str | None:
        """Get full name from first and last name."""
```

---

## Troubleshooting

### Error: JWKS URL is required

**Cause**: Missing issuer or JWKS URL in configuration

**Fix**:
1. Ensure `CLERK_ISSUER` is set in `.env`
2. Or explicitly set `CLERK_JWKS_URL`
3. Verify the URL is accessible

### Error: Token validation failed

**Cause**: Invalid or expired JWT

**Fix**:
1. Ensure token is from correct Clerk application
2. Check token hasn't expired
3. Verify issuer matches your Clerk app
4. Try refreshing the token from frontend

### Error: Invalid signature

**Cause**: Token signed with different key or wrong issuer

**Fix**:
1. Verify `CLERK_ISSUER` matches your Clerk app exactly
2. Check JWKS URL returns valid keys
3. Ensure you're using the correct environment (dev vs prod)

### Error: Token expired

**Cause**: JWT past expiration time

**Fix**:
1. Tokens are short-lived by design (default 1 hour)
2. Frontend should refresh tokens automatically
3. Increase `CLERK_TOKEN_LEEWAY` for clock skew tolerance

### Error: Authentication not configured

**Cause**: Clerk config not set up

**Fix**:
1. Ensure `CLERK_SECRET_KEY` is set
2. Ensure `CLERK_ISSUER` or `CLERK_JWKS_URL` is set
3. Check `config.is_configured` returns `True`

---

## Security Best Practices

### Credential Storage

1. **Never commit secrets** to version control
2. Add to `.gitignore`:
   ```
   .env
   .env.local
   .env.*.local
   ```

3. Use environment variables in production

### Token Handling

1. Always use HTTPS in production
2. Don't log full tokens (only first/last few chars)
3. Validate tokens on every request
4. Don't store JWTs in localStorage (use httpOnly cookies)

### API Keys

1. Rotate API keys regularly
2. Use different keys for different services
3. Limit API key permissions via middleware if needed

### Frontend Security

1. Use Clerk's official SDK for frontend
2. Enable session security features:
   - Multi-factor authentication
   - Session timeout
   - Device tracking

---

## Production Considerations

### Environment Separation

Use different Clerk applications for each environment:

```bash
# Development
CLERK_SECRET_KEY=sk_test_xxx
CLERK_ISSUER=https://your-app-dev.clerk.accounts.dev

# Production
CLERK_SECRET_KEY=sk_live_xxx
CLERK_ISSUER=https://your-app.clerk.accounts.dev
```

### Rate Limiting

Clerk applies rate limits:

| Resource | Limit |
|----------|-------|
| JWKS fetch | Cached (5 min TTL) |
| Token validation | Unlimited (local) |
| API calls | Per your Clerk plan |

### JWKS Caching

The JWT validator uses `PyJWKClient` which caches JWKS:
- Default cache TTL: 5 minutes
- Automatic refresh on key rotation
- No configuration needed

### Monitoring

1. Monitor failed authentications in logs
2. Set up alerts for unusual patterns
3. Review Clerk dashboard analytics

---

## Environment Variables Reference

```bash
# Required
CLERK_SECRET_KEY=sk_test_xxx...           # Clerk secret key
CLERK_ISSUER=https://xxx.clerk.accounts.dev  # Your Clerk issuer URL

# Optional
CLERK_JWKS_URL=                           # Override JWKS URL (derived from issuer)
CLERK_AUDIENCE=                           # Expected audience claim
CLERK_TOKEN_LEEWAY=30                     # Clock skew tolerance in seconds
CLERK_API_KEYS=                           # Comma-separated API keys for S2S auth
```

---

## Frontend Integration (React Example)

### Install Clerk SDK

```bash
npm install @clerk/clerk-react
```

### Configure Provider

```tsx
// App.tsx
import { ClerkProvider } from '@clerk/clerk-react';

const clerkPubKey = import.meta.env.VITE_CLERK_PUBLISHABLE_KEY;

function App() {
  return (
    <ClerkProvider publishableKey={clerkPubKey}>
      <YourApp />
    </ClerkProvider>
  );
}
```

### Get Token for API Calls

```tsx
import { useAuth } from '@clerk/clerk-react';

function ApiComponent() {
  const { getToken } = useAuth();

  const callApi = async () => {
    const token = await getToken();

    const response = await fetch('/api/connections', {
      headers: {
        'Authorization': `Bearer ${token}`,
      },
    });

    return response.json();
  };

  return <button onClick={callApi}>Call API</button>;
}
```

---

## Next Steps

After completing this setup:

1. **Test authentication**: Verify tokens work end-to-end
2. **Connect providers**: Use `/connections` API to link email providers
3. **Sync emails**: Start syncing user emails via provider connections
4. **Add frontend**: Integrate Clerk SDK in your frontend app

See [ARCHITECTURE_PLAN.md](./ARCHITECTURE_PLAN.md) for full implementation plan.
