# Environment Variables Security

## Overview
This document outlines security best practices for environment variables in this project.

## Current Environment Variables

### Public Variables (Safe for Frontend)
- `VITE_API_URL` - Public API endpoint (safe to expose)
- `VITE_WEBSOCKET_ENABLED` - Feature flag (safe to expose)
- `VITE_ANALYTICS_ENABLED` - Feature flag (safe to expose)

### Security Notes
1. **Never expose secrets in frontend code** - All environment variables prefixed with `VITE_` are bundled into the client code
2. **API keys should be server-side only** - Any sensitive keys (e.g., GEMINI_API_KEY) should only be used in backend code
3. **Use .env files** - Store environment-specific values in `.env.local` or `.env.production` files
4. **Gitignore .env files** - Never commit `.env` files containing secrets

## Best Practices

1. ✅ Use `VITE_` prefix only for public variables
2. ✅ Keep secrets in backend environment variables
3. ✅ Use `.env.example` to document required variables
4. ✅ Review `.gitignore` to ensure `.env*` files are excluded
5. ❌ Never hardcode API keys in source code
6. ❌ Never commit `.env` files with secrets

## Review Checklist
- [x] All frontend env vars use `VITE_` prefix
- [x] No secrets in frontend code
- [x] `.env.example` documents all variables
- [x] `.gitignore` excludes `.env*` files
- [ ] Backend secrets are properly secured
