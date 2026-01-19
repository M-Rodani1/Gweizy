# Deployment Configuration

This directory contains documentation and guides for deployment configurations.

## Deployment Config Files

The following configuration files are located in the **project root** (not in this directory) because deployment services require them there:

- `railway.toml` / `railway.json` - Railway.app backend deployment configuration
- `netlify.toml` - Netlify frontend deployment configuration (if used)
- `nixpacks.toml` - Nixpacks build configuration for Railway
- `wrangler.toml` - Cloudflare Workers/Pages configuration

## Deployment Platforms

### Backend - Railway
- **Config**: `railway.toml`, `railway.json`, `nixpacks.toml`
- **Procfile**: `Procfile` (in root)
- **Runtime**: Python 3.x
- **Build**: Auto-detected via Nixpacks

### Frontend - Cloudflare Pages
- **Config**: `wrangler.toml`
- **Build Command**: Defined in `frontend/package.json`
- **Publish Directory**: `frontend/dist`

## Documentation

- [Cloudflare Setup Guide](./CLOUDFLARE_SETUP.md)
- [Cloudflare Manual Setup](./CLOUDFLARE_MANUAL_SETUP.md)

## Environment Variables

Environment variables should be configured in the respective platform dashboards:
- Railway: Project → Variables
- Cloudflare Pages: Settings → Environment Variables

## Notes

- Configuration files **must** remain in the project root for services to detect them
- This directory serves as documentation and reference for deployment configurations
