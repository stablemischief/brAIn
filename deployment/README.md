# brAIn v2.0 Production Deployment Guide

## 🚀 Overview

This guide provides comprehensive instructions for deploying brAIn v2.0 to production using Docker, with automated SSL, monitoring, backups, and CI/CD.

## 📋 Prerequisites

- **Server Requirements:**
  - Ubuntu 22.04 LTS or similar
  - Minimum 4 CPU cores, 8GB RAM
  - 100GB SSD storage
  - Docker & Docker Compose installed
  - Domain name with DNS configured

- **Required Secrets:**
  - OpenAI API key
  - Supabase credentials
  - Database passwords
  - JWT secret
  - SSL email address

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Internet Traffic                      │
└────────────────────────┬────────────────────────────────┘
                         │
                    ┌────▼────┐
                    │  Caddy  │ (SSL/Reverse Proxy)
                    └────┬────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
   ┌────▼────┐    ┌─────▼────┐    ┌─────▼─────┐
   │ FastAPI │    │ Frontend │    │ Grafana   │
   └────┬────┘    └──────────┘    └───────────┘
        │
   ┌────▼────────────────┐
   │     PostgreSQL       │
   │   (with pgvector)    │
   └─────────────────────┘
```

## 🛠️ Deployment Steps

### 1. Clone Repository

```bash
git clone https://github.com/your-org/brain-app.git
cd brain-app
```

### 2. Configure Environment

```bash
# Copy environment template
cp deployment/secrets/.env.production.example deployment/secrets/.env.production

# Edit with your actual values
nano deployment/secrets/.env.production

# Validate secrets
python deployment/secrets/validate_secrets.py
```

### 3. Initial Database Setup

```bash
# Start only the database first
docker compose -f deployment/docker-compose.prod.yml up -d postgres

# Wait for it to be ready
docker compose -f deployment/docker-compose.prod.yml exec postgres \
  pg_isready -U brain_user -d brain_prod

# Run migrations
docker compose -f deployment/docker-compose.prod.yml run --rm brain-app \
  python -m alembic upgrade head
```

### 4. Deploy Full Stack

```bash
# Deploy all services
docker compose -f deployment/docker-compose.prod.yml up -d

# Check status
docker compose -f deployment/docker-compose.prod.yml ps

# View logs
docker compose -f deployment/docker-compose.prod.yml logs -f
```

### 5. Verify Deployment

```bash
# Check health endpoints
curl http://localhost:8000/health
curl https://your-domain.com/api/health

# Access services:
# - Main App: https://your-domain.com
# - Grafana: https://your-domain.com/grafana
# - Prometheus: https://your-domain.com/prometheus (requires auth)
```

## 📊 Monitoring

### Grafana Access

1. Navigate to: `https://your-domain.com/grafana`
2. Login with credentials from `.env.production`
3. Pre-configured dashboards available:
   - API Metrics
   - Cost Tracking
   - System Health
   - LLM Usage

### Prometheus Metrics

Available metrics endpoints:
- `/metrics` - Application metrics
- `/prometheus/metrics` - System metrics

Key metrics to monitor:
- `brain_api_requests_total` - API request count
- `brain_processing_cost_total` - Total processing costs
- `brain_llm_tokens_total` - LLM token usage
- `brain_api_request_duration_seconds` - Response times

## 🔒 Security

### SSL/TLS Configuration

- Automatic SSL via Let's Encrypt
- HTTP/3 support enabled
- Security headers configured
- HSTS enabled

### Secrets Management

- All secrets in `.env.production`
- Never commit secrets to Git
- Use strong, unique passwords
- Rotate keys regularly

### Security Headers

Configured headers:
- Content-Security-Policy
- X-Frame-Options: DENY
- X-Content-Type-Options: nosniff
- Strict-Transport-Security

## 💾 Backup & Recovery

### Automatic Backups

Daily backups run at 3 AM automatically:
- PostgreSQL full dumps
- Compressed with gzip
- 30-day retention
- Optional S3 upload

### Manual Backup

```bash
# Create manual backup
docker compose -f deployment/docker-compose.prod.yml exec postgres-backup \
  /scripts/backup-entrypoint.sh

# List backups
ls -la deployment/backup/postgres/
```

### Restore from Backup

```bash
# Stop application
docker compose -f deployment/docker-compose.prod.yml stop brain-app

# Restore database
gunzip -c deployment/backup/postgres/backup_brain_prod_20240101_030000.sql.gz | \
  docker compose -f deployment/docker-compose.prod.yml exec -T postgres \
  psql -U brain_user -d brain_prod

# Restart application
docker compose -f deployment/docker-compose.prod.yml start brain-app
```

## 🔄 Updates & Maintenance

### Rolling Update

```bash
# Pull latest images
docker compose -f deployment/docker-compose.prod.yml pull

# Deploy with zero downtime
docker compose -f deployment/docker-compose.prod.yml up -d --no-deps --scale brain-app=2 brain-app
sleep 30
docker compose -f deployment/docker-compose.prod.yml up -d --no-deps --scale brain-app=1 brain-app
```

### Rollback

```bash
# Rollback to previous version
docker compose -f deployment/docker-compose.prod.yml up -d --no-deps brain-app:previous

# Restore database if needed
gunzip -c deployment/backup/postgres/pre-deploy-backup.sql.gz | \
  docker compose -f deployment/docker-compose.prod.yml exec -T postgres \
  psql -U brain_user -d brain_prod
```

## 🚨 Troubleshooting

### Common Issues

**Container won't start:**
```bash
# Check logs
docker compose -f deployment/docker-compose.prod.yml logs brain-app

# Verify environment variables
docker compose -f deployment/docker-compose.prod.yml config
```

**Database connection issues:**
```bash
# Test connection
docker compose -f deployment/docker-compose.prod.yml exec brain-app \
  python -c "from sqlalchemy import create_engine; engine = create_engine('$DATABASE_URL'); engine.connect()"
```

**SSL certificate issues:**
```bash
# Check Caddy logs
docker compose -f deployment/docker-compose.prod.yml logs caddy

# Force certificate renewal
docker compose -f deployment/docker-compose.prod.yml exec caddy \
  caddy reload --config /etc/caddy/Caddyfile
```

### Health Checks

```bash
# Application health
curl http://localhost:8000/health

# Database health
docker compose -f deployment/docker-compose.prod.yml exec postgres \
  pg_isready -U brain_user -d brain_prod

# Redis health
docker compose -f deployment/docker-compose.prod.yml exec redis \
  redis-cli ping
```

## 📝 Maintenance Commands

```bash
# View all logs
docker compose -f deployment/docker-compose.prod.yml logs -f

# Restart specific service
docker compose -f deployment/docker-compose.prod.yml restart brain-app

# Clean up old images
docker system prune -a -f

# Database vacuum
docker compose -f deployment/docker-compose.prod.yml exec postgres \
  psql -U brain_user -d brain_prod -c "VACUUM ANALYZE;"

# Export metrics
curl -s http://localhost:9090/api/v1/query?query=up | jq .
```

## 🆘 Support

For issues or questions:
1. Check logs: `docker compose logs -f`
2. Review monitoring dashboards
3. Consult error tracking in Sentry (if configured)
4. Contact DevOps team

## 📚 Additional Resources

- [Docker Documentation](https://docs.docker.com/)
- [Caddy Documentation](https://caddyserver.com/docs/)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)

---

**Version:** 2.0.0
**Last Updated:** September 2024
**Maintained by:** brAIn DevOps Team