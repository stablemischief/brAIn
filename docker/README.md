# Docker Setup for brAIn v2.0

## Quick Start

1. **Clone and setup environment:**
```bash
cp .env.example .env
# Edit .env with your configuration values
```

2. **Build and run development environment:**
```bash
docker-compose up --build
```

3. **Access services:**
- Backend API: http://localhost:8000
- Frontend: http://localhost:3000
- API Documentation: http://localhost:8000/docs
- Database Admin (optional): http://localhost:8090

## Docker Architecture

### Multi-Stage Build
The Dockerfile uses a multi-stage build pattern optimized for AI/ML dependencies:

1. **Python Builder Stage**: Installs all Python dependencies using Poetry
2. **Node Builder Stage**: Builds the React frontend application
3. **Production Stage**: Minimal runtime with only necessary dependencies
4. **Development Stage**: Includes additional development tools

### Services

#### Main Application Container
- FastAPI backend on port 8000
- React frontend on port 3000
- WebSocket server on port 8080
- Managed by Supervisor for multi-process coordination

#### PostgreSQL with pgvector
- Vector similarity search support
- Optimized for AI embeddings storage
- Automatic migrations on startup

#### Redis Cache
- Session management
- Task queue backend
- Real-time data caching

## Development Commands

### Start services:
```bash
# Start all services
docker-compose up

# Start in background
docker-compose up -d

# Rebuild containers
docker-compose up --build

# View logs
docker-compose logs -f brain-app
```

### Database management:
```bash
# Access PostgreSQL
docker-compose exec postgres psql -U brain_user -d brain_db

# Run migrations
docker-compose exec brain-app alembic upgrade head

# Create new migration
docker-compose exec brain-app alembic revision --autogenerate -m "description"
```

### Container management:
```bash
# Enter container shell
docker-compose exec brain-app bash

# Run tests
docker-compose exec brain-app pytest

# Format code
docker-compose exec brain-app black .
docker-compose exec brain-app ruff check --fix .
```

## Production Deployment

### Build production image:
```bash
docker build -f docker/Dockerfile --target production -t brain-v2:latest .
```

### Run production stack:
```bash
docker-compose -f docker-compose.prod.yml up -d
```

### SSL/TLS Setup:
1. Place certificates in `./ssl/` directory
2. Update nginx configuration
3. Enable nginx profile in docker-compose.prod.yml

## Health Monitoring

### Check service health:
```bash
# Run health check
docker-compose exec brain-app python scripts/health-check.py

# Check specific service
curl http://localhost:8000/api/health
```

### View supervisor status:
```bash
docker-compose exec brain-app supervisorctl status
```

## Troubleshooting

### Container won't start:
```bash
# Check logs
docker-compose logs brain-app

# Verify environment variables
docker-compose config

# Reset volumes
docker-compose down -v
docker-compose up --build
```

### Database connection issues:
```bash
# Test connection
docker-compose exec brain-app python -c "from src.database import test_connection; test_connection()"

# Reset database
docker-compose down
docker volume rm brain-postgres-data
docker-compose up
```

### Performance issues:
```bash
# Check resource usage
docker stats

# Increase resource limits in docker-compose.yml
# Adjust worker counts in supervisord.conf
```

## Environment Variables

Key environment variables (see .env.example for full list):

- `ENVIRONMENT`: development/production
- `DEBUG`: Enable debug mode
- `SUPABASE_URL`: Supabase project URL
- `OPENAI_API_KEY`: OpenAI API key
- `LANGFUSE_PUBLIC_KEY`: Langfuse monitoring key
- `DATABASE_URL`: PostgreSQL connection string
- `REDIS_URL`: Redis connection string

## Security Notes

1. Never commit `.env` files
2. Use strong passwords for production
3. Enable SSL/TLS for production
4. Regularly update base images
5. Use non-root user in containers
6. Implement rate limiting
7. Configure CORS properly

## Maintenance

### Update dependencies:
```bash
# Update Python dependencies
docker-compose exec brain-app poetry update

# Update Node dependencies
docker-compose exec brain-app npm update

# Rebuild after updates
docker-compose up --build
```

### Backup database:
```bash
# Create backup
docker-compose exec postgres pg_dump -U brain_user brain_db > backup.sql

# Restore backup
docker-compose exec -T postgres psql -U brain_user brain_db < backup.sql
```

## Support

For issues or questions:
1. Check logs: `docker-compose logs`
2. Review health status: `scripts/health-check.py`
3. Consult documentation in `/docs`
4. Contact the BMad team