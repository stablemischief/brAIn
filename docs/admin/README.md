# Administrator Guide - brAIn v2.0

This guide provides comprehensive information for system administrators managing brAIn v2.0 installations. Whether you're deploying for a small team or managing a large-scale production environment, this guide covers everything you need to know.

## 🎯 Administrator Responsibilities

### Core Tasks
- **System Deployment**: Set up production environments
- **User Management**: Create accounts and manage permissions
- **Performance Monitoring**: Track system health and performance
- **Cost Management**: Monitor and optimize AI usage costs
- **Security Management**: Maintain security best practices
- **Backup & Recovery**: Ensure data protection and availability

### Daily Operations
- Monitor system health dashboard
- Review cost analytics and budgets
- Check processing queue status
- Validate backup completion
- Review security alerts

## 🚀 Production Deployment

### Infrastructure Requirements

#### Minimum Production Specs
```yaml
System Requirements:
  CPU: 4 cores (8+ recommended)
  RAM: 16GB (32GB recommended)
  Storage: 100GB SSD (500GB+ recommended)
  Network: 1Gbps connection
  OS: Ubuntu 20.04+ or RHEL 8+

Database Requirements:
  PostgreSQL: 15+ with pgvector extension
  Memory: 8GB dedicated
  Storage: 50GB+ SSD with backup
  Connections: 100 concurrent
```

#### Recommended Production Architecture
```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Load Balancer │◄──►│   brAIn App      │◄──►│  PostgreSQL     │
│   (nginx/HAProxy│    │   Containers     │    │  Primary + Hot  │
│   SSL Termination)   │   (2+ instances) │    │  Standby        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   CDN/Static    │    │  Redis Cluster   │    │  Backup Storage │
│   Assets        │    │  (Sessions/Cache)│    │  (S3/NFS)       │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Production Deployment Steps

#### 1. Server Preparation
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Create application directory
sudo mkdir -p /opt/brain
sudo chown $USER:$USER /opt/brain
cd /opt/brain
```

#### 2. Environment Configuration
```bash
# Clone repository
git clone https://github.com/stablemischief/brain-rag-v2 .

# Copy production configuration
cp .env.production.example .env.production

# Edit configuration
sudo nano .env.production
```

**Production Environment Variables**:
```env
# Environment
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=INFO

# Security
JWT_SECRET=your_secure_jwt_secret_here_minimum_32_characters
CORS_ORIGINS=https://yourdomain.com,https://app.yourdomain.com

# Database
DATABASE_URL=postgresql://brain_user:secure_password@localhost:5432/brain_prod
REDIS_URL=redis://localhost:6379/0

# External Services
OPENAI_API_KEY=sk-your_openai_key_here
ANTHROPIC_API_KEY=sk-ant-your_anthropic_key_here
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_supabase_anon_key_here
LANGFUSE_PUBLIC_KEY=pk-your_langfuse_key_here
LANGFUSE_SECRET_KEY=sk-your_langfuse_secret_here

# Performance
WORKERS=4
MAX_CONCURRENT_PROCESSES=10
RATE_LIMIT_PER_MINUTE=1000

# Cost Management
DEFAULT_DAILY_BUDGET=50.00
DEFAULT_MONTHLY_BUDGET=1500.00
COST_ALERT_THRESHOLD=0.8
```

#### 3. SSL/TLS Setup
```bash
# Install Certbot
sudo apt install certbot

# Generate SSL certificates
sudo certbot certonly --standalone -d yourdomain.com -d app.yourdomain.com

# Create certificate renewal script
sudo crontab -e
# Add: 0 2 * * * /usr/bin/certbot renew --quiet
```

#### 4. Production Launch
```bash
# Build and start production stack
docker-compose -f docker-compose.prod.yml up -d --build

# Verify all services
docker-compose -f docker-compose.prod.yml ps

# Check application health
curl https://yourdomain.com/api/health

# Monitor logs
docker-compose -f docker-compose.prod.yml logs -f
```

## 👥 User Management

### User Roles and Permissions

#### Role Definitions
- **Admin**: Full system access, user management, configuration
- **Manager**: User management, cost oversight, advanced features
- **User**: Document processing, search, basic features
- **Viewer**: Read-only access to documents and search

#### Creating Users
```bash
# Using the CLI tool
docker-compose exec brain-app python scripts/create_user.py \
  --email user@company.com \
  --role user \
  --budget 100.00

# Using the API
curl -X POST https://yourdomain.com/api/admin/users \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@company.com",
    "role": "user",
    "daily_budget": 25.00,
    "monthly_budget": 500.00
  }'
```

#### Managing User Budgets
```sql
-- Update user budget
UPDATE users
SET monthly_budget_limit = 1000.00,
    daily_budget_limit = 50.00
WHERE email = 'user@company.com';

-- Check current usage
SELECT u.email,
       COALESCE(SUM(l.cost), 0) as current_month_cost,
       u.monthly_budget_limit
FROM users u
LEFT JOIN llm_usage l ON u.id = l.user_id
  AND l.created_at >= date_trunc('month', CURRENT_DATE)
GROUP BY u.id, u.email, u.monthly_budget_limit;
```

### Bulk User Operations
```bash
# Bulk user import from CSV
docker-compose exec brain-app python scripts/import_users.py \
  --file /path/to/users.csv \
  --default-role user \
  --default-budget 500.00

# Bulk budget updates
docker-compose exec brain-app python scripts/update_budgets.py \
  --department engineering \
  --monthly-budget 2000.00
```

## 📊 System Monitoring

### Health Monitoring Dashboard

#### Key Metrics to Monitor
- **System Health**: Service uptime, response times, error rates
- **Processing Performance**: Queue length, processing times, success rates
- **Cost Analytics**: Daily/monthly spending, budget utilization
- **User Activity**: Active users, document processing volume
- **Database Performance**: Query times, connection counts, storage usage

#### Setting Up Monitoring
```bash
# Enable detailed logging
export LOG_LEVEL=INFO
export ENABLE_METRICS=true

# Set up log rotation
sudo nano /etc/logrotate.d/brain
```

```
/opt/brain/logs/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 644 brain brain
    postrotate
        docker-compose -f /opt/brain/docker-compose.prod.yml restart brain-app
    endscript
}
```

#### Alerting Configuration
```yaml
# alerts.yml
alerts:
  high_error_rate:
    metric: error_rate
    threshold: 5.0  # percentage
    duration: 5m
    action: email

  budget_exceeded:
    metric: daily_cost_percentage
    threshold: 90.0
    duration: 1m
    action: slack

  processing_queue_backup:
    metric: queue_length
    threshold: 100
    duration: 10m
    action: email

  database_connections_high:
    metric: db_connections
    threshold: 80
    duration: 5m
    action: slack
```

### Performance Optimization

#### Database Optimization
```sql
-- Analyze database performance
SELECT generate_health_report();

-- Check slow queries
SELECT query, calls, total_time, mean_time
FROM pg_stat_statements
ORDER BY total_time DESC
LIMIT 10;

-- Optimize indexes
REINDEX DATABASE brain_prod;
ANALYZE;

-- Clean up old data
DELETE FROM llm_usage WHERE created_at < NOW() - INTERVAL '90 days';
DELETE FROM system_health WHERE created_at < NOW() - INTERVAL '30 days';
```

#### Application Optimization
```bash
# Adjust worker processes based on load
docker-compose exec brain-app supervisorctl status
docker-compose exec brain-app supervisorctl restart processing_worker:*

# Monitor resource usage
docker stats --no-stream

# Optimize container resources
# Edit docker-compose.prod.yml
services:
  brain-app:
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: 4
        reservations:
          memory: 4G
          cpus: 2
```

## 🔒 Security Management

### Security Checklist

#### Authentication & Authorization
- [ ] JWT secrets are strong and rotated regularly
- [ ] User roles and permissions properly configured
- [ ] API rate limiting enabled
- [ ] Session timeouts configured appropriately

#### Network Security
- [ ] HTTPS/TLS enabled with valid certificates
- [ ] CORS properly configured for production domains
- [ ] Firewall rules restrict unnecessary access
- [ ] Database not accessible from public internet

#### Data Protection
- [ ] Environment variables secured (no secrets in code)
- [ ] Database encryption at rest enabled
- [ ] Backup encryption configured
- [ ] PII handling complies with regulations

#### Application Security
- [ ] SQL injection protection active
- [ ] XSS protection headers configured
- [ ] CSRF protection enabled
- [ ] Input validation implemented

### Security Monitoring
```bash
# Check for security updates
docker-compose pull
docker-compose up -d --force-recreate

# Monitor failed login attempts
docker-compose exec postgres psql -U brain_user -d brain_prod \
  -c "SELECT COUNT(*) FROM auth_logs WHERE success = false AND created_at > NOW() - INTERVAL '1 hour';"

# Review user activity
docker-compose exec postgres psql -U brain_user -d brain_prod \
  -c "SELECT user_id, COUNT(*) as requests FROM request_logs WHERE created_at > NOW() - INTERVAL '24 hours' GROUP BY user_id ORDER BY requests DESC;"
```

## 💾 Backup & Recovery

### Automated Backup Strategy

#### Database Backups
```bash
#!/bin/bash
# backup_database.sh

BACKUP_DIR="/opt/brain/backups"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="brain_backup_${DATE}.sql"

# Create backup
docker-compose exec -T postgres pg_dump \
  -U brain_user \
  -h localhost \
  brain_prod > "${BACKUP_DIR}/${BACKUP_FILE}"

# Compress backup
gzip "${BACKUP_DIR}/${BACKUP_FILE}"

# Upload to S3 (optional)
aws s3 cp "${BACKUP_DIR}/${BACKUP_FILE}.gz" \
  s3://your-backup-bucket/database/

# Clean up old backups (keep 30 days)
find ${BACKUP_DIR} -name "brain_backup_*.sql.gz" -mtime +30 -delete

echo "Backup completed: ${BACKUP_FILE}.gz"
```

#### Full System Backup
```bash
#!/bin/bash
# backup_system.sh

BACKUP_ROOT="/opt/brain/backups/system"
DATE=$(date +%Y%m%d_%H%M%S)

# Create system backup
tar -czf "${BACKUP_ROOT}/brain_system_${DATE}.tar.gz" \
  --exclude='/opt/brain/logs' \
  --exclude='/opt/brain/backups' \
  /opt/brain/

# Backup Docker volumes
docker run --rm \
  -v brain_postgres_data:/data \
  -v ${BACKUP_ROOT}:/backup \
  alpine tar -czf /backup/postgres_data_${DATE}.tar.gz -C /data .

echo "System backup completed"
```

#### Restore Procedures
```bash
# Restore database
docker-compose exec -T postgres psql \
  -U brain_user \
  -d brain_prod < backup_file.sql

# Restore Docker volume
docker run --rm \
  -v brain_postgres_data:/data \
  -v /backup/location:/backup \
  alpine sh -c "cd /data && tar -xzf /backup/postgres_data_backup.tar.gz"

# Restart services
docker-compose restart
```

### Recovery Testing
```bash
# Monthly recovery test script
#!/bin/bash
# test_recovery.sh

echo "Starting recovery test..."

# Create test restore environment
docker-compose -f docker-compose.test.yml down -v
docker-compose -f docker-compose.test.yml up -d postgres

# Restore latest backup
LATEST_BACKUP=$(ls -t backups/brain_backup_*.sql.gz | head -1)
gunzip -c $LATEST_BACKUP | docker-compose -f docker-compose.test.yml exec -T postgres psql -U brain_user -d brain_test

# Verify data integrity
docker-compose -f docker-compose.test.yml exec postgres psql -U brain_user -d brain_test -c "SELECT generate_health_report();"

echo "Recovery test completed"
```

## 📈 Cost Management

### Cost Monitoring and Optimization

#### Daily Cost Reports
```sql
-- Daily cost summary
SELECT
    date_trunc('day', created_at) as date,
    operation_type,
    SUM(cost) as total_cost,
    COUNT(*) as operations,
    AVG(cost) as avg_cost_per_operation
FROM llm_usage
WHERE created_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY date_trunc('day', created_at), operation_type
ORDER BY date DESC;

-- User cost analysis
SELECT
    u.email,
    SUM(l.cost) as total_cost,
    COUNT(l.id) as total_operations,
    u.monthly_budget_limit,
    (SUM(l.cost) / u.monthly_budget_limit * 100) as budget_utilization
FROM users u
LEFT JOIN llm_usage l ON u.id = l.user_id
WHERE l.created_at >= date_trunc('month', CURRENT_DATE)
GROUP BY u.id, u.email, u.monthly_budget_limit
ORDER BY budget_utilization DESC;
```

#### Cost Optimization Strategies
1. **Batch Processing**: Group operations to reduce API calls
2. **Model Selection**: Use cheaper models when appropriate
3. **Caching**: Avoid re-processing identical content
4. **Budget Alerts**: Proactive cost management
5. **Usage Analytics**: Identify optimization opportunities

#### Setting Budget Alerts
```bash
# Create cost alert script
cat > /opt/brain/scripts/check_budgets.sh << 'EOF'
#!/bin/bash
docker-compose exec -T postgres psql -U brain_user -d brain_prod << 'SQL'
SELECT
    u.email,
    u.daily_budget_limit,
    COALESCE(SUM(l.cost), 0) as todays_cost
FROM users u
LEFT JOIN llm_usage l ON u.id = l.user_id
    AND l.created_at >= CURRENT_DATE
GROUP BY u.id, u.email, u.daily_budget_limit
HAVING COALESCE(SUM(l.cost), 0) > u.daily_budget_limit * 0.8;
SQL
EOF

# Add to crontab for hourly checks
echo "0 * * * * /opt/brain/scripts/check_budgets.sh" | crontab -
```

## 🔧 Maintenance Procedures

### Routine Maintenance Tasks

#### Weekly Tasks
```bash
#!/bin/bash
# weekly_maintenance.sh

echo "Starting weekly maintenance..."

# Database maintenance
docker-compose exec postgres psql -U brain_user -d brain_prod \
  -c "SELECT maintenance_vacuum_analyze();"

# Clean up old logs
find /opt/brain/logs -name "*.log" -mtime +7 -delete

# Update system packages
sudo apt update && sudo apt upgrade -y

# Check Docker images for updates
docker-compose pull

echo "Weekly maintenance completed"
```

#### Monthly Tasks
- Review user access and permissions
- Analyze cost trends and optimize budgets
- Test backup and recovery procedures
- Update SSL certificates if needed
- Review and update security configurations

#### Quarterly Tasks
- Performance review and optimization
- Capacity planning based on usage trends
- Security audit and penetration testing
- Disaster recovery plan testing
- Documentation updates

## 🆘 Troubleshooting for Admins

### Common Production Issues

#### High Load Issues
```bash
# Check system resources
htop
iostat -x 1
df -h

# Monitor database performance
docker-compose exec postgres psql -U brain_user -d brain_prod \
  -c "SELECT * FROM pg_stat_activity WHERE state = 'active';"

# Scale horizontally (if configured)
docker-compose -f docker-compose.prod.yml up -d --scale brain-app=3
```

#### Database Issues
```bash
# Check database connections
docker-compose exec postgres psql -U brain_user -d brain_prod \
  -c "SELECT count(*) FROM pg_stat_activity;"

# Check for long-running queries
docker-compose exec postgres psql -U brain_user -d brain_prod \
  -c "SELECT pid, now() - pg_stat_activity.query_start AS duration, query FROM pg_stat_activity WHERE (now() - pg_stat_activity.query_start) > interval '5 minutes';"

# Kill problematic queries
docker-compose exec postgres psql -U brain_user -d brain_prod \
  -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE pid = <problematic_pid>;"
```

## 📞 Admin Support

### Escalation Procedures
1. **Level 1**: Check monitoring dashboard and common solutions
2. **Level 2**: Review logs and run diagnostic scripts
3. **Level 3**: Contact development team with detailed diagnostics
4. **Critical**: For production outages, follow emergency procedures

### Diagnostic Information Collection
```bash
# Collect system diagnostics
./scripts/collect_diagnostics.sh

# This script gathers:
# - System resource usage
# - Docker container status
# - Application logs (last 1000 lines)
# - Database connection status
# - Network connectivity tests
# - Configuration validation
```

---

**For additional admin resources**: Check the [deployment scripts](../../deployment/) and [monitoring configuration](../../monitoring/) directories.