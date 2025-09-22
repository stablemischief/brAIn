# Troubleshooting Guide - brAIn v2.0

Having issues with brAIn v2.0? This comprehensive troubleshooting guide covers the most common problems and their solutions. Most issues can be resolved quickly using these steps.

## 🚨 Quick Emergency Fixes

### System Won't Start
```bash
# Quick reset - fixes 80% of startup issues
docker-compose down -v
docker-compose up --build

# Check if ports are in use
netstat -tulpn | grep :3000
netstat -tulpn | grep :8000
```

### Dashboard Not Loading
1. **Check services**: `docker-compose ps`
2. **Verify ports**: Frontend (3000), Backend (8000)
3. **Clear browser cache**: Ctrl+Shift+R (hard refresh)
4. **Check network**: Ensure localhost access works

### Can't Process Documents
1. **Verify API key**: Check OpenAI key in environment
2. **Check Google Drive**: Ensure folder permissions are correct
3. **Monitor costs**: Verify you haven't exceeded budget limits
4. **Restart processing**: Use the "Force Restart" button

## 📋 Issue Categories

### 🐳 Docker & Environment Issues
- **[Container Problems](#container-issues)**
- **[Port Conflicts](#port-conflicts)**
- **[Environment Variables](#environment-issues)**
- **[Volume Mount Issues](#volume-issues)**

### 🔐 Authentication & Access
- **[Login Problems](#login-issues)**
- **[API Key Issues](#api-key-issues)**
- **[Google Drive Access](#google-drive-issues)**
- **[Permission Errors](#permission-issues)**

### 📄 Document Processing
- **[Processing Failures](#processing-failures)**
- **[Quality Issues](#quality-issues)**
- **[Cost Problems](#cost-issues)**
- **[Performance Issues](#performance-issues)**

### 🔍 Search & Features
- **[Search Not Working](#search-issues)**
- **[No Results Returned](#no-results-issues)**
- **[Knowledge Graph Issues](#knowledge-graph-issues)**
- **[Real-time Updates](#realtime-issues)**

### 🗄️ Database Issues
- **[Connection Problems](#database-connection)**
- **[Migration Issues](#migration-issues)**
- **[Performance Problems](#database-performance)**
- **[Data Corruption](#data-corruption)**

## 🔧 Detailed Solutions

### Container Issues

#### Problem: Containers won't start
**Symptoms**: `docker-compose up` fails or containers exit immediately

**Solutions**:
```bash
# 1. Check Docker daemon
docker info

# 2. Clean up old containers
docker system prune -f

# 3. Reset everything
docker-compose down -v
docker system prune -a -f
docker-compose up --build

# 4. Check specific service logs
docker-compose logs brain-app
docker-compose logs postgres
```

#### Problem: Out of memory errors
**Symptoms**: Containers crash with OOM errors

**Solutions**:
```bash
# Increase Docker memory limits (Docker Desktop)
# Go to Settings > Resources > Memory > 8GB+

# Or modify docker-compose.yml
services:
  brain-app:
    deploy:
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 2G
```

### Port Conflicts

#### Problem: Port already in use
**Symptoms**: "Port 3000/8000 is already allocated"

**Solutions**:
```bash
# Find process using port
lsof -i :3000
lsof -i :8000

# Kill process (replace PID)
kill -9 <PID>

# Or change ports in docker-compose.yml
services:
  brain-app:
    ports:
      - "3001:3000"  # Change external port
      - "8001:8000"
```

### Environment Issues

#### Problem: Environment variables not loading
**Symptoms**: "Configuration error" or default values used

**Solutions**:
```bash
# 1. Verify .env file exists and has correct format
cat .env
ls -la .env

# 2. Check for Windows line endings (if on Linux/Mac)
dos2unix .env

# 3. Verify docker-compose.yml includes env_file
grep -A 5 env_file docker-compose.yml

# 4. Test environment loading
docker-compose config
```

#### Problem: API keys not working
**Symptoms**: "Invalid API key" or authentication errors

**Solutions**:
```bash
# 1. Verify key format
echo $OPENAI_API_KEY | wc -c  # Should be ~50+ characters

# 2. Test key manually
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"

# 3. Check for extra spaces/characters
sed -i 's/[[:space:]]*$//' .env

# 4. Restart containers after env changes
docker-compose restart
```

### Processing Failures

#### Problem: Documents fail to process
**Symptoms**: Red status, "Processing failed" messages

**Diagnostic Steps**:
```bash
# 1. Check processing logs
docker-compose logs brain-app | grep -i error

# 2. Verify file accessibility
curl -I "https://drive.google.com/file/d/FILE_ID/view"

# 3. Check cost limits
# Look at cost dashboard for budget issues

# 4. Test with simple document
# Try a basic .txt file first
```

**Common Solutions**:
- **File type not supported**: Check supported formats list
- **File too large**: Reduce file size or adjust limits
- **Network timeout**: Retry processing
- **Quota exceeded**: Wait or increase limits
- **Corrupted file**: Re-upload document

#### Problem: Poor extraction quality
**Symptoms**: Low quality scores, incomplete content

**Solutions**:
1. **Check file format**: PDF works better than images
2. **Verify text content**: Ensure documents have selectable text
3. **Increase quality settings**: Use higher-quality extraction modes
4. **Review original file**: Ensure source document is clear

### Search Issues

#### Problem: Search returns no results
**Symptoms**: Empty search results for known content

**Diagnostic Steps**:
```bash
# 1. Verify documents are processed
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/documents | jq '.data[].processing_status'

# 2. Check embeddings exist
docker-compose exec postgres psql -U brain_user -d brain_db \
  -c "SELECT COUNT(*) FROM documents WHERE embedding IS NOT NULL;"

# 3. Test simple search
curl -X POST -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "limit": 1}' \
  http://localhost:8000/api/search
```

**Solutions**:
- **Wait for indexing**: Allow 30 seconds after processing
- **Try different terms**: Use exact words from documents
- **Check search type**: Switch between semantic/keyword/hybrid
- **Verify permissions**: Ensure you can access the documents

#### Problem: Poor search relevance
**Symptoms**: Irrelevant results ranked highly

**Solutions**:
1. **Adjust search weights**: Try hybrid search with different weights
2. **Use more specific queries**: Include context and details
3. **Check document quality**: Poor extraction affects search
4. **Try different search types**: Semantic vs keyword vs hybrid

### Database Connection

#### Problem: Cannot connect to database
**Symptoms**: "Connection refused" or "Database unavailable"

**Solutions**:
```bash
# 1. Check if PostgreSQL is running
docker-compose ps postgres

# 2. Test connection manually
docker-compose exec postgres psql -U brain_user -d brain_db -c "SELECT 1;"

# 3. Verify environment variables
echo $DATABASE_URL

# 4. Check for port conflicts
netstat -tulpn | grep :5432

# 5. Reset database
docker-compose down
docker volume rm brain_postgres_data
docker-compose up postgres
```

#### Problem: Database performance issues
**Symptoms**: Slow queries, timeouts

**Solutions**:
```sql
-- 1. Check for missing indexes
SELECT schemaname, tablename, attname, n_distinct, correlation
FROM pg_stats WHERE tablename = 'documents';

-- 2. Analyze database
ANALYZE;

-- 3. Check for bloat
SELECT schemaname, tablename, pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;

-- 4. Rebuild indexes if needed
REINDEX DATABASE brain_db;
```

### Real-time Updates

#### Problem: Dashboard not updating in real-time
**Symptoms**: Stale data, manual refresh required

**Solutions**:
```bash
# 1. Check WebSocket connection
# Open browser dev tools, look for WebSocket errors

# 2. Verify backend WebSocket server
docker-compose logs brain-app | grep -i websocket

# 3. Test WebSocket manually
wscat -c ws://localhost:8000/ws/realtime

# 4. Check firewall/proxy settings
# Ensure WebSocket connections aren't blocked
```

## 🚀 Performance Optimization

### General Performance Issues

#### System running slowly
**Symptoms**: High response times, UI lag

**Solutions**:
```bash
# 1. Check resource usage
docker stats

# 2. Increase memory allocation
# Edit docker-compose.yml memory limits

# 3. Optimize database
docker-compose exec postgres psql -U brain_user -d brain_db \
  -c "SELECT maintenance_vacuum_analyze();"

# 4. Clear old data
# Archive old documents and logs
```

#### High CPU usage
**Solutions**:
1. **Reduce concurrent processing**: Lower worker counts
2. **Optimize embeddings**: Use smaller models if acceptable
3. **Cache frequent queries**: Enable query caching
4. **Batch operations**: Process documents in smaller batches

### Cost Optimization

#### High AI costs
**Symptoms**: Rapid budget consumption

**Solutions**:
1. **Review usage patterns**: Check cost analytics dashboard
2. **Optimize batch sizes**: Process multiple documents together
3. **Use cheaper models**: Switch to cost-effective embedding models
4. **Set strict budgets**: Configure daily/monthly limits
5. **Archive old data**: Remove documents no longer needed

## 🛠️ Diagnostic Tools

### Health Check Script
```bash
#!/bin/bash
# Save as check_health.sh

echo "=== brAIn v2.0 Health Check ==="

# Check Docker
echo "Docker status:"
docker --version
docker-compose ps

# Check services
echo -e "\nService health:"
curl -s http://localhost:8000/api/health | jq '.status'

# Check database
echo -e "\nDatabase connection:"
docker-compose exec -T postgres psql -U brain_user -d brain_db -c "SELECT 1;"

# Check ports
echo -e "\nPort status:"
nc -zv localhost 3000
nc -zv localhost 8000

echo -e "\nHealth check complete!"
```

### Log Analysis
```bash
# Get all logs
docker-compose logs > full_logs.txt

# Filter for errors
docker-compose logs | grep -i error

# Follow live logs
docker-compose logs -f brain-app

# Specific timeframe (Docker 20.10+)
docker-compose logs --since 1h brain-app
```

### Performance Monitoring
```bash
# Resource usage
docker stats --no-stream

# Database performance
docker-compose exec postgres psql -U brain_user -d brain_db \
  -c "SELECT generate_health_report();"

# API response times
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:8000/api/health
```

## 📞 Getting Additional Help

### Before Contacting Support
1. **Try the quick fixes** above
2. **Check recent logs** for error messages
3. **Note your environment** (OS, Docker version, etc.)
4. **Document the issue** with steps to reproduce

### Support Channels
- **Documentation**: Check [User Guide](../user_guide/) first
- **Community**: GitHub Issues and Discussions
- **Team Support**: Internal team channels
- **Emergency**: Critical production issues only

### Information to Include
When reporting issues, include:
- **Environment details**: OS, Docker version, system specs
- **Error messages**: Complete error logs
- **Steps to reproduce**: Exact sequence to trigger issue
- **Expected behavior**: What should happen
- **Configuration**: Relevant environment variables (redact secrets)

---

**Still having issues?** The [API documentation](../api/) and [user guide](../user_guide/) contain additional troubleshooting information for specific features.