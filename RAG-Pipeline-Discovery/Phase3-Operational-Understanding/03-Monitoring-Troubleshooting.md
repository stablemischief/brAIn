# Phase 3: Operational Understanding
## Part 3: Monitoring, Troubleshooting & Maintenance

---

# MONITORING & TROUBLESHOOTING GUIDE

## Overview
This document provides comprehensive guidance for monitoring RAG Pipeline operations, diagnosing issues, and maintaining optimal performance.

---

## 1. OPERATIONAL MONITORING

### System Health Indicators

#### Key Metrics to Monitor
```bash
# Document Processing Rate
Documents processed per minute
Average processing time per document
Success rate percentage

# API Performance
OpenAI embedding API response time
Google Drive API quota usage
Supabase response time

# Resource Usage
Memory consumption
CPU utilization
Disk I/O patterns
Network bandwidth usage

# Error Rates
Failed downloads
Extraction failures
Embedding API errors
Database connection failures
```

### Health Check Script
```python
#!/usr/bin/env python3
"""
RAG Pipeline Health Monitoring Script
Usage: python health_check.py [--json] [--detailed]
"""

import sys
import os
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('..')

def check_environment():
    """Check environment variables"""
    required_vars = [
        'SUPABASE_URL', 'SUPABASE_SERVICE_KEY',
        'EMBEDDING_API_KEY', 'EMBEDDING_MODEL_CHOICE'
    ]
    
    results = {}
    for var in required_vars:
        value = os.getenv(var)
        results[var] = {
            'status': 'OK' if value else 'MISSING',
            'configured': bool(value)
        }
    
    return results

def check_database_connection():
    """Test Supabase database connectivity"""
    try:
        from common.db_handler import supabase
        
        # Test basic connectivity
        start_time = time.time()
        result = supabase.table('documents').select('id').limit(1).execute()
        response_time = time.time() - start_time
        
        # Get document count
        count_result = supabase.table('documents').select('id', count='exact').limit(1).execute()
        doc_count = count_result.count
        
        # Test recent activity
        recent_docs = supabase.table('documents')\
            .select('metadata')\
            .order('id', desc=True)\
            .limit(10).execute()
        
        return {
            'status': 'OK',
            'response_time_ms': round(response_time * 1000, 2),
            'document_count': doc_count,
            'recent_activity': len(recent_docs.data),
            'error': None
        }
    except Exception as e:
        return {
            'status': 'ERROR',
            'error': str(e),
            'response_time_ms': None,
            'document_count': 0
        }

def check_embedding_api():
    """Test embedding API connectivity and performance"""
    try:
        from common.text_processor import create_embeddings
        
        # Test with small input
        start_time = time.time()
        test_embeddings = create_embeddings(['Health check test'])
        response_time = time.time() - start_time
        
        return {
            'status': 'OK',
            'response_time_ms': round(response_time * 1000, 2),
            'embedding_dimensions': len(test_embeddings[0]) if test_embeddings else 0,
            'error': None
        }
    except Exception as e:
        return {
            'status': 'ERROR',
            'error': str(e),
            'response_time_ms': None,
            'embedding_dimensions': 0
        }

def check_google_drive_auth():
    """Check Google Drive authentication status"""
    try:
        token_path = Path('Google_Drive/token.json')
        creds_path = Path('Google_Drive/credentials.json')
        
        return {
            'status': 'OK' if token_path.exists() and creds_path.exists() else 'WARNING',
            'token_exists': token_path.exists(),
            'credentials_exist': creds_path.exists(),
            'token_age_hours': (time.time() - token_path.stat().st_mtime) / 3600 if token_path.exists() else None
        }
    except Exception as e:
        return {
            'status': 'ERROR',
            'error': str(e)
        }

def check_processing_status():
    """Check recent processing activity"""
    try:
        config_files = ['Google_Drive/config.json', 'Local_Files/config.json']
        results = {}
        
        for config_file in config_files:
            if Path(config_file).exists():
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    last_check = config.get('last_check_time')
                    if last_check:
                        last_check_dt = datetime.strptime(last_check, '%Y-%m-%dT%H:%M:%S.%fZ')
                        hours_ago = (datetime.utcnow() - last_check_dt).total_seconds() / 3600
                        
                        results[config_file] = {
                            'last_check_time': last_check,
                            'hours_since_last_check': round(hours_ago, 2),
                            'status': 'OK' if hours_ago < 24 else 'WARNING'
                        }
        
        return results
    except Exception as e:
        return {'error': str(e)}

def run_health_check():
    """Run complete health check"""
    print("🔍 RAG Pipeline Health Check\n")
    
    # Environment check
    print("📋 Environment Variables:")
    env_results = check_environment()
    for var, result in env_results.items():
        status = "✅" if result['status'] == 'OK' else "❌"
        print(f"  {status} {var}: {result['status']}")
    
    # Database check
    print("\n💾 Database Connection:")
    db_result = check_database_connection()
    status = "✅" if db_result['status'] == 'OK' else "❌"
    print(f"  {status} Status: {db_result['status']}")
    if db_result['status'] == 'OK':
        print(f"      Response Time: {db_result['response_time_ms']}ms")
        print(f"      Document Count: {db_result['document_count']}")
    else:
        print(f"      Error: {db_result['error']}")
    
    # Embedding API check
    print("\n🤖 Embedding API:")
    embed_result = check_embedding_api()
    status = "✅" if embed_result['status'] == 'OK' else "❌"
    print(f"  {status} Status: {embed_result['status']}")
    if embed_result['status'] == 'OK':
        print(f"      Response Time: {embed_result['response_time_ms']}ms")
        print(f"      Vector Dimensions: {embed_result['embedding_dimensions']}")
    else:
        print(f"      Error: {embed_result['error']}")
    
    # Google Drive auth check
    print("\n🔐 Google Drive Authentication:")
    auth_result = check_google_drive_auth()
    status = "✅" if auth_result['status'] == 'OK' else "⚠️" if auth_result['status'] == 'WARNING' else "❌"
    print(f"  {status} Status: {auth_result['status']}")
    print(f"      Token Exists: {auth_result.get('token_exists', False)}")
    print(f"      Credentials Exist: {auth_result.get('credentials_exist', False)}")
    
    # Processing status
    print("\n⏰ Recent Processing Activity:")
    proc_results = check_processing_status()
    for config_file, result in proc_results.items():
        if 'error' not in result:
            hours = result['hours_since_last_check']
            status = "✅" if hours < 1 else "⚠️" if hours < 24 else "❌"
            print(f"  {status} {config_file}: {hours}h ago")
    
    print("\n🎉 Health check complete!")

if __name__ == "__main__":
    run_health_check()
```

---

## 2. LOG ANALYSIS & MONITORING

### Log File Locations
```bash
# Application logs
RAG_Pipeline/rag-pipeline.log          # Main application log
RAG_Pipeline/error.log                 # Error-specific log
RAG_Pipeline/performance.log           # Performance metrics

# System logs (if using systemd)
/var/log/syslog                        # System log
journalctl -u rag-pipeline            # Service-specific logs
```

### Log Monitoring Commands
```bash
# Real-time log monitoring
tail -f rag-pipeline.log

# Error pattern search
grep -i "error\|exception\|failed" rag-pipeline.log | tail -20

# Performance monitoring
grep "processed.*seconds" rag-pipeline.log | tail -10

# API rate limiting detection
grep -i "rate limit\|quota\|429" rag-pipeline.log

# Database connection issues
grep -i "connection\|timeout\|refused" rag-pipeline.log
```

### Log Analysis Script
```python
#!/usr/bin/env python3
"""
Log analysis and alerting script
Usage: python analyze_logs.py [--errors-only] [--since=1h]
"""

import re
import sys
import argparse
from datetime import datetime, timedelta
from collections import Counter, defaultdict

def analyze_log_file(log_file, since_hours=24):
    """Analyze log file for patterns and issues"""
    
    # Patterns to match
    patterns = {
        'errors': r'(ERROR|Exception|Failed|failed)',
        'warnings': r'(WARNING|WARN)',
        'processing': r'processed.*?(\d+).*?seconds',
        'api_calls': r'(OpenAI|Google Drive|Supabase).*?(\d+ms|\d+\.\d+s)',
        'rate_limits': r'(rate limit|quota|429|too many requests)',
        'files_processed': r'Successfully processed file.*?\'([^\']+)\'',
        'database_ops': r'(insert|delete|update).*?documents'
    }
    
    results = defaultdict(list)
    stats = Counter()
    
    cutoff_time = datetime.now() - timedelta(hours=since_hours)
    
    try:
        with open(log_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                # Extract timestamp if available
                timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', line)
                if timestamp_match:
                    try:
                        log_time = datetime.strptime(timestamp_match.group(1), '%Y-%m-%d %H:%M:%S')
                        if log_time < cutoff_time:
                            continue
                    except ValueError:
                        pass
                
                # Check against patterns
                for pattern_name, pattern in patterns.items():
                    matches = re.search(pattern, line, re.IGNORECASE)
                    if matches:
                        results[pattern_name].append({
                            'line_num': line_num,
                            'content': line.strip(),
                            'match': matches.group(0)
                        })
                        stats[pattern_name] += 1
    
    except FileNotFoundError:
        print(f"❌ Log file not found: {log_file}")
        return None, None
    
    return results, stats

def print_analysis_report(results, stats):
    """Print formatted analysis report"""
    print("📊 Log Analysis Report")
    print("=" * 50)
    
    # Summary stats
    print(f"\n📈 Summary (last 24h):")
    for pattern, count in stats.most_common():
        icon = "❌" if pattern == 'errors' else "⚠️" if pattern == 'warnings' else "📄"
        print(f"  {icon} {pattern}: {count}")
    
    # Error details
    if results['errors']:
        print(f"\n❌ Recent Errors ({len(results['errors'])}):")
        for error in results['errors'][-5:]:  # Last 5 errors
            print(f"  Line {error['line_num']}: {error['content'][:100]}...")
    
    # Performance insights
    if results['processing']:
        processing_times = []
        for proc in results['processing']:
            match = re.search(r'(\d+(?:\.\d+)?)', proc['match'])
            if match:
                processing_times.append(float(match.group(1)))
        
        if processing_times:
            avg_time = sum(processing_times) / len(processing_times)
            print(f"\n⏱️ Performance:")
            print(f"  Average processing time: {avg_time:.2f}s")
            print(f"  Min/Max: {min(processing_times):.2f}s / {max(processing_times):.2f}s")
    
    # Files processed
    if results['files_processed']:
        print(f"\n📁 Files Processed ({len(results['files_processed'])}):")
        for file_proc in results['files_processed'][-3:]:  # Last 3 files
            filename = file_proc['match']
            print(f"  ✅ {filename}")

def main():
    parser = argparse.ArgumentParser(description='Analyze RAG Pipeline logs')
    parser.add_argument('--log-file', default='rag-pipeline.log', help='Log file to analyze')
    parser.add_argument('--since', default='24h', help='Time window (e.g., 1h, 24h)')
    parser.add_argument('--errors-only', action='store_true', help='Show only errors')
    
    args = parser.parse_args()
    
    # Parse time window
    since_match = re.match(r'(\d+)([hm])', args.since)
    if since_match:
        value, unit = since_match.groups()
        hours = int(value) if unit == 'h' else int(value) / 60
    else:
        hours = 24
    
    results, stats = analyze_log_file(args.log_file, since_hours=hours)
    if results is not None:
        if args.errors_only:
            if results['errors']:
                print("❌ Errors found:")
                for error in results['errors']:
                    print(f"  {error['content']}")
            else:
                print("✅ No errors found")
        else:
            print_analysis_report(results, stats)

if __name__ == "__main__":
    main()
```

---

## 3. COMMON ISSUES & SOLUTIONS

### Google Drive Issues

#### Issue: Authentication Failures
**Symptoms:**
- "Invalid credentials" errors
- OAuth flow not working
- Token refresh failures

**Diagnosis:**
```bash
# Check credential files
ls -la Google_Drive/credentials.json Google_Drive/token.json

# Validate credential format
python -c "
import json
with open('Google_Drive/credentials.json') as f:
    creds = json.load(f)
print('Credentials type:', creds.get('type'))
print('Client ID present:', bool(creds.get('client_id')))
"

# Test authentication
python -c "
from Google_Drive.drive_watcher import GoogleDriveWatcher
watcher = GoogleDriveWatcher()
watcher.authenticate()
print('✅ Authentication successful')
"
```

**Solutions:**
1. **Re-download credentials:** Get fresh credentials.json from Google Cloud Console
2. **Delete token:** Remove token.json to force re-authentication
3. **Check scopes:** Ensure OAuth consent screen includes required scopes
4. **Verify project:** Confirm Google Cloud project has Drive API enabled

#### Issue: Shared Drive Access
**Symptoms:**
- "File not found" for existing folders
- Unable to access team/shared drives
- Partial file listings

**Solution:**
```python
# Add to all API calls
supportsAllDrives=True,
includeItemsFromAllDrives=True
```

#### Issue: Rate Limiting
**Symptoms:**
- "Quota exceeded" errors
- HTTP 429 responses
- Slow processing

**Solutions:**
```python
# Implement exponential backoff
import time
import random

def with_retry(func, max_retries=5):
    for attempt in range(max_retries):
        try:
            return func()
        except HttpError as e:
            if e.resp.status == 429:
                wait_time = (2 ** attempt) + random.uniform(0, 1)
                time.sleep(wait_time)
            else:
                raise
    raise Exception("Max retries exceeded")
```

### Database Issues

#### Issue: Connection Failures
**Symptoms:**
- "Connection refused" errors
- Timeout errors
- SSL certificate issues

**Diagnosis:**
```bash
# Test direct connection
curl -H "apikey: $SUPABASE_SERVICE_KEY" \
     -H "Authorization: Bearer $SUPABASE_SERVICE_KEY" \
     "$SUPABASE_URL/rest/v1/documents?limit=1"

# Check SSL
openssl s_client -connect your-project.supabase.co:443 -servername your-project.supabase.co
```

**Solutions:**
1. **Verify credentials:** Check SUPABASE_URL and SUPABASE_SERVICE_KEY
2. **Network connectivity:** Ensure firewall allows HTTPS outbound
3. **SSL issues:** Update CA certificates
4. **Supabase status:** Check supabase.com status page

#### Issue: Embedding Dimension Mismatch
**Symptoms:**
- "vector dimension mismatch" errors
- Cannot insert embeddings
- Schema errors

**Diagnosis:**
```sql
-- Check current table schema
SELECT column_name, data_type, character_maximum_length 
FROM information_schema.columns 
WHERE table_name = 'documents' AND column_name = 'embedding';
```

**Solution:**
```sql
-- Update vector dimension
ALTER TABLE documents ALTER COLUMN embedding TYPE vector(1536);

-- Or recreate table if needed
DROP TABLE documents;
-- Re-run documents.sql
```

### Embedding API Issues

#### Issue: API Key Problems
**Symptoms:**
- "Invalid API key" errors
- Authentication failures
- Quota exceeded

**Diagnosis:**
```bash
# Validate API key format
echo $EMBEDDING_API_KEY | wc -c  # Should be ~51 chars for OpenAI

# Test direct API call
curl -H "Authorization: Bearer $EMBEDDING_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"input":"test","model":"text-embedding-3-small"}' \
     https://api.openai.com/v1/embeddings
```

#### Issue: Model Compatibility
**Symptoms:**
- "Model not found" errors
- Incorrect dimensions
- Performance issues

**Solution:**
```env
# Verify model names
# OpenAI models:
EMBEDDING_MODEL_CHOICE=text-embedding-3-small  # 1536 dims
EMBEDDING_MODEL_CHOICE=text-embedding-3-large  # 3072 dims

# Ollama models:
EMBEDDING_MODEL_CHOICE=nomic-embed-text        # 768 dims
```

### Processing Issues

#### Issue: Memory Exhaustion
**Symptoms:**
- Out of memory errors
- Slow processing
- System freezing

**Diagnosis:**
```bash
# Monitor memory usage
ps aux | grep python
free -h
top -p $(pgrep -f "main.py")
```

**Solutions:**
1. **Reduce batch size:** Process fewer files concurrently
2. **Increase swap:** Add system swap space
3. **File size limits:** Skip very large files
4. **Streaming processing:** Process large files in chunks

#### Issue: Text Extraction Failures
**Symptoms:**
- "No text extracted" messages
- Empty documents in database
- Format-specific errors

**Diagnosis:**
```bash
# Test extraction on specific file
python -c "
from common.text_processor import extract_text_from_file
with open('problem_file.pdf', 'rb') as f:
    content = f.read()
    text = extract_text_from_file(content, 'application/pdf')
    print(f'Extracted {len(text)} characters')
    print(f'Sample: {text[:200]}')
"
```

**Solutions:**
1. **Update dependencies:** Ensure latest versions of pypdf, python-docx
2. **Alternative extraction:** Try different PDF libraries
3. **File corruption:** Verify file integrity
4. **Format support:** Check if MIME type is supported

---

## 4. PERFORMANCE TROUBLESHOOTING

### Slow Processing Diagnosis

#### Performance Profiling Script
```python
#!/usr/bin/env python3
"""
Performance profiling for RAG Pipeline
"""

import time
import cProfile
import pstats
from io import StringIO

def profile_text_processing():
    """Profile text processing pipeline"""
    from common.text_processor import extract_text_from_file, chunk_text, create_embeddings
    
    # Load test file
    with open('test_document.pdf', 'rb') as f:
        content = f.read()
    
    # Profile each stage
    pr = cProfile.Profile()
    
    print("🔍 Profiling text processing pipeline...\n")
    
    # Stage 1: Text extraction
    pr.enable()
    start = time.time()
    text = extract_text_from_file(content, 'application/pdf')
    extraction_time = time.time() - start
    pr.disable()
    
    print(f"📄 Text Extraction: {extraction_time:.3f}s")
    print(f"   Extracted {len(text)} characters")
    
    # Stage 2: Chunking
    start = time.time()
    chunks = chunk_text(text, chunk_size=400)
    chunking_time = time.time() - start
    
    print(f"✂️ Text Chunking: {chunking_time:.3f}s")
    print(f"   Created {len(chunks)} chunks")
    
    # Stage 3: Embedding (limit to 5 chunks for testing)
    test_chunks = chunks[:5]
    start = time.time()
    embeddings = create_embeddings(test_chunks)
    embedding_time = time.time() - start
    
    print(f"🤖 Embedding Generation: {embedding_time:.3f}s")
    print(f"   Generated {len(embeddings)} embeddings")
    print(f"   Rate: {len(test_chunks)/embedding_time:.2f} chunks/second")
    
    # Total time projection
    total_embedding_time = (len(chunks) / len(test_chunks)) * embedding_time
    total_time = extraction_time + chunking_time + total_embedding_time
    
    print(f"\n📊 Full Document Projection:")
    print(f"   Total time estimate: {total_time:.3f}s")
    print(f"   Processing rate: {1/total_time:.2f} docs/second")
    
    # Show profiling results
    s = StringIO()
    ps = pstats.Stats(pr, stream=s)
    ps.sort_stats('tottime')
    ps.print_stats(10)  # Top 10 functions
    print(f"\n🔬 Performance Hotspots:")
    print(s.getvalue())

if __name__ == "__main__":
    profile_text_processing()
```

### Database Performance Optimization

#### Slow Query Diagnosis
```sql
-- Check for slow queries
SELECT query, mean_exec_time, calls, total_exec_time
FROM pg_stat_statements 
WHERE query LIKE '%documents%'
ORDER BY mean_exec_time DESC;

-- Analyze table statistics
ANALYZE documents;

-- Check index usage
SELECT 
    schemaname,
    tablename,
    attname,
    n_distinct,
    correlation
FROM pg_stats
WHERE tablename = 'documents';
```

#### Index Optimization
```sql
-- Create vector index for similarity search
CREATE INDEX CONCURRENTLY idx_documents_embedding 
ON documents USING ivfflat (embedding vector_cosine_ops) 
WITH (lists = 100);

-- Create partial index for metadata queries
CREATE INDEX CONCURRENTLY idx_documents_file_id 
ON documents USING gin ((metadata->>'file_id'));

-- Analyze after index creation
ANALYZE documents;
```

---

## 5. ALERTING & NOTIFICATIONS

### Error Detection Script
```python
#!/usr/bin/env python3
"""
Error detection and alerting system
"""

import json
import smtplib
from email.mime.text import MimeText
from datetime import datetime
import requests

class AlertManager:
    def __init__(self, config_file='alerts.json'):
        with open(config_file, 'r') as f:
            self.config = json.load(f)
    
    def check_error_patterns(self, log_file):
        """Check for error patterns in logs"""
        critical_errors = []
        warning_count = 0
        
        with open(log_file, 'r') as f:
            for line in f:
                if any(pattern in line.lower() for pattern in ['critical', 'fatal', 'exception']):
                    critical_errors.append(line.strip())
                elif 'warning' in line.lower():
                    warning_count += 1
        
        return critical_errors, warning_count
    
    def send_email_alert(self, subject, message):
        """Send email alert"""
        if not self.config.get('email', {}).get('enabled'):
            return
        
        email_config = self.config['email']
        msg = MimeText(message)
        msg['Subject'] = f"RAG Pipeline Alert: {subject}"
        msg['From'] = email_config['from']
        msg['To'] = email_config['to']
        
        with smtplib.SMTP(email_config['smtp_server'], email_config['port']) as server:
            if email_config.get('use_tls'):
                server.starttls()
            if email_config.get('username'):
                server.login(email_config['username'], email_config['password'])
            server.send_message(msg)
    
    def send_slack_alert(self, message):
        """Send Slack alert"""
        if not self.config.get('slack', {}).get('enabled'):
            return
        
        webhook_url = self.config['slack']['webhook_url']
        payload = {
            'text': f"🚨 RAG Pipeline Alert: {message}",
            'channel': self.config['slack'].get('channel', '#alerts')
        }
        
        requests.post(webhook_url, json=payload)
    
    def check_and_alert(self):
        """Main alerting logic"""
        # Check for errors
        critical_errors, warning_count = self.check_error_patterns('rag-pipeline.log')
        
        if critical_errors:
            message = f"Critical errors detected:\n" + "\n".join(critical_errors[:5])
            self.send_email_alert("Critical Errors", message)
            self.send_slack_alert(f"{len(critical_errors)} critical errors detected")
        
        if warning_count > 50:  # Threshold for warnings
            message = f"High warning count: {warning_count} warnings in recent logs"
            self.send_slack_alert(message)

# Example alerts.json configuration
alerts_config = {
    "email": {
        "enabled": True,
        "smtp_server": "smtp.gmail.com",
        "port": 587,
        "use_tls": True,
        "from": "alerts@yourcompany.com",
        "to": "admin@yourcompany.com",
        "username": "alerts@yourcompany.com",
        "password": "app_password"
    },
    "slack": {
        "enabled": True,
        "webhook_url": "https://hooks.slack.com/services/...",
        "channel": "#rag-pipeline-alerts"
    }
}
```

---

## 6. MAINTENANCE PROCEDURES

### Regular Maintenance Tasks

#### Daily Tasks
```bash
#!/bin/bash
# daily_maintenance.sh

echo "🔧 Daily RAG Pipeline Maintenance"

# 1. Check service status
systemctl status rag-pipeline

# 2. Analyze recent logs
python analyze_logs.py --since=24h

# 3. Check disk space
df -h | grep -E "(/$|/var|/tmp)"

# 4. Database health
python health_check.py --json > /tmp/health_$(date +%Y%m%d).json

# 5. Rotate logs if needed
if [ -f "rag-pipeline.log" ] && [ $(stat -f%z "rag-pipeline.log") -gt 104857600 ]; then
    mv rag-pipeline.log "rag-pipeline.log.$(date +%Y%m%d)"
    touch rag-pipeline.log
fi

echo "✅ Daily maintenance complete"
```

#### Weekly Tasks
```bash
#!/bin/bash
# weekly_maintenance.sh

echo "🔧 Weekly RAG Pipeline Maintenance"

# 1. Database statistics update
python -c "
from common.db_handler import supabase
supabase.rpc('update_statistics').execute()
print('✅ Database statistics updated')
"

# 2. Check for orphaned records
python -c "
from common.db_handler import supabase

# Find documents without metadata
orphaned = supabase.table('documents').select('metadata->file_id').execute()
file_ids = set(doc['metadata']['file_id'] for doc in orphaned.data)

metadata_result = supabase.table('document_metadata').select('id').execute()
metadata_ids = set(meta['id'] for meta in metadata_result.data)

orphaned_ids = file_ids - metadata_ids
print(f'Found {len(orphaned_ids)} orphaned documents')
"

# 3. Performance report
python -c "
from common.db_handler import supabase
import json

# Get processing stats
result = supabase.table('documents').select('metadata', count='exact').execute()
print(f'Total documents: {result.count}')

# Get recent activity (last 7 days)
recent = supabase.table('documents').select('id').gte('id', result.count - 1000).execute()
print(f'Recent documents: {len(recent.data)}')
"

echo "✅ Weekly maintenance complete"
```

#### Monthly Tasks
```bash
#!/bin/bash
# monthly_maintenance.sh

echo "🔧 Monthly RAG Pipeline Maintenance"

# 1. Full database backup
pg_dump $DATABASE_URL > "backup_$(date +%Y%m%d).sql"

# 2. Update dependencies
pip list --outdated

# 3. Security audit
pip audit

# 4. Performance review
python analyze_logs.py --since=720h > "monthly_report_$(date +%Y%m%d).txt"

echo "✅ Monthly maintenance complete"
```

---

## 7. DISASTER RECOVERY

### Backup Procedures
```bash
#!/bin/bash
# backup_procedure.sh

# 1. Database backup
pg_dump $DATABASE_URL | gzip > "db_backup_$(date +%Y%m%d_%H%M%S).sql.gz"

# 2. Configuration backup
tar -czf "config_backup_$(date +%Y%m%d).tar.gz" \
    .env \
    Google_Drive/config.json \
    Local_Files/config.json \
    Google_Drive/credentials.json

# 3. Upload to cloud storage (example with AWS S3)
aws s3 cp "db_backup_$(date +%Y%m%d_%H%M%S).sql.gz" s3://your-backup-bucket/rag-pipeline/
aws s3 cp "config_backup_$(date +%Y%m%d).tar.gz" s3://your-backup-bucket/rag-pipeline/
```

### Recovery Procedures
```bash
#!/bin/bash
# recovery_procedure.sh

echo "🔄 RAG Pipeline Recovery Procedure"

# 1. Stop services
systemctl stop rag-pipeline

# 2. Restore database
gunzip -c db_backup_20250115_120000.sql.gz | psql $DATABASE_URL

# 3. Restore configuration
tar -xzf config_backup_20250115.tar.gz

# 4. Verify restoration
python health_check.py

# 5. Restart services
systemctl start rag-pipeline

echo "✅ Recovery complete"
```

---

**End of Monitoring & Troubleshooting Guide**