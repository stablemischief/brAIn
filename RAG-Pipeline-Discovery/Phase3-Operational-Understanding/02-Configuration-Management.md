# Phase 3: Operational Understanding
## Part 2: Configuration Management & Tuning

---

# CONFIGURATION MANAGEMENT

## Overview
This document provides comprehensive guidance for configuring, tuning, and optimizing the RAG Pipeline across all environments and use cases.

---

## 1. CONFIGURATION ARCHITECTURE

### Configuration Hierarchy
```
Environment Variables (.env)
    ↓
Pipeline Config (config.json)
    ↓
Runtime Arguments (CLI flags)
    ↓
Default Values (code constants)
```

### Configuration Flow
1. **Environment Loading:** Parent `.env` file loaded first
2. **Pipeline Configs:** Module-specific JSON files
3. **CLI Overrides:** Runtime command-line arguments
4. **State Persistence:** Last check times and dynamic settings

---

## 2. ENVIRONMENT CONFIGURATION

### Core Environment Variables

#### Essential Settings
```env
# Embedding Configuration (REQUIRED)
EMBEDDING_PROVIDER=openai                     # Provider: openai, ollama
EMBEDDING_BASE_URL=https://api.openai.com/v1 # API endpoint
EMBEDDING_API_KEY=sk-...                     # Your API key
EMBEDDING_MODEL_CHOICE=text-embedding-3-small # Model name

# Database Configuration (REQUIRED)
SUPABASE_URL=https://xxx.supabase.co         # Your Supabase URL
SUPABASE_SERVICE_KEY=eyJ...                  # Service role key
```

#### Advanced Settings
```env
# Alternative Database (optional)
DATABASE_URL=postgresql://user:pass@host:5432/db

# LLM Configuration (for parent agent)
LLM_PROVIDER=openai
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=sk-...
LLM_CHOICE=gpt-4o-mini
VISION_LLM_CHOICE=gpt-4o-mini

# Search Configuration (optional)
BRAVE_API_KEY=your_brave_key
SEARXNG_BASE_URL=http://localhost:8081
```

### Provider-Specific Configurations

#### OpenAI Configuration
```env
EMBEDDING_PROVIDER=openai
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_API_KEY=sk-your_openai_key
EMBEDDING_MODEL_CHOICE=text-embedding-3-small  # 1536 dimensions
```

#### Ollama Configuration
```env
EMBEDDING_PROVIDER=ollama
EMBEDDING_BASE_URL=http://localhost:11434/v1
EMBEDDING_API_KEY=ollama
EMBEDDING_MODEL_CHOICE=nomic-embed-text  # 768 dimensions
```

#### OpenRouter Configuration
```env
EMBEDDING_PROVIDER=openai  # Use OpenAI-compatible endpoint
EMBEDDING_BASE_URL=https://openrouter.ai/api/v1
EMBEDDING_API_KEY=sk-or-your_openrouter_key
EMBEDDING_MODEL_CHOICE=text-embedding-3-small
```

---

## 3. PIPELINE CONFIGURATIONS

### Google Drive Configuration (`Google_Drive/config.json`)

#### Complete Configuration Template
```json
{
  "supported_mime_types": [
    "application/pdf",
    "text/plain",
    "text/html",
    "text/csv",
    "text/markdown",
    "application/vnd.google-apps.document",
    "application/vnd.google-apps.spreadsheet",
    "application/vnd.google-apps.presentation",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/msword",
    "application/vnd.ms-excel",
    "image/png",
    "image/jpeg",
    "image/jpg",
    "image/svg+xml"
  ],
  "export_mime_types": {
    "application/vnd.google-apps.document": "text/html",
    "application/vnd.google-apps.spreadsheet": "text/csv",
    "application/vnd.google-apps.presentation": "text/html"
  },
  "text_processing": {
    "default_chunk_size": 400,
    "default_chunk_overlap": 0
  },
  "watch_folder_id": "0AB4g0OsAn_LrUk9PVA",
  "last_check_time": "2025-09-08T22:21:15.417758Z"
}
```

#### Configuration Options Explained

**supported_mime_types:**
- Controls which file types are processed
- Add/remove types based on your needs
- Images require OCR capability

**export_mime_types:**
- Google Workspace file conversion mappings
- HTML preserves formatting and links
- CSV maintains tabular structure

**text_processing:**
- `default_chunk_size`: Characters per chunk (400 optimal for most embeddings)
- `default_chunk_overlap`: Characters to overlap between chunks (0 = no overlap)

**watch_folder_id:**
- Google Drive folder ID to monitor
- `null` = entire Drive (not recommended)
- Use specific folder for targeted processing

### Local Files Configuration (`Local_Files/config.json`)

```json
{
  "supported_mime_types": [
    "application/pdf",
    "text/plain",
    "text/html",
    "text/csv",
    "text/markdown",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-excel",
    "image/png",
    "image/jpeg"
  ],
  "text_processing": {
    "default_chunk_size": 400,
    "default_chunk_overlap": 0
  },
  "watch_directory": "/Users/username/Documents/RAG_Documents",
  "last_check_time": "2025-09-08T22:00:00.000Z"
}
```

---

## 4. PERFORMANCE TUNING

### Chunk Size Optimization

#### Embedding Model Guidelines
```json
{
  "text-embedding-3-small": {"optimal_chunk": 400, "max_chunk": 8000},
  "text-embedding-3-large": {"optimal_chunk": 500, "max_chunk": 8000},
  "text-embedding-ada-002": {"optimal_chunk": 400, "max_chunk": 8000},
  "nomic-embed-text": {"optimal_chunk": 300, "max_chunk": 2000}
}
```

#### Use Case Specific Tuning
```json
// For detailed documents (technical docs, reports)
{
  "default_chunk_size": 600,
  "default_chunk_overlap": 50
}

// For short documents (emails, notes)
{
  "default_chunk_size": 200,
  "default_chunk_overlap": 0
}

// For code documentation
{
  "default_chunk_size": 800,
  "default_chunk_overlap": 100
}
```

### Processing Interval Optimization

#### Interval Guidelines by Use Case
```bash
# Real-time collaboration (high activity)
--interval 30    # 30 seconds

# Regular business documents  
--interval 300   # 5 minutes

# Archive processing (low activity)
--interval 3600  # 1 hour

# One-time import
--interval 0     # Process once and exit
```

### Memory Optimization

#### Large File Handling
```json
{
  "text_processing": {
    "max_file_size_mb": 100,
    "stream_processing": true,
    "chunk_batch_size": 50
  }
}
```

---

## 5. ADVANCED CONFIGURATIONS

### Custom File Type Support

#### Adding New MIME Types
```json
{
  "supported_mime_types": [
    // Standard types...
    "application/x-custom-format",
    "text/x-special-text"
  ]
}
```

#### Custom Extraction Logic
```python
# In text_processor.py, add new extractor
def extract_text_from_custom(file_content: bytes) -> str:
    # Your custom extraction logic
    return processed_text

# Update the router function
def extract_text_from_file(file_content: bytes, mime_type: str):
    if mime_type == 'application/x-custom-format':
        return extract_text_from_custom(file_content)
    # ... existing logic
```

### Multi-Environment Configuration

#### Development Environment
```json
{
  "text_processing": {
    "default_chunk_size": 200,  // Smaller for testing
    "default_chunk_overlap": 0
  },
  "supported_mime_types": [
    "text/plain",               // Limited types for dev
    "application/pdf"
  ]
}
```

#### Production Environment
```json
{
  "text_processing": {
    "default_chunk_size": 400,  // Optimized size
    "default_chunk_overlap": 0
  },
  "supported_mime_types": [
    // Full list of supported types
  ],
  "performance": {
    "batch_size": 100,
    "concurrent_files": 5,
    "timeout_seconds": 300
  }
}
```

### Database Configuration Tuning

#### Vector Dimensions by Model
```sql
-- For text-embedding-3-small (1536 dimensions)
CREATE TABLE documents (
  embedding vector(1536)
);

-- For nomic-embed-text (768 dimensions)  
CREATE TABLE documents (
  embedding vector(768)
);

-- For text-embedding-3-large (3072 dimensions)
CREATE TABLE documents (
  embedding vector(3072)
);
```

#### Similarity Search Optimization
```sql
-- Create index for faster similarity search
CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- For larger datasets
CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops)
WITH (m = 16, ef_construction = 64);
```

---

## 6. SECURITY CONFIGURATIONS

### API Key Management

#### Key Rotation Strategy
```bash
# Environment variable backup
cp .env .env.backup.$(date +%Y%m%d)

# Update keys with new values
sed -i 's/EMBEDDING_API_KEY=sk-old.../EMBEDDING_API_KEY=sk-new.../' .env

# Verify new configuration
python -c "from common.text_processor import create_embeddings; print('✓ New key works')"
```

#### Access Control
```env
# Use read-only keys when possible
GOOGLE_DRIVE_READONLY=true
SUPABASE_READ_ONLY_KEY=your_anon_key  # Instead of service key for queries
```

### Network Security
```json
{
  "security": {
    "allowed_domains": [
      "api.openai.com",
      "*.supabase.co",
      "googleapis.com"
    ],
    "ssl_verify": true,
    "timeout_seconds": 30
  }
}
```

---

## 7. MONITORING CONFIGURATIONS

### Logging Configuration
```python
# Add to config.json
{
  "logging": {
    "level": "INFO",           // DEBUG, INFO, WARNING, ERROR
    "file": "rag-pipeline.log",
    "max_size_mb": 100,
    "backup_count": 5
  }
}
```

### Health Check Configuration
```json
{
  "health_checks": {
    "database_timeout": 10,
    "embedding_timeout": 30,
    "file_access_timeout": 60,
    "check_interval": 300
  }
}
```

### Metrics Collection
```json
{
  "metrics": {
    "enable_metrics": true,
    "metrics_file": "metrics.json",
    "collect_interval": 300,
    "retention_days": 30
  }
}
```

---

## 8. TROUBLESHOOTING CONFIGURATIONS

### Debug Mode Configuration
```json
{
  "debug": {
    "enabled": true,
    "save_intermediate_files": true,
    "verbose_logging": true,
    "skip_cleanup": true
  }
}
```

### Error Recovery Configuration
```json
{
  "error_handling": {
    "max_retries": 3,
    "retry_delay_seconds": 5,
    "fallback_extraction": true,
    "continue_on_error": true,
    "error_notification": {
      "email": "admin@example.com",
      "webhook": "https://hooks.slack.com/..."
    }
  }
}
```

---

## 9. DEPLOYMENT CONFIGURATIONS

### Docker Configuration
```dockerfile
# Dockerfile with configuration mounting
FROM python:3.11-slim

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV RAG_CONFIG_PATH=/app/config

# Mount configuration
VOLUME ["/app/config"]
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . /app
WORKDIR /app/RAG_Pipeline

# Default command with config override
CMD ["python", "Google_Drive/main.py", "--config", "/app/config/google_drive.json"]
```

### Production Systemd Configuration
```ini
[Unit]
Description=RAG Pipeline - Google Drive
After=network.target

[Service]
Type=simple
User=rag-user
Group=rag-group
WorkingDirectory=/opt/rag-pipeline/RAG_Pipeline
Environment=RAG_ENV=production
Environment=RAG_CONFIG_PATH=/etc/rag-pipeline
EnvironmentFile=/etc/rag-pipeline/.env
ExecStart=/opt/rag-pipeline/venv/bin/python Google_Drive/main.py --config /etc/rag-pipeline/google_drive.json
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

---

## 10. CONFIGURATION VALIDATION

### Validation Script
```python
#!/usr/bin/env python3
"""
Configuration validation script
Usage: python validate_config.py [config_file]
"""

import json
import sys
import os
from pathlib import Path

def validate_google_drive_config(config_path):
    """Validate Google Drive configuration"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    required_fields = [
        'supported_mime_types',
        'export_mime_types', 
        'text_processing',
        'last_check_time'
    ]
    
    for field in required_fields:
        if field not in config:
            print(f"❌ Missing required field: {field}")
            return False
    
    # Validate chunk size
    chunk_size = config['text_processing']['default_chunk_size']
    if not isinstance(chunk_size, int) or chunk_size < 50 or chunk_size > 10000:
        print(f"❌ Invalid chunk_size: {chunk_size} (must be 50-10000)")
        return False
    
    print("✅ Google Drive configuration is valid")
    return True

def validate_environment():
    """Validate environment variables"""
    required_vars = [
        'SUPABASE_URL',
        'SUPABASE_SERVICE_KEY', 
        'EMBEDDING_API_KEY',
        'EMBEDDING_MODEL_CHOICE'
    ]
    
    missing = []
    for var in required_vars:
        if not os.getenv(var):
            missing.append(var)
    
    if missing:
        print(f"❌ Missing environment variables: {', '.join(missing)}")
        return False
    
    print("✅ Environment configuration is valid")
    return True

if __name__ == "__main__":
    config_file = sys.argv[1] if len(sys.argv) > 1 else "Google_Drive/config.json"
    
    if not Path(config_file).exists():
        print(f"❌ Configuration file not found: {config_file}")
        sys.exit(1)
    
    valid_config = validate_google_drive_config(config_file)
    valid_env = validate_environment()
    
    if valid_config and valid_env:
        print("🎉 All configurations are valid!")
        sys.exit(0)
    else:
        print("❌ Configuration validation failed")
        sys.exit(1)
```

---

## 11. BEST PRACTICES

### Configuration Management
1. **Version Control:** Keep config templates in git, not actual values
2. **Environment Separation:** Different configs for dev/staging/prod
3. **Secret Management:** Use proper secret management systems
4. **Validation:** Always validate configurations before deployment
5. **Backup:** Maintain configuration backups

### Performance Optimization
1. **Chunk Size:** Test different sizes for your specific content
2. **Batch Processing:** Group similar operations
3. **Caching:** Cache embeddings for unchanged content
4. **Monitoring:** Track performance metrics continuously

### Security
1. **Principle of Least Privilege:** Use read-only keys when possible
2. **Key Rotation:** Regularly rotate API keys
3. **Access Logging:** Log all configuration changes
4. **Encryption:** Encrypt sensitive configuration data

---

**End of Configuration Management Guide**