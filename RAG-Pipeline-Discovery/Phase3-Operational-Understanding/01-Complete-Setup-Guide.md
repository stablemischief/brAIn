# Phase 3: Operational Understanding
## Part 1: Complete Setup & Installation Guide

---

# RAG PIPELINE SETUP & DEPLOYMENT

## Pre-Requisites Overview
This guide provides step-by-step instructions for deploying the RAG Pipeline in any environment, from development to production.

---

## 1. SYSTEM REQUIREMENTS

### Hardware Requirements
- **Memory:** 4GB+ RAM (8GB+ recommended for large files)
- **Storage:** 2GB+ free space for dependencies
- **Network:** Stable internet for API calls
- **CPU:** Multi-core recommended for concurrent processing

### Software Requirements
- **Python:** 3.11 or higher
- **Operating System:** macOS, Linux, or Windows
- **Virtual Environment:** venv or conda
- **Database:** Supabase account or PostgreSQL with PGVector

---

## 2. ENVIRONMENT SETUP

### Step 1: Clone and Navigate
```bash
# Clone the repository
git clone [repository-url]
cd 4_Pydantic_AI_Agent/RAG_Pipeline

# Verify structure
ls -la
# Should see: Google_Drive/, Local_Files/, common/, tests/, README.md
```

### Step 2: Python Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Linux/macOS:
source venv/bin/activate
# Windows:
# venv\Scripts\activate

# Verify activation
which python
# Should show path to venv/bin/python
```

### Step 3: Dependencies Installation
```bash
# Install all required packages
pip install -r requirements.txt

# Verify critical packages
pip list | grep -E "(supabase|openai|google-api-python-client|pypdf)"
```

**Expected Output:**
```
google-api-python-client==2.166.0
openai==1.71.0
pypdf==5.4.0
supabase==2.15.0
```

---

## 3. DATABASE SETUP

### Option A: Supabase Cloud Setup
1. **Create Account:** Go to [supabase.com](https://supabase.com)
2. **New Project:** Create a new project
3. **Get Credentials:** Project Settings → API
   - `SUPABASE_URL`: Your project URL
   - `SUPABASE_SERVICE_KEY`: Service role secret key
4. **Execute Schema:** Run SQL files in order:

```sql
-- In Supabase SQL Editor, run these in order:
-- 1. Execute: sql/documents.sql
-- 2. Execute: sql/document_metadata.sql  
-- 3. Execute: sql/document_rows.sql
-- 4. Execute: sql/execute_sql_rpc.sql (if needed)
```

### Option B: Local PostgreSQL Setup
```bash
# Install PostgreSQL with PGVector
# macOS:
brew install postgresql pgvector

# Ubuntu:
sudo apt-get install postgresql-14 postgresql-14-pgvector

# Start PostgreSQL
brew services start postgresql  # macOS
sudo service postgresql start   # Ubuntu

# Create database and user
createdb rag_pipeline
psql rag_pipeline
```

```sql
-- In psql, create user and enable extensions
CREATE USER rag_user WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE rag_pipeline TO rag_user;

-- Enable PGVector
CREATE EXTENSION IF NOT EXISTS vector;

-- Execute schema files (copy/paste contents)
\i sql/documents.sql
\i sql/document_metadata.sql
\i sql/document_rows.sql
```

---

## 4. ENVIRONMENT CONFIGURATION

### Step 1: Create Environment File
```bash
# Navigate to parent directory
cd ../

# Copy example environment file
cp .env.example .env

# Edit with your settings
nano .env  # or vim .env or your preferred editor
```

### Step 2: Configure Environment Variables

#### Essential Configuration
```env
# Embedding Configuration (REQUIRED)
EMBEDDING_PROVIDER=openai
EMBEDDING_BASE_URL=https://api.openai.com/v1
EMBEDDING_API_KEY=your_openai_api_key_here
EMBEDDING_MODEL_CHOICE=text-embedding-3-small

# Supabase Configuration (REQUIRED)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your_service_key_here
```

#### Optional Configuration
```env
# Database URL (if using local PostgreSQL)
DATABASE_URL=postgresql://rag_user:secure_password@localhost:5432/rag_pipeline

# LLM Configuration (for parent AI agent)
LLM_PROVIDER=openai
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=your_openai_api_key_here
LLM_CHOICE=gpt-4o-mini

# Vision Model (for image processing)
VISION_LLM_CHOICE=gpt-4o-mini
```

### Step 3: Validate Configuration
```bash
cd RAG_Pipeline

# Test database connection
python -c "
from common.db_handler import supabase
result = supabase.table('documents').select('id').limit(1).execute()
print(f'✓ Database connection successful: {len(result.data)} rows')
"

# Test embedding API
python -c "
from common.text_processor import create_embeddings
result = create_embeddings(['test text'])
print(f'✓ Embedding API successful: {len(result[0])} dimensions')
"
```

---

## 5. GOOGLE DRIVE SETUP (Optional)

### Step 1: Google Cloud Console Setup
1. **Go to:** [Google Cloud Console](https://console.cloud.google.com/)
2. **Create Project:** New project or select existing
3. **Enable APIs:** 
   - Google Drive API
   - (Optional) Google Docs API, Google Sheets API
4. **Create Credentials:**
   - Credentials → Create Credentials → OAuth 2.0 Client ID
   - Application type: Desktop application
   - Download JSON file

### Step 2: Credential Configuration
```bash
# Place credentials file
mv ~/Downloads/credentials.json RAG_Pipeline/Google_Drive/credentials.json

# Verify placement
ls -la Google_Drive/credentials.json
```

### Step 3: OAuth Authorization
```bash
# First run will trigger browser authentication
python Google_Drive/main.py --folder-id "your_folder_id" --interval 120

# After success, token.json will be created
ls -la Google_Drive/token.json
```

---

## 6. CONFIGURATION FILES

### Google Drive Configuration
Edit `Google_Drive/config.json`:
```json
{
  "supported_mime_types": [
    "application/pdf",
    "text/plain",
    "text/html",
    "text/csv",
    "application/vnd.google-apps.document",
    "application/vnd.google-apps.spreadsheet",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
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
  "watch_folder_id": "your_google_drive_folder_id",
  "last_check_time": "1970-01-01T00:00:00.000Z"
}
```

### Local Files Configuration
Edit `Local_Files/config.json`:
```json
{
  "supported_mime_types": [
    "application/pdf",
    "text/plain",
    "text/html",
    "text/csv",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
  ],
  "text_processing": {
    "default_chunk_size": 400,
    "default_chunk_overlap": 0
  },
  "watch_directory": "/path/to/your/documents",
  "last_check_time": "1970-01-01T00:00:00.000Z"
}
```

---

## 7. DEPLOYMENT OPTIONS

### Option A: Development Deployment
```bash
# Activate virtual environment
source venv/bin/activate

# Run Google Drive pipeline
python Google_Drive/main.py --folder-id "folder_id" --interval 60

# Or run Local Files pipeline (in separate terminal)
python Local_Files/main.py --directory "/path/to/docs" --interval 120
```

### Option B: Production Deployment with systemd
Create service file `/etc/systemd/system/rag-pipeline.service`:
```ini
[Unit]
Description=RAG Pipeline Google Drive Watcher
After=network.target

[Service]
Type=simple
User=rag-user
WorkingDirectory=/opt/rag-pipeline/RAG_Pipeline
Environment=PATH=/opt/rag-pipeline/venv/bin
ExecStart=/opt/rag-pipeline/venv/bin/python Google_Drive/main.py --folder-id "folder_id"
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# Enable and start service
sudo systemctl enable rag-pipeline
sudo systemctl start rag-pipeline

# Check status
sudo systemctl status rag-pipeline
```

### Option C: Docker Deployment
Create `Dockerfile`:
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["python", "Google_Drive/main.py"]
```

```bash
# Build and run
docker build -t rag-pipeline .
docker run -d --env-file .env rag-pipeline
```

---

## 8. INITIAL DATA LOAD

### Step 1: Test Connection
```bash
# Test with single file processing
python -c "
from common.text_processor import extract_text_from_file
from common.db_handler import process_file_for_rag

# Test file processing (provide test file path)
with open('test.txt', 'r') as f:
    content = f.read().encode()
    
text = extract_text_from_file(content, 'text/plain')
success = process_file_for_rag(content, text, 'test-id', 'test-url', 'test.txt', 'text/plain')
print(f'Test processing: {success}')
"
```

### Step 2: Full Initialization
```bash
# Google Drive: Full folder sync
python Google_Drive/main.py --folder-id "folder_id" --interval 0

# Local Files: Full directory scan
python Local_Files/main.py --directory "/docs" --interval 0
```

### Step 3: Verify Data Load
```bash
# Check document count
python -c "
from common.db_handler import supabase
result = supabase.table('documents').select('id', count='exact').execute()
print(f'Total documents: {result.count}')

# Check sample documents
sample = supabase.table('documents').select('*').limit(5).execute()
for doc in sample.data:
    print(f\"ID: {doc['id']}, Content: {doc['content'][:50]}...\")
"
```

---

## 9. VALIDATION & TESTING

### System Health Check
```bash
# Run comprehensive health check
python -c "
import sys
sys.path.append('.')

# Test all components
print('🔍 Testing RAG Pipeline Components...\n')

# 1. Environment variables
import os
from dotenv import load_dotenv
load_dotenv('../.env', override=True)

required_vars = ['SUPABASE_URL', 'SUPABASE_SERVICE_KEY', 'EMBEDDING_API_KEY']
for var in required_vars:
    if os.getenv(var):
        print(f'✅ {var}: Configured')
    else:
        print(f'❌ {var}: Missing')

# 2. Database connection
try:
    from common.db_handler import supabase
    result = supabase.table('documents').select('id').limit(1).execute()
    print(f'✅ Database: Connected ({result.count} documents)')
except Exception as e:
    print(f'❌ Database: {e}')

# 3. Embedding API
try:
    from common.text_processor import create_embeddings
    result = create_embeddings(['test'])
    print(f'✅ Embeddings: Working ({len(result[0])} dimensions)')
except Exception as e:
    print(f'❌ Embeddings: {e}')

print('\n🎉 Health check complete!')
"
```

### Performance Test
```bash
# Test processing speed
python -c "
import time
from common.text_processor import extract_text_from_file, chunk_text, create_embeddings

# Prepare test data
test_text = 'This is a test document. ' * 100  # ~2400 chars
test_content = test_text.encode()

# Time text extraction
start = time.time()
extracted = extract_text_from_file(test_content, 'text/plain')
extract_time = time.time() - start

# Time chunking
start = time.time()
chunks = chunk_text(extracted, chunk_size=400)
chunk_time = time.time() - start

# Time embedding
start = time.time()
embeddings = create_embeddings(chunks[:5])  # Test with 5 chunks
embed_time = time.time() - start

print(f'Performance Test Results:')
print(f'Text Extraction: {extract_time:.3f}s')
print(f'Chunking: {chunk_time:.3f}s')
print(f'Embeddings (5 chunks): {embed_time:.3f}s')
print(f'Total chunks: {len(chunks)}')
"
```

---

## 10. TROUBLESHOOTING SETUP ISSUES

### Common Issues

#### 1. Import Errors
```bash
# Error: Module not found
# Solution: Verify virtual environment
which python
pip list | grep supabase

# Reinstall if needed
pip install --force-reinstall -r requirements.txt
```

#### 2. Database Connection Errors
```bash
# Error: Connection refused
# Check 1: Verify credentials
python -c "import os; print(os.getenv('SUPABASE_URL'))"

# Check 2: Test direct connection
curl -H "apikey: $SUPABASE_SERVICE_KEY" "$SUPABASE_URL/rest/v1/documents?limit=1"
```

#### 3. Google Drive Authentication
```bash
# Error: Invalid credentials
# Solution: Re-download credentials.json from Google Cloud Console

# Error: OAuth scope issues
# Solution: Delete token.json and re-authenticate
rm Google_Drive/token.json
python Google_Drive/main.py --folder-id "test"
```

#### 4. Embedding API Issues
```bash
# Error: Invalid API key
# Check 1: Verify key format
echo $EMBEDDING_API_KEY | wc -c  # Should be ~51 characters for OpenAI

# Check 2: Test direct API call
curl -H "Authorization: Bearer $EMBEDDING_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"input":"test","model":"text-embedding-3-small"}' \
     https://api.openai.com/v1/embeddings
```

### Log Analysis
```bash
# Enable verbose logging
export PYTHONUNBUFFERED=1
python Google_Drive/main.py --folder-id "test" 2>&1 | tee rag-pipeline.log

# Monitor in real-time
tail -f rag-pipeline.log
```

---

## 11. NEXT STEPS

Once setup is complete:

1. **Start Monitoring:** Begin continuous file watching
2. **Set Up Alerts:** Monitor for processing errors
3. **Schedule Maintenance:** Regular database cleanup
4. **Configure Backups:** Protect your vector database
5. **Optimize Performance:** Tune chunk sizes and intervals

**Setup is now complete!** Your RAG Pipeline is ready to process documents and generate embeddings for your AI agent.

---

**End of Complete Setup Guide**