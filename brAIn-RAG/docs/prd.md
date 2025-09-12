# brAIn - Product Requirements Document (PRD)

## Version 1.0 - Internal MVP

---

## 1. Product Overview

### 1.1 Product Name
**brAIn** - Internal RAG Pipeline Management System

### 1.2 Product Description
brAIn is a containerized web application that wraps an existing CLI-based RAG Pipeline tool, providing a user-friendly dashboard for managing document vectorization from Google Drive folders into a Supabase vector database.

### 1.3 Target Users
- **Primary:** James (Product Owner)
- **Secondary:** Mitch (Partner)
- **Tertiary:** Selected colleague for testing
- **Team Size:** 2-3 concurrent users maximum

### 1.4 Deployment Environment
- **Platform:** Docker container on VPS
- **Access:** Web-based, public URL with authentication
- **Database:** Existing Supabase instance with PGVector
- **APIs:** OpenAI embeddings, Google Drive API

---

## 2. User Stories & Acceptance Criteria

### Story 1: Google Drive Folder Management
**As a team member, I want to manage Google Drive folders via web interface**

**Acceptance Criteria:**
- ✅ NO file upload functionality in MVP
- ✅ Web dashboard allows manual entry of Google Drive folder IDs
- ✅ Display list of watched folders with names (paths optional)
- ✅ Remove folders from watch list with confirmation dialog
- ✅ Initial folder scan queues all files as "new" for processing

**UI Components:**
```javascript
// Folder Management Component
const FolderManager = () => {
  return (
    <div className="folder-manager">
      <input 
        type="text" 
        placeholder="Enter Google Drive Folder ID"
        pattern="[a-zA-Z0-9_-]+"
      />
      <button onClick={addFolder}>Add Folder</button>
      <FolderList 
        folders={watchedFolders}
        onRemove={confirmAndRemoveFolder}
      />
    </div>
  );
};
```

### Story 2: Processing Status & Logs
**As a team member, I want to see processing status and logs**

**Acceptance Criteria:**
- ✅ Display current tool status (running/idle, current action)
- ✅ Show counts from prior cycle (files deleted/vectorized/failed)
- ✅ Export logs for past X hours (up to 24) as CSV
- ✅ Cumulative failed files list with clear button
- ✅ Store failure records in DB to prevent re-attempts

**Database Schema:**
```sql
-- Failure tracking with retry prevention
CREATE TABLE vectorization_failures (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  file_id TEXT NOT NULL,
  file_name TEXT NOT NULL,
  file_type TEXT,
  folder_id TEXT NOT NULL,
  failure_date TIMESTAMP DEFAULT NOW(),
  failure_reason TEXT,
  last_modified TIMESTAMP,
  retry_eligible BOOLEAN DEFAULT FALSE,
  is_cleared BOOLEAN DEFAULT FALSE,
  UNIQUE(file_id, folder_id)
);

-- Processing logs for export
CREATE TABLE processing_logs (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  poll_cycle_id UUID NOT NULL,
  timestamp TIMESTAMP DEFAULT NOW(),
  files_processed INTEGER DEFAULT 0,
  files_added INTEGER DEFAULT 0,
  files_deleted INTEGER DEFAULT 0,
  files_failed INTEGER DEFAULT 0,
  processing_time_ms INTEGER,
  details JSONB
);
```

### Story 3: Configuration Without CLI
**As a team admin, I want to configure the pipeline without CLI**

**Acceptance Criteria:**
- ✅ Installation wizard with form inputs for all configuration
- ✅ Generate complete SQL script for Supabase setup
- ✅ Support reinstall detection via DB check
- ✅ Validate all connections before starting

**Installation Configuration:**
```python
# config_schema.py
from pydantic import BaseModel, HttpUrl, SecretStr

class InstallationConfig(BaseModel):
    supabase_url: HttpUrl
    supabase_service_key: SecretStr
    openai_api_key: SecretStr
    google_service_account_json: str  # Base64 encoded or pasted JSON
    initial_folder_id: str
    admin_email: str
    is_reinstall: bool = False
    
    class Config:
        json_encoders = {
            SecretStr: lambda v: v.get_secret_value() if v else None
        }
```

**SQL Generation Template:**
```sql
-- Generated SQL for Supabase Setup
-- Copy and run this entire script in Supabase SQL Editor

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pg_stat_statements;

-- Main documents table with vector storage
CREATE TABLE IF NOT EXISTS documents (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  content TEXT NOT NULL,
  metadata JSONB,
  embedding vector(1536),
  created_at TIMESTAMP DEFAULT NOW(),
  updated_at TIMESTAMP DEFAULT NOW()
);

-- Document metadata table
CREATE TABLE IF NOT EXISTS document_metadata (
  id TEXT PRIMARY KEY,
  title TEXT,
  url TEXT,
  schema TEXT,
  created_at TIMESTAMP DEFAULT NOW()
);

-- Spreadsheet rows table
CREATE TABLE IF NOT EXISTS document_rows (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  dataset_id TEXT REFERENCES document_metadata(id) ON DELETE CASCADE,
  row_data JSONB NOT NULL,
  row_index INTEGER,
  created_at TIMESTAMP DEFAULT NOW()
);

-- Monitored folders configuration
CREATE TABLE IF NOT EXISTS monitored_folders (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  folder_id TEXT UNIQUE NOT NULL,
  folder_name TEXT,
  added_date TIMESTAMP DEFAULT NOW(),
  added_by TEXT,
  is_active BOOLEAN DEFAULT TRUE,
  last_poll TIMESTAMP
);

-- System metadata for reinstall detection
CREATE TABLE IF NOT EXISTS system_metadata (
  key TEXT PRIMARY KEY,
  value TEXT,
  updated_at TIMESTAMP DEFAULT NOW()
);

-- Insert initial schema version
INSERT INTO system_metadata (key, value) 
VALUES ('schema_version', '1.0.0')
ON CONFLICT (key) DO UPDATE SET value = '1.0.0', updated_at = NOW();

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_documents_metadata ON documents USING gin(metadata);
CREATE INDEX IF NOT EXISTS idx_documents_embedding ON documents USING ivfflat(embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_failures_folder ON vectorization_failures(folder_id, is_cleared);

-- Create search function
CREATE OR REPLACE FUNCTION search_documents(
  query_embedding vector(1536),
  match_count int DEFAULT 10,
  filter_metadata jsonb DEFAULT '{}'
)
RETURNS TABLE (
  id UUID,
  content TEXT,
  metadata JSONB,
  similarity FLOAT
)
LANGUAGE plpgsql
AS $$
BEGIN
  RETURN QUERY
  SELECT 
    d.id,
    d.content,
    d.metadata,
    1 - (d.embedding <=> query_embedding) AS similarity
  FROM documents d
  WHERE 
    CASE 
      WHEN filter_metadata = '{}'::jsonb THEN TRUE
      ELSE d.metadata @> filter_metadata
    END
  ORDER BY d.embedding <=> query_embedding
  LIMIT match_count;
END;
$$;
```

### Story 4: Search Functionality Testing
**As a team member, I want to test search functionality**

**Acceptance Criteria:**
- ✅ Search interface for file names and content
- ✅ Results displayed in markdown code blocks
- ✅ No chunk limitations for file recreation

**Search API Endpoint:**
```python
# api/search.py
from fastapi import APIRouter, Query, Depends
from typing import Optional, List
import markdown

router = APIRouter()

@router.get("/api/search")
async def search_documents(
    query: str = Query(..., description="Search query"),
    search_type: str = Query("content", regex="^(content|filename)$"),
    limit: int = Query(10, ge=1, le=100),
    current_user: User = Depends(get_current_user)
):
    """
    Search documents by content or filename
    Returns results formatted as markdown
    """
    if search_type == "filename":
        results = await search_by_filename(query, limit)
    else:
        results = await search_by_content(query, limit)
    
    # Format results as markdown
    markdown_output = format_search_results_markdown(results)
    
    return {
        "query": query,
        "type": search_type,
        "count": len(results),
        "results": markdown_output
    }

def format_search_results_markdown(results: List[dict]) -> str:
    """Format search results as markdown code blocks"""
    output = []
    for r in results:
        output.append(f"```markdown\n# {r['title']}\n")
        output.append(f"**File ID:** {r['file_id']}\n")
        output.append(f"**Similarity:** {r['similarity']:.2%}\n\n")
        output.append(f"{r['content']}\n```\n\n")
    return "\n".join(output)
```

### Story 5: Dashboard Status Monitoring
**As a team member, I want to easily check tool and connection status**

**Acceptance Criteria:**
- ✅ Status indicators: Down (Red), Starting (Blue), Unstable (Yellow), Up (Green)
- ✅ Monitor all connections (OpenAI, Supabase, Google Drive)
- ✅ Real-time updates via WebSocket or polling

**Status Monitoring Components:**
```python
# monitoring/health_checks.py
from enum import Enum
from typing import Dict, Any
import asyncio
from datetime import datetime

class ServiceStatus(Enum):
    DOWN = "down"
    STARTING = "starting"
    UNSTABLE = "unstable"
    UP = "up"

class HealthMonitor:
    def __init__(self):
        self.services = {
            "pipeline": ServiceStatus.STARTING,
            "supabase": ServiceStatus.DOWN,
            "openai": ServiceStatus.DOWN,
            "google_drive": ServiceStatus.DOWN
        }
    
    async def check_supabase(self) -> ServiceStatus:
        """Check Supabase connection"""
        try:
            # Test query
            result = await supabase.table('system_metadata').select('*').limit(1).execute()
            return ServiceStatus.UP if result else ServiceStatus.DOWN
        except Exception as e:
            logger.error(f"Supabase health check failed: {e}")
            return ServiceStatus.DOWN
    
    async def check_openai(self) -> ServiceStatus:
        """Check OpenAI API availability"""
        try:
            # Test with minimal tokens
            response = await openai.embeddings.create(
                model="text-embedding-3-small",
                input="health check"
            )
            return ServiceStatus.UP if response else ServiceStatus.DOWN
        except Exception as e:
            logger.error(f"OpenAI health check failed: {e}")
            return ServiceStatus.DOWN
    
    async def check_google_drive(self) -> ServiceStatus:
        """Check Google Drive API access"""
        try:
            # Test folder access with any monitored folder
            folders = await get_monitored_folders()
            if not folders:
                return ServiceStatus.UP  # No folders to check
            
            # Try to list files in first folder
            service = get_drive_service()
            result = service.files().list(
                q=f"'{folders[0]['folder_id']}' in parents",
                pageSize=1
            ).execute()
            return ServiceStatus.UP
        except Exception as e:
            logger.error(f"Google Drive health check failed: {e}")
            return ServiceStatus.DOWN
    
    async def run_health_checks(self) -> Dict[str, str]:
        """Run all health checks in parallel"""
        tasks = {
            "supabase": self.check_supabase(),
            "openai": self.check_openai(),
            "google_drive": self.check_google_drive()
        }
        
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        
        for service, result in zip(tasks.keys(), results):
            if isinstance(result, Exception):
                self.services[service] = ServiceStatus.DOWN
            else:
                self.services[service] = result
        
        # Determine overall pipeline status
        if all(s == ServiceStatus.UP for s in [
            self.services["supabase"],
            self.services["openai"], 
            self.services["google_drive"]
        ]):
            self.services["pipeline"] = ServiceStatus.UP
        elif any(s == ServiceStatus.DOWN for s in self.services.values()):
            self.services["pipeline"] = ServiceStatus.UNSTABLE
        
        return {k: v.value for k, v in self.services.items()}
```

### Story 6: File Type Handling
**As a team member, I want robust file type handling without corruption**

**Acceptance Criteria:**
- ✅ Strip images from docs before vectorization
- ✅ Reject unrecognized file types with clear error
- ✅ Handle spreadsheets with proper row/metadata structure

**File Processing Logic:**
```python
# processing/file_handlers.py
from typing import Optional, Dict, Any
import mimetypes
from pathlib import Path

SUPPORTED_FORMATS = {
    'application/pdf': 'pdf',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': 'docx',
    'application/msword': 'doc',
    'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': 'xlsx',
    'application/vnd.ms-excel': 'xls',
    'application/vnd.openxmlformats-officedocument.presentationml.presentation': 'pptx',
    'application/vnd.ms-powerpoint': 'ppt',
    'text/plain': 'txt',
    'text/csv': 'csv',
    'text/html': 'html',
    'text/markdown': 'md',
    'application/vnd.google-apps.document': 'gdoc',
    'application/vnd.google-apps.spreadsheet': 'gsheet',
    'application/vnd.google-apps.presentation': 'gslides'
}

class FileProcessor:
    def __init__(self):
        self.failed_files = []
        self.on_hold_files = []
    
    def validate_file_type(self, file_path: str, mime_type: str) -> bool:
        """Check if file type is supported"""
        if mime_type not in SUPPORTED_FORMATS:
            self.log_failure(
                file_path, 
                "Unrecognized file type", 
                mime_type
            )
            return False
        return True
    
    def check_file_size(self, file_path: str, size_bytes: int) -> str:
        """Check file size and determine processing status"""
        MAX_SIZE = 100 * 1024 * 1024  # 100MB
        
        if size_bytes > MAX_SIZE:
            self.add_to_hold(file_path, size_bytes, "size_exceeded")
            return "on_hold"
        return "process"
    
    def strip_images_from_document(self, content: str, file_type: str) -> str:
        """Remove image data from documents to prevent corruption"""
        if file_type in ['docx', 'gdoc', 'pdf']:
            # Remove base64 encoded images
            import re
            # Pattern for base64 images
            img_pattern = r'data:image/[^;]+;base64,[A-Za-z0-9+/=]+'
            content = re.sub(img_pattern, '[IMAGE_REMOVED]', content)
            
            # Remove image tags
            content = re.sub(r'<img[^>]*>', '[IMAGE_REMOVED]', content)
            
        return content
    
    def process_spreadsheet(self, file_path: str, file_id: str) -> Dict[str, Any]:
        """Special handling for spreadsheet files"""
        # Extract using existing pipeline logic
        # Reference: RAG-Pipeline-Discovery/Phase2-Technical-Analysis/02-database-vector-storage.md
        
        metadata = {
            "file_id": file_id,
            "file_type": "spreadsheet",
            "sheets": []
        }
        
        rows_data = []
        
        # Process each sheet
        for sheet_name, sheet_data in extract_sheets(file_path).items():
            metadata["sheets"].append(sheet_name)
            
            for idx, row in enumerate(sheet_data):
                rows_data.append({
                    "dataset_id": file_id,
                    "row_data": row,
                    "row_index": idx,
                    "sheet_name": sheet_name
                })
        
        return {
            "metadata": metadata,
            "rows": rows_data
        }
```

### Story 7: Folder Association Cleanup
**As a team member, I want automatic cleanup when folders are removed**

**Acceptance Criteria:**
- ✅ Delete all documents when folder removed from watch list
- ✅ Show confirmation dialog with document count
- ✅ Complete removal from database (no soft delete)

**Cleanup Implementation:**
```python
# cleanup/folder_operations.py
async def remove_folder_and_cleanup(folder_id: str, user_email: str) -> Dict[str, Any]:
    """
    Remove folder from watch list and cleanup all associated data
    Returns count of deleted documents
    """
    # Get document count for confirmation
    count_query = """
        SELECT COUNT(*) as doc_count 
        FROM documents 
        WHERE metadata->>'folder_id' = $1
    """
    
    doc_count = await database.fetch_one(
        query=count_query,
        values={"folder_id": folder_id}
    )
    
    # After user confirmation, proceed with deletion
    async with database.transaction():
        # Delete from documents table
        delete_docs = """
            DELETE FROM documents 
            WHERE metadata->>'folder_id' = $1
        """
        
        # Delete from document_metadata
        delete_metadata = """
            DELETE FROM document_metadata
            WHERE id IN (
                SELECT id FROM document_metadata
                WHERE id LIKE $1 || '%'
            )
        """
        
        # Delete from document_rows (cascades from metadata)
        # Already handled by ON DELETE CASCADE
        
        # Delete from failures table
        delete_failures = """
            DELETE FROM vectorization_failures
            WHERE folder_id = $1
        """
        
        # Remove from monitored folders
        remove_folder = """
            DELETE FROM monitored_folders
            WHERE folder_id = $1
        """
        
        # Execute all deletions
        await database.execute(delete_docs, folder_id)
        await database.execute(delete_metadata, folder_id)
        await database.execute(delete_failures, folder_id)
        await database.execute(remove_folder, folder_id)
        
        # Log the action
        await log_folder_removal(folder_id, user_email, doc_count)
    
    return {
        "folder_id": folder_id,
        "documents_deleted": doc_count,
        "status": "success"
    }
```

---

## 3. Technical Architecture

### 3.1 System Architecture

```
┌─────────────────────────────────────────────────┐
│              Docker Container                    │
│  ┌─────────────────────────────────────────────┐ │
│  │          FastAPI Backend (Port 8000)        │ │
│  │  ┌─────────────────────────────────────────┐│ │
│  │  │   API Endpoints                        ││ │
│  │  │   - /api/auth/*  (Supabase Auth)      ││ │
│  │  │   - /api/folders/* (CRUD)             ││ │
│  │  │   - /api/status/* (Health)            ││ │
│  │  │   - /api/logs/* (Export)              ││ │
│  │  │   - /api/search/* (Testing)           ││ │
│  │  │   - /api/config/* (Install)           ││ │
│  │  │   - /ws/status (WebSocket)            ││ │
│  │  └─────────────────────────────────────────┘│ │
│  │  ┌─────────────────────────────────────────┐│ │
│  │  │   RAG Pipeline Core (Existing)         ││ │
│  │  │   - Google Drive Monitor               ││ │
│  │  │   - Text Extraction Engine             ││ │
│  │  │   - Embedding Generator                ││ │
│  │  │   - Vector DB Operations               ││ │
│  │  └─────────────────────────────────────────┘│ │
│  └─────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────┐ │
│  │       React Frontend (Port 3000)            │ │
│  │   - Responsive Dashboard                    │ │
│  │   - Tailwind CSS Styling                   │ │
│  │   - Real-time Status Updates               │ │
│  └─────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────┐ │
│  │          Background Services                │ │
│  │   - Polling Scheduler                      │ │
│  │   - Health Monitor                         │ │
│  │   - Email Queue Processor                  │ │
│  └─────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
   Supabase DB    OpenAI API    Google Drive API
```

### 3.2 Directory Structure

```
brAIn/
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── backend/
│   ├── main.py                 # FastAPI application
│   ├── api/
│   │   ├── auth.py             # Supabase auth endpoints
│   │   ├── folders.py          # Folder management
│   │   ├── status.py           # Health monitoring
│   │   ├── logs.py             # Log export
│   │   ├── search.py           # Search testing
│   │   └── config.py           # Installation wizard
│   ├── core/
│   │   └── [existing RAG pipeline files]
│   ├── models/
│   │   ├── database.py         # SQLAlchemy models
│   │   ├── schemas.py          # Pydantic schemas
│   │   └── config.py           # Configuration models
│   ├── services/
│   │   ├── polling.py          # Polling scheduler
│   │   ├── health.py           # Health monitoring
│   │   ├── email.py            # Email notifications
│   │   └── cleanup.py          # Folder cleanup
│   └── utils/
│       ├── database.py         # DB connections
│       ├── auth.py             # Auth helpers
│       └── logging.py          # Logging config
├── frontend/
│   ├── package.json
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── Dashboard.jsx
│   │   │   ├── FolderManager.jsx
│   │   │   ├── StatusIndicator.jsx
│   │   │   ├── LogViewer.jsx
│   │   │   ├── SearchInterface.jsx
│   │   │   └── InstallWizard.jsx
│   │   ├── services/
│   │   │   ├── api.js
│   │   │   ├── auth.js
│   │   │   └── websocket.js
│   │   └── styles/
│   │       └── tailwind.css
│   └── public/
├── scripts/
│   ├── generate_sql.py         # SQL generation script
│   └── install.sh              # Installation helper
├── config/
│   ├── config.example.json    # Example configuration
│   └── requirements.txt       # Python dependencies
└── docs/
    ├── project-brief.md
    ├── prd.md
    └── deployment.md
```

### 3.3 Source File References

**IMPORTANT:** These files should be copied/referenced but NOT linked to the new build:

1. **TypingMind Plugin Configuration:**
   - Source: `enhanced_kb_search_FIXED.json`
   - Usage: Reference for search API structure

2. **Existing RAG Pipeline Core:**
   - Source: `/RAG-Pipeline-Discovery/` (entire structure)
   - Key files to copy:
     ```
     - text_processing.py (957 lines)
     - database_operations.py (361 lines)
     - google_drive_integration.py (552 lines)
     - file_extractors/*.py (all format handlers)
     ```

3. **Configuration Templates:**
   - Source: Existing `config.json` from RAG Pipeline
   - Adapt for web-based configuration

4. **Database Schema:**
   - Source: Existing Supabase structure
   - Enhance with new tables (see SQL above)

---

## 4. Implementation Details

### 4.1 Docker Configuration

```dockerfile
# Dockerfile
FROM python:3.11-slim as backend-build

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY backend/ ./backend/

# Build frontend
FROM node:18-alpine as frontend-build

WORKDIR /app

COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build

# Final image
FROM python:3.11-slim

WORKDIR /app

# Copy from build stages
COPY --from=backend-build /app /app
COPY --from=frontend-build /app/dist /app/static

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libpq-dev \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Copy supervisor config
COPY docker/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000

# Start supervisor
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
```

### 4.2 Polling Configuration

```python
# services/polling.py
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from datetime import datetime, timedelta
import asyncio

class PollingScheduler:
    def __init__(self):
        self.scheduler = AsyncIOScheduler()
        self.current_interval = 3600  # Default 1 hour
        self.is_running = False
        self.last_poll = None
        
    def configure_polling(self, interval_seconds: int):
        """Configure polling interval (1 second to 24 hours)"""
        MIN_INTERVAL = 1
        MAX_INTERVAL = 86400  # 24 hours
        
        interval_seconds = max(MIN_INTERVAL, min(interval_seconds, MAX_INTERVAL))
        self.current_interval = interval_seconds
        
        # Restart scheduler with new interval
        if self.is_running:
            self.scheduler.remove_all_jobs()
            self.scheduler.add_job(
                self.poll_folders,
                'interval',
                seconds=interval_seconds,
                id='folder_polling',
                replace_existing=True
            )
    
    def format_interval_display(self) -> str:
        """Format interval for dashboard display"""
        interval = self.current_interval
        
        if interval < 60:
            return f"{interval} seconds"
        elif interval < 3600:
            return f"{interval // 60} minutes"
        else:
            return f"{interval // 3600} hours"
    
    async def poll_folders(self):
        """Main polling logic"""
        self.is_running = True
        poll_id = str(uuid.uuid4())
        
        try:
            # Get active folders
            folders = await get_active_folders()
            
            stats = {
                "files_processed": 0,
                "files_added": 0,
                "files_deleted": 0,
                "files_failed": 0
            }
            
            for folder in folders:
                folder_stats = await process_folder(folder['folder_id'])
                for key in stats:
                    stats[key] += folder_stats.get(key, 0)
            
            # Log results
            await log_poll_cycle(poll_id, stats)
            
            # Send notifications if failures
            if stats["files_failed"] > 0:
                await queue_failure_notification(stats["files_failed"])
            
            self.last_poll = datetime.now()
            
        except Exception as e:
            logger.error(f"Polling error: {e}")
            await log_poll_error(poll_id, str(e))
        
        finally:
            self.is_running = False
    
    async def manual_trigger(self, user_email: str) -> bool:
        """Manual polling trigger with single instance enforcement"""
        if self.is_running:
            return False  # Already running
        
        # Run polling immediately
        await self.poll_folders()
        
        # Log manual trigger
        await log_manual_trigger(user_email)
        
        return True
```

### 4.3 Authentication Implementation

```python
# api/auth.py
from fastapi import APIRouter, Depends, HTTPException
from supabase import create_client
import os

router = APIRouter()

# Initialize Supabase client
supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

@router.post("/api/auth/login")
async def login(email: str):
    """Send magic link to user email"""
    try:
        response = supabase.auth.sign_in_with_otp({
            "email": email,
            "options": {
                "email_redirect_to": f"{os.getenv('APP_URL')}/dashboard"
            }
        })
        return {"message": "Check your email for login link"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/api/auth/verify")
async def verify_token(token: str):
    """Verify magic link token"""
    try:
        response = supabase.auth.verify_otp({
            "token": token,
            "type": "magiclink"
        })
        return {
            "access_token": response.session.access_token,
            "user": response.user
        }
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid token")

async def get_current_user(authorization: str = Header(None)):
    """Dependency to get current authenticated user"""
    if not authorization:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    try:
        token = authorization.replace("Bearer ", "")
        user = supabase.auth.get_user(token)
        return user
    except:
        raise HTTPException(status_code=401, detail="Invalid token")
```

### 4.4 Frontend Dashboard Component

```jsx
// components/Dashboard.jsx
import React, { useState, useEffect } from 'react';
import { StatusIndicator } from './StatusIndicator';
import { FolderManager } from './FolderManager';
import { LogViewer } from './LogViewer';
import { useWebSocket } from '../hooks/useWebSocket';

export const Dashboard = () => {
  const [status, setStatus] = useState({});
  const [pollingInterval, setPollingInterval] = useState(3600);
  const [isPolling, setIsPolling] = useState(false);
  
  // WebSocket for real-time updates
  const { messages } = useWebSocket('/ws/status');
  
  useEffect(() => {
    // Update status from WebSocket messages
    if (messages.length > 0) {
      const latest = messages[messages.length - 1];
      setStatus(latest.status);
      setIsPolling(latest.isPolling);
    }
  }, [messages]);
  
  const handleManualPoll = async () => {
    if (isPolling) {
      alert('Polling already in progress');
      return;
    }
    
    const response = await fetch('/api/polling/trigger', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${localStorage.getItem('token')}`
      }
    });
    
    if (response.ok) {
      setIsPolling(true);
    }
  };
  
  const formatInterval = (seconds) => {
    if (seconds < 60) return `${seconds} seconds`;
    if (seconds < 3600) return `${Math.floor(seconds / 60)} minutes`;
    return `${Math.floor(seconds / 3600)} hours`;
  };
  
  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header with Status Indicators */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
          <div className="flex items-center justify-between">
            <h1 className="text-2xl font-bold text-gray-900">brAIn Dashboard</h1>
            <div className="flex items-center space-x-4">
              <StatusIndicator 
                service="Pipeline" 
                status={status.pipeline || 'down'} 
              />
              <StatusIndicator 
                service="Supabase" 
                status={status.supabase || 'down'} 
              />
              <StatusIndicator 
                service="OpenAI" 
                status={status.openai || 'down'} 
              />
              <StatusIndicator 
                service="Google Drive" 
                status={status.google_drive || 'down'} 
              />
            </div>
          </div>
        </div>
      </header>
      
      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          
          {/* Polling Control */}
          <div className="lg:col-span-3">
            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-lg font-semibold">Polling Control</h2>
                  <p className="text-sm text-gray-600 mt-1">
                    Current interval: {formatInterval(pollingInterval)}
                  </p>
                  {isPolling && (
                    <p className="text-sm text-blue-600 mt-1 animate-pulse">
                      Polling in progress...
                    </p>
                  )}
                </div>
                <button
                  onClick={handleManualPoll}
                  disabled={isPolling}
                  className={`px-4 py-2 rounded-md font-medium ${
                    isPolling
                      ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                      : 'bg-blue-600 text-white hover:bg-blue-700'
                  }`}
                >
                  {isPolling ? 'Processing...' : 'Manual Poll'}
                </button>
              </div>
            </div>
          </div>
          
          {/* Folder Manager */}
          <div className="lg:col-span-2">
            <FolderManager />
          </div>
          
          {/* Recent Activity */}
          <div className="lg:col-span-1">
            <LogViewer />
          </div>
          
        </div>
      </main>
    </div>
  );
};
```

### 4.5 Email Notification Service

```python
# services/email.py
from typing import List, Dict
import asyncio
from datetime import datetime, timedelta
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class EmailNotificationService:
    def __init__(self):
        self.queue = []
        self.batch_interval = 900  # 15 minutes
        self.admin_email = os.getenv("ADMIN_EMAIL")
        
    async def queue_failure_notification(self, failures: List[Dict]):
        """Queue failure notifications for batching"""
        self.queue.extend(failures)
        
        # Schedule batch send if not already scheduled
        if not hasattr(self, '_batch_task'):
            self._batch_task = asyncio.create_task(
                self._send_batch_after_delay()
            )
    
    async def _send_batch_after_delay(self):
        """Wait for batch interval then send all queued notifications"""
        await asyncio.sleep(self.batch_interval)
        
        if self.queue:
            await self.send_failure_summary(self.queue)
            self.queue = []
        
        delattr(self, '_batch_task')
    
    async def send_failure_summary(self, failures: List[Dict]):
        """Send summary email of all failures"""
        if not self.admin_email:
            logger.warning("No admin email configured")
            return
        
        # Create email content
        subject = f"brAIn: {len(failures)} File Processing Failures"
        
        html_content = f"""
        <html>
          <body>
            <h2>File Processing Failure Report</h2>
            <p>The following files failed to process:</p>
            <table border="1" cellpadding="5">
              <tr>
                <th>File Name</th>
                <th>File Type</th>
                <th>Failure Reason</th>
                <th>Timestamp</th>
              </tr>
        """
        
        for failure in failures:
            html_content += f"""
              <tr>
                <td>{failure['file_name']}</td>
                <td>{failure['file_type']}</td>
                <td>{failure['reason']}</td>
                <td>{failure['timestamp']}</td>
              </tr>
            """
        
        html_content += f"""
            </table>
            <p><a href="{os.getenv('APP_URL')}/dashboard#failures">
              View in Dashboard
            </a></p>
          </body>
        </html>
        """
        
        await self._send_email(subject, html_content)
    
    async def send_critical_alert(self, alert_type: str, details: str):
        """Send immediate critical alerts"""
        if not self.admin_email:
            return
        
        subject = f"brAIn CRITICAL: {alert_type}"
        
        html_content = f"""
        <html>
          <body>
            <h2 style="color: red;">Critical Alert</h2>
            <p><strong>Type:</strong> {alert_type}</p>
            <p><strong>Details:</strong> {details}</p>
            <p><strong>Time:</strong> {datetime.now().isoformat()}</p>
            <p>
              <a href="{os.getenv('APP_URL')}/dashboard">
                Open Dashboard
              </a>
            </p>
          </body>
        </html>
        """
        
        await self._send_email(subject, html_content, is_critical=True)
```

---

## 5. Testing Requirements

### 5.1 Unit Tests

```python
# tests/test_folder_operations.py
import pytest
from unittest.mock import Mock, patch

@pytest.mark.asyncio
async def test_add_folder_validation():
    """Test folder ID validation"""
    # Valid folder ID
    valid_id = "1A2B3C4D5E6F7G8H9I0J"
    assert validate_folder_id(valid_id) == True
    
    # Invalid folder IDs
    assert validate_folder_id("") == False
    assert validate_folder_id("invalid-chars!") == False
    assert validate_folder_id(None) == False

@pytest.mark.asyncio
async def test_folder_removal_cascades():
    """Test that folder removal deletes all associated data"""
    folder_id = "test_folder_123"
    
    # Mock database responses
    with patch('database.fetch_one') as mock_fetch:
        mock_fetch.return_value = {"doc_count": 42}
        
        with patch('database.execute') as mock_execute:
            result = await remove_folder_and_cleanup(folder_id, "test@example.com")
            
            # Verify all delete queries were called
            assert mock_execute.call_count == 4  # docs, metadata, failures, folder
            assert result["documents_deleted"] == 42

@pytest.mark.asyncio
async def test_polling_single_instance():
    """Test that only one polling instance can run at a time"""
    scheduler = PollingScheduler()
    
    # Start first poll
    task1 = asyncio.create_task(scheduler.manual_trigger("user1@example.com"))
    
    # Try to start second poll immediately
    task2 = asyncio.create_task(scheduler.manual_trigger("user2@example.com"))
    
    result1 = await task1
    result2 = await task2
    
    assert result1 == True  # First should succeed
    assert result2 == False  # Second should be blocked
```

### 5.2 Integration Tests

```python
# tests/test_integration.py
@pytest.mark.integration
async def test_end_to_end_document_processing():
    """Test complete document processing flow"""
    # Add folder
    folder_id = "test_integration_folder"
    await add_monitored_folder(folder_id, "Test Folder")
    
    # Trigger polling
    await trigger_manual_poll("test@example.com")
    
    # Verify documents were processed
    docs = await get_documents_by_folder(folder_id)
    assert len(docs) > 0
    
    # Verify vector embeddings exist
    for doc in docs:
        assert doc['embedding'] is not None
        assert len(doc['embedding']) == 1536  # OpenAI dimension
    
    # Cleanup
    await remove_folder_and_cleanup(folder_id, "test@example.com")
```

### 5.3 Load Testing

```python
# tests/test_performance.py
@pytest.mark.performance
async def test_large_folder_processing():
    """Test processing folder with 1000+ files"""
    # Create test folder with many files
    folder_id = create_test_folder_with_files(count=1000)
    
    start_time = time.time()
    
    # Process folder
    await process_folder(folder_id)
    
    processing_time = time.time() - start_time
    
    # Should complete within reasonable time
    assert processing_time < 600  # 10 minutes for 1000 files
    
    # Verify memory usage stayed reasonable
    memory_usage = get_memory_usage()
    assert memory_usage < 2048  # Less than 2GB
```

---

## 6. Deployment Configuration

### 6.1 Environment Variables

```bash
# .env.example
# Supabase Configuration
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_KEY=your-service-key
SUPABASE_ANON_KEY=your-anon-key

# OpenAI Configuration
OPENAI_API_KEY=sk-your-api-key
OPENAI_MODEL=text-embedding-3-small

# Google Drive Configuration
GOOGLE_SERVICE_ACCOUNT_JSON=base64-encoded-json
GOOGLE_DRIVE_SCOPES=https://www.googleapis.com/auth/drive.readonly

# Application Configuration
APP_URL=https://your-domain.com
ADMIN_EMAIL=admin@example.com
SECRET_KEY=your-secret-key-for-sessions
LOG_LEVEL=INFO

# Database Configuration
DATABASE_POOL_SIZE=10
DATABASE_MAX_OVERFLOW=20

# Polling Configuration
DEFAULT_POLLING_INTERVAL=3600
MAX_FILE_SIZE_MB=100

# Email Configuration (Optional)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=app-specific-password
```

### 6.2 Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  brain:
    build:
      context: .
      dockerfile: docker/Dockerfile
    container_name: brain-rag
    ports:
      - "8000:8000"
    environment:
      - NODE_ENV=production
      - PYTHONUNBUFFERED=1
    env_file:
      - .env
    volumes:
      - ./config:/app/config:ro
      - ./logs:/app/logs
      - ./data:/app/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    networks:
      - brain-network

networks:
  brain-network:
    driver: bridge
```

### 6.3 Deployment Steps

```bash
#!/bin/bash
# deploy.sh - Deployment script for VPS

# 1. Clone repository
git clone https://github.com/your-repo/brain.git
cd brain

# 2. Create environment file
cp .env.example .env
# Edit .env with your configuration

# 3. Build Docker image
docker-compose build

# 4. Run database migrations
docker-compose run --rm brain python scripts/generate_sql.py > setup.sql
# Copy setup.sql content to Supabase SQL Editor

# 5. Start services
docker-compose up -d

# 6. Check health
docker-compose ps
docker-compose logs -f brain

# 7. Set up SSL with nginx (optional)
# Configure nginx reverse proxy with Let's Encrypt
```

---

## 7. Monitoring & Maintenance

### 7.1 Health Check Endpoints

```python
# api/health.py
@router.get("/health")
async def health_check():
    """Basic health check"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@router.get("/health/detailed")
async def detailed_health_check(current_user: User = Depends(get_current_user)):
    """Detailed health check with service status"""
    monitor = HealthMonitor()
    statuses = await monitor.run_health_checks()
    
    return {
        "services": statuses,
        "memory_usage_mb": get_memory_usage(),
        "active_folders": await count_active_folders(),
        "pending_files": await count_pending_files(),
        "last_poll": await get_last_poll_time()
    }
```

### 7.2 Logging Configuration

```python
# utils/logging.py
import logging
from logging.handlers import RotatingFileHandler

def setup_logging():
    """Configure application logging"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        'logs/brain.log',
        maxBytes=10485760,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.DEBUG,
        format=log_format,
        handlers=[console_handler, file_handler]
    )
    
    # Set third-party loggers to WARNING
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('googleapiclient').setLevel(logging.WARNING)
```

---

## 8. Security Considerations

### 8.1 Authentication & Authorization

- Supabase Auth with magic links (no password storage)
- JWT tokens with 7-day expiry
- Admin vs User role separation
- Rate limiting on API endpoints

### 8.2 Data Security

- All credentials in environment variables
- No sensitive data in logs
- HTTPS only for production
- Input validation on all endpoints
- SQL injection prevention via parameterized queries

### 8.3 Container Security

- Non-root user in container
- Minimal base image (python:3.11-slim)
- No unnecessary packages
- Regular security updates
- Health checks for monitoring

---

## 9. Performance Specifications

### 9.1 Expected Performance

- **Document Processing:** 6-20 documents/minute
- **API Response Time:** <500ms for most endpoints
- **WebSocket Latency:** <100ms for status updates
- **Memory Usage:** 512MB baseline + file processing overhead
- **Concurrent Users:** 2-3 without performance degradation

### 9.2 Optimization Strategies

- Database connection pooling
- Caching frequently accessed data
- Batch processing for large folders
- Async operations throughout
- Efficient vector similarity search with indexes

---

## 10. Acceptance Testing Checklist

### Installation & Setup
- [ ] Installation wizard loads correctly
- [ ] All configuration fields validate properly
- [ ] SQL script generates completely
- [ ] Reinstall detection works
- [ ] Initial setup completes successfully

### Authentication
- [ ] Magic link login works
- [ ] Session persists across page refreshes
- [ ] Logout functions properly
- [ ] Unauthorized access blocked

### Folder Management
- [ ] Add folder with valid ID succeeds
- [ ] Invalid folder ID shows error
- [ ] Folder list displays correctly
- [ ] Remove folder shows confirmation dialog
- [ ] Removal cascades all deletions

### Processing & Monitoring
- [ ] Polling runs on schedule
- [ ] Manual trigger works (single instance)
- [ ] Status indicators update in real-time
- [ ] Processing logs display correctly
- [ ] Failed files list accumulates properly
- [ ] Clear failed files works

### Search & Export
- [ ] Search by filename returns results
- [ ] Search by content works
- [ ] Results display in markdown format
- [ ] Log export generates valid CSV
- [ ] Export includes requested timeframe

### File Handling
- [ ] Supported file types process successfully
- [ ] Unsupported types marked as failed
- [ ] Files >100MB go to hold list
- [ ] Password-protected files skipped
- [ ] Corrupted files don't crash system

### Email Notifications
- [ ] Admin receives failure summaries
- [ ] Batching prevents spam
- [ ] Critical alerts send immediately
- [ ] Email contains dashboard links

### Responsive Design
- [ ] Dashboard works on mobile
- [ ] Tablet layout displays correctly
- [ ] Desktop view fully functional
- [ ] All interactions work on touch devices

---

## Appendix A: Source Code References

### Existing RAG Pipeline Files to Copy:

From `/RAG-Pipeline-Discovery/`:
- `core/text_processing.py` - Text extraction engine
- `core/database_operations.py` - Vector DB operations  
- `core/google_drive_integration.py` - Drive API handling
- `extractors/*.py` - All file format extractors
- `config/config.json` - Configuration template

### TypingMind Plugin Reference:

From `enhanced_kb_search_FIXED.json`:
- Search API structure
- Query formatting
- Response schema
- Authentication flow

---

## Appendix B: Database Migration Path

For existing installations upgrading to brAIn:

```sql
-- Migration script for existing RAG Pipeline databases
BEGIN;

-- Add new tables if not exist
CREATE TABLE IF NOT EXISTS monitored_folders (...);
CREATE TABLE IF NOT EXISTS vectorization_failures (...);
CREATE TABLE IF NOT EXISTS processing_logs (...);
CREATE TABLE IF NOT EXISTS files_on_hold (...);
CREATE TABLE IF NOT EXISTS system_metadata (...);

-- Add new columns to existing tables
ALTER TABLE documents 
ADD COLUMN IF NOT EXISTS folder_id TEXT;

-- Create new indexes
CREATE INDEX IF NOT EXISTS idx_documents_folder 
ON documents((metadata->>'folder_id'));

-- Set schema version
INSERT INTO system_metadata (key, value)
VALUES ('schema_version', '1.0.0')
ON CONFLICT (key) DO UPDATE 
SET value = '1.0.0', updated_at = NOW();

COMMIT;
```

---

*End of Product Requirements Document v1.0*
*Total Implementation Timeline: 15 Days*
*Target Deployment: Docker Container on VPS*