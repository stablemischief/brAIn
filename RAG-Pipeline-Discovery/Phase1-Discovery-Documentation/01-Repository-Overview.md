# Phase 1: Repository Discovery & Documentation
## RAG Pipeline Comprehensive Analysis

---

# 1. REPOSITORY OVERVIEW

## Project Identity
- **Project Name:** RAG Pipeline (Google Drive & Local Files)
- **Primary Purpose:** A Retrieval Augmented Generation (RAG) pipeline that processes files from Google Drive or local directories, extracts text, generates embeddings, and stores them in Supabase with PGVector for semantic search
- **Repository Location:** `/Users/james/Documents/GitHub/ai-agent-mastery/4_Pydantic_AI_Agent/RAG_Pipeline`
- **Parent Context:** Part of a larger AI Agent Mastery course/project focusing on Pydantic AI Agents

## Core Functionality
The system provides dual-mode document processing capabilities:
1. **Google Drive Integration:** Monitors and processes files from Google Drive folders
2. **Local File Processing:** Watches local directories for file changes
3. **Vector Database Storage:** Stores processed content with embeddings in Supabase
4. **Semantic Search Enablement:** Prepares data for RAG-based AI applications

## Key Features
- **Automated File Monitoring:** Continuous watching for file creation, updates, and deletion
- **Multi-Format Support:** Processes PDFs, documents, spreadsheets, presentations, and more
- **Text Extraction & Chunking:** Intelligent content extraction with configurable chunking
- **Embedding Generation:** Creates vector embeddings using OpenAI or compatible models
- **Database Synchronization:** Maintains consistency between source files and vector database
- **Configurable Processing:** JSON-based configuration for customization

---

# 2. REPOSITORY STRUCTURE

## Root Directory Layout
```
RAG_Pipeline/
├── Google_Drive/          # Google Drive monitoring module
├── Local_Files/           # Local file monitoring module
├── common/                # Shared utilities and handlers
├── tests/                 # Integration and unit tests
├── extract-test/          # Test extraction utilities
├── venv/                  # Python virtual environment
├── *.py                   # Utility scripts and testing files
├── *.json                 # Configuration files
├── README.md              # Main documentation
├── requirements.txt       # Python dependencies
└── SUPABASE_TROUBLESHOOTING_PROMPT.md
```

## Module Organization

### Google_Drive Module (`/Google_Drive`)
- **main.py:** Entry point for Google Drive pipeline
- **drive_watcher.py:** Core Google Drive monitoring logic
- **config.json:** Configuration for Google Drive processing
- **credentials.json:** Google API credentials
- **token.json:** OAuth token storage
- **TROUBLESHOOTING.md:** Google Drive specific troubleshooting
- **tests/:** Module-specific tests

### Local_Files Module (`/Local_Files`)
- **main.py:** Entry point for local file pipeline
- **file_watcher.py:** Local directory monitoring logic
- **config.json:** Configuration for local file processing
- **tests/:** Module-specific tests

### Common Module (`/common`)
- **db_handler.py:** Database operations and Supabase integration
- **text_processor.py:** Text extraction and chunking logic

### Testing Infrastructure
- **tests/:** Integration tests for the entire pipeline
- **test_*.py:** Individual test files for specific functionality
- **check_*.py:** Validation and verification scripts
- **demo_new_features.py:** Feature demonstration script

## Configuration Files

### Environment Configuration (../.env)
Required environment variables stored one level above RAG_Pipeline:
- Embedding model settings (EMBEDDING_MODEL_NAME, EMBEDDING_BASE_URL, EMBEDDING_API_KEY)
- Supabase credentials (SUPABASE_URL, SUPABASE_SERVICE_KEY)
- Additional AI/ML service configurations

### Module Configurations
- **Google_Drive/config.json:** Google Drive specific settings
- **Local_Files/config.json:** Local file processing settings
- **enhanced_kb_search_*.json:** Enhanced knowledge base search configurations

---

# 3. TECHNOLOGY STACK

## Core Technologies

### Programming Language
- **Python 3.11+** - Primary development language
- Virtual environment support for dependency isolation

### Cloud Services
- **Supabase:** PostgreSQL database with PGVector extension for vector storage
- **Google Drive API:** For Google Drive file access and monitoring
- **OpenAI API:** For embedding generation (configurable to other providers)

### Key Python Libraries

#### Google Integration
- google-api-python-client (2.166.0)
- google-auth (2.38.0)
- google-auth-oauthlib (1.2.1)

#### AI/ML
- openai (1.71.0) - Embedding generation
- pydantic (2.11.3) - Data validation and modeling

#### Document Processing
- pypdf (5.4.0) - PDF text extraction
- python-docx (1.1.2) - Word document processing
- python-pptx (1.0.2) - PowerPoint processing
- xlrd (2.0.1) & openpyxl (3.1.2) - Excel file handling
- pdfplumber (0.11.4) - Advanced PDF extraction
- markdownify (0.12.1) - HTML to Markdown conversion
- mammoth (1.8.0) - DOCX to HTML conversion

#### Database & API
- supabase (2.15.0) - Supabase client
- requests (2.32.3) - HTTP requests
- python-dotenv (1.1.0) - Environment variable management

#### Testing
- pytest (8.3.5) - Testing framework
- pytest-mock (3.14.0) - Mocking utilities

---

# 4. KEY ARCHITECTURAL DECISIONS

## Design Patterns

### 1. Modular Architecture
- Separate modules for Google Drive and Local Files
- Shared common utilities to avoid code duplication
- Clear separation of concerns between file watching, processing, and storage

### 2. Configuration-Driven Design
- JSON configuration files for runtime customization
- Environment variables for sensitive credentials
- Configurable chunk sizes, intervals, and file type support

### 3. Pipeline Pattern
- Sequential processing: Monitor → Extract → Chunk → Embed → Store
- Each stage is independently testable and replaceable

### 4. Database Schema Design
- Single `documents` table with JSONB metadata
- Vector storage for embeddings
- File identification through metadata (file_id, file_path)

## Processing Flow

### File Discovery
1. Periodic polling (configurable interval)
2. Change detection through timestamp comparison
3. Support for both creation and modification events

### Content Processing
1. File download/read based on source
2. Format-specific text extraction
3. Configurable text chunking
4. Embedding generation per chunk
5. Batch database operations

### Synchronization Strategy
- Delete existing records before inserting updates
- Maintain referential integrity through metadata
- Track last check time for incremental processing

---

# 5. FILE INVENTORY

## Core Implementation Files

### Entry Points
- `Google_Drive/main.py` - Google Drive pipeline launcher
- `Local_Files/main.py` - Local file pipeline launcher

### Primary Processing Logic
- `Google_Drive/drive_watcher.py` (23KB) - Google Drive monitoring and processing
- `Local_Files/file_watcher.py` - Local directory monitoring
- `common/text_processor.py` (36KB) - Text extraction and chunking
- `common/db_handler.py` (14KB) - Database operations

### Utility Scripts
- `demo_new_features.py` - Feature demonstration
- `check_actual_count.py` - Document count verification
- `check_match_documents.py` - Document matching validation
- `test_file_retrieval.py` - File retrieval testing
- `test_supabase_urllib.py` - Supabase connection testing

### Configuration Files
- `Google_Drive/config.json` - Google Drive settings
- `Local_Files/config.json` - Local file settings
- `enhanced_kb_search_*.json` - Knowledge base configurations

### Documentation
- `README.md` (10KB) - Main documentation
- `SUPABASE_TROUBLESHOOTING_PROMPT.md` - Troubleshooting guide
- `Google_Drive/TROUBLESHOOTING.md` - Google Drive specific issues

---

# 6. EXTERNAL DEPENDENCIES

## API Integrations

### Google Drive API
- OAuth 2.0 authentication flow
- Drive file listing and downloading
- Support for Google Workspace file exports
- Folder hierarchy traversal

### OpenAI API (or compatible)
- Embedding model access
- Configurable model selection
- Support for alternative providers (Ollama, OpenRouter)

### Supabase
- PostgreSQL with PGVector extension
- Service key authentication
- CRUD operations on documents table
- Vector similarity search support

## Parent Project Integration
The RAG Pipeline exists within a larger Pydantic AI Agent project that includes:
- `agent.py` - Main agent implementation
- `tools.py` - Agent tool definitions
- `clients.py` - Client configurations
- `streamlit_ui.py` - User interface
- SQL schema definitions in `sql/` directory

---

# 7. KEY OBSERVATIONS & INSIGHTS

## Strengths
1. **Dual-mode flexibility:** Supports both cloud and local file processing
2. **Comprehensive format support:** Handles most common document types
3. **Production-ready features:** Error handling, logging, configuration management
4. **Modular design:** Easy to extend or modify individual components
5. **Well-documented:** Clear README and inline documentation

## Architecture Highlights
1. **Separation of concerns:** Clear boundaries between modules
2. **Reusable components:** Common utilities shared across pipelines
3. **Configuration flexibility:** JSON-based settings for easy customization
4. **Scalable design:** Can handle large document collections
5. **Testing infrastructure:** Comprehensive test coverage

## Integration Points
1. **Database dependency:** Requires Supabase with PGVector
2. **API dependencies:** Google Drive and OpenAI APIs
3. **Environment configuration:** Relies on parent directory .env file
4. **Schema requirements:** Expects specific database table structure

## Development Patterns
1. **Asynchronous processing:** Uses async patterns for efficiency
2. **Batch operations:** Groups database operations for performance
3. **Error recovery:** Handles API failures and retries
4. **State management:** Tracks processing state via timestamps
5. **Incremental updates:** Processes only changed files

---

# END OF PHASE 1 DOCUMENTATION

This completes the initial discovery and documentation phase. The repository has been thoroughly mapped and inventoried, providing a solid foundation for deeper technical analysis in Phase 2.