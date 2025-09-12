# Phase 2: Technical Deep-Dive
## Part 2: Database & Vector Storage Architecture

---

# DATABASE HANDLER (db_handler.py)

## System Architecture
The `db_handler.py` module (361 lines) implements the data persistence layer, managing all interactions with Supabase and PGVector for vector storage and retrieval.

---

## 1. INITIALIZATION & CONNECTION

### Supabase Client Setup
```python
# Environment loading with path resolution
project_root = Path(__file__).resolve().parent.parent.parent
dotenv_path = project_root / '.env'
load_dotenv(dotenv_path, override=True)

# Supabase initialization
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)
```

**Key Points:**
- Service key authentication (admin access)
- Single client instance for all operations
- Environment-based configuration

---

## 2. DATA MODEL & SCHEMA

### Primary Tables

#### documents Table
```sql
documents:
- id: UUID/SERIAL PRIMARY KEY
- content: TEXT (chunk text)
- metadata: JSONB (flexible metadata)
- embedding: VECTOR (embedding vectors)
```

#### Metadata Structure
```python
metadata = {
    "file_id": str,          # Unique identifier
    "file_url": str,         # Access URL
    "file_title": str,       # Display name
    "mime_type": str,        # Content type
    "chunk_index": int,      # Order in document
    "file_contents": str     # Base64 for images (optional)
}
```

#### document_metadata Table
```sql
document_metadata:
- id: TEXT PRIMARY KEY (file_id)
- title: TEXT
- url: TEXT
- schema: JSONB (for tabular files)
```

#### document_rows Table
```sql
document_rows:
- id: SERIAL PRIMARY KEY
- dataset_id: TEXT (foreign key to document_metadata)
- data: JSONB (row data)
```

---

## 3. CORE OPERATIONS

### Delete Operations (Idempotent Design)
```python
def delete_document_by_file_id(file_id: str) -> None:
    # Delete pattern: Remove all related data
    # 1. Delete document chunks
    supabase.table("documents").delete().eq("metadata->>file_id", file_id).execute()
    
    # 2. Delete tabular rows
    supabase.table("document_rows").delete().eq("dataset_id", file_id).execute()
    
    # 3. Delete metadata
    supabase.table("document_metadata").delete().eq("id", file_id).execute()
```

**Design Pattern:** Complete cleanup ensures no orphaned data

### Existence Checking
```python
def check_document_exists(file_id: str) -> bool:
    # Dual-table checking for reliability
    response = supabase.table("documents").select("id").eq("metadata->>file_id", file_id).limit(1).execute()
    if response.data:
        return True
    
    # Backup check in metadata table
    metadata_response = supabase.table("document_metadata").select("id").eq("id", file_id).limit(1).execute()
    return bool(metadata_response.data)
```

**Purpose:** Prevents duplicate processing and ensures data consistency

---

## 4. VECTOR STORAGE IMPLEMENTATION

### Chunk Insertion with Embeddings
```python
def insert_document_chunks(chunks: List[str], embeddings: List[List[float]], 
                          file_id: str, file_url: str, file_title: str, 
                          mime_type: str, file_contents: bytes = None) -> None:
    # Validation
    if len(chunks) != len(embeddings):
        raise ValueError("Chunks and embeddings count mismatch")
    
    # Prepare batch data
    data = []
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        # Optional base64 encoding for images
        file_bytes_str = base64.b64encode(file_contents).decode('utf-8') if file_contents else None
        
        data.append({
            "content": chunk,
            "metadata": {
                "file_id": file_id,
                "file_url": file_url,
                "file_title": file_title,
                "mime_type": mime_type,
                "chunk_index": i,
                **({"file_contents": file_bytes_str} if file_bytes_str else {})
            },
            "embedding": embedding
        })
    
    # Insert individually (for error isolation)
    for item in data:
        supabase.table("documents").insert(item).execute()
```

**Key Features:**
- Chunk ordering via index
- Optional binary storage for images
- Individual insertion for error tracking
- JSONB metadata for flexibility

---

## 5. MASTER PROCESSING PIPELINE

### Main Processing Function
```python
def process_file_for_rag(file_content: bytes, text: str, file_id: str, 
                         file_url: str, file_title: str, mime_type: str = None, 
                         config: Dict[str, Any] = None) -> None:
    # Phase 1: Cleanup
    delete_document_by_file_id(file_id)
    
    # Phase 2: Type Detection
    is_tabular = is_tabular_file(mime_type, config)
    
    # Phase 3: Metadata Creation
    schema = extract_schema_from_csv(file_content) if is_tabular else None
    insert_or_update_document_metadata(file_id, file_title, file_url, schema)
    
    # Phase 4: Tabular Data Processing
    if is_tabular:
        rows = extract_rows_from_csv(file_content)
        insert_document_rows(file_id, rows)
    
    # Phase 5: Text Chunking
    chunk_size = config.get('text_processing', {}).get('default_chunk_size', 400)
    chunk_overlap = config.get('text_processing', {}).get('default_chunk_overlap', 0)
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap)
    
    # Phase 6: Embedding Generation
    embeddings = create_embeddings(chunks)
    
    # Phase 7: Storage
    if mime_type.startswith("image"):
        # Special handling: store binary with chunks
        insert_document_chunks(chunks, embeddings, file_id, file_url, 
                             file_title, mime_type, file_content)
    else:
        insert_document_chunks(chunks, embeddings, file_id, file_url, 
                             file_title, mime_type)
```

**Processing Flow:**
1. **Delete-First Pattern:** Ensures clean state
2. **Type-Specific Handling:** Different paths for tabular/image/text
3. **Metadata First:** Satisfies foreign key constraints
4. **Configurable Processing:** Runtime customization
5. **Error Isolation:** Each phase can fail independently

---

## 6. RETRIEVAL OPERATIONS

### Full Document Reconstruction
```python
def retrieve_full_file_content(file_id: str) -> Dict[str, Any]:
    # Query all chunks ordered by index
    response = supabase.table("documents").select("*")\
        .eq("metadata->>file_id", file_id)\
        .order("metadata->>chunk_index").execute()
    
    # Reconstruct document
    chunks_data = []
    for chunk in response.data:
        chunks_data.append({
            "content": chunk["content"],
            "chunk_index": chunk["metadata"].get("chunk_index", 0),
            "embedding": chunk.get("embedding", [])
        })
    
    # Sort and join
    chunks_data.sort(key=lambda x: x["chunk_index"])
    full_content = "\n".join([chunk["content"] for chunk in chunks_data])
    
    return {
        "success": True,
        "content": full_content,
        "metadata": file_metadata,
        "chunks": chunks_data
    }
```

**Capabilities:**
- Ordered reconstruction via chunk_index
- Metadata preservation
- Embedding retrieval for similarity operations

### Similarity Search Support
```python
def similarity_search(query_embedding: List[float], limit: int = 5) -> List[Dict]:
    # PostgreSQL function call via Supabase
    response = supabase.rpc('similarity_search', {
        'query_embedding': query_embedding,
        'limit': limit
    }).execute()
    return response.data
```

**Note:** Requires PostgreSQL function with PGVector:
```sql
CREATE FUNCTION similarity_search(query_embedding vector, limit int)
RETURNS TABLE (...) AS $$
  SELECT * FROM documents
  ORDER BY embedding <=> query_embedding
  LIMIT limit;
$$ LANGUAGE SQL;
```

---

## 7. SPECIALIZED HANDLERS

### Tabular Data Processing
```python
def insert_document_rows(dataset_id: str, rows: List[Dict[str, Any]]) -> None:
    # Batch insert for tabular data
    data = [{"dataset_id": dataset_id, "data": row} for row in rows]
    
    # Batch size optimization (100 rows)
    batch_size = 100
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        supabase.table("document_rows").insert(batch).execute()
```

**Purpose:** Preserves structured data for SQL queries alongside vector search

### Image Handling
```python
# Special case for images
if mime_type.startswith("image"):
    # Store binary in metadata
    file_bytes_str = base64.b64encode(file_contents).decode('utf-8')
    metadata["file_contents"] = file_bytes_str
```

**Rationale:** Enables image retrieval for multimodal AI operations

---

## 8. ERROR HANDLING & RESILIENCE

### Transaction-like Behavior
```python
try:
    # Delete existing (rollback point)
    delete_document_by_file_id(file_id)
    
    # Process in phases
    # ... processing steps ...
    
    return True
except Exception as e:
    traceback.print_exc()
    print(f"Error processing file: {e}")
    return False  # Indicates failure to caller
```

### Error Isolation Pattern
```python
# Each operation in try-except
try:
    supabase.table("document_rows").delete().eq("dataset_id", file_id).execute()
except Exception as e:
    print(f"Error deleting document rows: {e}")
    # Continue with other deletions
```

**Design:** Partial failures don't stop entire operation

---

## 9. PERFORMANCE CHARACTERISTICS

### Database Operations
- **Individual Inserts:** ~10-50ms per chunk
- **Batch Deletes:** Single query for all chunks
- **JSONB Queries:** Uses PostgreSQL indexes
- **Vector Operations:** Hardware-dependent (CPU vs GPU)

### Optimization Strategies
1. **Batch Inserts:** Could reduce API calls
2. **Connection Pooling:** Reuse database connections
3. **Async Operations:** Parallel processing
4. **Prepared Statements:** Reduce query parsing

### Current Limitations
- Sequential chunk insertion
- No transaction support
- Single-threaded operations
- No connection retry logic

---

## 10. SECURITY CONSIDERATIONS

### Authentication
- **Service Key:** Full admin access to Supabase
- **No Row-Level Security:** Direct table access
- **Environment Variables:** Credentials in .env

### Data Safety
- **Input Sanitization:** Via text_processor
- **SQL Injection Protection:** Parameterized queries via Supabase client
- **Base64 Encoding:** Safe binary storage

### Potential Vulnerabilities
1. Service key exposure grants full access
2. No encryption for stored content
3. No access logging
4. Metadata stored in plaintext

---

## 11. INTEGRATION PATTERNS

### With Text Processor
```python
from text_processor import chunk_text, create_embeddings, is_tabular_file
```
- Delegates all text operations
- Receives sanitized, processed data

### With Pipeline Modules
```python
from common.db_handler import process_file_for_rag, delete_document_by_file_id
```
- Provides unified interface
- Handles all storage concerns

### With Vector Search
- Stores embeddings in standard format
- Supports similarity operations via PGVector
- Compatible with OpenAI embedding dimensions

---

# KEY TECHNICAL INSIGHTS

## Architecture Strengths
1. **Clean Separation:** Database logic isolated from processing
2. **Flexible Metadata:** JSONB allows schema evolution
3. **Multi-Format Support:** Handles text, tabular, and images
4. **Idempotent Operations:** Safe to retry

## Design Patterns
1. **Delete-Insert Pattern:** Simplifies updates
2. **Metadata-First:** Maintains referential integrity
3. **Error Isolation:** Partial failures handled gracefully
4. **Configuration-Driven:** Runtime customization

## Performance Profile
- **Optimized for Correctness:** Reliability over speed
- **Memory Bound:** Full document in memory
- **Network Bound:** Individual API calls
- **Suitable for:** Medium-scale deployments (thousands of documents)

## Future Improvements
1. **Batch Operations:** Reduce API calls
2. **Streaming Support:** Handle large files
3. **Transaction Support:** Atomic operations
4. **Caching Layer:** Reduce redundant processing
5. **Async Processing:** Parallel operations

---

**End of Database & Vector Storage Analysis**