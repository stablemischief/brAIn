# Phase 2: Technical Deep-Dive
## Part 4: Complete Data Flow Analysis

---

# END-TO-END DATA FLOW ARCHITECTURE

## System Overview
This document traces the complete journey of a document from source to vector storage, analyzing each transformation and decision point.

---

## 1. DATA FLOW VISUALIZATION

### High-Level Pipeline
```
[Google Drive/Local File]
         ↓
[File Discovery & Monitoring]
         ↓
[File Download/Read]
         ↓
[Format Detection]
         ↓
[Text Extraction]
         ↓
[Content Sanitization]
         ↓
[Text Chunking]
         ↓
[Embedding Generation]
         ↓
[Vector Storage]
         ↓
[Ready for RAG]
```

---

## 2. STAGE 1: FILE DISCOVERY

### Google Drive Path
```python
# Initial Discovery
GoogleDriveWatcher.initialize_and_sync_files()
  → get_all_files_in_folder(folder_id)
    → Recursive traversal
    → Returns: List[file_metadata]

# Continuous Monitoring
GoogleDriveWatcher.watch_for_changes()
  → get_folder_contents(folder_id, last_check_time)
    → Time-based filtering
    → Returns: List[changed_files]
```

**Data Structure at Discovery:**
```python
file_metadata = {
    'id': 'unique_file_id',
    'name': 'document.pdf',
    'mimeType': 'application/pdf',
    'webViewLink': 'https://drive.google.com/...',
    'modifiedTime': '2025-01-15T10:30:00.000Z',
    'createdTime': '2025-01-10T08:00:00.000Z',
    'trashed': False
}
```

### Local Files Path
```python
LocalFileWatcher.scan_directory()
  → os.walk(directory_path)
    → File system traversal
    → Returns: List[file_paths]

# File metadata construction
file_metadata = {
    'id': '/absolute/path/to/file.pdf',
    'name': 'file.pdf',
    'mime_type': 'application/pdf',  # Inferred from extension
    'modified_time': os.path.getmtime(),
    'size': os.path.getsize()
}
```

---

## 3. STAGE 2: FILE RETRIEVAL

### Google Drive Download
```python
def download_file(file_id, mime_type):
    if mime_type in export_mime_types:
        # Google Workspace files
        request = service.files().export_media(
            fileId=file_id,
            mimeType=export_mime_types[mime_type]
        )
    else:
        # Binary files
        request = service.files().get_media(fileId=file_id)
    
    # Stream download
    file_content = io.BytesIO()
    downloader = MediaIoBaseDownload(file_content, request)
    
    return file_content.getvalue()  # Returns: bytes
```

**Export Transformations:**
- Google Docs → HTML (text/html)
- Google Sheets → CSV (text/csv)
- Google Slides → HTML (text/html)

### Local File Read
```python
def read_file(file_path):
    with open(file_path, 'rb') as f:
        return f.read()  # Returns: bytes
```

---

## 4. STAGE 3: TEXT EXTRACTION

### Extraction Router
```python
def extract_text_from_file(file_content: bytes, mime_type: str):
    if mime_type == 'application/pdf':
        return extract_text_from_pdf(file_content)
    elif mime_type.startswith('application/vnd.openxmlformats'):
        if 'wordprocessingml' in mime_type:
            return extract_text_from_docx(file_content)
        elif 'spreadsheetml' in mime_type:
            return extract_text_from_xlsx(file_content)
        elif 'presentationml' in mime_type:
            return extract_text_from_pptx(file_content)
    elif mime_type == 'text/csv':
        return extract_text_from_csv(file_content)
    elif mime_type == 'text/html':
        return extract_text_from_html(file_content)
    else:
        return file_content.decode('utf-8', errors='ignore')
```

### Format-Specific Processing

#### PDF Extraction Flow
```python
# Primary: pdfplumber (advanced)
with pdfplumber.open(io.BytesIO(file_content)) as pdf:
    for page in pdf.pages:
        text = page.extract_text()
        # Preserve layout and hyperlinks
        
# Fallback: pypdf (basic)
reader = pypdf.PdfReader(io.BytesIO(file_content))
for page in reader.pages:
    text = page.extract_text()
```

#### DOCX Extraction Flow
```python
# XML parsing approach
with zipfile.ZipFile(io.BytesIO(file_content)) as zipf:
    with zipf.open('word/document.xml') as xml_file:
        tree = ET.parse(xml_file)
        # Extract all text nodes
        texts = [elem.text for elem in tree.iter('{...}t')]
```

#### XLSX Extraction Flow
```python
# openpyxl with formula evaluation
workbook = load_workbook(temp_file, data_only=True)
for sheet in workbook.worksheets:
    for row in sheet.iter_rows():
        row_data = [str(cell.value) for cell in row if cell.value]
        # Join with " | " separator
```

---

## 5. STAGE 4: TEXT SANITIZATION

### Sanitization Pipeline
```python
def sanitize_text(text: str) -> str:
    # Step 1: Remove null bytes
    text = text.replace('\x00', '')
    
    # Step 2: Remove control characters
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', text)
    
    # Step 3: Normalize whitespace
    text = re.sub(r' +', ' ', text)
    
    # Step 4: Strip edges
    return text.strip()
```

**Data Transformation:**
```
Input:  "Hello\x00World\n\n  Multiple   Spaces\t\tTabs"
Output: "Hello World Multiple Spaces Tabs"
```

---

## 6. STAGE 5: TEXT CHUNKING

### Chunking Algorithm
```python
def chunk_text(text: str, chunk_size: int = 400, overlap: int = 0):
    chunks = []
    start = 0
    text_length = len(text)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap if overlap > 0 else end
    
    return chunks
```

**Chunking Example:**
```
Text: "This is a long document with multiple sentences..." (1000 chars)
Settings: chunk_size=400, overlap=0

Chunks:
[0:400]   "This is a long document with..."
[400:800] "...continues here with more..."
[800:1000] "...and ends with this text."
```

---

## 7. STAGE 6: EMBEDDING GENERATION

### Embedding Process
```python
def create_embeddings(texts: List[str]) -> List[List[float]]:
    # Pre-processing
    sanitized_texts = [sanitize_text(text) for text in texts]
    sanitized_texts = [t for t in sanitized_texts if t.strip()]
    
    # API call
    response = openai_client.embeddings.create(
        model=os.getenv("EMBEDDING_MODEL_CHOICE"),  # e.g., "text-embedding-3-small"
        input=sanitized_texts
    )
    
    # Extract vectors
    embeddings = [item.embedding for item in response.data]
    return embeddings
```

**Data Transformation:**
```
Input:  ["chunk1 text", "chunk2 text", "chunk3 text"]
Output: [[0.012, -0.034, ...], [0.023, 0.045, ...], [-0.001, 0.067, ...]]
         (Each vector: 1536 dimensions for text-embedding-3-small)
```

---

## 8. STAGE 7: DATABASE OPERATIONS

### Delete-Insert Pattern
```python
def process_file_for_rag(file_content, text, file_id, ...):
    # Phase 1: Clean slate
    delete_document_by_file_id(file_id)
    
    # Phase 2: Metadata first (foreign key)
    insert_or_update_document_metadata(file_id, title, url, schema)
    
    # Phase 3: Process content
    chunks = chunk_text(text, chunk_size=400)
    embeddings = create_embeddings(chunks)
    
    # Phase 4: Store vectors
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        data = {
            "content": chunk,
            "metadata": {
                "file_id": file_id,
                "file_url": url,
                "file_title": title,
                "mime_type": mime_type,
                "chunk_index": i
            },
            "embedding": embedding
        }
        supabase.table("documents").insert(data).execute()
```

---

## 9. SPECIAL CASE FLOWS

### Tabular Data Flow
```python
if is_tabular_file(mime_type):
    # Extract structure
    schema = extract_schema_from_csv(file_content)
    rows = extract_rows_from_csv(file_content)
    
    # Store in specialized table
    insert_document_metadata(file_id, title, url, schema)
    insert_document_rows(file_id, rows)
    
    # Also process as text for RAG
    text = csv_to_text(file_content)
    # Continue normal flow...
```

### Image Data Flow
```python
if mime_type.startswith("image"):
    # Store binary in metadata
    file_bytes_str = base64.b64encode(file_content).decode('utf-8')
    
    # Use filename as searchable text
    chunks = [file_title]
    embeddings = create_embeddings(chunks)
    
    # Store with binary
    insert_document_chunks(chunks, embeddings, file_id, 
                          url, title, mime_type, file_content)
```

### Trashed File Flow
```python
if file.get('trashed', False):
    # Immediate cleanup
    delete_document_by_file_id(file_id)
    
    # Update state
    if file_id in known_files:
        del known_files[file_id]
    
    # Skip further processing
    return
```

---

## 10. DATA VOLUME ANALYSIS

### Per-Document Metrics
```
Average Document: 10,000 characters
Chunk Size: 400 characters
Overlap: 0

Results:
- Chunks per document: ~25
- Embeddings per document: 25 vectors
- Storage per chunk: ~6.5KB (1536 floats * 4 bytes + metadata)
- Total per document: ~162.5KB
```

### API Call Analysis
```
Per Document:
- Google Drive API: 1-2 calls (list + download)
- OpenAI Embedding API: 1 call (batch)
- Supabase API: 27 calls (1 metadata + 25 chunks + 1 delete)

Total: ~30 API calls per document
```

### Processing Time Estimates
```
File Download: 1-5 seconds (size dependent)
Text Extraction: 0.5-2 seconds
Chunking: <0.1 seconds
Embedding Generation: 0.5-1 second
Database Storage: 1-2 seconds

Total: 3-10 seconds per document
```

---

## 11. ERROR PROPAGATION PATHS

### Critical Failure Points
1. **Authentication Failure** → Complete stop
2. **File Download Failure** → Skip file, continue
3. **Text Extraction Failure** → Try fallback, then skip
4. **Embedding API Failure** → Retry, then fail file
5. **Database Failure** → Retry, then fail file

### Recovery Mechanisms
```python
# Example: Embedding retry logic
for attempt in range(3):
    try:
        embeddings = create_embeddings(chunks)
        break
    except Exception as e:
        if attempt == 2:
            raise
        time.sleep(2 ** attempt)
```

---

## 12. OPTIMIZATION OPPORTUNITIES

### Current Bottlenecks
1. **Sequential Processing:** One file at a time
2. **Full File Loading:** Memory intensive
3. **Individual DB Inserts:** Network overhead
4. **No Caching:** Repeated embeddings for unchanged content

### Proposed Optimizations
1. **Parallel Processing:**
   - Multi-threaded file processing
   - Batch API calls
   
2. **Streaming Processing:**
   - Chunk files during download
   - Process while downloading
   
3. **Batch Operations:**
   ```python
   # Batch insert example
   supabase.table("documents").insert(all_chunks).execute()
   ```
   
4. **Content Hashing:**
   ```python
   content_hash = hashlib.sha256(file_content).hexdigest()
   if content_hash == stored_hash:
       skip_processing()
   ```

---

# KEY INSIGHTS

## Data Flow Characteristics
1. **Transform-Heavy:** Multiple format conversions
2. **Memory-Bound:** Full document in memory
3. **Network-Intensive:** Multiple API calls
4. **Stateful:** Tracking via timestamps and known_files

## Design Decisions
1. **Delete-First Updates:** Simplifies consistency
2. **Chunk-Level Storage:** Enables partial retrieval
3. **Metadata Preservation:** Rich context for RAG
4. **Format Normalization:** Unified processing

## Performance Profile
- **Throughput:** ~6-20 documents/minute
- **Latency:** 3-10 seconds per document
- **Storage:** ~160KB per document
- **API Usage:** ~30 calls per document

---

**End of Complete Data Flow Analysis**