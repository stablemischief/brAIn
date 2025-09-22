# brAIn v2.0 API Documentation

Welcome to the brAIn v2.0 API! This RESTful API provides comprehensive access to document processing, search, analytics, and real-time monitoring capabilities.

## 🚀 Quick Start

### Base URL
- **Development**: `http://localhost:8000`
- **Production**: `https://your-domain.com`

### Interactive Documentation
- **Swagger UI**: `/docs` - Interactive API explorer
- **ReDoc**: `/redoc` - Clean API documentation

### Authentication
All API endpoints require JWT authentication:
```bash
# Get authentication token
curl -X POST "/api/auth/login" \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "your_secure_password"}'

# Use token in subsequent requests
curl -H "Authorization: Bearer YOUR_JWT_TOKEN" \
  "/api/folders"
```

## 📚 API Sections

### Core Resources
- **[Authentication](./auth.md)** - Login, logout, session management
- **[Folders](./folders.md)** - Google Drive folder management
- **[Documents](./documents.md)** - Document processing and metadata
- **[Search](./search.md)** - Semantic and hybrid search

### Monitoring & Analytics
- **[Health](./health.md)** - System health and status monitoring
- **[Analytics](./analytics.md)** - Cost tracking and performance metrics
- **[Processing](./processing.md)** - Processing status and queue management

### Advanced Features
- **[Knowledge Graph](./knowledge-graph.md)** - Entity and relationship APIs
- **[Configuration](./configuration.md)** - AI-powered system configuration
- **[WebSocket](./websocket.md)** - Real-time updates and subscriptions

## 🔧 Quick Reference

### Common Operations

#### List Folders
```bash
GET /api/folders
Authorization: Bearer <token>
```

#### Process Documents
```bash
POST /api/processing/trigger
Authorization: Bearer <token>
Content-Type: application/json

{
  "folder_id": "uuid",
  "force_reprocess": false
}
```

#### Semantic Search
```bash
POST /api/search
Authorization: Bearer <token>
Content-Type: application/json

{
  "query": "project timeline",
  "limit": 10,
  "search_type": "semantic"
}
```

#### Get System Health
```bash
GET /api/health
Authorization: Bearer <token>
```

### Response Format

All API responses follow this structure:
```json
{
  "success": true,
  "data": { /* response data */ },
  "message": "Operation completed successfully",
  "timestamp": "2025-09-22T10:00:00Z"
}
```

Error responses:
```json
{
  "success": false,
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid folder ID format",
    "details": { /* error details */ }
  },
  "timestamp": "2025-09-22T10:00:00Z"
}
```

## 📊 Data Models

### Core Models
```typescript
interface Document {
  id: string;
  folder_id: string;
  title: string;
  content: string;
  file_type: string;
  processing_status: 'pending' | 'processing' | 'completed' | 'failed';
  extraction_quality: number; // 0-1 quality score
  processing_cost: number;
  token_count: number;
  created_at: string;
  updated_at: string;
}

interface Folder {
  id: string;
  name: string;
  google_drive_id: string;
  sync_enabled: boolean;
  last_sync: string;
  document_count: number;
  total_cost: number;
}

interface SearchResult {
  id: string;
  title: string;
  content_excerpt: string;
  similarity_score: number;
  document_type: string;
  metadata: object;
}
```

### Processing Models
```typescript
interface ProcessingStatus {
  folder_id: string;
  total_files: number;
  processed_files: number;
  failed_files: number;
  current_status: string;
  estimated_completion: string;
  total_cost: number;
}

interface CostAnalytics {
  daily_cost: number;
  monthly_cost: number;
  total_cost: number;
  cost_by_operation: object;
  budget_remaining: number;
  cost_trend: number[]; // 30-day trend
}
```

## 🔌 WebSocket API

### Connection
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/realtime');

ws.onopen = () => {
  // Subscribe to updates
  ws.send(JSON.stringify({
    action: 'subscribe',
    channels: ['processing_status', 'cost_updates']
  }));
};

ws.onmessage = (event) => {
  const update = JSON.parse(event.data);
  // Handle real-time updates
};
```

### Channels
- `processing_status` - Document processing updates
- `cost_updates` - Real-time cost tracking
- `system_health` - Service health changes
- `search_analytics` - Search performance metrics

## 🛠️ SDK and Integrations

### Python SDK
```python
from brain_api import BrainClient

client = BrainClient(
    base_url="http://localhost:8000",
    api_key="your_jwt_token"
)

# Process documents
result = client.folders.process_documents(folder_id="uuid")

# Search documents
results = client.search.semantic_search(
    query="project timeline",
    limit=10
)

# Get real-time updates
client.realtime.subscribe(['processing_status'], callback=handle_update)
```

### JavaScript SDK
```javascript
import { BrainClient } from '@brain/api-client';

const client = new BrainClient({
  baseUrl: 'http://localhost:8000',
  apiKey: 'your_jwt_token'
});

// Process documents
const result = await client.folders.processDocuments('folder-uuid');

// Search documents
const results = await client.search.semanticSearch({
  query: 'project timeline',
  limit: 10
});
```

## 🔒 Security

### Authentication Flow
1. **Login**: POST `/api/auth/login` with credentials
2. **Token**: Receive JWT token in response
3. **Requests**: Include `Authorization: Bearer <token>` header
4. **Refresh**: Use refresh token before expiration

### Rate Limiting
- **General API**: 1000 requests/hour per user
- **Search API**: 100 requests/minute per user
- **Processing API**: 50 requests/minute per user

### CORS Configuration
Configure allowed origins in environment:
```env
CORS_ORIGINS=http://localhost:3000,https://yourdomain.com
```

## 📈 Performance

### Response Times
- **Authentication**: <100ms
- **Folder Operations**: <200ms
- **Search Queries**: <500ms
- **Document Processing**: Async (use WebSocket for updates)

### Pagination
Large result sets are paginated:
```bash
GET /api/documents?page=1&limit=50&sort=created_at&order=desc
```

Response includes pagination metadata:
```json
{
  "data": [...],
  "pagination": {
    "page": 1,
    "limit": 50,
    "total": 1250,
    "pages": 25,
    "has_next": true,
    "has_prev": false
  }
}
```

## 🧪 Testing

### API Testing with curl
```bash
# Set token variable
export TOKEN="your_jwt_token_here"

# Test authentication
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/health

# Test search
curl -X POST \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "test search", "limit": 5}' \
  http://localhost:8000/api/search
```

### Integration Tests
See `/tests/api/` for comprehensive test suites covering:
- Authentication flows
- CRUD operations
- Search functionality
- Error handling
- Rate limiting

## 📋 Error Codes

### Common Error Codes
- `400` - Bad Request (validation errors)
- `401` - Unauthorized (authentication required)
- `403` - Forbidden (insufficient permissions)
- `404` - Not Found (resource doesn't exist)
- `429` - Rate Limited (too many requests)
- `500` - Internal Server Error (system error)

### Custom Error Codes
- `FOLDER_NOT_FOUND` - Google Drive folder not accessible
- `PROCESSING_IN_PROGRESS` - Cannot modify during processing
- `QUOTA_EXCEEDED` - Daily/monthly limits reached
- `INVALID_SEARCH_QUERY` - Search query format invalid

## 📚 Additional Resources

- **[Authentication Guide](./auth.md)** - Detailed auth flows and security
- **[Search Guide](./search.md)** - Advanced search capabilities
- **[WebSocket Guide](./websocket.md)** - Real-time integration patterns
- **[Integration Examples](../examples/)** - Complete integration examples

---

**Need help?** Check the [troubleshooting guide](../troubleshooting/) or review the [interactive API docs](http://localhost:8000/docs).