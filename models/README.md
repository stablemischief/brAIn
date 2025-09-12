# brAIn v2.0 Pydantic Models Documentation

## Overview

This directory contains the comprehensive Pydantic model layer for brAIn v2.0, providing robust data validation, serialization, and API contract definitions for the AI-enhanced RAG pipeline management system.

## Architecture

### Model Categories

1. **Base Models** (`base.py`) - Foundation classes, mixins, and common functionality
2. **Core Models** (`core.py`) - Business domain entities (User, Document, Folder, etc.)
3. **API Models** (`api.py`) - Request/response models for REST API endpoints
4. **Monitoring Models** (`monitoring.py`) - System health, cost tracking, and analytics
5. **Custom Validators** (`validators/custom_validators.py`) - Business logic validation
6. **Examples** (`examples.py`) - Comprehensive usage examples and demonstrations

### Key Features

- **AI-First Design**: Optimized for vector embeddings, quality scoring, and cost tracking
- **Comprehensive Validation**: Input validation, business rules, and data integrity
- **API Contract Definition**: Complete request/response models for all endpoints
- **Real-time Support**: WebSocket message models for live updates
- **Monitoring Integration**: Cost tracking, usage analytics, and system health
- **Knowledge Graph Ready**: Models for entity extraction and relationship mapping

## Model Hierarchy

```
BrainBaseModel (base configuration)
├── BaseEntityModel (UUID + timestamps + user scoping)
│   ├── User
│   ├── Folder
│   └── BaseDeletableModel (soft delete support)
│       └── Document
├── BaseProcessingModel (processing + quality + cost tracking)
│   ├── ProcessingTask
│   ├── KnowledgeNode
│   └── KnowledgeEdge
└── API Request/Response Models
    ├── Authentication (Login, Register, etc.)
    ├── Document Management (Upload, Search, etc.)
    ├── Processing (Tasks, Analytics, etc.)
    └── WebSocket Messages
```

## Core Components

### Base Configuration (`BrainBaseModel`)

All models inherit from `BrainBaseModel` which provides:

- **Alias Generation**: Automatic camelCase/snake_case conversion
- **Validation**: Assignment validation and default value validation  
- **Serialization**: JSON serialization with proper formatting
- **API Compatibility**: Consistent field naming and structure

```python
from models.base import BrainBaseModel

class ExampleModel(BrainBaseModel):
    field_name: str  # Becomes "fieldName" in JSON
    
# Usage
model = ExampleModel(field_name="test")
print(model.model_dump_camel())  # {"fieldName": "test"}
print(model.model_dump_snake())  # {"field_name": "test"}
```

### Mixins for Common Functionality

#### TimestampMixin
```python
created_at: datetime  # Auto-populated
updated_at: Optional[datetime]  # Manual updates
age_seconds: float  # Computed field
```

#### UUIDMixin
```python
id: UUID  # Auto-generated UUID4
```

#### ProcessingMixin
```python
processing_status: ProcessingStatus
processing_started_at: Optional[datetime]
processing_completed_at: Optional[datetime]
processing_error_message: Optional[str]
processing_retry_count: int
processing_duration_seconds: Optional[float]  # Computed
is_processing_complete: bool  # Computed
```

#### QualityMixin
```python
extraction_quality: Optional[float]  # 0.0-1.0
extraction_confidence: Optional[float]  # 0.0-1.0
quality_grade: Optional[str]  # A-F grade, computed
```

#### CostMixin
```python
processing_cost: Optional[float]  # USD amount
token_count: Optional[int]
cost_per_token: Optional[float]  # Computed
```

## Model Usage Examples

### Document Processing Workflow

```python
from models.core import Document
from models.api import DocumentProcessingRequest
from models.base import ProcessingStatus

# 1. Create document
document = Document(
    user_id=user_id,
    title="Research Paper",
    filename="paper.pdf",
    document_type=DocumentType.PDF,
    processing_status=ProcessingStatus.PENDING
)

# 2. Configure processing
processing_config = DocumentProcessingRequest(
    extract_text=True,
    generate_embeddings=True,
    extract_entities=True,
    quality_threshold=0.8
)

# 3. Update after processing
document.processing_status = ProcessingStatus.COMPLETED
document.extraction_quality = 0.95
document.processing_cost = 0.0125
document.embedding = [0.1, 0.2, 0.3, ...]  # 1536 dimensions

# 4. Check computed properties
print(f"Quality Grade: {document.quality_grade}")  # "A"
print(f"Cost per Token: {document.cost_per_token}")  # 0.0000059
```

### API Request/Response Pattern

```python
from models.api import SemanticSearchRequest, SearchResponse

# Request validation
search_request = SemanticSearchRequest(
    query="machine learning transformers",
    limit=20,
    similarity_threshold=0.75,
    document_types=[DocumentType.PDF]
)

# Response with results
search_response = SearchResponse(
    results=[...],  # List[SearchResult]
    total_results=47,
    search_duration_ms=125,
    query=search_request.query,
    filters_applied={"document_types": ["pdf"]}
)

# Computed properties
print(f"Has results: {search_response.has_results}")
print(f"Avg similarity: {search_response.average_similarity_score}")
```

### Real-time WebSocket Messages

```python
from models.api import ProcessingStatusMessage

# Processing update
status_message = ProcessingStatusMessage(
    document_id=document.id,
    status=ProcessingStatus.PROCESSING,
    progress_percentage=67.5,
    data={
        "stage": "entity_extraction",
        "entities_found": 23
    }
)
```

### Cost Tracking and Analytics

```python
from models.monitoring import LLMUsage, DailyCostSummary

# Track individual operation
llm_usage = LLMUsage(
    user_id=user_id,
    operation_type="document_processing",
    model_name="gpt-4-turbo",
    input_tokens=2500,
    output_tokens=750,
    cost=0.0975
)

# Daily summary
daily_summary = DailyCostSummary(
    user_id=user_id,
    date=datetime.now().date(),
    total_cost=12.45,
    operation_costs={
        "document_processing": 8.75,
        "search_queries": 2.30
    },
    budget_limit=25.00
)

# Check budget status
if daily_summary.is_over_budget:
    print("Budget limit exceeded!")
```

## Validation Features

### Built-in Validations

```python
from models.api import LoginRequest

# Email format validation
try:
    login = LoginRequest(
        email="invalid-email",  # Fails validation
        password="test123"
    )
except ValidationError as e:
    print(e.errors())
```

### Custom Validators

```python
from validators.custom_validators import GoogleDriveIdValidator

# Google Drive ID validation
folder_id = GoogleDriveIdValidator.validate("1BxiMVs0XRA...")
```

### Business Rule Validation

```python
from models.core import Document

# Embedding dimension validation
document = Document(
    embedding=[0.1] * 1535  # Fails - must be 1536 dimensions
)
```

## Serialization and API Integration

### FastAPI Integration

```python
from fastapi import FastAPI
from models.api import LoginRequest, LoginResponse

app = FastAPI()

@app.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    # Request automatically validated
    # Response automatically serialized
    return LoginResponse(
        access_token="...",
        refresh_token="...",
        expires_in=3600,
        user=authenticated_user
    )
```

### JSON Schema Generation

```python
from models.core import Document

# Get JSON schema for API documentation
schema = Document.model_json_schema()
print(schema["properties"]["embedding"])
# {
#   "type": "array", 
#   "items": {"type": "number"},
#   "minItems": 1536,
#   "maxItems": 1536
# }
```

## Performance Considerations

### Memory Optimization

- **Lazy Loading**: Use `defer_build=True` in model config
- **Selective Serialization**: Use `exclude` and `include` parameters
- **Computed Fields**: Calculate derived values on-demand

### Vector Embedding Handling

```python
# Efficient embedding operations
document = Document(
    embedding=embedding_vector,  # Store as List[float]
    # ... other fields
)

# Serialize without embeddings for list views
lightweight_doc = document.model_dump(exclude={"embedding"})
```

## Testing and Quality Assurance

### Model Testing

```python
import pytest
from models.core import Document
from models.examples import create_example_document

def test_document_creation():
    doc = create_example_document()
    assert doc.quality_grade == "A"
    assert doc.age_seconds > 0
    
def test_document_validation():
    with pytest.raises(ValidationError):
        Document(embedding=[0.1] * 1000)  # Wrong dimension
```

### Example-Driven Development

```python
from models.examples import get_all_examples

# Get comprehensive examples for testing
examples = get_all_examples()
print(examples["base_models"]["document"])
```

## Migration and Versioning

### Schema Evolution

- **Backward Compatibility**: Use `Optional` fields for new attributes
- **Default Values**: Provide sensible defaults for new fields
- **Deprecation**: Mark deprecated fields with documentation

### Database Integration

```python
# SQLAlchemy integration example
from sqlalchemy.orm import declarative_base
from models.core import Document

# Convert Pydantic models to SQLAlchemy
Base = declarative_base()

class DocumentTable(Base):
    __tablename__ = "documents"
    # Map Pydantic fields to SQLAlchemy columns
```

## Best Practices

### Model Design

1. **Single Responsibility**: Each model should represent one clear concept
2. **Composition over Inheritance**: Use mixins for shared functionality
3. **Validation at Boundaries**: Validate data at API entry points
4. **Computed Fields**: Use for derived values that don't need storage

### API Design

1. **Request/Response Pairs**: Every endpoint should have dedicated models
2. **Nested Validation**: Use nested models for complex request structures
3. **Pagination**: Use consistent pagination models across endpoints
4. **Error Handling**: Provide structured error responses

### Performance

1. **Field Selection**: Use `exclude`/`include` for large models
2. **Lazy Loading**: Don't compute expensive fields unless needed
3. **Caching**: Cache model schemas and validation results
4. **Bulk Operations**: Use specialized models for bulk operations

## Integration Points

### Database Layer
- SQLAlchemy model generation
- Migration script support
- Constraint validation

### API Layer  
- FastAPI route definitions
- OpenAPI schema generation
- Request/response validation

### Frontend Layer
- TypeScript type generation
- Form validation schemas
- Real-time update models

### Monitoring Layer
- Cost tracking integration
- Performance metrics collection
- Health check validation

## Future Enhancements

### Planned Features

1. **Advanced Validation**: Custom validation rules engine
2. **Schema Registry**: Centralized schema management
3. **Version Migration**: Automatic model version migration
4. **Performance Profiling**: Model validation performance tracking

### Extensibility Points

1. **Custom Mixins**: Add domain-specific functionality
2. **Validation Plugins**: Pluggable validation rules
3. **Serialization Formats**: Support additional output formats
4. **Integration Adapters**: Connect to external systems

For detailed examples and usage patterns, see `examples.py` and the individual model files.