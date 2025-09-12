"""
brAIn v2.0 Pydantic Model Examples
Comprehensive examples demonstrating usage of all model types and validation features.
"""

from datetime import datetime, timedelta
from uuid import uuid4
from typing import Dict, List, Any

from .base import (
    ProcessingStatus,
    DocumentType,
    LanguageCode,
    ServiceStatus,
    ValidationResult,
    PaginationParams,
    ErrorDetail,
    ApiError,
    ApiResponse
)
from .core import (
    User,
    Folder,
    Document,
    ProcessingTask,
    SearchResult,
    KnowledgeNode,
    KnowledgeEdge
)
from .api import (
    LoginRequest,
    LoginResponse,
    CreateFolderRequest,
    SemanticSearchRequest,
    SearchResponse,
    ProcessingStatusMessage,
    BulkOperationResponse
)
from .monitoring import (
    LLMUsage,
    SystemHealth,
    ProcessingAnalytics,
    DailyCostSummary
)


# ========================================
# BASE MODEL EXAMPLES
# ========================================

def create_example_user() -> User:
    """Create an example user with all fields populated"""
    return User(
        id=uuid4(),
        email="john.doe@example.com",
        full_name="John Doe",
        is_active=True,
        is_verified=True,
        timezone="America/New_York",
        language_preference=LanguageCode.EN,
        monthly_budget_limit=500.00,
        notification_preferences={
            "email_updates": True,
            "processing_complete": True,
            "budget_alerts": True,
            "weekly_summary": False
        },
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )


def create_example_folder() -> Folder:
    """Create an example folder with sync configuration"""
    return Folder(
        id=uuid4(),
        user_id=uuid4(),
        google_folder_id="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
        name="Research Documents",
        auto_sync_enabled=True,
        sync_frequency_hours=24,
        last_sync_at=datetime.utcnow() - timedelta(hours=12),
        sync_status="completed",
        document_count=142,
        total_size_bytes=524288000,  # ~500MB
        tags=["research", "papers", "machine-learning"],
        metadata={
            "sync_errors": 0,
            "last_full_sync": "2025-09-10T08:00:00Z",
            "avg_processing_time": 2.5
        },
        created_at=datetime.utcnow() - timedelta(days=30),
        updated_at=datetime.utcnow()
    )


def create_example_document() -> Document:
    """Create an example document with comprehensive metadata"""
    return Document(
        id=uuid4(),
        user_id=uuid4(),
        folder_id=uuid4(),
        google_file_id="1mGdI9q_dQK8XGK9pLkO_8uQ9n4YkLcFw",
        title="Transformer Architecture Deep Dive",
        filename="transformer_architecture_analysis.pdf",
        file_extension=".pdf",
        file_size=2048576,  # 2MB
        content_hash="sha256:a1b2c3d4e5f6789012345678901234567890abcdef",
        document_type=DocumentType.PDF,
        language_code=LanguageCode.EN,
        extracted_text="The Transformer architecture has revolutionized natural language processing...",
        extracted_metadata={
            "author": "Dr. Jane Smith",
            "creation_date": "2025-09-01",
            "page_count": 24,
            "word_count": 8450
        },
        embedding=[0.1234] * 1536,  # Example 1536-dimensional vector
        processing_status=ProcessingStatus.COMPLETED,
        processing_started_at=datetime.utcnow() - timedelta(minutes=15),
        processing_completed_at=datetime.utcnow() - timedelta(minutes=5),
        processing_retry_count=0,
        extraction_quality=0.95,
        extraction_confidence=0.88,
        processing_cost=0.0125,
        token_count=2100,
        tags=["transformer", "nlp", "architecture", "research"],
        metadata={
            "processing_model": "gpt-4-turbo",
            "extraction_method": "comprehensive",
            "quality_checks_passed": 8,
            "entities_extracted": 45
        },
        created_at=datetime.utcnow() - timedelta(days=2),
        updated_at=datetime.utcnow() - timedelta(minutes=5)
    )


def create_example_processing_task() -> ProcessingTask:
    """Create an example processing task"""
    return ProcessingTask(
        id=uuid4(),
        user_id=uuid4(),
        document_id=uuid4(),
        task_type="extract_entities",
        priority=7,
        processing_status=ProcessingStatus.PROCESSING,
        processing_started_at=datetime.utcnow() - timedelta(minutes=3),
        configuration={
            "extract_people": True,
            "extract_organizations": True,
            "extract_locations": True,
            "confidence_threshold": 0.8,
            "max_entities": 100
        },
        result_data={},
        processing_cost=0.0045,
        token_count=850,
        metadata={
            "worker_id": "worker-001",
            "attempt_number": 1,
            "estimated_completion": "2025-09-11T11:15:00Z"
        },
        created_at=datetime.utcnow() - timedelta(minutes=5)
    )


# ========================================
# API REQUEST/RESPONSE EXAMPLES
# ========================================

def create_example_login_request() -> LoginRequest:
    """Example login request"""
    return LoginRequest(
        email="user@example.com",
        password="SecurePassword123!",
        remember_me=True
    )


def create_example_login_response() -> LoginResponse:
    """Example successful login response"""
    return LoginResponse(
        access_token="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
        refresh_token="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
        expires_in=3600,
        user=create_example_user()
    )


def create_example_folder_request() -> CreateFolderRequest:
    """Example folder creation request"""
    return CreateFolderRequest(
        google_folder_id="1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms",
        name="AI Research Papers",
        auto_sync_enabled=True,
        sync_frequency_hours=12,
        tags=["ai", "research", "papers"]
    )


def create_example_semantic_search() -> SemanticSearchRequest:
    """Example semantic search request"""
    return SemanticSearchRequest(
        query="transformer architecture attention mechanisms",
        limit=20,
        similarity_threshold=0.75,
        document_types=[DocumentType.PDF, DocumentType.DOCX],
        tags=["research", "nlp"],
        created_after=datetime.utcnow() - timedelta(days=30)
    )


def create_example_search_response() -> SearchResponse:
    """Example search results response"""
    return SearchResponse(
        results=[
            SearchResult(
                document_id=uuid4(),
                title="Attention Is All You Need",
                similarity_score=0.92,
                document_type=DocumentType.PDF,
                excerpt="The Transformer, a model architecture eschewing recurrence...",
                highlight_text="<mark>attention</mark> mechanisms in <mark>transformer</mark> models",
                metadata={
                    "author": "Vaswani et al.",
                    "year": "2017"
                }
            ),
            SearchResult(
                document_id=uuid4(),
                title="BERT: Pre-training Deep Bidirectional Transformers",
                similarity_score=0.89,
                document_type=DocumentType.PDF,
                excerpt="We introduce BERT, which stands for Bidirectional...",
                highlight_text="bidirectional <mark>transformer</mark> representations",
                metadata={
                    "author": "Devlin et al.",
                    "year": "2018"
                }
            )
        ],
        total_results=47,
        search_duration_ms=125,
        query="transformer architecture attention mechanisms",
        filters_applied={
            "document_types": ["pdf", "docx"],
            "tags": ["research", "nlp"],
            "created_after": "2025-08-12T00:00:00Z"
        }
    )


# ========================================
# WEBSOCKET MESSAGE EXAMPLES
# ========================================

def create_example_processing_status_message() -> ProcessingStatusMessage:
    """Example WebSocket processing status message"""
    return ProcessingStatusMessage(
        document_id=uuid4(),
        status=ProcessingStatus.PROCESSING,
        progress_percentage=67.5,
        data={
            "stage": "entity_extraction",
            "entities_found": 23,
            "estimated_completion": "2025-09-11T11:12:30Z"
        }
    )


def create_example_bulk_operation_response() -> BulkOperationResponse:
    """Example bulk operation response"""
    return BulkOperationResponse(
        operation_id=uuid4(),
        total_items=50,
        processed_items=42,
        failed_items=3,
        errors=[
            {
                "document_id": str(uuid4()),
                "error": "Invalid file format",
                "details": "File appears to be corrupted"
            },
            {
                "document_id": str(uuid4()),
                "error": "Processing timeout",
                "details": "Document too large for current processing limits"
            }
        ],
        started_at=datetime.utcnow() - timedelta(minutes=45),
        completed_at=datetime.utcnow()
    )


# ========================================
# MONITORING EXAMPLES
# ========================================

def create_example_llm_usage() -> LLMUsage:
    """Example LLM usage tracking record"""
    return LLMUsage(
        id=uuid4(),
        user_id=uuid4(),
        operation_type="document_processing",
        model_name="gpt-4-turbo",
        input_tokens=2500,
        output_tokens=750,
        total_tokens=3250,
        cost=0.0975,
        processing_duration_ms=8500,
        request_metadata={
            "document_id": str(uuid4()),
            "processing_stage": "text_extraction",
            "quality_threshold": 0.8
        },
        response_metadata={
            "quality_score": 0.94,
            "confidence_score": 0.87,
            "entities_extracted": 34
        },
        created_at=datetime.utcnow()
    )


def create_example_system_health() -> SystemHealth:
    """Example system health monitoring record"""
    return SystemHealth(
        id=uuid4(),
        service_name="document_processor",
        status=ServiceStatus.HEALTHY,
        response_time_ms=245,
        cpu_usage_percent=45.2,
        memory_usage_percent=62.8,
        error_rate_percent=0.3,
        active_connections=12,
        health_details={
            "queue_length": 8,
            "workers_active": 4,
            "workers_idle": 2,
            "last_error": None,
            "uptime_seconds": 3600
        },
        metadata={
            "version": "2.1.4",
            "deployment": "production",
            "region": "us-east-1"
        },
        created_at=datetime.utcnow()
    )


def create_example_daily_cost_summary() -> DailyCostSummary:
    """Example daily cost summary"""
    return DailyCostSummary(
        id=uuid4(),
        user_id=uuid4(),
        date=datetime.utcnow().date(),
        total_cost=12.45,
        operation_costs={
            "document_processing": 8.75,
            "search_queries": 2.30,
            "entity_extraction": 1.40
        },
        total_tokens=28500,
        operation_counts={
            "documents_processed": 15,
            "search_queries": 48,
            "entity_extractions": 12
        },
        budget_limit=25.00,
        metadata={
            "avg_cost_per_document": 0.58,
            "peak_usage_hour": 14,
            "cost_trend": "increasing"
        },
        created_at=datetime.utcnow()
    )


# ========================================
# KNOWLEDGE GRAPH EXAMPLES
# ========================================

def create_example_knowledge_node() -> KnowledgeNode:
    """Example knowledge graph node"""
    return KnowledgeNode(
        id=uuid4(),
        user_id=uuid4(),
        node_type="person",
        name="Geoffrey Hinton",
        embedding=[0.2345] * 384,  # Example 384-dimensional vector
        confidence_score=0.95,
        source_document_ids=[uuid4(), uuid4()],
        extraction_count=8,
        properties={
            "full_name": "Geoffrey Everest Hinton",
            "affiliation": "University of Toronto",
            "field": "Machine Learning",
            "known_for": "Deep Learning, Backpropagation"
        },
        metadata={
            "first_mentioned": "2025-09-01T10:30:00Z",
            "last_updated": "2025-09-11T09:15:00Z",
            "frequency_score": 0.87
        },
        created_at=datetime.utcnow() - timedelta(days=10)
    )


def create_example_knowledge_edge() -> KnowledgeEdge:
    """Example knowledge graph relationship"""
    return KnowledgeEdge(
        id=uuid4(),
        user_id=uuid4(),
        source_node_id=uuid4(),
        target_node_id=uuid4(),
        relationship_type="mentor_of",
        confidence_score=0.89,
        source_document_ids=[uuid4()],
        extraction_count=3,
        properties={
            "relationship_strength": "strong",
            "time_period": "1980s-present",
            "context": "academic_supervision"
        },
        metadata={
            "supporting_text": "Hinton supervised Bengio's doctoral research...",
            "validation_score": 0.92
        },
        created_at=datetime.utcnow() - timedelta(days=5)
    )


# ========================================
# VALIDATION EXAMPLES
# ========================================

def create_example_validation_result() -> ValidationResult:
    """Example validation result with errors and warnings"""
    return ValidationResult(
        is_valid=False,
        errors=[
            "Email format is invalid",
            "Password must be at least 8 characters"
        ],
        warnings=[
            "Password does not contain special characters",
            "Email domain not recognized"
        ],
        details={
            "field_count": 5,
            "validated_at": datetime.utcnow().isoformat(),
            "validation_time_ms": 12
        }
    )


def create_example_api_error() -> ApiError:
    """Example API error response"""
    return ApiError(
        success=False,
        error="Validation failed",
        details=[
            ErrorDetail(
                code="INVALID_EMAIL",
                message="Email address format is invalid",
                field="email",
                context={"provided_value": "invalid-email"}
            ),
            ErrorDetail(
                code="PASSWORD_TOO_SHORT",
                message="Password must be at least 8 characters long",
                field="password",
                context={"min_length": 8, "provided_length": 6}
            )
        ],
        timestamp=datetime.utcnow(),
        request_id="req_abc123def456"
    )


def create_example_api_response() -> ApiResponse:
    """Example successful API response"""
    return ApiResponse(
        success=True,
        data={
            "user_id": str(uuid4()),
            "message": "User created successfully"
        },
        message="Operation completed successfully",
        timestamp=datetime.utcnow()
    )


# ========================================
# PAGINATION EXAMPLES
# ========================================

def create_example_pagination_params() -> PaginationParams:
    """Example pagination parameters"""
    return PaginationParams(
        page=2,
        per_page=25
    )


# ========================================
# USAGE EXAMPLES DOCUMENTATION
# ========================================

def demonstrate_model_serialization():
    """
    Demonstrate different serialization options for models.
    
    Returns:
        Dict with serialization examples
    """
    user = create_example_user()
    
    return {
        "camel_case": user.model_dump_camel(),
        "snake_case": user.model_dump_snake(),
        "exclude_sensitive": user.model_dump_camel(exclude={"id", "email"}),
        "json_schema": user.model_json_schema()
    }


def demonstrate_validation_examples():
    """
    Demonstrate validation scenarios with examples.
    
    Returns:
        Dict with validation examples
    """
    examples = {}
    
    # Valid login request
    try:
        valid_login = LoginRequest(
            email="user@example.com",
            password="SecurePass123!",
            remember_me=True
        )
        examples["valid_login"] = valid_login.model_dump()
    except Exception as e:
        examples["valid_login_error"] = str(e)
    
    # Invalid login request
    try:
        invalid_login = LoginRequest(
            email="invalid-email",  # Invalid email format
            password="123",         # Too short
            remember_me=True
        )
        examples["invalid_login"] = invalid_login.model_dump()
    except Exception as e:
        examples["invalid_login_error"] = str(e)
    
    return examples


def demonstrate_computed_fields():
    """
    Demonstrate computed field functionality.
    
    Returns:
        Dict with computed field examples
    """
    document = create_example_document()
    bulk_op = create_example_bulk_operation_response()
    search = create_example_search_response()
    
    return {
        "document_age_seconds": document.age_seconds,
        "document_quality_grade": document.quality_grade,
        "document_cost_per_token": document.cost_per_token,
        "bulk_operation_success_rate": bulk_op.success_rate,
        "bulk_operation_complete": bulk_op.is_complete,
        "search_has_results": search.has_results,
        "search_avg_similarity": search.average_similarity_score
    }


# ========================================
# EXAMPLE COLLECTIONS
# ========================================

def get_all_examples() -> Dict[str, Any]:
    """
    Get all example models organized by category.
    
    Returns:
        Dictionary containing all example models
    """
    return {
        "base_models": {
            "user": create_example_user().model_dump(),
            "folder": create_example_folder().model_dump(),
            "document": create_example_document().model_dump(),
            "processing_task": create_example_processing_task().model_dump()
        },
        "api_requests": {
            "login": create_example_login_request().model_dump(),
            "create_folder": create_example_folder_request().model_dump(),
            "semantic_search": create_example_semantic_search().model_dump()
        },
        "api_responses": {
            "login": create_example_login_response().model_dump(),
            "search": create_example_search_response().model_dump(),
            "bulk_operation": create_example_bulk_operation_response().model_dump()
        },
        "monitoring": {
            "llm_usage": create_example_llm_usage().model_dump(),
            "system_health": create_example_system_health().model_dump(),
            "daily_cost": create_example_daily_cost_summary().model_dump()
        },
        "knowledge_graph": {
            "node": create_example_knowledge_node().model_dump(),
            "edge": create_example_knowledge_edge().model_dump()
        },
        "websocket_messages": {
            "processing_status": create_example_processing_status_message().model_dump()
        },
        "validation": {
            "validation_result": create_example_validation_result().model_dump(),
            "api_error": create_example_api_error().model_dump(),
            "api_response": create_example_api_response().model_dump()
        },
        "demonstrations": {
            "serialization": demonstrate_model_serialization(),
            "validation": demonstrate_validation_examples(),
            "computed_fields": demonstrate_computed_fields()
        }
    }