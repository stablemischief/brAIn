"""
brAIn v2.0 API Request/Response Models
Pydantic models for API validation, serialization, and documentation.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID

from pydantic import BaseModel, Field, computed_field

from .base import (
    BrainBaseModel, 
    PaginationParams, 
    PaginatedResponse,
    ProcessingStatus,
    DocumentType,
    LanguageCode,
    ServiceStatus,
    ApiResponse,
    ApiError
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


# ========================================
# AUTHENTICATION & USER MANAGEMENT
# ========================================

class LoginRequest(BrainBaseModel):
    """User login request"""
    
    email: str = Field(
        description="User's email address",
        examples=["user@example.com"]
    )
    
    password: str = Field(
        description="User's password",
        min_length=8,
        examples=["SecurePass123!"]
    )
    
    remember_me: bool = Field(
        default=False,
        description="Whether to create a long-lived session"
    )


class LoginResponse(BrainBaseModel):
    """Successful login response"""
    
    access_token: str = Field(
        description="JWT access token for API authentication"
    )
    
    refresh_token: str = Field(
        description="JWT refresh token for token renewal"
    )
    
    expires_in: int = Field(
        description="Token expiration time in seconds",
        examples=[3600]
    )
    
    user: User = Field(
        description="Authenticated user information"
    )


class RegisterRequest(BrainBaseModel):
    """User registration request"""
    
    email: str = Field(
        description="User's email address",
        examples=["newuser@example.com"]
    )
    
    password: str = Field(
        description="User's password",
        min_length=8,
        examples=["SecurePass123!"]
    )
    
    full_name: str = Field(
        description="User's full name",
        min_length=2,
        max_length=100,
        examples=["John Doe"]
    )
    
    agree_to_terms: bool = Field(
        description="User agreement to terms of service"
    )


class RefreshTokenRequest(BrainBaseModel):
    """Token refresh request"""
    
    refresh_token: str = Field(
        description="Valid refresh token"
    )


class PasswordResetRequest(BrainBaseModel):
    """Password reset initiation request"""
    
    email: str = Field(
        description="User's email address"
    )


class PasswordResetConfirmRequest(BrainBaseModel):
    """Password reset confirmation request"""
    
    token: str = Field(
        description="Password reset token from email"
    )
    
    new_password: str = Field(
        description="New password",
        min_length=8
    )


# ========================================
# USER PROFILE MANAGEMENT
# ========================================

class UpdateUserProfileRequest(BrainBaseModel):
    """User profile update request"""
    
    full_name: Optional[str] = Field(
        default=None,
        min_length=2,
        max_length=100,
        description="Updated full name"
    )
    
    timezone: Optional[str] = Field(
        default=None,
        description="User's timezone (IANA format)",
        examples=["America/New_York", "Europe/London"]
    )
    
    language_preference: Optional[LanguageCode] = Field(
        default=None,
        description="Preferred language for the interface"
    )
    
    notification_preferences: Optional[Dict[str, bool]] = Field(
        default=None,
        description="Notification preferences by type",
        examples=[{"email_updates": True, "processing_complete": False}]
    )


class UserPreferencesRequest(BrainBaseModel):
    """User preferences update request"""
    
    default_folder_sync: bool = Field(
        default=True,
        description="Auto-sync new folders by default"
    )
    
    processing_quality_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum quality threshold for document processing"
    )
    
    monthly_budget_limit: Optional[float] = Field(
        default=None,
        ge=0.0,
        description="Monthly processing budget limit in USD"
    )
    
    auto_retry_failed: bool = Field(
        default=True,
        description="Automatically retry failed document processing"
    )


# ========================================
# FOLDER MANAGEMENT
# ========================================

class CreateFolderRequest(BrainBaseModel):
    """Create new folder request"""
    
    google_folder_id: str = Field(
        description="Google Drive folder ID",
        min_length=15,
        examples=["1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms"]
    )
    
    name: str = Field(
        description="Display name for the folder",
        min_length=1,
        max_length=255,
        examples=["Research Documents"]
    )
    
    auto_sync_enabled: bool = Field(
        default=True,
        description="Enable automatic synchronization"
    )
    
    sync_frequency_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="Sync frequency in hours (1-168)"
    )
    
    tags: List[str] = Field(
        default_factory=list,
        description="Tags for folder organization",
        examples=[["research", "papers"]]
    )


class UpdateFolderRequest(BrainBaseModel):
    """Update folder request"""
    
    name: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=255,
        description="Updated display name"
    )
    
    auto_sync_enabled: Optional[bool] = Field(
        default=None,
        description="Enable/disable automatic sync"
    )
    
    sync_frequency_hours: Optional[int] = Field(
        default=None,
        ge=1,
        le=168,
        description="Updated sync frequency"
    )
    
    tags: Optional[List[str]] = Field(
        default=None,
        description="Updated tags list"
    )


class FolderSyncRequest(BrainBaseModel):
    """Manual folder sync request"""
    
    force_full_sync: bool = Field(
        default=False,
        description="Force full resync ignoring timestamps"
    )
    
    process_documents: bool = Field(
        default=True,
        description="Process new documents immediately"
    )


class FoldersListResponse(PaginatedResponse):
    """Paginated folder list response"""
    
    items: List[Folder] = Field(
        description="List of folders for this page"
    )


# ========================================
# DOCUMENT MANAGEMENT
# ========================================

class UploadDocumentRequest(BrainBaseModel):
    """Document upload request metadata"""
    
    filename: str = Field(
        description="Original filename",
        min_length=1,
        max_length=255,
        examples=["research_paper.pdf"]
    )
    
    folder_id: Optional[UUID] = Field(
        default=None,
        description="Folder to upload document to"
    )
    
    document_type: DocumentType = Field(
        description="Type of document being uploaded"
    )
    
    process_immediately: bool = Field(
        default=True,
        description="Start processing immediately after upload"
    )
    
    tags: List[str] = Field(
        default_factory=list,
        description="Tags for document organization"
    )
    
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional document metadata"
    )


class DocumentProcessingRequest(BrainBaseModel):
    """Document processing configuration request"""
    
    extract_text: bool = Field(
        default=True,
        description="Extract text content from document"
    )
    
    generate_embeddings: bool = Field(
        default=True,
        description="Generate vector embeddings for semantic search"
    )
    
    extract_entities: bool = Field(
        default=True,
        description="Extract entities and relationships for knowledge graph"
    )
    
    quality_threshold: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Minimum quality threshold for processing"
    )
    
    language_hint: Optional[LanguageCode] = Field(
        default=None,
        description="Language hint to improve processing accuracy"
    )


class DocumentUpdateRequest(BrainBaseModel):
    """Document update request"""
    
    title: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=255,
        description="Updated document title"
    )
    
    tags: Optional[List[str]] = Field(
        default=None,
        description="Updated tags list"
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Updated metadata"
    )
    
    folder_id: Optional[UUID] = Field(
        default=None,
        description="Move document to different folder"
    )


class DocumentsListRequest(PaginationParams):
    """Documents list request with filters"""
    
    folder_id: Optional[UUID] = Field(
        default=None,
        description="Filter by folder ID"
    )
    
    document_type: Optional[DocumentType] = Field(
        default=None,
        description="Filter by document type"
    )
    
    processing_status: Optional[ProcessingStatus] = Field(
        default=None,
        description="Filter by processing status"
    )
    
    language_code: Optional[LanguageCode] = Field(
        default=None,
        description="Filter by detected language"
    )
    
    tags: Optional[List[str]] = Field(
        default=None,
        description="Filter by tags (AND operation)"
    )
    
    created_after: Optional[datetime] = Field(
        default=None,
        description="Filter documents created after this date"
    )
    
    created_before: Optional[datetime] = Field(
        default=None,
        description="Filter documents created before this date"
    )
    
    min_quality: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum extraction quality score"
    )
    
    search_query: Optional[str] = Field(
        default=None,
        description="Text search query"
    )


class DocumentsListResponse(PaginatedResponse):
    """Paginated document list response"""
    
    items: List[Document] = Field(
        description="List of documents for this page"
    )
    
    @computed_field
    @property
    def total_size_bytes(self) -> int:
        """Calculate total size of all documents in bytes"""
        return sum(doc.file_size or 0 for doc in self.items)


# ========================================
# SEARCH FUNCTIONALITY
# ========================================

class SemanticSearchRequest(BrainBaseModel):
    """Semantic search request"""
    
    query: str = Field(
        description="Natural language search query",
        min_length=2,
        max_length=1000,
        examples=["machine learning papers about transformers"]
    )
    
    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum number of results to return"
    )
    
    similarity_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for results"
    )
    
    folder_ids: Optional[List[UUID]] = Field(
        default=None,
        description="Limit search to specific folders"
    )
    
    document_types: Optional[List[DocumentType]] = Field(
        default=None,
        description="Limit search to specific document types"
    )
    
    tags: Optional[List[str]] = Field(
        default=None,
        description="Filter by tags (AND operation)"
    )
    
    created_after: Optional[datetime] = Field(
        default=None,
        description="Only search documents created after this date"
    )


class HybridSearchRequest(SemanticSearchRequest):
    """Hybrid search request (semantic + keyword)"""
    
    keyword_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for keyword search (0.0 = pure semantic)"
    )
    
    semantic_weight: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Weight for semantic search"
    )
    
    @computed_field
    @property
    def weights_sum_to_one(self) -> bool:
        """Validate that weights sum to approximately 1.0"""
        return abs((self.keyword_weight + self.semantic_weight) - 1.0) < 0.01


class SearchResponse(BrainBaseModel):
    """Search results response"""
    
    results: List[SearchResult] = Field(
        description="List of search results"
    )
    
    total_results: int = Field(
        description="Total number of matching documents",
        ge=0
    )
    
    search_duration_ms: int = Field(
        description="Search execution time in milliseconds",
        ge=0
    )
    
    query: str = Field(
        description="Original search query"
    )
    
    filters_applied: Dict[str, Any] = Field(
        default_factory=dict,
        description="Summary of applied search filters"
    )
    
    @computed_field
    @property
    def has_results(self) -> bool:
        """Check if search returned any results"""
        return len(self.results) > 0
    
    @computed_field
    @property
    def average_similarity_score(self) -> Optional[float]:
        """Calculate average similarity score of results"""
        if not self.results:
            return None
        return sum(result.similarity_score for result in self.results) / len(self.results)


# ========================================
# PROCESSING MANAGEMENT
# ========================================

class ProcessingTaskRequest(BrainBaseModel):
    """Manual processing task creation request"""
    
    document_id: UUID = Field(
        description="Document to process"
    )
    
    task_type: str = Field(
        description="Type of processing task",
        examples=["extract_text", "generate_embeddings", "extract_entities"]
    )
    
    priority: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Task priority (1=lowest, 10=highest)"
    )
    
    configuration: Dict[str, Any] = Field(
        default_factory=dict,
        description="Task-specific configuration parameters"
    )


class ProcessingTasksListRequest(PaginationParams):
    """Processing tasks list request with filters"""
    
    status: Optional[ProcessingStatus] = Field(
        default=None,
        description="Filter by processing status"
    )
    
    task_type: Optional[str] = Field(
        default=None,
        description="Filter by task type"
    )
    
    document_id: Optional[UUID] = Field(
        default=None,
        description="Filter by document ID"
    )
    
    created_after: Optional[datetime] = Field(
        default=None,
        description="Filter tasks created after this date"
    )


class ProcessingTasksListResponse(PaginatedResponse):
    """Paginated processing tasks list response"""
    
    items: List[ProcessingTask] = Field(
        description="List of processing tasks for this page"
    )
    
    @computed_field
    @property
    def status_summary(self) -> Dict[str, int]:
        """Count of tasks by status"""
        summary = {}
        for task in self.items:
            status = task.processing_status
            summary[status] = summary.get(status, 0) + 1
        return summary


class RetryProcessingRequest(BrainBaseModel):
    """Retry failed processing request"""
    
    task_ids: List[UUID] = Field(
        description="List of task IDs to retry",
        min_length=1
    )
    
    reset_retry_count: bool = Field(
        default=False,
        description="Reset retry count to zero"
    )


# ========================================
# KNOWLEDGE GRAPH
# ========================================

class KnowledgeGraphQueryRequest(BrainBaseModel):
    """Knowledge graph query request"""
    
    entity_query: Optional[str] = Field(
        default=None,
        description="Search for specific entities",
        examples=["machine learning"]
    )
    
    relationship_types: Optional[List[str]] = Field(
        default=None,
        description="Filter by relationship types",
        examples=[["mentions", "related_to"]]
    )
    
    min_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score for relationships"
    )
    
    max_depth: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Maximum traversal depth from starting nodes"
    )
    
    limit: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Maximum number of nodes to return"
    )


class KnowledgeGraphResponse(BrainBaseModel):
    """Knowledge graph query response"""
    
    nodes: List[KnowledgeNode] = Field(
        description="Knowledge nodes in the graph"
    )
    
    edges: List[KnowledgeEdge] = Field(
        description="Knowledge edges connecting the nodes"
    )
    
    query_summary: Dict[str, Any] = Field(
        description="Summary of the query executed"
    )
    
    @computed_field
    @property
    def node_count(self) -> int:
        """Total number of nodes"""
        return len(self.nodes)
    
    @computed_field
    @property
    def edge_count(self) -> int:
        """Total number of edges"""
        return len(self.edges)


# ========================================
# ANALYTICS AND MONITORING
# ========================================

class AnalyticsRequest(BrainBaseModel):
    """Analytics data request"""
    
    start_date: datetime = Field(
        description="Start date for analytics period"
    )
    
    end_date: datetime = Field(
        description="End date for analytics period"
    )
    
    metrics: List[str] = Field(
        description="List of metrics to include",
        examples=[["document_count", "processing_cost", "search_queries"]]
    )
    
    group_by: Optional[str] = Field(
        default=None,
        description="Group results by time period",
        examples=["day", "week", "month"]
    )


class CostAnalyticsResponse(BrainBaseModel):
    """Cost analytics response"""
    
    total_cost: float = Field(
        description="Total cost for the period",
        ge=0.0
    )
    
    cost_by_operation: Dict[str, float] = Field(
        description="Cost breakdown by operation type"
    )
    
    daily_costs: List[Dict[str, Any]] = Field(
        description="Daily cost breakdown"
    )
    
    token_usage: Dict[str, int] = Field(
        description="Token usage by operation type"
    )
    
    budget_status: Dict[str, Any] = Field(
        description="Current budget status and alerts"
    )


class SystemHealthResponse(BrainBaseModel):
    """System health status response"""
    
    overall_status: ServiceStatus = Field(
        description="Overall system health status"
    )
    
    services: Dict[str, Dict[str, Any]] = Field(
        description="Individual service health information"
    )
    
    performance_metrics: Dict[str, float] = Field(
        description="System performance metrics"
    )
    
    last_check: datetime = Field(
        description="When the health check was last performed"
    )
    
    alerts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Active system alerts"
    )


# ========================================
# WEBSOCKET MESSAGES
# ========================================

class WebSocketMessage(BrainBaseModel):
    """Base WebSocket message"""
    
    type: str = Field(
        description="Message type identifier"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Message timestamp"
    )
    
    data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Message payload"
    )


class ProcessingStatusMessage(WebSocketMessage):
    """Processing status update message"""
    
    type: str = Field(
        default="processing_status",
        description="Message type"
    )
    
    document_id: UUID = Field(
        description="Document being processed"
    )
    
    status: ProcessingStatus = Field(
        description="Current processing status"
    )
    
    progress_percentage: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Processing progress percentage"
    )
    
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if processing failed"
    )


class CostUpdateMessage(WebSocketMessage):
    """Real-time cost update message"""
    
    type: str = Field(
        default="cost_update",
        description="Message type"
    )
    
    operation_cost: float = Field(
        description="Cost of the current operation",
        ge=0.0
    )
    
    daily_total: float = Field(
        description="Total cost for today",
        ge=0.0
    )
    
    monthly_total: float = Field(
        description="Total cost for this month",
        ge=0.0
    )
    
    budget_alert: Optional[str] = Field(
        default=None,
        description="Budget alert message if threshold exceeded"
    )


# ========================================
# BULK OPERATIONS
# ========================================

class BulkDocumentProcessingRequest(BrainBaseModel):
    """Bulk document processing request"""
    
    document_ids: List[UUID] = Field(
        description="List of document IDs to process",
        min_length=1,
        max_length=100
    )
    
    processing_config: DocumentProcessingRequest = Field(
        description="Processing configuration for all documents"
    )
    
    batch_size: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of documents to process concurrently"
    )


class BulkOperationResponse(BrainBaseModel):
    """Bulk operation response"""
    
    operation_id: UUID = Field(
        description="Unique identifier for the bulk operation"
    )
    
    total_items: int = Field(
        description="Total number of items to process",
        ge=0
    )
    
    processed_items: int = Field(
        description="Number of items processed successfully",
        ge=0
    )
    
    failed_items: int = Field(
        description="Number of items that failed processing",
        ge=0
    )
    
    errors: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of errors encountered"
    )
    
    started_at: datetime = Field(
        description="When the operation started"
    )
    
    completed_at: Optional[datetime] = Field(
        default=None,
        description="When the operation completed"
    )
    
    @computed_field
    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage"""
        if self.total_items == 0:
            return 0.0
        return (self.processed_items / self.total_items) * 100.0
    
    @computed_field
    @property
    def is_complete(self) -> bool:
        """Check if operation is complete"""
        return self.completed_at is not None


# ========================================
# EXPORT FUNCTIONALITY
# ========================================

class ExportRequest(BrainBaseModel):
    """Data export request"""
    
    export_type: str = Field(
        description="Type of export",
        examples=["documents", "knowledge_graph", "analytics"]
    )
    
    format: str = Field(
        description="Export format",
        examples=["json", "csv", "pdf"]
    )
    
    filters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Filters to apply to exported data"
    )
    
    include_metadata: bool = Field(
        default=True,
        description="Include metadata in export"
    )


class ExportResponse(BrainBaseModel):
    """Export operation response"""
    
    export_id: UUID = Field(
        description="Unique identifier for the export"
    )
    
    download_url: Optional[str] = Field(
        default=None,
        description="URL to download the exported file"
    )
    
    expires_at: Optional[datetime] = Field(
        default=None,
        description="When the download URL expires"
    )
    
    file_size: Optional[int] = Field(
        default=None,
        ge=0,
        description="Size of the exported file in bytes"
    )
    
    status: ProcessingStatus = Field(
        description="Export processing status"
    )


# ========================================
# API RESPONSE WRAPPERS
# ========================================

# Create specific response models for common endpoints
UserResponse = ApiResponse.model_validate({"data": User})
FolderResponse = ApiResponse.model_validate({"data": Folder})
DocumentResponse = ApiResponse.model_validate({"data": Document})
ProcessingTaskResponse = ApiResponse.model_validate({"data": ProcessingTask})

# Type aliases for complex response types
DocumentListResponse = DocumentsListResponse
FolderListResponse = FoldersListResponse
ProcessingTaskListResponse = ProcessingTasksListResponse