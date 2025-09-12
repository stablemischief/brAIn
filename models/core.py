"""
brAIn v2.0 Core Domain Models
Core business entities: Users, Folders, Documents, and Sessions.
"""

from datetime import datetime
from typing import Dict, List, Optional, Union
from uuid import UUID

from pydantic import Field, field_validator, computed_field, EmailStr
from validators.custom_validators import GoogleDriveIdValidator, ContentHashValidator

from .base import (
    BrainBaseModel, 
    BaseEntityModel, 
    BaseProcessingModel, 
    BaseDeletableModel,
    ProcessingStatus,
    DocumentType,
    LanguageCode,
    TimestampMixin,
    MetadataMixin
)


# ========================================
# USER MODELS
# ========================================

class User(BaseEntityModel):
    """User account with authentication and preference management"""
    
    email: EmailStr = Field(
        description="User's email address (must be unique)",
        examples=["user@example.com", "admin@brain.ai"]
    )
    
    display_name: Optional[str] = Field(
        default=None,
        description="User's display name",
        min_length=1,
        max_length=100,
        examples=["John Doe", "Jane Smith", "AI Assistant"]
    )
    
    avatar_url: Optional[str] = Field(
        default=None,
        description="URL to user's avatar image",
        examples=["https://example.com/avatar.jpg"]
    )
    
    role: str = Field(
        default="user",
        description="User role for permissions",
        examples=["user", "admin", "viewer"]
    )
    
    # Authentication fields
    auth_provider: str = Field(
        default="supabase",
        description="Authentication provider",
        examples=["supabase", "google", "email"]
    )
    
    external_id: Optional[str] = Field(
        default=None,
        description="External authentication provider ID",
        examples=["google_123456789", "supabase_abc123"]
    )
    
    email_verified: bool = Field(
        default=False,
        description="Whether the user's email is verified"
    )
    
    # User preferences and settings
    preferences: Dict[str, Union[str, bool, int]] = Field(
        default_factory=dict,
        description="User preferences (theme, language, etc.)",
        examples=[{"theme": "dark", "language": "en", "notifications": True}]
    )
    
    settings: Dict[str, Union[str, bool, int]] = Field(
        default_factory=dict,
        description="User settings (auto_sync, batch_size, etc.)",
        examples=[{"auto_sync": True, "batch_size": 10}]
    )
    
    # Budget and cost tracking
    monthly_budget_limit: float = Field(
        default=100.00,
        description="Monthly budget limit in USD",
        ge=0.0,
        decimal_places=2,
        examples=[100.00, 500.00, 25.00]
    )
    
    current_month_spend: float = Field(
        default=0.00,
        description="Current month's spending in USD",
        ge=0.0,
        decimal_places=2
    )
    
    # Activity tracking
    last_seen_at: Optional[datetime] = Field(
        default=None,
        description="When the user was last active"
    )
    
    # Soft delete
    deleted_at: Optional[datetime] = Field(
        default=None,
        description="Soft delete timestamp"
    )
    
    @computed_field
    @property
    def budget_remaining(self) -> float:
        """Calculate remaining budget for current month"""
        return max(0.0, self.monthly_budget_limit - self.current_month_spend)
    
    @computed_field
    @property
    def budget_utilization_percent(self) -> float:
        """Calculate budget utilization as percentage"""
        if self.monthly_budget_limit <= 0:
            return 0.0
        return (self.current_month_spend / self.monthly_budget_limit) * 100
    
    @computed_field
    @property
    def is_budget_exceeded(self) -> bool:
        """Check if monthly budget is exceeded"""
        return self.current_month_spend > self.monthly_budget_limit
    
    @field_validator('role')
    @classmethod
    def validate_role(cls, v: str) -> str:
        """Validate user role"""
        allowed_roles = {'user', 'admin', 'viewer', 'editor'}
        if v not in allowed_roles:
            raise ValueError(f"Role must be one of: {', '.join(allowed_roles)}")
        return v


class UserSession(BaseEntityModel):
    """User session for activity tracking and security"""
    
    session_token: str = Field(
        description="Unique session token",
        min_length=32,
        examples=["abc123def456ghi789jkl012mno345pqr678"]
    )
    
    ip_address: Optional[str] = Field(
        default=None,
        description="Client IP address",
        examples=["192.168.1.1", "203.0.113.1"]
    )
    
    user_agent: Optional[str] = Field(
        default=None,
        description="Client user agent string",
        max_length=1000
    )
    
    # Session data
    session_data: Dict[str, Union[str, int, bool]] = Field(
        default_factory=dict,
        description="Flexible session data storage"
    )
    
    # Session lifecycle
    last_accessed_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the session was last accessed"
    )
    
    expires_at: datetime = Field(
        description="When the session expires"
    )
    
    # Activity metrics
    page_views: int = Field(
        default=0,
        description="Number of page views in this session",
        ge=0
    )
    
    api_calls: int = Field(
        default=0,
        description="Number of API calls in this session",
        ge=0
    )
    
    last_activity: Dict[str, Union[str, datetime]] = Field(
        default_factory=dict,
        description="Last activity details"
    )
    
    @computed_field
    @property
    def is_expired(self) -> bool:
        """Check if session is expired"""
        return datetime.utcnow() > self.expires_at
    
    @computed_field
    @property
    def session_duration_seconds(self) -> float:
        """Calculate session duration in seconds"""
        return (self.last_accessed_at - self.created_at).total_seconds()


# ========================================
# FOLDER MODELS
# ========================================

class Folder(BaseEntityModel, MetadataMixin):
    """Google Drive folder with sync configuration and monitoring"""
    
    # Google Drive fields
    google_folder_id: str = Field(
        description="Google Drive folder ID",
        examples=["1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms"]
    )
    
    folder_name: str = Field(
        description="Display name of the folder",
        min_length=1,
        max_length=255,
        examples=["My Documents", "Project Files", "Research Papers"]
    )
    
    folder_path: Optional[str] = Field(
        default=None,
        description="Full path to the folder in Google Drive",
        examples=["/Documents/Projects", "/Research/2025"]
    )
    
    parent_folder_id: Optional[str] = Field(
        default=None,
        description="Google Drive ID of parent folder"
    )
    
    # Sync configuration
    auto_sync_enabled: bool = Field(
        default=True,
        description="Whether automatic sync is enabled"
    )
    
    sync_frequency_minutes: int = Field(
        default=60,
        description="How often to sync in minutes",
        ge=5,
        le=1440,  # Max 24 hours
        examples=[30, 60, 120, 720]
    )
    
    include_subfolders: bool = Field(
        default=True,
        description="Whether to include subfolders in sync"
    )
    
    file_type_filters: List[str] = Field(
        default_factory=list,
        description="File types to include (empty = all types)",
        examples=[["pdf", "docx"], ["txt", "md"], []]
    )
    
    max_file_size_mb: int = Field(
        default=50,
        description="Maximum file size to process in MB",
        ge=1,
        le=200,
        examples=[10, 50, 100]
    )
    
    # Status and monitoring
    last_sync_at: Optional[datetime] = Field(
        default=None,
        description="When the folder was last synced"
    )
    
    last_successful_sync_at: Optional[datetime] = Field(
        default=None,
        description="When the last successful sync occurred"
    )
    
    sync_status: ProcessingStatus = Field(
        default=ProcessingStatus.PENDING,
        description="Current sync status"
    )
    
    sync_error_message: Optional[str] = Field(
        default=None,
        description="Error message if sync failed",
        max_length=1000
    )
    
    # Statistics
    total_files: int = Field(
        default=0,
        description="Total number of files in the folder",
        ge=0
    )
    
    processed_files: int = Field(
        default=0,
        description="Number of successfully processed files",
        ge=0
    )
    
    failed_files: int = Field(
        default=0,
        description="Number of files that failed processing",
        ge=0
    )
    
    total_size_bytes: int = Field(
        default=0,
        description="Total size of all files in bytes",
        ge=0
    )
    
    @computed_field
    @property
    def processing_success_rate(self) -> float:
        """Calculate processing success rate as percentage"""
        if self.total_files == 0:
            return 0.0
        return (self.processed_files / self.total_files) * 100
    
    @computed_field
    @property
    def total_size_mb(self) -> float:
        """Convert total size to megabytes"""
        return self.total_size_bytes / (1024 * 1024)
    
    @computed_field
    @property
    def sync_health_score(self) -> float:
        """Calculate folder sync health score (0-100)"""
        score = 100.0
        
        # Deduct for failed files
        if self.total_files > 0:
            failure_rate = (self.failed_files / self.total_files) * 100
            score -= failure_rate
        
        # Deduct for sync failures
        if self.sync_status == ProcessingStatus.FAILED:
            score -= 20
        
        # Deduct for stale syncs (more than 2x frequency)
        if self.last_sync_at:
            minutes_since_sync = (datetime.utcnow() - self.last_sync_at).total_seconds() / 60
            if minutes_since_sync > (self.sync_frequency_minutes * 2):
                score -= 10
        
        return max(0.0, min(100.0, score))
    
    @field_validator('google_folder_id')
    @classmethod
    def validate_google_folder_id(cls, v: str) -> str:
        """Validate Google Drive folder ID format"""
        return GoogleDriveIdValidator.validate(v)


# ========================================
# DOCUMENT MODELS
# ========================================

class Document(BaseProcessingModel, BaseDeletableModel):
    """Enhanced document with AI processing, embeddings, and cost tracking"""
    
    folder_id: UUID = Field(
        description="ID of the folder containing this document"
    )
    
    # Google Drive metadata
    google_file_id: str = Field(
        description="Google Drive file ID",
        examples=["1BxiMVs0XRA5nFMdKvBdBZjgmUUqptlbs74OgvE2upms"]
    )
    
    file_name: str = Field(
        description="Name of the file",
        min_length=1,
        max_length=255,
        examples=["document.pdf", "presentation.pptx", "data.xlsx"]
    )
    
    file_path: Optional[str] = Field(
        default=None,
        description="Full path to the file in Google Drive",
        examples=["/Documents/Projects/document.pdf"]
    )
    
    mime_type: Optional[str] = Field(
        default=None,
        description="MIME type of the file",
        examples=["application/pdf", "text/plain", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]
    )
    
    file_size_bytes: Optional[int] = Field(
        default=None,
        description="File size in bytes",
        ge=0,
        examples=[1024000, 5242880, 102400]
    )
    
    google_modified_at: Optional[datetime] = Field(
        default=None,
        description="When the file was last modified in Google Drive"
    )
    
    # Enhanced AI-focused fields
    document_type: Optional[DocumentType] = Field(
        default=None,
        description="Detected document type"
    )
    
    content_hash: Optional[str] = Field(
        default=None,
        description="SHA-256 hash for duplicate detection",
        examples=["e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"]
    )
    
    language_code: LanguageCode = Field(
        default=LanguageCode.UNKNOWN,
        description="Detected language of the document content"
    )
    
    # Text extraction
    raw_text: Optional[str] = Field(
        default=None,
        description="Raw extracted text from the document"
    )
    
    processed_text: Optional[str] = Field(
        default=None,
        description="Cleaned and processed text for embedding"
    )
    
    text_length: Optional[int] = Field(
        default=None,
        description="Length of processed text in characters",
        ge=0
    )
    
    # Vector embeddings (OpenAI text-embedding-3-small: 1536 dimensions)
    embedding: Optional[List[float]] = Field(
        default=None,
        description="Vector embedding for semantic search (1536 dimensions)",
        min_length=1536,
        max_length=1536
    )
    
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Model used to generate the embedding",
        examples=["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"]
    )
    
    embedding_created_at: Optional[datetime] = Field(
        default=None,
        description="When the embedding was created"
    )
    
    # Processing notes and context
    processing_notes: Dict[str, Union[str, int, float]] = Field(
        default_factory=dict,
        description="Additional processing information and notes"
    )
    
    @computed_field
    @property
    def file_size_mb(self) -> Optional[float]:
        """Convert file size to megabytes"""
        if self.file_size_bytes is None:
            return None
        return self.file_size_bytes / (1024 * 1024)
    
    @computed_field
    @property
    def has_embedding(self) -> bool:
        """Check if document has a vector embedding"""
        return self.embedding is not None and len(self.embedding) == 1536
    
    @computed_field
    @property
    def extraction_efficiency(self) -> Optional[float]:
        """Calculate text extraction efficiency (characters per byte)"""
        if self.file_size_bytes and self.text_length and self.file_size_bytes > 0:
            return self.text_length / self.file_size_bytes
        return None
    
    @computed_field
    @property
    def cost_per_character(self) -> Optional[float]:
        """Calculate cost per character of extracted text"""
        if self.text_length and self.processing_cost and self.text_length > 0:
            return self.processing_cost / self.text_length
        return None
    
    @computed_field
    @property
    def is_ready_for_search(self) -> bool:
        """Check if document is ready for semantic search"""
        return (
            self.processing_status == ProcessingStatus.COMPLETED and
            self.has_embedding and
            self.processed_text is not None and
            len(self.processed_text) > 0
        )
    
    @field_validator('google_file_id')
    @classmethod
    def validate_google_file_id(cls, v: str) -> str:
        """Validate Google Drive file ID format"""
        return GoogleDriveIdValidator.validate(v)
    
    @field_validator('content_hash')
    @classmethod
    def validate_content_hash(cls, v: Optional[str]) -> Optional[str]:
        """Validate content hash format"""
        if v is None:
            return v
        return ContentHashValidator.validate(v)
    
    @field_validator('embedding')
    @classmethod
    def validate_embedding_dimensions(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        """Validate embedding dimensions"""
        if v is None:
            return v
        
        if len(v) != 1536:
            raise ValueError("Embedding must have exactly 1536 dimensions")
        
        # Check for valid float values
        for i, val in enumerate(v):
            if not isinstance(val, (int, float)):
                raise ValueError(f"Embedding value at index {i} must be a number")
            if not (-2.0 <= val <= 2.0):  # OpenAI embeddings are typically in this range
                raise ValueError(f"Embedding value at index {i} is out of expected range [-2.0, 2.0]")
        
        return v


# ========================================
# PROCESSING QUEUE MODELS
# ========================================

class ProcessingTask(BaseEntityModel):
    """Background processing task with priority and retry logic"""
    
    document_id: Optional[UUID] = Field(
        default=None,
        description="ID of the document to process (if applicable)"
    )
    
    folder_id: Optional[UUID] = Field(
        default=None,
        description="ID of the folder to process (if applicable)"
    )
    
    # Task details
    task_type: str = Field(
        description="Type of processing task",
        examples=["extract", "embed", "sync", "analyze", "cleanup"]
    )
    
    task_data: Dict[str, Union[str, int, float, bool]] = Field(
        default_factory=dict,
        description="Task-specific data and parameters"
    )
    
    priority: int = Field(
        default=5,
        description="Task priority (1-10, higher = more urgent)",
        ge=1,
        le=10,
        examples=[1, 5, 8, 10]
    )
    
    # Processing details
    status: ProcessingStatus = Field(
        default=ProcessingStatus.PENDING,
        description="Current task status"
    )
    
    started_at: Optional[datetime] = Field(
        default=None,
        description="When task processing started"
    )
    
    completed_at: Optional[datetime] = Field(
        default=None,
        description="When task processing completed"
    )
    
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if task failed",
        max_length=1000
    )
    
    retry_count: int = Field(
        default=0,
        description="Number of retry attempts",
        ge=0
    )
    
    max_retries: int = Field(
        default=3,
        description="Maximum number of retry attempts",
        ge=0,
        le=10
    )
    
    # Scheduling
    scheduled_for: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the task should be executed"
    )
    
    worker_id: Optional[str] = Field(
        default=None,
        description="ID of the worker processing this task",
        examples=["worker-1", "worker-abc123"]
    )
    
    @computed_field
    @property
    def can_retry(self) -> bool:
        """Check if task can be retried"""
        return (
            self.status == ProcessingStatus.FAILED and
            self.retry_count < self.max_retries
        )
    
    @computed_field
    @property
    def processing_duration_seconds(self) -> Optional[float]:
        """Calculate task processing duration in seconds"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    @computed_field
    @property
    def is_overdue(self) -> bool:
        """Check if task is overdue for processing"""
        return (
            self.status == ProcessingStatus.PENDING and
            datetime.utcnow() > self.scheduled_for
        )
    
    @field_validator('task_type')
    @classmethod
    def validate_task_type(cls, v: str) -> str:
        """Validate task type"""
        allowed_types = {
            'extract', 'embed', 'sync', 'analyze', 'cleanup', 
            'backup', 'health_check', 'refresh_views'
        }
        if v not in allowed_types:
            raise ValueError(f"Task type must be one of: {', '.join(allowed_types)}")
        return v