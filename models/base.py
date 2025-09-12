"""
brAIn v2.0 Base Pydantic Models
Base models, mixins, and common functionality for all Pydantic models.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional, Union
from uuid import UUID, uuid4
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, computed_field, field_validator
from pydantic.alias_generators import to_camel, to_snake
from pydantic_core import core_schema


# ========================================
# ENUMS AND TYPES
# ========================================

class ProcessingStatus(str, Enum):
    """Processing status for documents and tasks"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class DocumentType(str, Enum):
    """Supported document types"""
    PDF = "pdf"
    DOCX = "docx"
    XLSX = "xlsx"
    PPTX = "pptx"
    TXT = "txt"
    MD = "md"
    HTML = "html"
    CSV = "csv"
    JSON = "json"
    XML = "xml"
    EPUB = "epub"
    RTF = "rtf"
    ODT = "odt"
    OTHER = "other"


class LanguageCode(str, Enum):
    """ISO 639-1 language codes"""
    EN = "en"
    ES = "es"
    FR = "fr"
    DE = "de"
    IT = "it"
    PT = "pt"
    RU = "ru"
    ZH = "zh"
    JA = "ja"
    KO = "ko"
    AR = "ar"
    HI = "hi"
    UNKNOWN = "unknown"


class ServiceStatus(str, Enum):
    """Service health status"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    DOWN = "down"
    MAINTENANCE = "maintenance"


# ========================================
# BASE MODEL CONFIGURATIONS
# ========================================

class BrainBaseModel(BaseModel):
    """
    Base model for all brAIn models with common configuration.
    
    Features:
    - Automatic alias generation (camelCase for API, snake_case for Python)
    - UUID validation and generation
    - Timestamp handling
    - JSON serialization configuration
    - Validation configuration
    """
    
    model_config = ConfigDict(
        # Alias configuration for API compatibility
        alias_generator=to_camel,
        populate_by_name=True,
        
        # Validation configuration
        validate_assignment=True,
        validate_default=True,
        use_enum_values=True,
        
        # Serialization configuration
        ser_json_timedelta='float',
        ser_json_bytes='base64',
        
        # Allow extra fields for extensibility
        extra='forbid',
        
        # Performance optimization
        defer_build=True,
        
        # Documentation
        title="brAIn Base Model",
        
        # JSON schema configuration
        json_schema_extra={
            "examples": [{}]
        }
    )
    
    def model_dump_camel(self, **kwargs) -> Dict[str, Any]:
        """Dump model to dict with camelCase keys for API responses"""
        return self.model_dump(by_alias=True, **kwargs)
    
    def model_dump_snake(self, **kwargs) -> Dict[str, Any]:
        """Dump model to dict with snake_case keys for internal use"""
        return self.model_dump(by_alias=False, **kwargs)


# ========================================
# MIXINS FOR COMMON FUNCTIONALITY
# ========================================

class TimestampMixin(BaseModel):
    """Mixin for models with timestamp fields"""
    
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Timestamp when the record was created",
        examples=["2025-09-11T10:30:00Z"]
    )
    
    updated_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp when the record was last updated",
        examples=["2025-09-11T10:35:00Z"]
    )
    
    @computed_field
    @property
    def age_seconds(self) -> float:
        """Calculate age of record in seconds"""
        return (datetime.utcnow() - self.created_at).total_seconds()


class UUIDMixin(BaseModel):
    """Mixin for models with UUID primary keys"""
    
    id: UUID = Field(
        default_factory=uuid4,
        description="Unique identifier for the record",
        examples=["550e8400-e29b-41d4-a716-446655440000"]
    )


class UserScopedMixin(BaseModel):
    """Mixin for models that belong to a specific user"""
    
    user_id: UUID = Field(
        description="ID of the user who owns this record",
        examples=["550e8400-e29b-41d4-a716-446655440000"]
    )


class SoftDeleteMixin(BaseModel):
    """Mixin for models with soft delete functionality"""
    
    deleted_at: Optional[datetime] = Field(
        default=None,
        description="Timestamp when the record was soft deleted",
        examples=[None, "2025-09-11T10:40:00Z"]
    )
    
    @computed_field
    @property
    def is_deleted(self) -> bool:
        """Check if record is soft deleted"""
        return self.deleted_at is not None


class MetadataMixin(BaseModel):
    """Mixin for models with flexible metadata storage"""
    
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Flexible metadata storage as key-value pairs",
        examples=[{"key": "value", "number": 42}]
    )
    
    tags: List[str] = Field(
        default_factory=list,
        description="List of tags for categorization",
        examples=[["tag1", "tag2", "category"]]
    )


# ========================================
# PROCESSING AND QUALITY MIXINS
# ========================================

class ProcessingMixin(BaseModel):
    """Mixin for models with processing status and error handling"""
    
    processing_status: ProcessingStatus = Field(
        default=ProcessingStatus.PENDING,
        description="Current processing status"
    )
    
    processing_started_at: Optional[datetime] = Field(
        default=None,
        description="When processing started"
    )
    
    processing_completed_at: Optional[datetime] = Field(
        default=None,
        description="When processing completed"
    )
    
    processing_error_message: Optional[str] = Field(
        default=None,
        description="Error message if processing failed",
        max_length=1000
    )
    
    processing_retry_count: int = Field(
        default=0,
        description="Number of processing retries attempted",
        ge=0
    )
    
    @computed_field
    @property
    def processing_duration_seconds(self) -> Optional[float]:
        """Calculate processing duration in seconds"""
        if self.processing_started_at and self.processing_completed_at:
            return (self.processing_completed_at - self.processing_started_at).total_seconds()
        return None
    
    @computed_field
    @property
    def is_processing_complete(self) -> bool:
        """Check if processing is complete (success or failure)"""
        return self.processing_status in [ProcessingStatus.COMPLETED, ProcessingStatus.FAILED]


class QualityMixin(BaseModel):
    """Mixin for models with quality assessment metrics"""
    
    extraction_quality: Optional[float] = Field(
        default=None,
        description="AI-assessed quality score (0.0-1.0)",
        ge=0.0,
        le=1.0,
        examples=[0.95, 0.78, 0.82]
    )
    
    extraction_confidence: Optional[float] = Field(
        default=None,
        description="Confidence in the quality assessment (0.0-1.0)",
        ge=0.0,
        le=1.0,
        examples=[0.92, 0.88, 0.75]
    )
    
    @computed_field
    @property
    def quality_grade(self) -> Optional[str]:
        """Convert quality score to letter grade"""
        if self.extraction_quality is None:
            return None
        
        if self.extraction_quality >= 0.9:
            return "A"
        elif self.extraction_quality >= 0.8:
            return "B"
        elif self.extraction_quality >= 0.7:
            return "C"
        elif self.extraction_quality >= 0.6:
            return "D"
        else:
            return "F"


class CostMixin(BaseModel):
    """Mixin for models with cost tracking"""
    
    processing_cost: Optional[float] = Field(
        default=0.0,
        description="Cost in USD for processing this item",
        ge=0.0,
        decimal_places=4,
        examples=[0.0025, 0.0158, 0.0003]
    )
    
    token_count: Optional[int] = Field(
        default=0,
        description="Number of tokens processed",
        ge=0,
        examples=[1250, 2400, 850]
    )
    
    @computed_field
    @property
    def cost_per_token(self) -> Optional[float]:
        """Calculate cost per token"""
        if self.token_count and self.token_count > 0 and self.processing_cost:
            return self.processing_cost / self.token_count
        return None


# ========================================
# COMMON COMPOSED MODELS
# ========================================

class BaseEntityModel(BrainBaseModel, UUIDMixin, TimestampMixin, UserScopedMixin):
    """Base model for entities with UUID, timestamps, and user scoping"""
    pass


class BaseProcessingModel(BaseEntityModel, ProcessingMixin, QualityMixin, CostMixin, MetadataMixin):
    """Base model for entities that undergo processing with quality and cost tracking"""
    pass


class BaseDeletableModel(BaseEntityModel, SoftDeleteMixin):
    """Base model for entities that can be soft deleted"""
    pass


# ========================================
# VALIDATION HELPERS
# ========================================

class ValidationResult(BrainBaseModel):
    """Result of a validation operation"""
    
    is_valid: bool = Field(
        description="Whether the validation passed"
    )
    
    errors: List[str] = Field(
        default_factory=list,
        description="List of validation error messages"
    )
    
    warnings: List[str] = Field(
        default_factory=list,
        description="List of validation warnings"
    )
    
    details: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional validation details"
    )
    
    @computed_field
    @property
    def has_errors(self) -> bool:
        """Check if there are validation errors"""
        return len(self.errors) > 0
    
    @computed_field
    @property
    def has_warnings(self) -> bool:
        """Check if there are validation warnings"""
        return len(self.warnings) > 0


# ========================================
# PAGINATION HELPERS
# ========================================

class PaginationParams(BrainBaseModel):
    """Parameters for paginated queries"""
    
    page: int = Field(
        default=1,
        description="Page number (1-based)",
        ge=1,
        examples=[1, 2, 5]
    )
    
    per_page: int = Field(
        default=20,
        description="Number of items per page",
        ge=1,
        le=100,
        examples=[10, 20, 50]
    )
    
    @computed_field
    @property
    def offset(self) -> int:
        """Calculate offset for database queries"""
        return (self.page - 1) * self.per_page
    
    @computed_field
    @property
    def limit(self) -> int:
        """Alias for per_page for database queries"""
        return self.per_page


class PaginatedResponse(BrainBaseModel):
    """Generic paginated response wrapper"""
    
    items: List[Any] = Field(
        description="List of items for this page"
    )
    
    total_count: int = Field(
        description="Total number of items across all pages",
        ge=0
    )
    
    page: int = Field(
        description="Current page number",
        ge=1
    )
    
    per_page: int = Field(
        description="Number of items per page",
        ge=1
    )
    
    @computed_field
    @property
    def total_pages(self) -> int:
        """Calculate total number of pages"""
        if self.per_page == 0:
            return 0
        return (self.total_count + self.per_page - 1) // self.per_page
    
    @computed_field
    @property
    def has_next(self) -> bool:
        """Check if there is a next page"""
        return self.page < self.total_pages
    
    @computed_field
    @property
    def has_prev(self) -> bool:
        """Check if there is a previous page"""
        return self.page > 1


# ========================================
# ERROR MODELS
# ========================================

class ErrorDetail(BrainBaseModel):
    """Detailed error information"""
    
    code: str = Field(
        description="Error code for programmatic handling"
    )
    
    message: str = Field(
        description="Human-readable error message"
    )
    
    field: Optional[str] = Field(
        default=None,
        description="Field name if error is field-specific"
    )
    
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional error context"
    )


class ApiError(BrainBaseModel):
    """Standardized API error response"""
    
    success: bool = Field(
        default=False,
        description="Always false for errors"
    )
    
    error: str = Field(
        description="High-level error message"
    )
    
    details: List[ErrorDetail] = Field(
        default_factory=list,
        description="Detailed error information"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the error occurred"
    )
    
    request_id: Optional[str] = Field(
        default=None,
        description="Request ID for tracking"
    )


# ========================================
# SUCCESS RESPONSE WRAPPER
# ========================================

class ApiResponse(BrainBaseModel):
    """Generic successful API response wrapper"""
    
    success: bool = Field(
        default=True,
        description="Always true for successful responses"
    )
    
    data: Any = Field(
        description="Response data"
    )
    
    message: Optional[str] = Field(
        default=None,
        description="Optional success message"
    )
    
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="Response timestamp"
    )


# ========================================
# UTILITY FUNCTIONS
# ========================================

def create_response_model(data_model: type, list_response: bool = False) -> type:
    """
    Create a standardized API response model for a given data model.
    
    Args:
        data_model: The Pydantic model for the response data
        list_response: Whether the response contains a list of items
        
    Returns:
        A new Pydantic model class for API responses
    """
    if list_response:
        data_field_type = List[data_model]
        model_name = f"{data_model.__name__}ListResponse"
    else:
        data_field_type = data_model
        model_name = f"{data_model.__name__}Response"
    
    return type(model_name, (ApiResponse,), {
        'data': Field(..., description=f"The {data_model.__name__.lower()} data")
    })


def create_paginated_response_model(data_model: type) -> type:
    """
    Create a paginated response model for a given data model.
    
    Args:
        data_model: The Pydantic model for the response items
        
    Returns:
        A new Pydantic model class for paginated API responses
    """
    model_name = f"{data_model.__name__}PaginatedResponse"
    
    class PaginatedModel(PaginatedResponse):
        items: List[data_model] = Field(
            description=f"List of {data_model.__name__.lower()} items for this page"
        )
    
    PaginatedModel.__name__ = model_name
    return PaginatedModel