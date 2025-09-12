"""
brAIn v2.0 Monitoring Models
Models for system health, LLM usage, cost tracking, and analytics.
"""

from datetime import datetime, date
from typing import Dict, List, Optional, Union, Any
from enum import Enum
from uuid import UUID

from pydantic import Field, computed_field, field_validator

from .base import (
    BrainBaseModel, 
    BaseEntityModel, 
    TimestampMixin,
    ServiceStatus
)
from validators.custom_validators import CostValidator, TokenValidator


# ========================================
# ENUMS FOR MONITORING
# ========================================

class OperationType(str, Enum):
    """Types of LLM operations"""
    EMBEDDING = "embedding"
    COMPLETION = "completion"
    EXTRACTION = "extraction"
    ANALYSIS = "analysis"
    CLASSIFICATION = "classification"
    SUMMARIZATION = "summarization"
    VALIDATION = "validation"


class AlertSeverity(str, Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


# ========================================
# LLM USAGE TRACKING
# ========================================

class LLMUsage(BaseEntityModel):
    """Detailed tracking of LLM operations with cost and performance metrics"""
    
    # Operation details
    operation_type: OperationType = Field(
        description="Type of LLM operation performed"
    )
    
    model_name: str = Field(
        description="Name of the model used",
        examples=["gpt-4-turbo-preview", "text-embedding-3-small", "claude-3-sonnet"]
    )
    
    provider: str = Field(
        default="openai",
        description="LLM provider",
        examples=["openai", "anthropic", "local"]
    )
    
    # Request/Response data
    input_text: Optional[str] = Field(
        default=None,
        description="Input text sent to the model"
    )
    
    output_text: Optional[str] = Field(
        default=None,
        description="Output text received from the model"
    )
    
    # Token usage
    input_tokens: int = Field(
        default=0,
        description="Number of input tokens",
        ge=0
    )
    
    output_tokens: int = Field(
        default=0,
        description="Number of output tokens", 
        ge=0
    )
    
    # Cost calculation
    input_cost_per_token: float = Field(
        default=0.0,
        description="Cost per input token in USD",
        ge=0.0,
        decimal_places=8
    )
    
    output_cost_per_token: float = Field(
        default=0.0,
        description="Cost per output token in USD",
        ge=0.0,
        decimal_places=8
    )
    
    # Performance metrics
    latency_ms: Optional[int] = Field(
        default=None,
        description="Response time in milliseconds",
        ge=0
    )
    
    started_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the operation started"
    )
    
    completed_at: Optional[datetime] = Field(
        default=None,
        description="When the operation completed"
    )
    
    # Context
    document_id: Optional[UUID] = Field(
        default=None,
        description="Associated document ID"
    )
    
    folder_id: Optional[UUID] = Field(
        default=None,
        description="Associated folder ID"
    )
    
    session_id: Optional[UUID] = Field(
        default=None,
        description="User session ID"
    )
    
    request_id: Optional[str] = Field(
        default=None,
        description="Provider's request ID",
        examples=["req_123456789", "anthropic_abc123"]
    )
    
    # Error handling
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if operation failed"
    )
    
    error_code: Optional[str] = Field(
        default=None,
        description="Error code from provider"
    )
    
    retry_count: int = Field(
        default=0,
        description="Number of retries attempted",
        ge=0
    )
    
    # Langfuse integration
    trace_id: Optional[str] = Field(
        default=None,
        description="Langfuse trace ID for debugging",
        examples=["trace_abc123def456"]
    )
    
    span_id: Optional[str] = Field(
        default=None,
        description="Langfuse span ID for debugging",
        examples=["span_def456ghi789"]
    )
    
    # Quality assessment
    response_quality_score: Optional[float] = Field(
        default=None,
        description="Assessed quality of the response (0.0-1.0)",
        ge=0.0,
        le=1.0
    )
    
    @computed_field
    @property
    def total_tokens(self) -> int:
        """Calculate total tokens (input + output)"""
        return self.input_tokens + self.output_tokens
    
    @computed_field
    @property
    def total_cost(self) -> float:
        """Calculate total cost in USD"""
        input_cost = self.input_tokens * self.input_cost_per_token
        output_cost = self.output_tokens * self.output_cost_per_token
        return round(input_cost + output_cost, 4)
    
    @computed_field
    @property
    def cost_per_token_avg(self) -> Optional[float]:
        """Calculate average cost per token"""
        if self.total_tokens > 0:
            return self.total_cost / self.total_tokens
        return None
    
    @computed_field
    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate operation duration in seconds"""
        if self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    @computed_field
    @property
    def tokens_per_second(self) -> Optional[float]:
        """Calculate tokens processed per second"""
        duration = self.duration_seconds
        if duration and duration > 0:
            return self.total_tokens / duration
        return None
    
    @computed_field
    @property
    def was_successful(self) -> bool:
        """Check if operation was successful"""
        return self.error_message is None
    
    @field_validator('input_tokens', 'output_tokens')
    @classmethod
    def validate_token_counts(cls, v: int) -> int:
        """Validate token counts"""
        return TokenValidator.validate_token_count(v)


# ========================================
# SYSTEM HEALTH MONITORING
# ========================================

class SystemHealth(BrainBaseModel, TimestampMixin):
    """System health monitoring for all services"""
    
    # Service identification
    service_name: str = Field(
        description="Name of the service being monitored",
        examples=["backend", "postgres", "redis", "websocket", "frontend"]
    )
    
    service_version: Optional[str] = Field(
        default=None,
        description="Version of the service",
        examples=["1.0.0", "2.1.3", "latest"]
    )
    
    host_name: str = Field(
        default="localhost",
        description="Hostname where the service is running"
    )
    
    # Health status
    status: ServiceStatus = Field(
        description="Current service status"
    )
    
    status_message: Optional[str] = Field(
        default=None,
        description="Detailed status message",
        max_length=500
    )
    
    # Performance metrics
    response_time_ms: Optional[int] = Field(
        default=None,
        description="Service response time in milliseconds",
        ge=0
    )
    
    cpu_percent: Optional[float] = Field(
        default=None,
        description="CPU usage percentage",
        ge=0.0,
        le=100.0
    )
    
    memory_mb: Optional[int] = Field(
        default=None,
        description="Memory usage in megabytes",
        ge=0
    )
    
    disk_usage_percent: Optional[float] = Field(
        default=None,
        description="Disk usage percentage",
        ge=0.0,
        le=100.0
    )
    
    # Network metrics
    active_connections: int = Field(
        default=0,
        description="Number of active connections",
        ge=0
    )
    
    requests_per_minute: int = Field(
        default=0,
        description="Requests processed per minute",
        ge=0
    )
    
    error_rate_percent: float = Field(
        default=0.0,
        description="Error rate as percentage",
        ge=0.0,
        le=100.0
    )
    
    # Database-specific metrics
    db_connections_active: Optional[int] = Field(
        default=None,
        description="Active database connections",
        ge=0
    )
    
    db_connections_max: Optional[int] = Field(
        default=None,
        description="Maximum database connections",
        ge=0
    )
    
    db_query_avg_time_ms: Optional[float] = Field(
        default=None,
        description="Average query time in milliseconds",
        ge=0.0
    )
    
    db_cache_hit_ratio: Optional[float] = Field(
        default=None,
        description="Database cache hit ratio (0.0-1.0)",
        ge=0.0,
        le=1.0
    )
    
    # Custom metrics
    custom_metrics: Dict[str, Union[str, int, float, bool]] = Field(
        default_factory=dict,
        description="Service-specific custom metrics"
    )
    
    # Health check details
    check_duration_ms: Optional[int] = Field(
        default=None,
        description="Time taken for health check in milliseconds",
        ge=0
    )
    
    last_error: Optional[str] = Field(
        default=None,
        description="Last error encountered",
        max_length=1000
    )
    
    consecutive_failures: int = Field(
        default=0,
        description="Number of consecutive failures",
        ge=0
    )
    
    measured_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When these metrics were measured"
    )
    
    @computed_field
    @property
    def is_healthy(self) -> bool:
        """Check if service is in a healthy state"""
        return self.status == ServiceStatus.HEALTHY
    
    @computed_field
    @property
    def needs_attention(self) -> bool:
        """Check if service needs attention"""
        return self.status in [ServiceStatus.WARNING, ServiceStatus.CRITICAL, ServiceStatus.DOWN]
    
    @computed_field
    @property
    def performance_score(self) -> float:
        """Calculate overall performance score (0-100)"""
        score = 100.0
        
        # Deduct for high response time
        if self.response_time_ms:
            if self.response_time_ms > 1000:  # 1 second
                score -= 20
            elif self.response_time_ms > 500:
                score -= 10
            elif self.response_time_ms > 200:
                score -= 5
        
        # Deduct for high CPU usage
        if self.cpu_percent:
            if self.cpu_percent > 90:
                score -= 15
            elif self.cpu_percent > 70:
                score -= 10
            elif self.cpu_percent > 50:
                score -= 5
        
        # Deduct for high error rate
        if self.error_rate_percent > 5:
            score -= 20
        elif self.error_rate_percent > 1:
            score -= 10
        
        # Deduct for consecutive failures
        if self.consecutive_failures > 0:
            score -= min(30, self.consecutive_failures * 5)
        
        return max(0.0, score)


# ========================================
# PROCESSING ANALYTICS
# ========================================

class ProcessingAnalytics(BrainBaseModel, TimestampMixin):
    """Hourly aggregated processing analytics"""
    
    user_id: Optional[UUID] = Field(
        default=None,
        description="User ID for user-specific analytics"
    )
    
    # Time aggregation
    date: date = Field(
        description="Date of the analytics"
    )
    
    hour: int = Field(
        description="Hour of the day (0-23)",
        ge=0,
        le=23
    )
    
    # Folder context
    folder_id: Optional[UUID] = Field(
        default=None,
        description="Folder ID for folder-specific analytics"
    )
    
    # Processing statistics
    files_processed: int = Field(
        default=0,
        description="Number of files successfully processed",
        ge=0
    )
    
    files_failed: int = Field(
        default=0,
        description="Number of files that failed processing",
        ge=0
    )
    
    files_skipped: int = Field(
        default=0,
        description="Number of files skipped",
        ge=0
    )
    
    # Performance metrics
    avg_processing_time_ms: float = Field(
        default=0.0,
        description="Average processing time in milliseconds",
        ge=0.0
    )
    
    min_processing_time_ms: int = Field(
        default=0,
        description="Minimum processing time in milliseconds",
        ge=0
    )
    
    max_processing_time_ms: int = Field(
        default=0,
        description="Maximum processing time in milliseconds",
        ge=0
    )
    
    total_processing_time_ms: int = Field(
        default=0,
        description="Total processing time in milliseconds",
        ge=0
    )
    
    # Size and content metrics
    total_file_size_bytes: int = Field(
        default=0,
        description="Total size of processed files in bytes",
        ge=0
    )
    
    total_text_extracted_chars: int = Field(
        default=0,
        description="Total characters extracted from text",
        ge=0
    )
    
    # Cost analytics
    total_cost: float = Field(
        default=0.0,
        description="Total processing cost in USD",
        ge=0.0,
        decimal_places=4
    )
    
    total_tokens: int = Field(
        default=0,
        description="Total tokens processed",
        ge=0
    )
    
    # Quality metrics
    avg_extraction_quality: float = Field(
        default=0.0,
        description="Average extraction quality score",
        ge=0.0,
        le=1.0
    )
    
    avg_confidence_score: float = Field(
        default=0.0,
        description="Average confidence score",
        ge=0.0,
        le=1.0
    )
    
    # Error analysis
    error_types: Dict[str, int] = Field(
        default_factory=dict,
        description="Count of different error types"
    )
    
    retry_count: int = Field(
        default=0,
        description="Total number of retries",
        ge=0
    )
    
    @computed_field
    @property
    def total_files(self) -> int:
        """Calculate total files processed (success + failed + skipped)"""
        return self.files_processed + self.files_failed + self.files_skipped
    
    @computed_field
    @property
    def success_rate(self) -> float:
        """Calculate processing success rate as percentage"""
        if self.total_files == 0:
            return 0.0
        return (self.files_processed / self.total_files) * 100
    
    @computed_field
    @property
    def avg_file_size_mb(self) -> float:
        """Calculate average file size in megabytes"""
        if self.files_processed == 0:
            return 0.0
        return (self.total_file_size_bytes / (1024 * 1024)) / self.files_processed
    
    @computed_field
    @property
    def avg_cost_per_document(self) -> float:
        """Calculate average cost per processed document"""
        if self.files_processed == 0:
            return 0.0
        return self.total_cost / self.files_processed
    
    @computed_field
    @property
    def throughput_files_per_minute(self) -> float:
        """Calculate processing throughput in files per minute"""
        if self.total_processing_time_ms == 0:
            return 0.0
        time_minutes = self.total_processing_time_ms / (1000 * 60)
        return self.files_processed / time_minutes
    
    @computed_field
    @property
    def efficiency_score(self) -> float:
        """Calculate processing efficiency score (0-100)"""
        score = 100.0
        
        # Factor in success rate
        score *= (self.success_rate / 100)
        
        # Factor in speed (penalize very slow processing)
        if self.avg_processing_time_ms > 30000:  # 30 seconds
            score *= 0.7
        elif self.avg_processing_time_ms > 10000:  # 10 seconds
            score *= 0.9
        
        # Factor in retry rate
        if self.total_files > 0:
            retry_rate = self.retry_count / self.total_files
            if retry_rate > 0.5:
                score *= 0.8
            elif retry_rate > 0.2:
                score *= 0.9
        
        return min(100.0, max(0.0, score))


# ========================================
# COST TRACKING
# ========================================

class DailyCostSummary(BrainBaseModel, TimestampMixin):
    """Daily cost summary with budget tracking"""
    
    user_id: UUID = Field(
        description="User ID for the cost summary"
    )
    
    date: date = Field(
        description="Date of the cost summary"
    )
    
    # Cost breakdown by operation type
    embedding_cost: float = Field(
        default=0.0,
        description="Cost for embedding operations in USD",
        ge=0.0,
        decimal_places=4
    )
    
    completion_cost: float = Field(
        default=0.0,
        description="Cost for completion operations in USD",
        ge=0.0,
        decimal_places=4
    )
    
    other_operations_cost: float = Field(
        default=0.0,
        description="Cost for other operations in USD",
        ge=0.0,
        decimal_places=4
    )
    
    # Token usage
    total_input_tokens: int = Field(
        default=0,
        description="Total input tokens used",
        ge=0
    )
    
    total_output_tokens: int = Field(
        default=0,
        description="Total output tokens used",
        ge=0
    )
    
    # Operation counts
    total_operations: int = Field(
        default=0,
        description="Total number of operations",
        ge=0
    )
    
    successful_operations: int = Field(
        default=0,
        description="Number of successful operations",
        ge=0
    )
    
    failed_operations: int = Field(
        default=0,
        description="Number of failed operations",
        ge=0
    )
    
    # Budget tracking
    budget_limit: Optional[float] = Field(
        default=None,
        description="Daily budget limit in USD",
        ge=0.0,
        decimal_places=2
    )
    
    # Performance metrics
    avg_latency_ms: float = Field(
        default=0.0,
        description="Average operation latency in milliseconds",
        ge=0.0
    )
    
    documents_processed: int = Field(
        default=0,
        description="Number of documents processed",
        ge=0
    )
    
    @computed_field
    @property
    def total_cost(self) -> float:
        """Calculate total cost for the day"""
        return self.embedding_cost + self.completion_cost + self.other_operations_cost
    
    @computed_field
    @property
    def total_tokens(self) -> int:
        """Calculate total tokens used"""
        return self.total_input_tokens + self.total_output_tokens
    
    @computed_field
    @property
    def success_rate(self) -> float:
        """Calculate operation success rate as percentage"""
        if self.total_operations == 0:
            return 0.0
        return (self.successful_operations / self.total_operations) * 100
    
    @computed_field
    @property
    def budget_remaining(self) -> Optional[float]:
        """Calculate remaining budget"""
        if self.budget_limit is None:
            return None
        return max(0.0, self.budget_limit - self.total_cost)
    
    @computed_field
    @property
    def budget_exceeded(self) -> bool:
        """Check if budget is exceeded"""
        if self.budget_limit is None:
            return False
        return self.total_cost > self.budget_limit
    
    @computed_field
    @property
    def budget_utilization(self) -> Optional[float]:
        """Calculate budget utilization as percentage"""
        if self.budget_limit is None or self.budget_limit == 0:
            return None
        return min(100.0, (self.total_cost / self.budget_limit) * 100)
    
    @computed_field
    @property
    def avg_cost_per_document(self) -> float:
        """Calculate average cost per document"""
        if self.documents_processed == 0:
            return 0.0
        return self.total_cost / self.documents_processed
    
    @computed_field
    @property
    def cost_efficiency_score(self) -> float:
        """Calculate cost efficiency score based on various factors"""
        score = 100.0
        
        # Factor in success rate (failed operations waste money)
        score *= (self.success_rate / 100)
        
        # Factor in budget utilization (penalize overspending)
        if self.budget_exceeded:
            score *= 0.5
        elif self.budget_utilization and self.budget_utilization > 90:
            score *= 0.8
        
        # Factor in cost per document (penalize expensive processing)
        if self.avg_cost_per_document > 0.10:  # $0.10 per document
            score *= 0.7
        elif self.avg_cost_per_document > 0.05:
            score *= 0.9
        
        return max(0.0, min(100.0, score))


# ========================================
# ALERT MANAGEMENT
# ========================================

class AlertRule(BaseEntityModel):
    """User-configurable alert rules"""
    
    rule_name: str = Field(
        description="Name of the alert rule",
        min_length=1,
        max_length=100,
        examples=["Daily Cost Limit", "High Error Rate", "Processing Delays"]
    )
    
    rule_type: str = Field(
        description="Type of alert rule",
        examples=["cost_threshold", "error_rate", "performance", "health"]
    )
    
    description: Optional[str] = Field(
        default=None,
        description="Description of the alert rule",
        max_length=500
    )
    
    # Alert conditions
    metric_name: str = Field(
        description="Name of the metric to monitor",
        examples=["daily_cost", "error_rate", "response_time", "success_rate"]
    )
    
    operator: str = Field(
        description="Comparison operator",
        examples=["greater_than", "less_than", "equals", "not_equals"]
    )
    
    threshold_value: float = Field(
        description="Threshold value for triggering the alert"
    )
    
    evaluation_window_minutes: int = Field(
        default=60,
        description="Time window for evaluation in minutes",
        ge=5,
        le=1440  # 24 hours
    )
    
    # Alert behavior
    enabled: bool = Field(
        default=True,
        description="Whether the alert rule is enabled"
    )
    
    severity: AlertSeverity = Field(
        default=AlertSeverity.WARNING,
        description="Severity level of the alert"
    )
    
    notification_channels: List[str] = Field(
        default_factory=lambda: ["email"],
        description="Channels for sending notifications",
        examples=[["email"], ["slack"], ["email", "webhook"]]
    )
    
    # Rate limiting
    cooldown_minutes: int = Field(
        default=60,
        description="Minimum time between alerts in minutes",
        ge=0
    )
    
    max_alerts_per_day: int = Field(
        default=10,
        description="Maximum alerts per day",
        ge=1,
        le=100
    )
    
    # Status tracking
    last_triggered_at: Optional[datetime] = Field(
        default=None,
        description="When the alert was last triggered"
    )
    
    alerts_sent_today: int = Field(
        default=0,
        description="Number of alerts sent today",
        ge=0
    )
    
    total_alerts_sent: int = Field(
        default=0,
        description="Total number of alerts sent",
        ge=0
    )
    
    # Additional metadata
    created_by: Optional[UUID] = Field(
        default=None,
        description="ID of the user who created the rule"
    )
    
    @computed_field
    @property
    def can_trigger(self) -> bool:
        """Check if alert can be triggered (not in cooldown)"""
        if not self.enabled:
            return False
        
        if self.alerts_sent_today >= self.max_alerts_per_day:
            return False
        
        if self.last_triggered_at:
            cooldown_ends = self.last_triggered_at + timedelta(minutes=self.cooldown_minutes)
            if datetime.utcnow() < cooldown_ends:
                return False
        
        return True
    
    @computed_field
    @property
    def next_trigger_available_at(self) -> Optional[datetime]:
        """Calculate when the alert can next be triggered"""
        if not self.last_triggered_at:
            return datetime.utcnow()
        
        return self.last_triggered_at + timedelta(minutes=self.cooldown_minutes)
    
    @field_validator('operator')
    @classmethod
    def validate_operator(cls, v: str) -> str:
        """Validate comparison operator"""
        allowed_operators = {'greater_than', 'less_than', 'equals', 'not_equals', 'greater_equal', 'less_equal'}
        if v not in allowed_operators:
            raise ValueError(f"Operator must be one of: {', '.join(allowed_operators)}")
        return v
    
    @field_validator('notification_channels')
    @classmethod
    def validate_channels(cls, v: List[str]) -> List[str]:
        """Validate notification channels"""
        allowed_channels = {'email', 'slack', 'webhook', 'sms'}
        for channel in v:
            if channel not in allowed_channels:
                raise ValueError(f"Invalid notification channel: {channel}")
        return v


class AlertHistory(BaseEntityModel):
    """Historical record of triggered alerts"""
    
    rule_id: UUID = Field(
        description="ID of the alert rule that triggered"
    )
    
    alert_message: str = Field(
        description="Alert message sent to users",
        max_length=1000
    )
    
    severity: AlertSeverity = Field(
        description="Severity level of the alert"
    )
    
    metric_value: Optional[float] = Field(
        default=None,
        description="Value of the metric that triggered the alert"
    )
    
    threshold_value: Optional[float] = Field(
        default=None,
        description="Threshold value that was exceeded"
    )
    
    # Notification details
    channels_notified: List[str] = Field(
        default_factory=list,
        description="Channels where notification was sent"
    )
    
    notification_status: Dict[str, Union[str, bool]] = Field(
        default_factory=dict,
        description="Status of notification per channel"
    )
    
    # Context
    context_data: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context data for the alert"
    )
    
    # Resolution
    acknowledged: bool = Field(
        default=False,
        description="Whether the alert has been acknowledged"
    )
    
    acknowledged_by: Optional[UUID] = Field(
        default=None,
        description="ID of the user who acknowledged the alert"
    )
    
    acknowledged_at: Optional[datetime] = Field(
        default=None,
        description="When the alert was acknowledged"
    )
    
    resolution_notes: Optional[str] = Field(
        default=None,
        description="Notes about alert resolution",
        max_length=1000
    )
    
    triggered_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the alert was triggered"
    )
    
    @computed_field
    @property
    def is_resolved(self) -> bool:
        """Check if alert is resolved/acknowledged"""
        return self.acknowledged
    
    @computed_field
    @property
    def time_to_acknowledgment(self) -> Optional[float]:
        """Calculate time to acknowledgment in hours"""
        if self.acknowledged_at:
            return (self.acknowledged_at - self.triggered_at).total_seconds() / 3600
        return None