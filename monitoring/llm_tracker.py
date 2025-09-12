"""
brAIn v2.0 LLM Operation Tracker
Comprehensive tracking system for all LLM operations with Langfuse integration.
"""

import time
import asyncio
import logging
import traceback
from typing import Optional, Dict, Any, Callable, Union, List
from functools import wraps
from datetime import datetime
from uuid import uuid4, UUID

from langfuse.decorators import observe, langfuse_context
from pydantic import BaseModel, Field

from .langfuse_client import get_langfuse_client, is_langfuse_enabled
from .cost_calculator import CostCalculator, TokenUsage
from ..models.monitoring import LLMUsage

logger = logging.getLogger(__name__)


class TraceMetadata(BaseModel):
    """Metadata for LLM operation traces"""
    
    operation_type: str = Field(
        description="Type of LLM operation",
        examples=["document_processing", "search", "chat", "embedding"]
    )
    
    user_id: Optional[UUID] = Field(
        default=None,
        description="User ID for operation tracking"
    )
    
    document_id: Optional[UUID] = Field(
        default=None,
        description="Document ID if processing a document"
    )
    
    model_name: str = Field(
        description="Name of the LLM model used",
        examples=["gpt-4-turbo", "text-embedding-3-small"]
    )
    
    session_id: Optional[str] = Field(
        default=None,
        description="Session ID for grouping related operations"
    )
    
    tags: List[str] = Field(
        default_factory=list,
        description="Tags for categorizing the operation"
    )
    
    custom_metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Custom metadata for the operation"
    )


class OperationResult(BaseModel):
    """Result of an LLM operation"""
    
    success: bool = Field(
        description="Whether the operation was successful"
    )
    
    output: Optional[str] = Field(
        default=None,
        description="Operation output (truncated for large outputs)"
    )
    
    token_usage: Optional[TokenUsage] = Field(
        default=None,
        description="Token usage statistics"
    )
    
    cost: Optional[float] = Field(
        default=None,
        description="Operation cost in USD"
    )
    
    duration_ms: Optional[float] = Field(
        default=None,
        description="Operation duration in milliseconds"
    )
    
    error_message: Optional[str] = Field(
        default=None,
        description="Error message if operation failed"
    )
    
    quality_score: Optional[float] = Field(
        default=None,
        description="Quality assessment of the operation result"
    )


class LLMTracker:
    """
    Centralized tracker for LLM operations with Langfuse integration.
    
    Features:
    - Automatic trace creation and management
    - Token counting and cost calculation
    - Performance metrics collection
    - Error tracking and debugging
    - Integration with local database
    """
    
    def __init__(self):
        self.cost_calculator = CostCalculator()
        self._session_operations: Dict[str, List[str]] = {}
    
    def create_trace(
        self,
        operation_name: str,
        metadata: TraceMetadata,
        input_data: Optional[str] = None
    ) -> Optional[Any]:
        """
        Create a new Langfuse trace for an LLM operation.
        
        Args:
            operation_name: Name of the operation
            metadata: Operation metadata
            input_data: Input data for the operation
            
        Returns:
            Langfuse trace object or None if disabled
        """
        if not is_langfuse_enabled():
            return None
        
        client = get_langfuse_client()
        if not client:
            return None
        
        try:
            trace = client.trace(
                name=operation_name,
                input=input_data,
                metadata={
                    "operation_type": metadata.operation_type,
                    "user_id": str(metadata.user_id) if metadata.user_id else None,
                    "document_id": str(metadata.document_id) if metadata.document_id else None,
                    "model_name": metadata.model_name,
                    "session_id": metadata.session_id,
                    "tags": metadata.tags,
                    "timestamp": datetime.utcnow().isoformat(),
                    **metadata.custom_metadata
                },
                tags=metadata.tags,
                user_id=str(metadata.user_id) if metadata.user_id else None,
                session_id=metadata.session_id
            )
            
            # Track session operations
            if metadata.session_id:
                if metadata.session_id not in self._session_operations:
                    self._session_operations[metadata.session_id] = []
                self._session_operations[metadata.session_id].append(trace.id)
            
            return trace
            
        except Exception as e:
            logger.error(f"Failed to create Langfuse trace: {e}")
            return None
    
    def create_generation(
        self,
        trace: Any,
        model_name: str,
        input_text: Optional[str] = None,
        prompt_name: Optional[str] = None,
        prompt_version: Optional[str] = None
    ) -> Optional[Any]:
        """
        Create a generation span within a trace.
        
        Args:
            trace: Parent trace object
            model_name: Name of the model used
            input_text: Input text for the generation
            prompt_name: Name of the prompt used
            prompt_version: Version of the prompt
            
        Returns:
            Generation object or None if disabled/failed
        """
        if not trace or not is_langfuse_enabled():
            return None
        
        try:
            generation = trace.generation(
                name=f"{model_name}-generation",
                model=model_name,
                input=input_text,
                prompt_name=prompt_name,
                prompt_version=prompt_version,
                start_time=datetime.utcnow()
            )
            
            return generation
            
        except Exception as e:
            logger.error(f"Failed to create Langfuse generation: {e}")
            return None
    
    def complete_operation(
        self,
        trace: Optional[Any],
        generation: Optional[Any],
        result: OperationResult,
        metadata: TraceMetadata
    ) -> None:
        """
        Complete an LLM operation by updating traces and storing analytics.
        
        Args:
            trace: Langfuse trace object
            generation: Langfuse generation object
            result: Operation result
            metadata: Operation metadata
        """
        try:
            # Update Langfuse objects if available
            if trace and is_langfuse_enabled():
                trace.update(
                    output=result.output,
                    metadata={
                        "success": result.success,
                        "duration_ms": result.duration_ms,
                        "cost": result.cost,
                        "token_usage": result.token_usage.model_dump() if result.token_usage else None,
                        "quality_score": result.quality_score,
                        "error": result.error_message
                    }
                )
            
            if generation and is_langfuse_enabled():
                generation.end(
                    output=result.output,
                    usage={
                        "input": result.token_usage.input_tokens if result.token_usage else 0,
                        "output": result.token_usage.output_tokens if result.token_usage else 0,
                        "total": result.token_usage.total_tokens if result.token_usage else 0
                    } if result.token_usage else None,
                    level="ERROR" if not result.success else "DEFAULT",
                    status_message=result.error_message if result.error_message else "Success"
                )
            
            # Store in local database for analytics
            self._store_local_analytics(metadata, result)
            
        except Exception as e:
            logger.error(f"Failed to complete operation tracking: {e}")
    
    def _store_local_analytics(
        self,
        metadata: TraceMetadata,
        result: OperationResult
    ) -> None:
        """
        Store operation analytics in local database.
        
        Args:
            metadata: Operation metadata
            result: Operation result
        """
        try:
            # Create LLMUsage record for local storage
            usage_record = LLMUsage(
                user_id=metadata.user_id,
                operation_type=metadata.operation_type,
                model_name=metadata.model_name,
                input_tokens=result.token_usage.input_tokens if result.token_usage else 0,
                output_tokens=result.token_usage.output_tokens if result.token_usage else 0,
                total_tokens=result.token_usage.total_tokens if result.token_usage else 0,
                cost=result.cost or 0.0,
                processing_duration_ms=int(result.duration_ms) if result.duration_ms else 0,
                success=result.success,
                error_message=result.error_message,
                request_metadata={
                    "session_id": metadata.session_id,
                    "document_id": str(metadata.document_id) if metadata.document_id else None,
                    "tags": metadata.tags,
                    **metadata.custom_metadata
                },
                response_metadata={
                    "quality_score": result.quality_score,
                    "output_length": len(result.output) if result.output else 0
                }
            )
            
            # TODO: Store in database (will be implemented with database integration)
            logger.debug(f"Local analytics stored for operation: {metadata.operation_type}")
            
        except Exception as e:
            logger.error(f"Failed to store local analytics: {e}")
    
    def add_user_feedback(
        self,
        trace_id: str,
        score: float,
        comment: Optional[str] = None
    ) -> bool:
        """
        Add user feedback to a trace.
        
        Args:
            trace_id: Langfuse trace ID
            score: Feedback score (0.0-1.0)
            comment: Optional feedback comment
            
        Returns:
            True if feedback was added successfully
        """
        if not is_langfuse_enabled():
            return False
        
        client = get_langfuse_client()
        if not client:
            return False
        
        try:
            client.score(
                trace_id=trace_id,
                name="user-feedback",
                value=score,
                comment=comment
            )
            
            client.flush()
            return True
            
        except Exception as e:
            logger.error(f"Failed to add user feedback: {e}")
            return False
    
    def get_session_summary(
        self,
        session_id: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get summary of operations for a session.
        
        Args:
            session_id: Session ID to summarize
            
        Returns:
            Session summary or None if not found
        """
        if session_id not in self._session_operations:
            return None
        
        operation_ids = self._session_operations[session_id]
        
        return {
            "session_id": session_id,
            "operation_count": len(operation_ids),
            "operation_ids": operation_ids,
            "created_at": datetime.utcnow().isoformat()
        }


# Global tracker instance
_tracker = LLMTracker()


def track_llm_operation(
    operation_type: str,
    model_name: str,
    user_id: Optional[UUID] = None,
    document_id: Optional[UUID] = None,
    session_id: Optional[str] = None,
    tags: Optional[List[str]] = None,
    **custom_metadata
):
    """
    Decorator for tracking LLM operations with Langfuse.
    
    Args:
        operation_type: Type of operation being tracked
        model_name: Name of the LLM model
        user_id: User ID for the operation
        document_id: Document ID if applicable
        session_id: Session ID for grouping operations
        tags: Tags for categorization
        **custom_metadata: Additional custom metadata
    """
    def decorator(func: Callable) -> Callable:
        if asyncio.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs) -> Any:
                return await _execute_tracked_operation(
                    func, args, kwargs, operation_type, model_name,
                    user_id, document_id, session_id, tags or [], custom_metadata
                )
            return async_wrapper
        else:
            @wraps(func)
            def sync_wrapper(*args, **kwargs) -> Any:
                return asyncio.run(_execute_tracked_operation(
                    func, args, kwargs, operation_type, model_name,
                    user_id, document_id, session_id, tags or [], custom_metadata
                ))
            return sync_wrapper
    
    return decorator


async def _execute_tracked_operation(
    func: Callable,
    args: tuple,
    kwargs: dict,
    operation_type: str,
    model_name: str,
    user_id: Optional[UUID],
    document_id: Optional[UUID],
    session_id: Optional[str],
    tags: List[str],
    custom_metadata: Dict[str, Any]
) -> Any:
    """Execute a function with LLM operation tracking"""
    
    # Extract input from function arguments
    input_text = kwargs.get('prompt') or kwargs.get('text') or kwargs.get('input')
    if isinstance(input_text, list):
        input_text = str(input_text)
    elif input_text and len(str(input_text)) > 2000:
        input_text = str(input_text)[:2000] + "..."
    
    # Create metadata
    metadata = TraceMetadata(
        operation_type=operation_type,
        user_id=user_id,
        document_id=document_id,
        model_name=model_name,
        session_id=session_id,
        tags=tags,
        custom_metadata=custom_metadata
    )
    
    # Create trace
    trace = _tracker.create_trace(
        operation_name=f"{operation_type}-{model_name}",
        metadata=metadata,
        input_data=input_text
    )
    
    # Create generation
    generation = _tracker.create_generation(
        trace=trace,
        model_name=model_name,
        input_text=input_text
    )
    
    start_time = time.time()
    result = None
    error = None
    
    try:
        # Execute the function
        if asyncio.iscoroutinefunction(func):
            result = await func(*args, **kwargs)
        else:
            result = func(*args, **kwargs)
        
        success = True
        
    except Exception as e:
        success = False
        error = str(e)
        logger.error(f"LLM operation failed: {e}\n{traceback.format_exc()}")
        raise
    
    finally:
        end_time = time.time()
        duration_ms = (end_time - start_time) * 1000
        
        # Calculate token usage and cost
        token_usage = None
        cost = None
        
        try:
            if result and hasattr(result, 'usage'):
                # OpenAI-style usage
                usage = result.usage
                token_usage = TokenUsage(
                    input_tokens=getattr(usage, 'prompt_tokens', 0),
                    output_tokens=getattr(usage, 'completion_tokens', 0)
                )
            elif result and isinstance(result, dict) and 'usage' in result:
                # Dictionary response with usage
                usage = result['usage']
                token_usage = TokenUsage(
                    input_tokens=usage.get('prompt_tokens', 0),
                    output_tokens=usage.get('completion_tokens', 0)
                )
            elif input_text:
                # Estimate tokens from input/output
                token_usage = _tracker.cost_calculator.estimate_tokens(
                    input_text=input_text,
                    output_text=str(result) if result else None
                )
            
            if token_usage:
                cost = _tracker.cost_calculator.calculate_cost(
                    model_name=model_name,
                    token_usage=token_usage
                )
                
        except Exception as e:
            logger.warning(f"Failed to calculate token usage/cost: {e}")
        
        # Create operation result
        operation_result = OperationResult(
            success=success,
            output=str(result)[:1000] if result else None,  # Truncate large outputs
            token_usage=token_usage,
            cost=cost,
            duration_ms=duration_ms,
            error_message=error
        )
        
        # Complete the operation tracking
        _tracker.complete_operation(trace, generation, operation_result, metadata)
    
    return result


# Convenience functions for common operations
def track_embedding_generation(
    user_id: Optional[UUID] = None,
    document_id: Optional[UUID] = None,
    session_id: Optional[str] = None
):
    """Decorator for tracking embedding generation operations"""
    return track_llm_operation(
        operation_type="embedding_generation",
        model_name="text-embedding-3-small",
        user_id=user_id,
        document_id=document_id,
        session_id=session_id,
        tags=["embedding", "vector"]
    )


def track_document_processing(
    user_id: Optional[UUID] = None,
    document_id: Optional[UUID] = None,
    processing_type: str = "extraction"
):
    """Decorator for tracking document processing operations"""
    return track_llm_operation(
        operation_type="document_processing",
        model_name="gpt-4-turbo",
        user_id=user_id,
        document_id=document_id,
        tags=["document", "processing", processing_type]
    )


def track_search_operation(
    user_id: Optional[UUID] = None,
    session_id: Optional[str] = None,
    search_type: str = "semantic"
):
    """Decorator for tracking search operations"""
    return track_llm_operation(
        operation_type="search",
        model_name="text-embedding-3-small",
        user_id=user_id,
        session_id=session_id,
        tags=["search", search_type]
    )


def track_ai_configuration(
    user_id: Optional[UUID] = None,
    session_id: Optional[str] = None
):
    """Decorator for tracking AI configuration assistant operations"""
    return track_llm_operation(
        operation_type="ai_configuration",
        model_name="claude-3-sonnet-20240229",
        user_id=user_id,
        session_id=session_id,
        tags=["configuration", "assistant"]
    )


# Direct tracking functions
def add_user_feedback(trace_id: str, score: float, comment: Optional[str] = None) -> bool:
    """Add user feedback to a specific trace"""
    return _tracker.add_user_feedback(trace_id, score, comment)


def get_session_summary(session_id: str) -> Optional[Dict[str, Any]]:
    """Get summary of operations for a session"""
    return _tracker.get_session_summary(session_id)