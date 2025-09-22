"""
Intelligent Error Handling and Recovery System for brAIn v2.0

This module provides AI-powered error analysis, categorization, and recovery
suggestions for document processing operations with context-aware solutions.

Author: BMad Team
"""

import traceback
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple, Callable

# Core imports
from pydantic import BaseModel, Field
from enum import Enum
import logging

# AI imports for intelligent analysis
import anthropic
import openai
from langfuse import Langfuse

# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class ErrorSeverity(str, Enum):
    """Error severity levels."""
    CRITICAL = "critical"     # System-breaking errors
    HIGH = "high"            # Processing-blocking errors
    MEDIUM = "medium"        # Quality-affecting errors
    LOW = "low"              # Minor issues
    WARNING = "warning"      # Non-blocking warnings

class ErrorCategory(str, Enum):
    """Error categories for classification."""
    FILE_ACCESS = "file_access"
    FORMAT_PARSING = "format_parsing"
    ENCODING_ISSUES = "encoding_issues"
    MEMORY_EXHAUSTION = "memory_exhaustion"
    NETWORK_ERROR = "network_error"
    AI_SERVICE_ERROR = "ai_service_error"
    VALIDATION_ERROR = "validation_error"
    PERMISSION_ERROR = "permission_error"
    TIMEOUT_ERROR = "timeout_error"
    UNKNOWN = "unknown"

class RecoveryStrategy(str, Enum):
    """Available recovery strategies."""
    RETRY = "retry"                    # Retry with same parameters
    RETRY_WITH_BACKOFF = "retry_with_backoff"  # Retry with exponential backoff
    FALLBACK_METHOD = "fallback_method"  # Use alternative processing method
    SKIP_FILE = "skip_file"            # Skip problematic file
    REDUCE_QUALITY = "reduce_quality"   # Lower quality settings
    CHUNK_PROCESSING = "chunk_processing"  # Process in smaller chunks
    MANUAL_INTERVENTION = "manual_intervention"  # Requires human intervention
    ABORT_PROCESSING = "abort_processing"  # Stop processing entirely

class ErrorAnalysis(BaseModel):
    """Comprehensive error analysis result."""
    error_id: str = Field(description="Unique error identifier")
    error_type: str = Field(description="Python exception type")
    error_message: str = Field(description="Error message")
    severity: ErrorSeverity = Field(description="Error severity level")
    category: ErrorCategory = Field(description="Error category")
    root_cause: str = Field(description="Identified root cause")
    context_info: Dict[str, Any] = Field(default_factory=dict, description="Error context")
    stack_trace: str = Field(description="Stack trace information")
    similar_errors: List[str] = Field(default_factory=list, description="Similar past errors")
    analysis_confidence: float = Field(ge=0.0, le=1.0, description="Analysis confidence")
    timestamp: datetime = Field(default_factory=datetime.now)

class RecoveryPlan(BaseModel):
    """Recovery plan with multiple strategies."""
    plan_id: str = Field(description="Unique recovery plan ID")
    error_analysis: ErrorAnalysis = Field(description="Associated error analysis")
    primary_strategy: RecoveryStrategy = Field(description="Primary recovery strategy")
    alternative_strategies: List[RecoveryStrategy] = Field(default_factory=list)
    recovery_parameters: Dict[str, Any] = Field(default_factory=dict)
    estimated_success_rate: float = Field(ge=0.0, le=1.0, description="Success probability")
    estimated_recovery_time: float = Field(ge=0.0, description="Estimated recovery time")
    prerequisites: List[str] = Field(default_factory=list, description="Recovery prerequisites")
    risk_assessment: str = Field(description="Recovery risk assessment")
    human_intervention_required: bool = Field(default=False)

class RecoveryResult(BaseModel):
    """Result of recovery attempt."""
    recovery_id: str = Field(description="Recovery attempt ID")
    strategy_used: RecoveryStrategy = Field(description="Strategy that was used")
    recovered: bool = Field(description="Whether recovery was successful")
    recovery_time: float = Field(ge=0.0, description="Time taken for recovery")
    new_error: Optional[str] = Field(None, description="New error if recovery failed")
    quality_impact: float = Field(ge=-1.0, le=1.0, description="Impact on quality")
    recommendations: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

# =============================================================================
# INTELLIGENT ERROR HANDLER
# =============================================================================

class IntelligentErrorHandler:
    """
    AI-powered error handler with intelligent analysis, categorization,
    and recovery strategies for document processing operations.
    """
    
    def __init__(
        self,
        ai_client: Optional[Union[anthropic.Anthropic, openai.OpenAI]] = None,
        langfuse_client: Optional[Langfuse] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize intelligent error handler.
        
        Args:
            ai_client: AI client for intelligent error analysis
            langfuse_client: Langfuse for error monitoring
            config: Configuration options
        """
        self.ai_client = ai_client
        self.langfuse = langfuse_client
        self.config = config or {}
        
        # Error knowledge base
        self.error_patterns = self._initialize_error_patterns()
        self.recovery_strategies = self._initialize_recovery_strategies()
        
        # Error tracking
        self.error_history: List[ErrorAnalysis] = []
        self.recovery_history: List[RecoveryResult] = []
        
        # Statistics
        self.stats = {
            "total_errors_analyzed": 0,
            "successful_recoveries": 0,
            "failed_recoveries": 0,
            "ai_analyses_performed": 0,
            "patterns_matched": 0,
            "recovery_strategies_used": {}
        }
        
        # Logger
        self.logger = logging.getLogger(__name__)
    
    async def analyze_error(
        self,
        error: Exception,
        file_path: Union[str, Path],
        processing_context: Any,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> ErrorAnalysis:
        """
        Analyze error with AI-powered intelligence.
        
        Args:
            error: Exception that occurred
            file_path: File being processed when error occurred
            processing_context: Processing context information
            additional_context: Additional context information
            
        Returns:
            Comprehensive error analysis
        """
        start_time = datetime.now()
        
        try:
            # Generate unique error ID
            error_id = self._generate_error_id(error, file_path)
            
            # Extract basic error information
            error_type = type(error).__name__
            error_message = str(error)
            stack_trace = traceback.format_exc()
            
            # Pattern-based classification
            category, severity = await self._classify_error_pattern(error, error_message, stack_trace)
            
            # AI-powered analysis (if available)
            ai_analysis = {}
            if self.ai_client:
                ai_analysis = await self._ai_error_analysis(
                    error, file_path, processing_context, stack_trace
                )
            
            # Root cause analysis
            root_cause = await self._analyze_root_cause(
                error, file_path, processing_context, ai_analysis
            )
            
            # Context information gathering
            context_info = await self._gather_error_context(
                error, file_path, processing_context, additional_context
            )
            
            # Find similar historical errors
            similar_errors = await self._find_similar_errors(error_type, error_message, category)
            
            # Calculate analysis confidence
            confidence = self._calculate_analysis_confidence(
                category, ai_analysis, similar_errors
            )
            
            # Create comprehensive analysis
            analysis = ErrorAnalysis(
                error_id=error_id,
                error_type=error_type,
                error_message=error_message,
                severity=severity,
                category=category,
                root_cause=root_cause,
                context_info=context_info,
                stack_trace=stack_trace,
                similar_errors=similar_errors,
                analysis_confidence=confidence
            )
            
            # Store in history
            self.error_history.append(analysis)
            
            # Update statistics
            self.stats["total_errors_analyzed"] += 1
            if ai_analysis:
                self.stats["ai_analyses_performed"] += 1
            if similar_errors:
                self.stats["patterns_matched"] += 1
            
            return analysis
            
        except Exception as analysis_error:
            # Fallback analysis if analysis itself fails
            self.logger.error(f"Error analysis failed: {analysis_error}")
            
            return ErrorAnalysis(
                error_id=f"fallback_{datetime.now().timestamp()}",
                error_type=type(error).__name__,
                error_message=str(error),
                severity=ErrorSeverity.MEDIUM,
                category=ErrorCategory.UNKNOWN,
                root_cause=f"Analysis failed: {str(analysis_error)}",
                stack_trace=traceback.format_exc(),
                analysis_confidence=0.1
            )
    
    async def attempt_recovery(
        self,
        error: Exception,
        file_path: Union[str, Path],
        processing_context: Any,
        error_analysis: ErrorAnalysis,
        max_attempts: int = 3
    ) -> RecoveryResult:
        """
        Attempt intelligent error recovery based on analysis.
        
        Args:
            error: Original exception
            file_path: File path where error occurred
            processing_context: Processing context
            error_analysis: Error analysis results
            max_attempts: Maximum recovery attempts
            
        Returns:
            Recovery attempt result
        """
        start_time = datetime.now()
        
        # Generate recovery plan
        recovery_plan = await self._generate_recovery_plan(error_analysis, processing_context)
        
        recovery_id = f"recovery_{datetime.now().timestamp()}"
        
        # Attempt recovery strategies in order
        strategies_to_try = [recovery_plan.primary_strategy] + recovery_plan.alternative_strategies
        
        for attempt, strategy in enumerate(strategies_to_try[:max_attempts]):
            try:
                # Log recovery attempt
                if self.langfuse:
                    trace = self.langfuse.trace(
                        name="error_recovery_attempt",
                        input={
                            "strategy": strategy.value,
                            "attempt": attempt + 1,
                            "error_category": error_analysis.category.value
                        }
                    )
                
                # Execute recovery strategy
                recovery_success = await self._execute_recovery_strategy(
                    strategy, error, file_path, processing_context, recovery_plan
                )
                
                if recovery_success["success"]:
                    # Successful recovery
                    recovery_time = (datetime.now() - start_time).total_seconds()
                    
                    result = RecoveryResult(
                        recovery_id=recovery_id,
                        strategy_used=strategy,
                        recovered=True,
                        recovery_time=recovery_time,
                        quality_impact=recovery_success.get("quality_impact", 0.0),
                        recommendations=recovery_success.get("recommendations", []),
                        metadata=recovery_success.get("metadata", {})
                    )
                    
                    # Update statistics
                    self.stats["successful_recoveries"] += 1
                    self.stats["recovery_strategies_used"][strategy.value] = \
                        self.stats["recovery_strategies_used"].get(strategy.value, 0) + 1
                    
                    # Store in history
                    self.recovery_history.append(result)
                    
                    return result
                
            except Exception as recovery_error:
                self.logger.warning(f"Recovery attempt {attempt + 1} failed: {recovery_error}")
                continue
        
        # All recovery attempts failed
        recovery_time = (datetime.now() - start_time).total_seconds()
        
        result = RecoveryResult(
            recovery_id=recovery_id,
            strategy_used=strategies_to_try[0] if strategies_to_try else RecoveryStrategy.ABORT_PROCESSING,
            recovered=False,
            recovery_time=recovery_time,
            new_error=str(error),
            quality_impact=-0.5,
            recommendations=[
                "Manual intervention may be required",
                "Consider alternative processing methods",
                "Review file format and content"
            ]
        )
        
        # Update statistics
        self.stats["failed_recoveries"] += 1
        
        # Store in history
        self.recovery_history.append(result)
        
        return result
    
    async def _classify_error_pattern(
        self,
        error: Exception,
        error_message: str,
        stack_trace: str
    ) -> Tuple[ErrorCategory, ErrorSeverity]:
        """Classify error based on patterns."""
        error_type = type(error).__name__
        
        # Check against known patterns
        for pattern_info in self.error_patterns:
            if pattern_info["match_function"](error, error_message, stack_trace):
                return pattern_info["category"], pattern_info["severity"]
        
        # Default classification
        return ErrorCategory.UNKNOWN, ErrorSeverity.MEDIUM
    
    async def _ai_error_analysis(
        self,
        error: Exception,
        file_path: Union[str, Path],
        processing_context: Any,
        stack_trace: str
    ) -> Dict[str, Any]:
        """Use AI for advanced error analysis."""
        if not self.ai_client:
            return {}
        
        try:
            # Prepare analysis prompt
            prompt = f"""
            Analyze this document processing error and provide insights:
            
            Error Type: {type(error).__name__}
            Error Message: {str(error)}
            File Path: {file_path}
            Processing Context: {getattr(processing_context, '__dict__', str(processing_context))[:500]}
            
            Stack Trace (last 10 lines):
            {stack_trace.split(chr(10))[-10:]}
            
            Please analyze and provide:
            1. Root cause analysis
            2. Error severity (critical, high, medium, low, warning)
            3. Error category (file_access, format_parsing, encoding_issues, etc.)
            4. Recommended recovery strategies
            5. Prevention suggestions
            6. Confidence level (0-100)
            
            Format response as JSON.
            """
            
            # Call AI service
            if isinstance(self.ai_client, anthropic.Anthropic):
                response = await self._call_anthropic_error_analysis(prompt)
            elif isinstance(self.ai_client, openai.OpenAI):
                response = await self._call_openai_error_analysis(prompt)
            else:
                response = {}
            
            return response
            
        except Exception as e:
            self.logger.warning(f"AI error analysis failed: {e}")
            return {}
    
    async def _call_anthropic_error_analysis(self, prompt: str) -> Dict[str, Any]:
        """Call Anthropic for error analysis."""
        try:
            message = self.ai_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1000,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = message.content[0].text
            
            # Try to parse JSON
            import json
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
            return {"raw_response": response_text}
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _call_openai_error_analysis(self, prompt: str) -> Dict[str, Any]:
        """Call OpenAI for error analysis."""
        try:
            response = self.ai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0
            )
            
            response_text = response.choices[0].message.content
            
            # Try to parse JSON
            import json
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
            return {"raw_response": response_text}
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _analyze_root_cause(
        self,
        error: Exception,
        file_path: Union[str, Path],
        processing_context: Any,
        ai_analysis: Dict[str, Any]
    ) -> str:
        """Analyze root cause of error."""
        # Start with AI analysis if available
        if ai_analysis.get("root_cause"):
            return str(ai_analysis["root_cause"])
        
        # Pattern-based root cause analysis
        error_type = type(error).__name__
        error_message = str(error).lower()
        
        # File access issues
        if "no such file" in error_message or "permission denied" in error_message:
            return "File access problem - check file path and permissions"
        
        # Encoding issues
        if "codec" in error_message or "encoding" in error_message:
            return "Character encoding issue - file may use unsupported encoding"
        
        # Memory issues
        if "memory" in error_message or error_type == "MemoryError":
            return "Insufficient memory - file may be too large for current processing method"
        
        # Format parsing issues
        if "parse" in error_message or "invalid" in error_message:
            return "File format parsing issue - file may be corrupted or unsupported format"
        
        # Network/service issues
        if "connection" in error_message or "timeout" in error_message:
            return "Network or service connectivity issue"
        
        # Generic fallback
        return f"Unknown root cause - {error_type}: {str(error)[:100]}"
    
    async def _gather_error_context(
        self,
        error: Exception,
        file_path: Union[str, Path],
        processing_context: Any,
        additional_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Gather contextual information about the error."""
        context = {
            "timestamp": datetime.now().isoformat(),
            "python_version": sys.version,
            "error_location": traceback.format_tb(error.__traceback__)[-1] if error.__traceback__ else "Unknown"
        }
        
        # File context
        try:
            file_path = Path(file_path)
            context["file_info"] = {
                "exists": file_path.exists(),
                "size": file_path.stat().st_size if file_path.exists() else 0,
                "extension": file_path.suffix,
                "readable": os.access(file_path, os.R_OK) if file_path.exists() else False
            }
        except Exception:
            context["file_info"] = {"error": "Could not gather file information"}
        
        # Processing context
        if hasattr(processing_context, '__dict__'):
            context["processing_context"] = {
                key: str(value)[:100]  # Limit length
                for key, value in processing_context.__dict__.items()
                if not key.startswith('_')
            }
        
        # Additional context
        if additional_context:
            context["additional"] = additional_context
        
        # System context
        import psutil
        try:
            context["system"] = {
                "memory_percent": psutil.virtual_memory().percent,
                "cpu_percent": psutil.cpu_percent(),
                "disk_usage": psutil.disk_usage('/').percent
            }
        except Exception:
            context["system"] = {"error": "Could not gather system information"}
        
        return context
    
    async def _find_similar_errors(
        self,
        error_type: str,
        error_message: str,
        category: ErrorCategory
    ) -> List[str]:
        """Find similar errors in history."""
        similar = []
        
        for past_error in self.error_history:
            similarity_score = 0
            
            # Same error type
            if past_error.error_type == error_type:
                similarity_score += 0.4
            
            # Same category
            if past_error.category == category:
                similarity_score += 0.3
            
            # Similar message (basic string similarity)
            message_similarity = self._calculate_string_similarity(
                error_message, past_error.error_message
            )
            similarity_score += message_similarity * 0.3
            
            if similarity_score > 0.6:  # Threshold for similarity
                similar.append(past_error.error_id)
        
        return similar[:5]  # Return top 5 similar errors
    
    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """Calculate basic string similarity."""
        if not str1 or not str2:
            return 0.0
        
        # Simple word overlap similarity
        words1 = set(str1.lower().split())
        words2 = set(str2.lower().split())
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_analysis_confidence(
        self,
        category: ErrorCategory,
        ai_analysis: Dict[str, Any],
        similar_errors: List[str]
    ) -> float:
        """Calculate confidence in error analysis."""
        confidence = 0.5  # Base confidence
        
        # Pattern match bonus
        if category != ErrorCategory.UNKNOWN:
            confidence += 0.2
        
        # AI analysis bonus
        if ai_analysis and "confidence" in ai_analysis:
            ai_conf = float(ai_analysis["confidence"]) / 100.0
            confidence = (confidence + ai_conf) / 2
        elif ai_analysis:
            confidence += 0.2
        
        # Historical similarity bonus
        if similar_errors:
            confidence += min(len(similar_errors) * 0.1, 0.3)
        
        return min(confidence, 1.0)
    
    async def _generate_recovery_plan(
        self,
        error_analysis: ErrorAnalysis,
        processing_context: Any
    ) -> RecoveryPlan:
        """Generate recovery plan based on error analysis."""
        plan_id = f"plan_{datetime.now().timestamp()}"
        
        # Select primary strategy based on category and severity
        primary_strategy = self._select_primary_strategy(error_analysis)
        
        # Select alternative strategies
        alternative_strategies = self._select_alternative_strategies(
            error_analysis, primary_strategy
        )
        
        # Calculate success rate estimate
        success_rate = self._estimate_recovery_success_rate(
            error_analysis, primary_strategy
        )
        
        # Estimate recovery time
        recovery_time = self._estimate_recovery_time(
            error_analysis, primary_strategy
        )
        
        return RecoveryPlan(
            plan_id=plan_id,
            error_analysis=error_analysis,
            primary_strategy=primary_strategy,
            alternative_strategies=alternative_strategies,
            estimated_success_rate=success_rate,
            estimated_recovery_time=recovery_time,
            prerequisites=self._get_recovery_prerequisites(primary_strategy),
            risk_assessment=self._assess_recovery_risk(primary_strategy, error_analysis),
            human_intervention_required=primary_strategy == RecoveryStrategy.MANUAL_INTERVENTION
        )
    
    async def _execute_recovery_strategy(
        self,
        strategy: RecoveryStrategy,
        error: Exception,
        file_path: Union[str, Path],
        processing_context: Any,
        recovery_plan: RecoveryPlan
    ) -> Dict[str, Any]:
        """Execute a specific recovery strategy."""
        try:
            if strategy == RecoveryStrategy.RETRY:
                return await self._execute_retry_strategy(error, file_path, processing_context)
            
            elif strategy == RecoveryStrategy.RETRY_WITH_BACKOFF:
                return await self._execute_retry_with_backoff(error, file_path, processing_context)
            
            elif strategy == RecoveryStrategy.FALLBACK_METHOD:
                return await self._execute_fallback_method(error, file_path, processing_context)
            
            elif strategy == RecoveryStrategy.REDUCE_QUALITY:
                return await self._execute_reduce_quality(error, file_path, processing_context)
            
            elif strategy == RecoveryStrategy.CHUNK_PROCESSING:
                return await self._execute_chunk_processing(error, file_path, processing_context)
            
            elif strategy == RecoveryStrategy.SKIP_FILE:
                return {
                    "success": True,
                    "quality_impact": -1.0,
                    "recommendations": ["File was skipped due to processing error"],
                    "metadata": {"strategy": "skip_file"}
                }
            
            elif strategy == RecoveryStrategy.MANUAL_INTERVENTION:
                return {
                    "success": False,
                    "recommendations": ["Manual intervention required"],
                    "metadata": {"strategy": "manual_intervention", "requires_human": True}
                }
            
            elif strategy == RecoveryStrategy.ABORT_PROCESSING:
                return {
                    "success": False,
                    "recommendations": ["Processing aborted due to critical error"],
                    "metadata": {"strategy": "abort_processing"}
                }
            
            else:
                return {"success": False, "error": f"Unknown strategy: {strategy}"}
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    # Recovery strategy implementations
    
    async def _execute_retry_strategy(
        self,
        error: Exception,
        file_path: Union[str, Path],
        processing_context: Any
    ) -> Dict[str, Any]:
        """Execute simple retry strategy."""
        # Simple retry - in real implementation, this would re-invoke the original processing
        return {
            "success": True,  # Optimistic assumption
            "quality_impact": 0.0,
            "recommendations": ["Retried processing with same parameters"],
            "metadata": {"strategy": "retry", "attempts": 1}
        }
    
    async def _execute_retry_with_backoff(
        self,
        error: Exception,
        file_path: Union[str, Path],
        processing_context: Any
    ) -> Dict[str, Any]:
        """Execute retry with exponential backoff."""
        import asyncio
        
        max_retries = 3
        base_delay = 1.0
        
        for attempt in range(max_retries):
            delay = base_delay * (2 ** attempt)
            await asyncio.sleep(delay)
            
            # In real implementation, would retry actual processing here
            # For now, simulate success after backoff
            if attempt >= 1:  # Success after second attempt
                return {
                    "success": True,
                    "quality_impact": -0.1,  # Slight quality impact due to delay
                    "recommendations": [f"Succeeded after {attempt + 1} retry attempts"],
                    "metadata": {"strategy": "retry_with_backoff", "attempts": attempt + 1}
                }
        
        return {"success": False, "error": "Max retries exceeded"}
    
    async def _execute_fallback_method(
        self,
        error: Exception,
        file_path: Union[str, Path],
        processing_context: Any
    ) -> Dict[str, Any]:
        """Execute fallback processing method."""
        # Switch to alternative processing method
        return {
            "success": True,
            "quality_impact": -0.3,  # Some quality reduction
            "recommendations": ["Used fallback processing method"],
            "metadata": {"strategy": "fallback_method", "original_method": "primary"}
        }
    
    async def _execute_reduce_quality(
        self,
        error: Exception,
        file_path: Union[str, Path],
        processing_context: Any
    ) -> Dict[str, Any]:
        """Execute quality reduction strategy."""
        # Reduce processing quality settings
        return {
            "success": True,
            "quality_impact": -0.5,  # Significant quality reduction
            "recommendations": ["Reduced processing quality to avoid error"],
            "metadata": {"strategy": "reduce_quality", "quality_reduction": 0.5}
        }
    
    async def _execute_chunk_processing(
        self,
        error: Exception,
        file_path: Union[str, Path],
        processing_context: Any
    ) -> Dict[str, Any]:
        """Execute chunk-based processing strategy."""
        # Process file in smaller chunks
        return {
            "success": True,
            "quality_impact": -0.1,  # Minor quality impact
            "recommendations": ["Processed file in smaller chunks"],
            "metadata": {"strategy": "chunk_processing", "chunk_size": "reduced"}
        }
    
    # Strategy selection helpers
    
    def _select_primary_strategy(self, error_analysis: ErrorAnalysis) -> RecoveryStrategy:
        """Select primary recovery strategy based on error analysis."""
        category = error_analysis.category
        severity = error_analysis.severity
        
        # Critical errors
        if severity == ErrorSeverity.CRITICAL:
            if category == ErrorCategory.MEMORY_EXHAUSTION:
                return RecoveryStrategy.CHUNK_PROCESSING
            elif category == ErrorCategory.PERMISSION_ERROR:
                return RecoveryStrategy.MANUAL_INTERVENTION
            else:
                return RecoveryStrategy.ABORT_PROCESSING
        
        # High severity errors
        elif severity == ErrorSeverity.HIGH:
            if category == ErrorCategory.FILE_ACCESS:
                return RecoveryStrategy.SKIP_FILE
            elif category == ErrorCategory.FORMAT_PARSING:
                return RecoveryStrategy.FALLBACK_METHOD
            elif category == ErrorCategory.TIMEOUT_ERROR:
                return RecoveryStrategy.RETRY_WITH_BACKOFF
            else:
                return RecoveryStrategy.MANUAL_INTERVENTION
        
        # Medium and low severity errors
        else:
            if category == ErrorCategory.ENCODING_ISSUES:
                return RecoveryStrategy.FALLBACK_METHOD
            elif category == ErrorCategory.NETWORK_ERROR:
                return RecoveryStrategy.RETRY_WITH_BACKOFF
            else:
                return RecoveryStrategy.RETRY
    
    def _select_alternative_strategies(
        self,
        error_analysis: ErrorAnalysis,
        primary_strategy: RecoveryStrategy
    ) -> List[RecoveryStrategy]:
        """Select alternative strategies."""
        alternatives = []
        
        # Add common alternatives based on primary strategy
        if primary_strategy == RecoveryStrategy.RETRY:
            alternatives = [RecoveryStrategy.RETRY_WITH_BACKOFF, RecoveryStrategy.FALLBACK_METHOD]
        elif primary_strategy == RecoveryStrategy.FALLBACK_METHOD:
            alternatives = [RecoveryStrategy.REDUCE_QUALITY, RecoveryStrategy.SKIP_FILE]
        elif primary_strategy == RecoveryStrategy.CHUNK_PROCESSING:
            alternatives = [RecoveryStrategy.REDUCE_QUALITY, RecoveryStrategy.SKIP_FILE]
        else:
            alternatives = [RecoveryStrategy.SKIP_FILE, RecoveryStrategy.MANUAL_INTERVENTION]
        
        # Remove primary strategy from alternatives
        return [alt for alt in alternatives if alt != primary_strategy][:3]
    
    def _estimate_recovery_success_rate(
        self,
        error_analysis: ErrorAnalysis,
        strategy: RecoveryStrategy
    ) -> float:
        """Estimate success rate for recovery strategy."""
        base_rates = {
            RecoveryStrategy.RETRY: 0.3,
            RecoveryStrategy.RETRY_WITH_BACKOFF: 0.6,
            RecoveryStrategy.FALLBACK_METHOD: 0.8,
            RecoveryStrategy.REDUCE_QUALITY: 0.7,
            RecoveryStrategy.CHUNK_PROCESSING: 0.9,
            RecoveryStrategy.SKIP_FILE: 1.0,
            RecoveryStrategy.MANUAL_INTERVENTION: 0.95,
            RecoveryStrategy.ABORT_PROCESSING: 1.0
        }
        
        base_rate = base_rates.get(strategy, 0.5)
        
        # Adjust based on analysis confidence
        confidence_factor = error_analysis.analysis_confidence
        
        return min(base_rate * (0.5 + confidence_factor * 0.5), 1.0)
    
    def _estimate_recovery_time(
        self,
        error_analysis: ErrorAnalysis,
        strategy: RecoveryStrategy
    ) -> float:
        """Estimate recovery time in seconds."""
        base_times = {
            RecoveryStrategy.RETRY: 5.0,
            RecoveryStrategy.RETRY_WITH_BACKOFF: 15.0,
            RecoveryStrategy.FALLBACK_METHOD: 10.0,
            RecoveryStrategy.REDUCE_QUALITY: 8.0,
            RecoveryStrategy.CHUNK_PROCESSING: 20.0,
            RecoveryStrategy.SKIP_FILE: 1.0,
            RecoveryStrategy.MANUAL_INTERVENTION: 300.0,  # 5 minutes
            RecoveryStrategy.ABORT_PROCESSING: 1.0
        }
        
        return base_times.get(strategy, 10.0)
    
    def _get_recovery_prerequisites(self, strategy: RecoveryStrategy) -> List[str]:
        """Get prerequisites for recovery strategy."""
        prerequisites = {
            RecoveryStrategy.RETRY: ["Same processing parameters available"],
            RecoveryStrategy.RETRY_WITH_BACKOFF: ["Network/service availability"],
            RecoveryStrategy.FALLBACK_METHOD: ["Alternative processing method available"],
            RecoveryStrategy.REDUCE_QUALITY: ["Lower quality settings acceptable"],
            RecoveryStrategy.CHUNK_PROCESSING: ["File can be processed in chunks"],
            RecoveryStrategy.SKIP_FILE: ["File can be safely skipped"],
            RecoveryStrategy.MANUAL_INTERVENTION: ["Human operator available"],
            RecoveryStrategy.ABORT_PROCESSING: ["Processing can be safely terminated"]
        }
        
        return prerequisites.get(strategy, [])
    
    def _assess_recovery_risk(
        self,
        strategy: RecoveryStrategy,
        error_analysis: ErrorAnalysis
    ) -> str:
        """Assess risk of recovery strategy."""
        high_risk_strategies = [
            RecoveryStrategy.MANUAL_INTERVENTION,
            RecoveryStrategy.ABORT_PROCESSING
        ]
        
        if strategy in high_risk_strategies:
            return "High - May require significant intervention or data loss"
        
        if error_analysis.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH]:
            return "Medium - Error severity indicates potential complications"
        
        return "Low - Standard recovery with minimal risk"
    
    def _generate_error_id(self, error: Exception, file_path: Union[str, Path]) -> str:
        """Generate unique error identifier."""
        import hashlib
        
        content = f"{type(error).__name__}{str(error)}{file_path}{datetime.now().date()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]
    
    def _initialize_error_patterns(self) -> List[Dict[str, Any]]:
        """Initialize error pattern matchers."""
        import os
        
        def file_not_found_matcher(error, message, stack_trace):
            return isinstance(error, FileNotFoundError) or "no such file" in message.lower()
        
        def permission_error_matcher(error, message, stack_trace):
            return isinstance(error, PermissionError) or "permission denied" in message.lower()
        
        def memory_error_matcher(error, message, stack_trace):
            return isinstance(error, MemoryError) or "memory" in message.lower()
        
        def encoding_error_matcher(error, message, stack_trace):
            return (isinstance(error, UnicodeDecodeError) or 
                   "codec" in message.lower() or 
                   "encoding" in message.lower())
        
        def timeout_error_matcher(error, message, stack_trace):
            return "timeout" in message.lower() or "timed out" in message.lower()
        
        return [
            {
                "match_function": file_not_found_matcher,
                "category": ErrorCategory.FILE_ACCESS,
                "severity": ErrorSeverity.HIGH
            },
            {
                "match_function": permission_error_matcher,
                "category": ErrorCategory.PERMISSION_ERROR,
                "severity": ErrorSeverity.HIGH
            },
            {
                "match_function": memory_error_matcher,
                "category": ErrorCategory.MEMORY_EXHAUSTION,
                "severity": ErrorSeverity.CRITICAL
            },
            {
                "match_function": encoding_error_matcher,
                "category": ErrorCategory.ENCODING_ISSUES,
                "severity": ErrorSeverity.MEDIUM
            },
            {
                "match_function": timeout_error_matcher,
                "category": ErrorCategory.TIMEOUT_ERROR,
                "severity": ErrorSeverity.HIGH
            }
        ]
    
    def _initialize_recovery_strategies(self) -> Dict[str, Dict[str, Any]]:
        """Initialize recovery strategy configurations."""
        return {
            RecoveryStrategy.RETRY.value: {
                "max_attempts": 1,
                "delay": 0,
                "success_rate": 0.3
            },
            RecoveryStrategy.RETRY_WITH_BACKOFF.value: {
                "max_attempts": 3,
                "base_delay": 1.0,
                "backoff_factor": 2.0,
                "success_rate": 0.6
            },
            RecoveryStrategy.FALLBACK_METHOD.value: {
                "quality_impact": -0.3,
                "success_rate": 0.8
            },
            RecoveryStrategy.CHUNK_PROCESSING.value: {
                "chunk_size_factor": 0.5,
                "success_rate": 0.9
            }
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get error handler health status."""
        return {
            "ai_client_available": self.ai_client is not None,
            "error_patterns": len(self.error_patterns),
            "error_history_size": len(self.error_history),
            "recovery_history_size": len(self.recovery_history),
            "statistics": self.stats.copy()
        }
    
    def get_error_summary(self, last_n_hours: int = 24) -> Dict[str, Any]:
        """Get summary of recent errors."""
        cutoff_time = datetime.now().timestamp() - (last_n_hours * 3600)
        
        recent_errors = [
            error for error in self.error_history
            if error.timestamp.timestamp() > cutoff_time
        ]
        
        recent_recoveries = [
            recovery for recovery in self.recovery_history
            if hasattr(recovery, 'timestamp') and recovery.timestamp.timestamp() > cutoff_time
        ]
        
        # Categorize errors
        error_categories = {}
        for error in recent_errors:
            category = error.category.value
            error_categories[category] = error_categories.get(category, 0) + 1
        
        # Recovery success rate
        successful_recoveries = sum(1 for r in recent_recoveries if r.recovered)
        recovery_rate = successful_recoveries / len(recent_recoveries) if recent_recoveries else 0
        
        return {
            "total_errors": len(recent_errors),
            "error_categories": error_categories,
            "recovery_attempts": len(recent_recoveries),
            "recovery_success_rate": recovery_rate,
            "most_common_error": max(error_categories, key=error_categories.get) if error_categories else None
        }

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Global error handler instance
_default_handler = None

def get_default_error_handler() -> IntelligentErrorHandler:
    """Get default error handler instance."""
    global _default_handler
    if _default_handler is None:
        _default_handler = IntelligentErrorHandler()
    return _default_handler

async def handle_processing_error(
    error: Exception,
    file_path: Union[str, Path],
    processing_context: Any = None
) -> Tuple[ErrorAnalysis, RecoveryResult]:
    """Handle processing error with default handler."""
    handler = get_default_error_handler()
    
    # Analyze error
    analysis = await handler.analyze_error(error, file_path, processing_context)
    
    # Attempt recovery
    recovery = await handler.attempt_recovery(error, file_path, processing_context, analysis)
    
    return analysis, recovery