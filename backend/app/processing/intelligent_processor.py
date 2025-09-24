"""
Intelligent File Processing with AI Enhancement for brAIn v2.0

This module provides AI-enhanced file processing capabilities with intelligent format detection,
quality assessment, and processing optimization beyond traditional MIME type detection.

Author: BMad Team
"""

import asyncio
import hashlib
import mimetypes
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Core processing imports
from pydantic import BaseModel, Field
from decimal import Decimal

# AI and ML imports
import anthropic
import openai
from langfuse import Langfuse

# Internal imports
from .file_classification import AIFileClassifier, FileTypeAnalysis
from .quality_assessment import ProcessingQualityAnalyzer, QualityAnalysisResult
from .rules_engine import ExtractionRulesEngine, CustomRule
from .format_optimization import FormatOptimizationEngine, OptimizationStrategy
from .error_intelligence import IntelligentErrorHandler, ErrorAnalysis

# Integration with existing pipeline
try:
    from ..core.text_processor import EnhancedTextProcessor, ProcessingConfig
    from ..core.quality_assessor import QualityAssessmentEngine
    from ..core.duplicate_detector import DuplicateDetectionEngine

    CORE_AVAILABLE = True
except ImportError:
    CORE_AVAILABLE = False
    print("Warning: Core modules not available, running in standalone mode")

# =============================================================================
# PYDANTIC MODELS
# =============================================================================


class ProcessingContext(BaseModel):
    """Context information for intelligent processing."""

    user_preferences: Dict[str, Any] = Field(default_factory=dict)
    domain_knowledge: Optional[str] = Field(
        None, description="Domain context (legal, medical, etc.)"
    )
    priority_level: str = Field(default="normal", description="Processing priority")
    quality_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    custom_rules: List[str] = Field(default_factory=list)
    optimization_level: str = Field(
        default="balanced", description="speed, balanced, or quality"
    )


class IntelligentProcessingResult(BaseModel):
    """Result of intelligent processing with comprehensive analysis."""

    file_path: str = Field(description="Path to processed file")
    detected_format: str = Field(description="AI-detected file format")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Detection confidence")
    processing_quality: float = Field(
        ge=0.0, le=1.0, description="Overall processing quality"
    )
    extraction_method: str = Field(description="Method used for extraction")
    optimization_applied: List[str] = Field(default_factory=list)
    processing_time: float = Field(ge=0.0, description="Processing time in seconds")
    token_count: int = Field(ge=0, description="Estimated token count")
    processing_cost: Decimal = Field(ge=0, description="Estimated processing cost")
    content_preview: str = Field(description="Preview of extracted content")
    quality_metrics: Dict[str, Any] = Field(default_factory=dict)
    recommendations: List[str] = Field(default_factory=list)
    error_analysis: Optional[Dict[str, Any]] = Field(None)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProcessingJob(BaseModel):
    """Intelligent processing job definition."""

    job_id: str = Field(description="Unique job identifier")
    file_path: str = Field(description="Path to file to process")
    context: ProcessingContext = Field(default_factory=ProcessingContext)
    created_at: datetime = Field(default_factory=datetime.now)
    status: str = Field(default="pending", description="Job status")
    result: Optional[IntelligentProcessingResult] = Field(None)
    error_message: Optional[str] = Field(None)


# =============================================================================
# INTELLIGENT FILE PROCESSOR
# =============================================================================


class IntelligentFileProcessor:
    """
    AI-enhanced file processor with intelligent format detection,
    quality assessment, and processing optimization.
    """

    def __init__(
        self,
        ai_client: Optional[Union[anthropic.Anthropic, openai.OpenAI]] = None,
        langfuse_client: Optional[Langfuse] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize intelligent file processor.

        Args:
            ai_client: AI client for intelligent analysis
            langfuse_client: Langfuse client for monitoring
            config: Configuration options
        """
        self.config = config or {}
        self.ai_client = ai_client
        self.langfuse = langfuse_client

        # Initialize AI-powered components
        self.file_classifier = AIFileClassifier(ai_client)
        self.quality_analyzer = ProcessingQualityAnalyzer(ai_client)
        self.rules_engine = ExtractionRulesEngine()
        self.format_optimizer = FormatOptimizationEngine()
        self.error_handler = IntelligentErrorHandler(ai_client)

        # Initialize core processors if available
        if CORE_AVAILABLE:
            self.text_processor = EnhancedTextProcessor()
            self.quality_assessor = QualityAssessmentEngine()
            self.duplicate_detector = DuplicateDetectionEngine()
        else:
            self.text_processor = None
            self.quality_assessor = None
            self.duplicate_detector = None

        # Processing statistics
        self.processing_stats = {
            "total_processed": 0,
            "successful_classifications": 0,
            "quality_improvements": 0,
            "optimization_successes": 0,
            "error_recoveries": 0,
        }

    async def process_file_intelligently(
        self, file_path: Union[str, Path], context: Optional[ProcessingContext] = None
    ) -> IntelligentProcessingResult:
        """
        Process file with AI-enhanced intelligence.

        Args:
            file_path: Path to file to process
            context: Processing context and preferences

        Returns:
            Comprehensive processing result
        """
        start_time = datetime.now()
        file_path = Path(file_path)
        context = context or ProcessingContext()

        # Create processing job ID
        job_id = self._generate_job_id(file_path)

        try:
            # Start Langfuse trace if available
            trace = None
            if self.langfuse:
                trace = self.langfuse.trace(
                    name="intelligent_file_processing",
                    input={"file_path": str(file_path), "context": context.dict()},
                )

            # Step 1: AI-powered file classification
            classification_result = await self._classify_file_intelligently(
                file_path, context
            )

            # Step 2: Select optimal processing strategy
            strategy = await self._select_processing_strategy(
                file_path, classification_result, context
            )

            # Step 3: Apply format-specific optimizations
            optimizations = await self._apply_format_optimizations(
                file_path, classification_result, strategy, context
            )

            # Step 4: Execute intelligent extraction
            extraction_result = await self._execute_intelligent_extraction(
                file_path, classification_result, strategy, optimizations, context
            )

            # Step 5: Assess processing quality
            quality_assessment = await self._assess_processing_quality(
                extraction_result, classification_result, context
            )

            # Step 6: Generate optimization recommendations
            recommendations = await self._generate_recommendations(
                classification_result, quality_assessment, context
            )

            # Calculate processing metrics
            processing_time = (datetime.now() - start_time).total_seconds()

            result = IntelligentProcessingResult(
                file_path=str(file_path),
                detected_format=classification_result.detected_format,
                confidence_score=classification_result.confidence_score,
                processing_quality=quality_assessment.overall_score,
                extraction_method=strategy.method_name,
                optimization_applied=optimizations.applied_optimizations,
                processing_time=processing_time,
                token_count=extraction_result.get("token_count", 0),
                processing_cost=Decimal(str(extraction_result.get("cost", 0.0))),
                content_preview=extraction_result.get("preview", "")[:500],
                quality_metrics=quality_assessment.metrics,
                recommendations=recommendations,
                metadata={
                    "file_size": file_path.stat().st_size,
                    "mime_type": mimetypes.guess_type(file_path)[0],
                    "classification_metadata": classification_result.metadata,
                    "optimization_metadata": optimizations.metadata,
                },
            )

            # Update statistics
            self.processing_stats["total_processed"] += 1
            if classification_result.confidence_score > 0.8:
                self.processing_stats["successful_classifications"] += 1
            if quality_assessment.overall_score > context.quality_threshold:
                self.processing_stats["quality_improvements"] += 1

            # End Langfuse trace
            if trace:
                trace.update(output=result.dict(), metadata={"success": True})

            return result

        except Exception as e:
            # Intelligent error handling
            error_analysis = await self.error_handler.analyze_error(
                e, file_path, context
            )

            # Attempt error recovery if possible
            recovery_result = await self.error_handler.attempt_recovery(
                e, file_path, context, error_analysis
            )

            if recovery_result.recovered:
                self.processing_stats["error_recoveries"] += 1
                # Recursive call with recovery adjustments
                recovery_context = context.copy()
                recovery_context.optimization_level = "speed"  # Use faster processing
                return await self.process_file_intelligently(
                    file_path, recovery_context
                )

            # End Langfuse trace with error
            if trace:
                trace.update(
                    output={"error": str(e)},
                    metadata={
                        "success": False,
                        "error_analysis": error_analysis.dict(),
                    },
                )

            raise RuntimeError(f"Intelligent processing failed: {str(e)}")

    async def _classify_file_intelligently(
        self, file_path: Path, context: ProcessingContext
    ) -> FileTypeAnalysis:
        """Classify file using AI beyond MIME types."""
        return await self.file_classifier.classify_with_content_analysis(
            file_path, context.domain_knowledge
        )

    async def _select_processing_strategy(
        self,
        file_path: Path,
        classification: FileTypeAnalysis,
        context: ProcessingContext,
    ) -> OptimizationStrategy:
        """Select optimal processing strategy based on classification and context."""
        return await self.format_optimizer.select_strategy(
            classification, context.optimization_level, context.quality_threshold
        )

    async def _apply_format_optimizations(
        self,
        file_path: Path,
        classification: FileTypeAnalysis,
        strategy: OptimizationStrategy,
        context: ProcessingContext,
    ) -> Dict[str, Any]:
        """Apply format-specific optimizations."""
        return await self.format_optimizer.apply_optimizations(
            file_path, classification, strategy, context
        )

    async def _execute_intelligent_extraction(
        self,
        file_path: Path,
        classification: FileTypeAnalysis,
        strategy: OptimizationStrategy,
        optimizations: Dict[str, Any],
        context: ProcessingContext,
    ) -> Dict[str, Any]:
        """Execute extraction with intelligent method selection."""
        # Apply custom rules if specified
        if context.custom_rules:
            extraction_rules = await self.rules_engine.compile_rules(
                context.custom_rules
            )
            return await self.rules_engine.extract_with_rules(
                file_path, extraction_rules, strategy
            )

        # Use format-specific extraction
        if classification.detected_format in strategy.supported_formats:
            return await strategy.extract_content(file_path, optimizations)

        # Fallback to core processor if available
        if self.text_processor:
            config = ProcessingConfig(
                quality_threshold=context.quality_threshold,
                chunk_size=strategy.recommended_chunk_size,
            )
            result = await self.text_processor.process_file_with_validation(
                file_path, config
            )
            return {
                "content": result.extracted_content,
                "token_count": result.token_count,
                "cost": float(result.processing_cost),
                "preview": (
                    result.extracted_content[:500] if result.extracted_content else ""
                ),
            }

        # Basic extraction fallback
        return await self._basic_extraction_fallback(file_path)

    async def _assess_processing_quality(
        self,
        extraction_result: Dict[str, Any],
        classification: FileTypeAnalysis,
        context: ProcessingContext,
    ) -> QualityAnalysisResult:
        """Assess the quality of processing results."""
        return await self.quality_analyzer.analyze_processing_quality(
            extraction_result, classification, context
        )

    async def _generate_recommendations(
        self,
        classification: FileTypeAnalysis,
        quality_assessment: QualityAnalysisResult,
        context: ProcessingContext,
    ) -> List[str]:
        """Generate optimization recommendations based on results."""
        recommendations = []

        if classification.confidence_score < 0.7:
            recommendations.append(
                "Consider manual format specification for better accuracy"
            )

        if quality_assessment.overall_score < context.quality_threshold:
            recommendations.append(
                "Increase processing quality threshold for better results"
            )
            recommendations.append(
                "Consider using custom extraction rules for this document type"
            )

        if quality_assessment.processing_efficiency < 0.5:
            recommendations.append(
                "File structure may benefit from format-specific optimization"
            )

        return recommendations

    async def _basic_extraction_fallback(self, file_path: Path) -> Dict[str, Any]:
        """Basic text extraction as fallback method."""
        try:
            # Simple text extraction for common formats
            if file_path.suffix.lower() in [".txt", ".md", ".log"]:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                return {
                    "content": content,
                    "token_count": len(content.split()),
                    "cost": 0.0,
                    "preview": content[:500],
                }

            # For other formats, return minimal info
            return {
                "content": f"File: {file_path.name}",
                "token_count": 0,
                "cost": 0.0,
                "preview": f"Processing not supported for {file_path.suffix}",
            }

        except Exception as e:
            return {
                "content": "",
                "token_count": 0,
                "cost": 0.0,
                "preview": f"Extraction failed: {str(e)}",
            }

    def _generate_job_id(self, file_path: Path) -> str:
        """Generate unique job ID for processing."""
        content = f"{file_path}{datetime.now().isoformat()}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    async def batch_process_intelligently(
        self,
        file_paths: List[Union[str, Path]],
        context: Optional[ProcessingContext] = None,
    ) -> List[IntelligentProcessingResult]:
        """Process multiple files intelligently with optimized batching."""
        context = context or ProcessingContext()

        # Create processing tasks
        tasks = [
            self.process_file_intelligently(file_path, context)
            for file_path in file_paths
        ]

        # Execute with controlled concurrency
        semaphore = asyncio.Semaphore(self.config.get("max_concurrent", 3))

        async def process_with_semaphore(task):
            async with semaphore:
                return await task

        results = await asyncio.gather(
            *[process_with_semaphore(task) for task in tasks], return_exceptions=True
        )

        # Handle exceptions in results
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Create error result
                error_result = IntelligentProcessingResult(
                    file_path=str(file_paths[i]),
                    detected_format="unknown",
                    confidence_score=0.0,
                    processing_quality=0.0,
                    extraction_method="failed",
                    processing_time=0.0,
                    token_count=0,
                    processing_cost=Decimal("0.0"),
                    content_preview=f"Error: {str(result)}",
                    error_analysis={
                        "error": str(result),
                        "type": type(result).__name__,
                    },
                )
                processed_results.append(error_result)
            else:
                processed_results.append(result)

        return processed_results

    def get_processing_analytics(self) -> Dict[str, Any]:
        """Get processing analytics and performance metrics."""
        return {
            "statistics": self.processing_stats.copy(),
            "component_health": {
                "file_classifier": self.file_classifier.get_health_status(),
                "quality_analyzer": self.quality_analyzer.get_health_status(),
                "rules_engine": self.rules_engine.get_health_status(),
                "format_optimizer": self.format_optimizer.get_health_status(),
                "error_handler": self.error_handler.get_health_status(),
            },
            "recommendations": self._generate_system_recommendations(),
        }

    def _generate_system_recommendations(self) -> List[str]:
        """Generate system-wide optimization recommendations."""
        recommendations = []

        success_rate = self.processing_stats["successful_classifications"] / max(
            self.processing_stats["total_processed"], 1
        )

        if success_rate < 0.8:
            recommendations.append("Consider training custom classification models")

        if (
            self.processing_stats["error_recoveries"]
            > self.processing_stats["total_processed"] * 0.1
        ):
            recommendations.append("Review file input validation to reduce error rates")

        return recommendations


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Global processor instance
_default_processor = None


def get_default_intelligent_processor() -> IntelligentFileProcessor:
    """Get default intelligent processor instance."""
    global _default_processor
    if _default_processor is None:
        _default_processor = IntelligentFileProcessor()
    return _default_processor


async def process_file_with_ai(
    file_path: Union[str, Path], context: Optional[ProcessingContext] = None
) -> IntelligentProcessingResult:
    """Process single file with AI intelligence."""
    processor = get_default_intelligent_processor()
    return await processor.process_file_intelligently(file_path, context)


async def batch_process_with_ai(
    file_paths: List[Union[str, Path]], context: Optional[ProcessingContext] = None
) -> List[IntelligentProcessingResult]:
    """Process multiple files with AI intelligence."""
    processor = get_default_intelligent_processor()
    return await processor.batch_process_intelligently(file_paths, context)
