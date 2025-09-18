"""
Enhanced RAG Pipeline Core Module for brAIn v2.0

This module provides the enhanced RAG (Retrieval Augmented Generation) pipeline
with AI validation, duplicate detection, quality assessment, and cost tracking.

The enhanced pipeline builds upon the proven RAG Pipeline architecture with
modern AI validation, Pydantic models, and intelligent processing capabilities.

Author: BMad Team
"""

from typing import Tuple, List

from .text_processor import (
    EnhancedTextProcessor,
    ProcessingConfig,
    FileProcessingResult,
    TextChunk,
    ProcessingQuality,
    # Backward compatibility functions
    sanitize_text,
    chunk_text,
    create_embeddings,
    extract_text_from_file,
    get_default_processor
)

from .database_handler import (
    EnhancedDatabaseHandler,
    DocumentMetadata,
    DocumentChunk,
    DuplicateDetectionResult,
    ProcessingStats,
    # Backward compatibility functions
    check_document_exists,
    delete_document_by_file_id,
    get_default_handler
)

from .duplicate_detector import (
    DuplicateDetectionEngine,
    DuplicateMatch,
    DeduplicationResult,
    DuplicateDetectionConfig,
    # Convenience functions
    detect_duplicates,
    check_duplicate,
    get_default_detector
)

from .quality_assessor import (
    QualityAssessmentEngine,
    QualityAssessmentResult,
    ContentQualityMetrics,
    ExtractionQualityMetrics,
    ProcessingQualityMetrics,
    QualityThresholds,
    # Convenience functions
    assess_quality,
    quick_quality_check,
    get_default_assessor
)

from .processing_orchestrator import (
    EnhancedProcessingOrchestrator,
    ProcessingJob,
    ProcessingBatch,
    ProcessingSummary,
    ProcessingStatus,
    ProcessingPriority,
    # Convenience functions
    process_directory,
    process_files,
    get_default_orchestrator
)

# Version information
__version__ = "2.0.0"
__author__ = "BMad Team"
__description__ = "Enhanced RAG Pipeline with AI Validation and Quality Assessment"

# Main orchestrator for easy access (lazy loaded)
_orchestrator = None

def get_orchestrator():
    """Get the default orchestrator (lazy loaded)."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = get_default_orchestrator()
    return _orchestrator

# For backward compatibility
orchestrator = None  # Will be loaded when needed

# Quick access functions for common operations
def quick_process_file(file_path: str, priority: str = "normal") -> FileProcessingResult:
    """
    Quickly process a single file with default settings.
    
    Args:
        file_path: Path to file to process
        priority: Processing priority (low, normal, high, urgent)
        
    Returns:
        File processing result
    """
    import asyncio
    from pathlib import Path
    
    # Convert priority string to enum
    priority_map = {
        "low": ProcessingPriority.LOW,
        "normal": ProcessingPriority.NORMAL,
        "high": ProcessingPriority.HIGH,
        "urgent": ProcessingPriority.URGENT
    }
    
    priority_enum = priority_map.get(priority.lower(), ProcessingPriority.NORMAL)
    
    # Create and process job
    orch = get_orchestrator()
    job = orch.create_processing_job(file_path, priority_enum)
    
    async def process_async():
        return await orch.process_single_job(job)
    
    # Run in event loop
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If we're already in an async context, create a new loop
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, process_async())
                result_job = future.result()
        else:
            result_job = loop.run_until_complete(process_async())
    except RuntimeError:
        # No event loop, create one
        result_job = asyncio.run(process_async())
    
    if result_job.processing_result:
        return result_job.processing_result
    else:
        raise RuntimeError(f"Failed to process file: {result_job.error_message}")

def validate_text_quality(text: str, threshold: float = 0.7) -> Tuple[bool, float, List[str]]:
    """
    Validate text quality against threshold.
    
    Args:
        text: Text to validate
        threshold: Quality threshold (0.0 to 1.0)
        
    Returns:
        Tuple of (passes_validation, quality_score, recommendations)
    """
    quality_score = quick_quality_check(text)
    
    assessor = get_default_assessor()
    content_metrics = assessor.assess_content_quality(text)
    recommendations = []
    
    if content_metrics.readability_score < 30:
        recommendations.append("Improve readability")
    if content_metrics.coherence_score < 0.6:
        recommendations.append("Improve coherence")
    if content_metrics.information_density < 0.4:
        recommendations.append("Increase information density")
    
    passes = quality_score >= threshold
    
    return passes, quality_score, recommendations

def detect_content_duplicates(content_items: List[Tuple[str, str]], 
                            threshold: float = 0.95) -> List[Tuple[str, str, float]]:
    """
    Detect duplicates in a list of content items.
    
    Args:
        content_items: List of (id, content) tuples
        threshold: Similarity threshold for duplicate detection
        
    Returns:
        List of (original_id, duplicate_id, similarity) tuples
    """
    detector = get_default_detector()
    detector.config.similarity_threshold = threshold
    
    result = detector.deduplicate_content_batch(content_items)
    
    duplicates = []
    for group in result.duplicate_groups:
        if len(group) > 1:
            original = group[0]
            for duplicate in group[1:]:
                duplicates.append((original, duplicate, 1.0))  # Exact matches get 1.0
    
    return duplicates

# Configuration shortcuts
def configure_processing(chunk_size: int = 400, 
                        chunk_overlap: int = 0,
                        quality_threshold: float = 0.7,
                        enable_duplicate_detection: bool = True,
                        max_file_size_mb: int = 100) -> None:
    """
    Configure global processing settings.
    
    Args:
        chunk_size: Default chunk size in characters
        chunk_overlap: Overlap between chunks
        quality_threshold: Minimum quality threshold
        enable_duplicate_detection: Enable duplicate detection
        max_file_size_mb: Maximum file size in MB
    """
    global orchestrator
    
    # Update text processor config
    text_processor = get_default_processor()
    text_processor.config.chunk_size = chunk_size
    text_processor.config.chunk_overlap = chunk_overlap
    text_processor.config.quality_threshold = quality_threshold
    text_processor.config.max_file_size_mb = max_file_size_mb
    
    # Update orchestrator config
    orchestrator.config.enable_duplicate_detection = enable_duplicate_detection
    orchestrator.config.quality_threshold = quality_threshold
    orchestrator.config.max_file_size_mb = max_file_size_mb

# Export all major classes and functions
__all__ = [
    # Main classes
    "EnhancedTextProcessor",
    "EnhancedDatabaseHandler", 
    "DuplicateDetectionEngine",
    "QualityAssessmentEngine",
    "EnhancedProcessingOrchestrator",
    
    # Models
    "ProcessingConfig",
    "FileProcessingResult",
    "TextChunk",
    "ProcessingQuality",
    "DocumentMetadata",
    "DocumentChunk",
    "DuplicateDetectionResult",
    "ProcessingStats",
    "DuplicateMatch",
    "DeduplicationResult",
    "QualityAssessmentResult",
    "ContentQualityMetrics",
    "ExtractionQualityMetrics",
    "ProcessingQualityMetrics",
    "ProcessingJob",
    "ProcessingBatch",
    "ProcessingSummary",
    
    # Enums
    "ProcessingStatus",
    "ProcessingPriority",
    
    # Quick access functions
    "quick_process_file",
    "validate_text_quality",
    "detect_content_duplicates",
    "configure_processing",
    
    # Convenience functions
    "sanitize_text",
    "chunk_text", 
    "create_embeddings",
    "extract_text_from_file",
    "check_document_exists",
    "delete_document_by_file_id",
    "detect_duplicates",
    "check_duplicate",
    "assess_quality",
    "quick_quality_check",
    "process_directory",
    "process_files",
    
    # Default instances
    "orchestrator",
    
    # Getters for default instances
    "get_default_processor",
    "get_default_handler",
    "get_default_detector", 
    "get_default_assessor",
    "get_default_orchestrator",
]

# Module-level initialization
print(f"brAIn Enhanced RAG Pipeline v{__version__} initialized")
print(f"Available components: Text Processing, Database Operations, Duplicate Detection, Quality Assessment, Processing Orchestration")