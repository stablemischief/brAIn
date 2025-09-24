"""
Enhanced Processing Orchestrator for brAIn v2.0 RAG Pipeline

This module orchestrates the complete document processing pipeline with
AI validation, cost tracking, quality assessment, and intelligent error recovery.

Author: BMad Team
"""

import os
import asyncio
import time
from typing import List, Dict, Any, Optional, Union, Callable
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum

# Pydantic imports
from pydantic import BaseModel, Field, ValidationError

# Local imports
from .text_processor import (
    EnhancedTextProcessor,
    ProcessingConfig,
    FileProcessingResult,
)
from .database_handler import EnhancedDatabaseHandler, ProcessingStats
from .duplicate_detector import DuplicateDetectionEngine
from .quality_assessor import QualityAssessmentEngine

# Cost and monitoring integration
try:
    from ..monitoring.cost_calculator import CostCalculator
    from ..monitoring.langfuse_client import LangfuseClient

    MONITORING_AVAILABLE = True
except ImportError:
    MONITORING_AVAILABLE = False
    print("Warning: Monitoring modules not available")

# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================


class ProcessingStatus(str, Enum):
    """Status of processing operations."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class ProcessingPriority(str, Enum):
    """Priority levels for processing."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"


class ErrorRecoveryAction(str, Enum):
    """Error recovery actions."""

    RETRY = "retry"
    SKIP = "skip"
    FALLBACK = "fallback"
    ABORT = "abort"


# =============================================================================
# PYDANTIC MODELS
# =============================================================================


class ProcessingJob(BaseModel):
    """Individual processing job with validation."""

    job_id: str = Field(description="Unique job identifier")
    file_path: str = Field(description="Path to file to process")
    file_name: str = Field(description="Name of the file")
    file_size: int = Field(ge=0, description="File size in bytes")
    mime_type: Optional[str] = Field(default=None, description="MIME type")
    priority: ProcessingPriority = Field(default=ProcessingPriority.NORMAL)
    status: ProcessingStatus = Field(default=ProcessingStatus.PENDING)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)
    error_message: Optional[str] = Field(default=None)
    retry_count: int = Field(default=0, ge=0, le=10)
    processing_result: Optional[FileProcessingResult] = Field(default=None)
    cost_estimate: Decimal = Field(default=Decimal("0.00"), ge=0)

    def __init__(self, **data):
        if "job_id" not in data:
            # Generate job ID from file path and timestamp
            job_data = f"{data.get('file_path', '')}_{time.time()}"
            data["job_id"] = hashlib.sha256(job_data.encode()).hexdigest()[:16]
        super().__init__(**data)


class ProcessingBatch(BaseModel):
    """Batch of processing jobs."""

    batch_id: str = Field(description="Unique batch identifier")
    jobs: List[ProcessingJob] = Field(description="Jobs in this batch")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)
    total_cost: Decimal = Field(default=Decimal("0.00"), ge=0)
    success_count: int = Field(default=0, ge=0)
    failure_count: int = Field(default=0, ge=0)

    @property
    def total_jobs(self) -> int:
        return len(self.jobs)

    @property
    def is_complete(self) -> bool:
        return all(
            job.status
            in [
                ProcessingStatus.COMPLETED,
                ProcessingStatus.FAILED,
                ProcessingStatus.SKIPPED,
            ]
            for job in self.jobs
        )


class ProcessingConfig(BaseModel):
    """Enhanced processing configuration."""

    max_concurrent_jobs: int = Field(default=5, ge=1, le=50)
    max_retries: int = Field(default=3, ge=0, le=10)
    retry_delay_seconds: int = Field(default=5, ge=1, le=300)
    enable_duplicate_detection: bool = Field(default=True)
    enable_quality_assessment: bool = Field(default=True)
    enable_cost_tracking: bool = Field(default=True)
    enable_langfuse_monitoring: bool = Field(default=True)
    quality_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    max_file_size_mb: int = Field(default=100, ge=1, le=1000)
    supported_extensions: List[str] = Field(
        default_factory=lambda: [
            ".pdf",
            ".docx",
            ".doc",
            ".xlsx",
            ".xls",
            ".pptx",
            ".txt",
            ".html",
            ".csv",
            ".md",
        ]
    )
    exclude_patterns: List[str] = Field(
        default_factory=lambda: ["*.tmp", "*.temp", "*~", ".DS_Store"]
    )
    batch_size: int = Field(default=10, ge=1, le=100)
    enable_async_processing: bool = Field(default=True)


class ProcessingSummary(BaseModel):
    """Summary of processing operations."""

    batch_id: str
    total_jobs: int
    successful_jobs: int
    failed_jobs: int
    skipped_jobs: int
    total_files_size: int
    total_processing_time: float
    total_cost: Decimal
    average_quality_score: float
    duplicates_detected: int
    errors_encountered: List[str]
    performance_metrics: Dict[str, Any]


# =============================================================================
# PROCESSING ORCHESTRATOR CLASS
# =============================================================================


class EnhancedProcessingOrchestrator:
    """
    Enhanced processing orchestrator with AI validation and intelligent recovery.
    """

    def __init__(self, config: Optional[ProcessingConfig] = None):
        """
        Initialize the orchestrator.

        Args:
            config: Processing configuration
        """
        self.config = config or ProcessingConfig()

        # Initialize core components
        self.text_processor = EnhancedTextProcessor()
        self.database_handler = EnhancedDatabaseHandler()
        self.duplicate_detector = DuplicateDetectionEngine()
        self.quality_assessor = QualityAssessmentEngine()

        # Initialize monitoring if available
        self.cost_calculator = None
        self.langfuse_client = None

        if MONITORING_AVAILABLE:
            try:
                if self.config.enable_cost_tracking:
                    self.cost_calculator = CostCalculator()
                if self.config.enable_langfuse_monitoring:
                    self.langfuse_client = LangfuseClient()
            except Exception as e:
                print(f"Warning: Could not initialize monitoring: {e}")

        # Processing state
        self.active_batches: Dict[str, ProcessingBatch] = {}
        self.processing_stats = ProcessingStats(
            total_files_processed=0,
            total_chunks_created=0,
            duplicates_detected=0,
            total_processing_cost=Decimal("0.00"),
            average_quality_score=0.0,
            processing_time_seconds=0.0,
            errors_encountered=0,
        )

        # Event callbacks
        self.on_job_started: Optional[Callable[[ProcessingJob], None]] = None
        self.on_job_completed: Optional[Callable[[ProcessingJob], None]] = None
        self.on_job_failed: Optional[Callable[[ProcessingJob, str], None]] = None
        self.on_batch_completed: Optional[Callable[[ProcessingBatch], None]] = None

    def create_processing_job(
        self, file_path: str, priority: ProcessingPriority = ProcessingPriority.NORMAL
    ) -> ProcessingJob:
        """
        Create a processing job from a file path.

        Args:
            file_path: Path to the file to process
            priority: Processing priority

        Returns:
            Created processing job
        """
        file_path_obj = Path(file_path)

        if not file_path_obj.exists():
            raise ValueError(f"File does not exist: {file_path}")

        file_size = file_path_obj.stat().st_size

        # Check file size limits
        max_size = self.config.max_file_size_mb * 1024 * 1024
        if file_size > max_size:
            raise ValueError(f"File size {file_size} exceeds limit {max_size}")

        # Check supported extensions
        if file_path_obj.suffix.lower() not in self.config.supported_extensions:
            raise ValueError(f"Unsupported file extension: {file_path_obj.suffix}")

        # Create job
        job = ProcessingJob(
            file_path=str(file_path),
            file_name=file_path_obj.name,
            file_size=file_size,
            priority=priority,
        )

        return job

    def create_batch_from_directory(
        self, directory_path: str, recursive: bool = True
    ) -> ProcessingBatch:
        """
        Create a processing batch from a directory.

        Args:
            directory_path: Path to directory to scan
            recursive: Whether to scan recursively

        Returns:
            Created processing batch
        """
        directory = Path(directory_path)

        if not directory.exists() or not directory.is_dir():
            raise ValueError(f"Directory does not exist: {directory_path}")

        # Scan for files
        jobs = []
        pattern = "**/*" if recursive else "*"

        for file_path in directory.glob(pattern):
            if not file_path.is_file():
                continue

            # Check extension
            if file_path.suffix.lower() not in self.config.supported_extensions:
                continue

            # Check exclude patterns
            if any(
                file_path.match(pattern) for pattern in self.config.exclude_patterns
            ):
                continue

            try:
                job = self.create_processing_job(str(file_path))
                jobs.append(job)
            except Exception as e:
                print(f"Skipping file {file_path}: {e}")

        # Sort by priority and size
        jobs.sort(key=lambda j: (j.priority.value, j.file_size))

        # Create batch
        batch_id = hashlib.sha256(
            f"{directory_path}_{time.time()}".encode()
        ).hexdigest()[:16]
        batch = ProcessingBatch(batch_id=batch_id, jobs=jobs)

        return batch

    def estimate_job_cost(self, job: ProcessingJob) -> Decimal:
        """
        Estimate the cost of processing a job.

        Args:
            job: Processing job

        Returns:
            Estimated cost
        """
        if not self.cost_calculator:
            return Decimal("0.00")

        try:
            # Rough estimation based on file size
            # In production, this would be more sophisticated
            estimated_tokens = job.file_size // 4  # Rough character-to-token ratio
            cost = self.cost_calculator.calculate_embedding_cost(estimated_tokens)
            return cost
        except Exception as e:
            print(f"Error estimating cost for job {job.job_id}: {e}")
            return Decimal("0.00")

    def should_retry_job(self, job: ProcessingJob, error: str) -> ErrorRecoveryAction:
        """
        Determine if a job should be retried based on the error.

        Args:
            job: Failed processing job
            error: Error message

        Returns:
            Recovery action to take
        """
        if job.retry_count >= self.config.max_retries:
            return ErrorRecoveryAction.ABORT

        # Analyze error type
        error_lower = error.lower()

        # Network or temporary errors - retry
        if any(
            keyword in error_lower
            for keyword in [
                "timeout",
                "connection",
                "network",
                "temporary",
                "rate limit",
            ]
        ):
            return ErrorRecoveryAction.RETRY

        # File access errors - skip
        if any(
            keyword in error_lower
            for keyword in ["permission", "access denied", "file not found"]
        ):
            return ErrorRecoveryAction.SKIP

        # Validation errors - try fallback
        if any(
            keyword in error_lower for keyword in ["validation", "invalid", "corrupted"]
        ):
            return ErrorRecoveryAction.FALLBACK

        # Default to retry for unknown errors
        return ErrorRecoveryAction.RETRY

    async def process_single_job(self, job: ProcessingJob) -> ProcessingJob:
        """
        Process a single job with error handling and recovery.

        Args:
            job: Job to process

        Returns:
            Updated job with processing results
        """
        job.status = ProcessingStatus.PROCESSING
        job.started_at = datetime.now(timezone.utc)

        # Notify job started
        if self.on_job_started:
            self.on_job_started(job)

        # Start Langfuse trace if available
        trace = None
        if self.langfuse_client:
            trace = self.langfuse_client.start_trace(
                name="document_processing",
                metadata={"file_name": job.file_name, "job_id": job.job_id},
            )

        try:
            # Read file content
            file_content = Path(job.file_path).read_bytes()

            # Process file with text processor
            processing_result = self.text_processor.process_file_with_validation(
                file_content, job.file_name, job.job_id
            )

            # Create embeddings for chunks
            if processing_result.text_chunks:
                chunk_texts = [chunk.content for chunk in processing_result.text_chunks]
                embeddings = self.text_processor.create_embeddings(chunk_texts)

                # Update chunks with embeddings
                for chunk, embedding in zip(processing_result.text_chunks, embeddings):
                    # Store embedding in chunk metadata (would need to extend TextChunk model)
                    pass

            # Quality assessment
            if self.config.enable_quality_assessment:
                quality_score = self.quality_assessor.assess_processing_result(
                    processing_result
                )

                if quality_score < self.config.quality_threshold:
                    job.error_message = f"Quality score {quality_score:.3f} below threshold {self.config.quality_threshold}"
                    job.status = ProcessingStatus.FAILED
                    return job

            # Duplicate detection
            if self.config.enable_duplicate_detection:
                is_duplicate = self.duplicate_detector.check_document_duplicate(
                    processing_result.content_hash, processing_result.extracted_text
                )

                if is_duplicate:
                    job.status = ProcessingStatus.SKIPPED
                    job.error_message = "Document is a duplicate"
                    return job

            # Store in database
            success = self.database_handler.process_file_for_rag(processing_result)

            if success:
                job.status = ProcessingStatus.COMPLETED
                job.processing_result = processing_result
                job.cost_estimate = processing_result.cost_estimate

                # Update stats
                self.processing_stats.total_files_processed += 1
                self.processing_stats.total_chunks_created += len(
                    processing_result.text_chunks
                )
                self.processing_stats.total_processing_cost += (
                    processing_result.cost_estimate
                )

                # Notify completion
                if self.on_job_completed:
                    self.on_job_completed(job)
            else:
                job.status = ProcessingStatus.FAILED
                job.error_message = "Failed to store in database"

        except Exception as e:
            error_message = str(e)
            job.error_message = error_message

            # Determine recovery action
            recovery_action = self.should_retry_job(job, error_message)

            if recovery_action == ErrorRecoveryAction.RETRY:
                job.retry_count += 1
                job.status = ProcessingStatus.PENDING
                print(
                    f"Job {job.job_id} will be retried ({job.retry_count}/{self.config.max_retries})"
                )
            elif recovery_action == ErrorRecoveryAction.SKIP:
                job.status = ProcessingStatus.SKIPPED
                print(f"Job {job.job_id} skipped due to error: {error_message}")
            else:
                job.status = ProcessingStatus.FAILED
                print(f"Job {job.job_id} failed: {error_message}")

            # Update stats
            self.processing_stats.errors_encountered += 1

            # Notify failure
            if self.on_job_failed:
                self.on_job_failed(job, error_message)

        finally:
            job.completed_at = datetime.now(timezone.utc)

            # Complete Langfuse trace
            if trace and self.langfuse_client:
                self.langfuse_client.complete_trace(
                    trace,
                    output={"status": job.status, "cost": float(job.cost_estimate)},
                    metadata={"retry_count": job.retry_count},
                )

        return job

    async def process_batch(self, batch: ProcessingBatch) -> ProcessingSummary:
        """
        Process a batch of jobs with concurrency control.

        Args:
            batch: Batch to process

        Returns:
            Processing summary
        """
        batch.started_at = datetime.now(timezone.utc)
        self.active_batches[batch.batch_id] = batch

        start_time = time.time()

        try:
            if self.config.enable_async_processing:
                # Process jobs concurrently with semaphore
                semaphore = asyncio.Semaphore(self.config.max_concurrent_jobs)

                async def process_with_semaphore(job):
                    async with semaphore:
                        return await self.process_single_job(job)

                # Create tasks for all jobs
                tasks = [process_with_semaphore(job) for job in batch.jobs]

                # Process all tasks
                completed_jobs = await asyncio.gather(*tasks, return_exceptions=True)

                # Update batch with results
                for i, result in enumerate(completed_jobs):
                    if isinstance(result, ProcessingJob):
                        batch.jobs[i] = result
                    else:
                        # Handle exception
                        batch.jobs[i].status = ProcessingStatus.FAILED
                        batch.jobs[i].error_message = str(result)
            else:
                # Process jobs sequentially
                for i, job in enumerate(batch.jobs):
                    batch.jobs[i] = await self.process_single_job(job)

        finally:
            batch.completed_at = datetime.now(timezone.utc)
            processing_time = time.time() - start_time

            # Calculate batch statistics
            success_count = sum(
                1 for job in batch.jobs if job.status == ProcessingStatus.COMPLETED
            )
            failure_count = sum(
                1 for job in batch.jobs if job.status == ProcessingStatus.FAILED
            )
            skipped_count = sum(
                1 for job in batch.jobs if job.status == ProcessingStatus.SKIPPED
            )

            batch.success_count = success_count
            batch.failure_count = failure_count
            batch.total_cost = sum(job.cost_estimate for job in batch.jobs)

            # Create summary
            summary = ProcessingSummary(
                batch_id=batch.batch_id,
                total_jobs=batch.total_jobs,
                successful_jobs=success_count,
                failed_jobs=failure_count,
                skipped_jobs=skipped_count,
                total_files_size=sum(job.file_size for job in batch.jobs),
                total_processing_time=processing_time,
                total_cost=batch.total_cost,
                average_quality_score=0.0,  # Would calculate from results
                duplicates_detected=skipped_count,  # Rough approximation
                errors_encountered=[
                    job.error_message for job in batch.jobs if job.error_message
                ],
                performance_metrics={
                    "jobs_per_second": (
                        len(batch.jobs) / processing_time if processing_time > 0 else 0
                    ),
                    "average_job_time": (
                        processing_time / len(batch.jobs) if batch.jobs else 0
                    ),
                    "success_rate": (
                        success_count / len(batch.jobs) if batch.jobs else 0
                    ),
                },
            )

            # Notify batch completion
            if self.on_batch_completed:
                self.on_batch_completed(batch)

            # Clean up active batch
            if batch.batch_id in self.active_batches:
                del self.active_batches[batch.batch_id]

            return summary

    def get_batch_status(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of an active batch."""
        if batch_id not in self.active_batches:
            return None

        batch = self.active_batches[batch_id]

        completed_jobs = sum(
            1
            for job in batch.jobs
            if job.status
            in [
                ProcessingStatus.COMPLETED,
                ProcessingStatus.FAILED,
                ProcessingStatus.SKIPPED,
            ]
        )

        return {
            "batch_id": batch_id,
            "total_jobs": len(batch.jobs),
            "completed_jobs": completed_jobs,
            "progress": completed_jobs / len(batch.jobs) if batch.jobs else 0,
            "started_at": batch.started_at,
            "is_complete": batch.is_complete,
        }

    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get overall processing statistics."""
        return {
            "total_files_processed": self.processing_stats.total_files_processed,
            "total_chunks_created": self.processing_stats.total_chunks_created,
            "total_cost": float(self.processing_stats.total_processing_cost),
            "errors_encountered": self.processing_stats.errors_encountered,
            "active_batches": len(self.active_batches),
            "config": self.config.dict(),
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Global orchestrator instance
_default_orchestrator = None


def get_default_orchestrator() -> EnhancedProcessingOrchestrator:
    """Get the default orchestrator instance."""
    global _default_orchestrator
    if _default_orchestrator is None:
        _default_orchestrator = EnhancedProcessingOrchestrator()
    return _default_orchestrator


async def process_directory(
    directory_path: str, recursive: bool = True
) -> ProcessingSummary:
    """Process all supported files in a directory."""
    orchestrator = get_default_orchestrator()
    batch = orchestrator.create_batch_from_directory(directory_path, recursive)
    return await orchestrator.process_batch(batch)


async def process_files(file_paths: List[str]) -> ProcessingSummary:
    """Process a list of files."""
    orchestrator = get_default_orchestrator()

    jobs = []
    for file_path in file_paths:
        try:
            job = orchestrator.create_processing_job(file_path)
            jobs.append(job)
        except Exception as e:
            print(f"Skipping file {file_path}: {e}")

    batch_id = hashlib.sha256(f"manual_batch_{time.time()}".encode()).hexdigest()[:16]
    batch = ProcessingBatch(batch_id=batch_id, jobs=jobs)

    return await orchestrator.process_batch(batch)
