"""
Processing control API endpoints
Manages document processing, monitoring, and control
"""

from typing import List, Optional, Dict, Any
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status, Query, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, or_

from app.config.settings import get_settings
from app.models.api import (
    ProcessingStatusResponse,
    ProcessingJobResponse,
    ProcessingStatsResponse,
    ProcessingStartRequest,
    ProcessingQueueResponse,
)
from app.models.documents import Document, ProcessingJob
from database.connection import get_database_session
from api.auth import get_current_user

router = APIRouter()


@router.get("/status", response_model=ProcessingStatusResponse)
async def get_processing_status(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_database_session),
):
    """Get overall processing status and metrics."""
    try:
        # Get processing statistics
        total_docs_query = select(Document).where(
            Document.user_id == current_user["id"]
        )
        total_result = await db.execute(total_docs_query)
        total_documents = len(total_result.scalars().all())

        processed_docs_query = select(Document).where(
            and_(
                Document.user_id == current_user["id"],
                Document.processing_status == "completed",
            )
        )
        processed_result = await db.execute(processed_docs_query)
        processed_documents = len(processed_result.scalars().all())

        processing_docs_query = select(Document).where(
            and_(
                Document.user_id == current_user["id"],
                Document.processing_status.in_(["processing", "queued"]),
            )
        )
        processing_result = await db.execute(processing_docs_query)
        processing_documents = len(processing_result.scalars().all())

        failed_docs_query = select(Document).where(
            and_(
                Document.user_id == current_user["id"],
                Document.processing_status == "failed",
            )
        )
        failed_result = await db.execute(failed_docs_query)
        failed_documents = len(failed_result.scalars().all())

        return ProcessingStatusResponse(
            total_documents=total_documents,
            processed_documents=processed_documents,
            processing_documents=processing_documents,
            failed_documents=failed_documents,
            processing_rate=0.0,  # Would calculate actual rate
            estimated_completion=None,  # Would calculate estimate
            current_status="idle" if processing_documents == 0 else "processing",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get processing status: {str(e)}",
        )


@router.post("/start", response_model=Dict[str, Any])
async def start_processing(
    request: ProcessingStartRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_database_session),
):
    """Start processing documents with specified options."""
    try:
        # Validate processing options
        if request.folder_ids:
            # Process specific folders
            processing_scope = "folders"
            item_count = len(request.folder_ids)
        elif request.document_ids:
            # Process specific documents
            processing_scope = "documents"
            item_count = len(request.document_ids)
        else:
            # Process all user documents
            processing_scope = "all"
            total_query = select(Document).where(Document.user_id == current_user["id"])
            result = await db.execute(total_query)
            item_count = len(result.scalars().all())

        # Create processing job
        job = ProcessingJob(
            user_id=current_user["id"],
            job_type=processing_scope,
            total_items=item_count,
            processed_items=0,
            failed_items=0,
            status="queued",
            options={
                "force_reprocess": request.force_reprocess,
                "enable_ai_enhancement": request.enable_ai_enhancement,
                "batch_size": request.batch_size,
                "folder_ids": request.folder_ids or [],
                "document_ids": request.document_ids or [],
            },
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        db.add(job)
        await db.commit()
        await db.refresh(job)

        # Queue background processing
        background_tasks.add_task(execute_processing_job, job.id, db)

        return {
            "message": "Processing job started successfully",
            "job_id": job.id,
            "scope": processing_scope,
            "total_items": item_count,
            "status": "queued",
        }
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start processing: {str(e)}",
        )


@router.get("/jobs", response_model=List[ProcessingJobResponse])
async def list_processing_jobs(
    limit: int = Query(20, ge=1, le=100),
    status_filter: Optional[str] = Query(None, description="Filter by job status"),
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_database_session),
):
    """List processing jobs for the current user."""
    try:
        query = select(ProcessingJob).where(ProcessingJob.user_id == current_user["id"])

        if status_filter:
            query = query.where(ProcessingJob.status == status_filter)

        query = query.order_by(ProcessingJob.created_at.desc()).limit(limit)
        result = await db.execute(query)
        jobs = result.scalars().all()

        return [
            ProcessingJobResponse(
                id=job.id,
                job_type=job.job_type,
                status=job.status,
                total_items=job.total_items,
                processed_items=job.processed_items,
                failed_items=job.failed_items,
                progress_percentage=(
                    (job.processed_items / job.total_items * 100)
                    if job.total_items > 0
                    else 0
                ),
                created_at=job.created_at,
                updated_at=job.updated_at,
                completed_at=job.completed_at,
                options=job.options,
            )
            for job in jobs
        ]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list processing jobs: {str(e)}",
        )


@router.get("/jobs/{job_id}", response_model=ProcessingJobResponse)
async def get_processing_job(
    job_id: int,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_database_session),
):
    """Get details of a specific processing job."""
    try:
        query = select(ProcessingJob).where(
            and_(
                ProcessingJob.id == job_id, ProcessingJob.user_id == current_user["id"]
            )
        )
        result = await db.execute(query)
        job = result.scalar_one_or_none()

        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Processing job not found"
            )

        return ProcessingJobResponse(
            id=job.id,
            job_type=job.job_type,
            status=job.status,
            total_items=job.total_items,
            processed_items=job.processed_items,
            failed_items=job.failed_items,
            progress_percentage=(
                (job.processed_items / job.total_items * 100)
                if job.total_items > 0
                else 0
            ),
            created_at=job.created_at,
            updated_at=job.updated_at,
            completed_at=job.completed_at,
            options=job.options,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get processing job: {str(e)}",
        )


@router.post("/jobs/{job_id}/cancel")
async def cancel_processing_job(
    job_id: int,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_database_session),
):
    """Cancel a processing job."""
    try:
        query = select(ProcessingJob).where(
            and_(
                ProcessingJob.id == job_id, ProcessingJob.user_id == current_user["id"]
            )
        )
        result = await db.execute(query)
        job = result.scalar_one_or_none()

        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Processing job not found"
            )

        if job.status in ["completed", "failed", "cancelled"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Cannot cancel job with status: {job.status}",
            )

        job.status = "cancelled"
        job.updated_at = datetime.now(timezone.utc)
        job.completed_at = datetime.now(timezone.utc)

        await db.commit()

        return {
            "message": "Processing job cancelled successfully",
            "job_id": job_id,
            "status": "cancelled",
        }
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel processing job: {str(e)}",
        )


@router.get("/stats", response_model=ProcessingStatsResponse)
async def get_processing_stats(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_database_session),
):
    """Get detailed processing statistics."""
    try:
        # Calculate various statistics
        return ProcessingStatsResponse(
            total_documents_processed=0,  # Would calculate actual stats
            total_processing_time_seconds=0,
            average_processing_time_per_document=0.0,
            successful_processing_rate=0.0,
            most_common_file_types={},
            processing_errors_by_type={},
            daily_processing_counts={},
            cost_statistics={
                "total_cost": 0.0,
                "cost_per_document": 0.0,
                "cost_by_model": {},
            },
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get processing stats: {str(e)}",
        )


@router.get("/queue", response_model=ProcessingQueueResponse)
async def get_processing_queue(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_database_session),
):
    """Get current processing queue status."""
    try:
        # Get queued and processing items
        queue_query = (
            select(Document)
            .where(
                and_(
                    Document.user_id == current_user["id"],
                    Document.processing_status.in_(["queued", "processing"]),
                )
            )
            .order_by(Document.created_at)
        )

        result = await db.execute(queue_query)
        queue_items = result.scalars().all()

        return ProcessingQueueResponse(
            total_items=len(queue_items),
            processing_items=[
                item for item in queue_items if item.processing_status == "processing"
            ],
            queued_items=[
                item for item in queue_items if item.processing_status == "queued"
            ],
            estimated_wait_time_seconds=len(queue_items) * 30,  # Rough estimate
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get processing queue: {str(e)}",
        )


async def execute_processing_job(job_id: int, db: AsyncSession):
    """Background task to execute a processing job."""
    # This would integrate with the actual processing pipeline
    # For now, we'll just update the job status
    try:
        query = select(ProcessingJob).where(ProcessingJob.id == job_id)
        result = await db.execute(query)
        job = result.scalar_one_or_none()

        if job:
            job.status = "processing"
            job.updated_at = datetime.now(timezone.utc)
            await db.commit()

            # Simulate processing
            # In reality, this would call the actual processing pipeline

            job.status = "completed"
            job.processed_items = job.total_items
            job.completed_at = datetime.now(timezone.utc)
            await db.commit()
    except Exception as e:
        # Handle processing errors
        if job:
            job.status = "failed"
            job.error_message = str(e)
            job.completed_at = datetime.now(timezone.utc)
            await db.commit()
