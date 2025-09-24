"""
Folder management API endpoints
Handles Google Drive folder operations and database management
"""

from typing import List, Optional
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_

from config.settings import get_settings
from models.api import (
    FolderResponse,
    FolderCreateRequest,
    FolderUpdateRequest,
    FolderListResponse,
)
from models.documents import Folder
from database.connection import get_database_session
from api.auth import get_current_user

router = APIRouter()


@router.get("/", response_model=FolderListResponse)
async def list_folders(
    skip: int = Query(0, ge=0, description="Number of folders to skip"),
    limit: int = Query(100, ge=1, le=1000, description="Number of folders to return"),
    search: Optional[str] = Query(None, description="Search folders by name"),
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_database_session),
):
    """List all folders with optional filtering."""
    try:
        # Build query
        query = select(Folder).where(Folder.user_id == current_user["id"])

        if search:
            query = query.where(Folder.name.ilike(f"%{search}%"))

        # Execute query
        query = query.offset(skip).limit(limit)
        result = await db.execute(query)
        folders = result.scalars().all()

        # Count total
        count_query = select(Folder).where(Folder.user_id == current_user["id"])
        if search:
            count_query = count_query.where(Folder.name.ilike(f"%{search}%"))

        count_result = await db.execute(count_query)
        total = len(count_result.scalars().all())

        folder_responses = [
            FolderResponse(
                id=folder.id,
                name=folder.name,
                google_drive_id=folder.google_drive_id,
                total_documents=0,  # Would calculate actual count
                processed_documents=0,  # Would calculate actual count
                status=folder.status,
                created_at=folder.created_at,
                updated_at=folder.updated_at,
            )
            for folder in folders
        ]

        return FolderListResponse(
            folders=folder_responses, total=total, skip=skip, limit=limit
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list folders: {str(e)}",
        )


@router.post("/", response_model=FolderResponse)
async def create_folder(
    request: FolderCreateRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_database_session),
):
    """Create a new folder."""
    try:
        # Validate Google Drive ID format
        if not request.google_drive_id or len(request.google_drive_id) < 10:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid Google Drive ID format",
            )

        # Check if folder already exists
        existing_query = select(Folder).where(
            and_(
                Folder.google_drive_id == request.google_drive_id,
                Folder.user_id == current_user["id"],
            )
        )
        existing_result = await db.execute(existing_query)
        if existing_result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Folder with this Google Drive ID already exists",
            )

        # Create new folder
        new_folder = Folder(
            name=request.name,
            google_drive_id=request.google_drive_id,
            user_id=current_user["id"],
            status="pending",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        db.add(new_folder)
        await db.commit()
        await db.refresh(new_folder)

        return FolderResponse(
            id=new_folder.id,
            name=new_folder.name,
            google_drive_id=new_folder.google_drive_id,
            total_documents=0,
            processed_documents=0,
            status=new_folder.status,
            created_at=new_folder.created_at,
            updated_at=new_folder.updated_at,
        )
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create folder: {str(e)}",
        )


@router.get("/{folder_id}", response_model=FolderResponse)
async def get_folder(
    folder_id: int,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_database_session),
):
    """Get a specific folder by ID."""
    try:
        query = select(Folder).where(
            and_(Folder.id == folder_id, Folder.user_id == current_user["id"])
        )
        result = await db.execute(query)
        folder = result.scalar_one_or_none()

        if not folder:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Folder not found"
            )

        return FolderResponse(
            id=folder.id,
            name=folder.name,
            google_drive_id=folder.google_drive_id,
            total_documents=0,  # Would calculate actual count
            processed_documents=0,  # Would calculate actual count
            status=folder.status,
            created_at=folder.created_at,
            updated_at=folder.updated_at,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get folder: {str(e)}",
        )


@router.put("/{folder_id}", response_model=FolderResponse)
async def update_folder(
    folder_id: int,
    request: FolderUpdateRequest,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_database_session),
):
    """Update a folder's information."""
    try:
        query = select(Folder).where(
            and_(Folder.id == folder_id, Folder.user_id == current_user["id"])
        )
        result = await db.execute(query)
        folder = result.scalar_one_or_none()

        if not folder:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Folder not found"
            )

        # Update fields if provided
        if request.name is not None:
            folder.name = request.name

        folder.updated_at = datetime.now(timezone.utc)

        await db.commit()
        await db.refresh(folder)

        return FolderResponse(
            id=folder.id,
            name=folder.name,
            google_drive_id=folder.google_drive_id,
            total_documents=0,
            processed_documents=0,
            status=folder.status,
            created_at=folder.created_at,
            updated_at=folder.updated_at,
        )
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update folder: {str(e)}",
        )


@router.delete("/{folder_id}")
async def delete_folder(
    folder_id: int,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_database_session),
):
    """Delete a folder and all its documents."""
    try:
        query = select(Folder).where(
            and_(Folder.id == folder_id, Folder.user_id == current_user["id"])
        )
        result = await db.execute(query)
        folder = result.scalar_one_or_none()

        if not folder:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Folder not found"
            )

        await db.delete(folder)
        await db.commit()

        return {"message": "Folder deleted successfully", "folder_id": folder_id}
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete folder: {str(e)}",
        )


@router.post("/{folder_id}/process")
async def process_folder(
    folder_id: int,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_database_session),
):
    """Trigger processing of all documents in a folder."""
    try:
        query = select(Folder).where(
            and_(Folder.id == folder_id, Folder.user_id == current_user["id"])
        )
        result = await db.execute(query)
        folder = result.scalar_one_or_none()

        if not folder:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Folder not found"
            )

        # Update folder status to processing
        folder.status = "processing"
        folder.updated_at = datetime.now(timezone.utc)
        await db.commit()

        # Trigger processing (would integrate with processing pipeline)
        # This would typically queue the folder for processing

        return {
            "message": "Folder processing initiated",
            "folder_id": folder_id,
            "status": "processing",
        }
    except HTTPException:
        raise
    except Exception as e:
        await db.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process folder: {str(e)}",
        )
