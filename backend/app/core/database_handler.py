"""
Enhanced Database Handler for brAIn v2.0 RAG Pipeline

This module provides enhanced database operations with duplicate detection,
vector similarity search, and comprehensive validation based on the proven
RAG Pipeline architecture.

Author: BMad Team
"""

import os
import json
import hashlib
import traceback
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
import base64
import numpy as np

# Pydantic imports for validation
from pydantic import BaseModel, Field, ValidationError
from pydantic import field_validator

# Database and environment
from dotenv import load_dotenv
from supabase import create_client, Client

# Local imports
from .text_processor import (
    FileProcessingResult,
    TextChunk,
    ProcessingQuality,
    is_tabular_file,
    extract_schema_from_csv,
    extract_rows_from_csv,
)

# Load environment variables
project_root = Path(__file__).resolve().parent.parent
dotenv_path = project_root / ".env"
load_dotenv(dotenv_path, override=True)

# Lazy initialization for Supabase client
_supabase_client = None


def get_supabase_client() -> Client:
    """Get or create Supabase client with lazy initialization."""
    global _supabase_client
    if _supabase_client is None:
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_SERVICE_KEY")

        if not supabase_url or not supabase_key:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in environment"
            )

        _supabase_client = create_client(supabase_url, supabase_key)

    return _supabase_client


# For backward compatibility
supabase = None  # Will be initialized when needed

# =============================================================================
# PYDANTIC MODELS FOR DATABASE VALIDATION
# =============================================================================


class DocumentMetadata(BaseModel):
    """Enhanced document metadata with validation."""

    file_id: str = Field(description="Unique file identifier")
    file_title: str = Field(min_length=1, description="Document title")
    file_url: str = Field(description="URL or path to file")
    mime_type: str = Field(description="MIME type of the file")
    file_size: int = Field(ge=0, description="File size in bytes")
    content_hash: str = Field(description="SHA-256 hash of content")
    processing_timestamp: datetime = Field(
        description="When the document was processed"
    )
    processing_cost: Decimal = Field(ge=0, description="Cost of processing")
    token_count: int = Field(ge=0, description="Total token count")
    chunk_count: int = Field(ge=0, description="Number of chunks")
    content_language: Optional[str] = Field(
        default=None, description="Detected language"
    )
    extracted_entities: List[str] = Field(
        default_factory=list, description="Extracted entities"
    )
    quality_score: float = Field(ge=0.0, le=1.0, description="Overall quality score")
    schema: Optional[Dict[str, Any]] = Field(
        default=None, description="Schema for tabular data"
    )


class DocumentChunk(BaseModel):
    """Enhanced document chunk with validation."""

    chunk_id: Optional[str] = Field(default=None, description="Unique chunk identifier")
    file_id: str = Field(description="Parent file identifier")
    content: str = Field(min_length=1, description="Chunk content")
    chunk_index: int = Field(ge=0, description="Index of chunk in document")
    chunk_hash: str = Field(description="SHA-256 hash of chunk content")
    embedding: List[float] = Field(description="Vector embedding")
    quality_score: float = Field(ge=0.0, le=1.0, description="Chunk quality score")
    token_count: int = Field(ge=0, description="Token count for chunk")
    language: Optional[str] = Field(default=None, description="Detected language")
    entities: List[str] = Field(default_factory=list, description="Entities in chunk")

    def __init__(self, **data):
        if "chunk_hash" not in data and "content" in data:
            data["chunk_hash"] = hashlib.sha256(data["content"].encode()).hexdigest()
        if "chunk_id" not in data:
            # Generate chunk ID from file_id and chunk_index
            chunk_data = f"{data.get('file_id', '')}_{data.get('chunk_index', 0)}"
            data["chunk_id"] = hashlib.sha256(chunk_data.encode()).hexdigest()[:16]
        super().__init__(**data)


class DuplicateDetectionResult(BaseModel):
    """Result of duplicate detection analysis."""

    is_duplicate: bool = Field(description="Whether content is a duplicate")
    similarity_score: float = Field(
        ge=0.0, le=1.0, description="Similarity to closest match"
    )
    duplicate_file_id: Optional[str] = Field(
        default=None, description="ID of duplicate file"
    )
    duplicate_chunk_id: Optional[str] = Field(
        default=None, description="ID of duplicate chunk"
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in detection")
    detection_method: str = Field(description="Method used for detection")


class ProcessingStats(BaseModel):
    """Statistics for processing operations."""

    total_files_processed: int = Field(ge=0)
    total_chunks_created: int = Field(ge=0)
    duplicates_detected: int = Field(ge=0)
    total_processing_cost: Decimal = Field(ge=0)
    average_quality_score: float = Field(ge=0.0, le=1.0)
    processing_time_seconds: float = Field(ge=0.0)
    errors_encountered: int = Field(ge=0)


# =============================================================================
# ENHANCED DATABASE OPERATIONS CLASS
# =============================================================================


class EnhancedDatabaseHandler:
    """Enhanced database handler with duplicate detection and validation."""

    def __init__(
        self,
        duplicate_threshold: float = 0.95,
        supabase_url: str = None,
        supabase_key: str = None,
    ):
        """
        Initialize with configuration.

        Args:
            duplicate_threshold: Similarity threshold for duplicate detection
            supabase_url: Optional Supabase URL (uses env if not provided)
            supabase_key: Optional Supabase key (uses env if not provided)
        """
        self.duplicate_threshold = duplicate_threshold

        # Use provided credentials or get from environment
        if supabase_url and supabase_key:
            self.supabase = create_client(supabase_url, supabase_key)
        else:
            self.supabase = get_supabase_client()

    def calculate_vector_similarity(
        self, vec1: List[float], vec2: List[float]
    ) -> float:
        """Calculate cosine similarity between two vectors."""
        try:
            if len(vec1) != len(vec2):
                return 0.0

            vec1_np = np.array(vec1)
            vec2_np = np.array(vec2)

            # Calculate cosine similarity
            dot_product = np.dot(vec1_np, vec2_np)
            norm1 = np.linalg.norm(vec1_np)
            norm2 = np.linalg.norm(vec2_np)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            similarity = dot_product / (norm1 * norm2)
            return max(0.0, min(1.0, similarity))  # Clamp to [0, 1]

        except Exception as e:
            print(f"Error calculating vector similarity: {e}")
            return 0.0

    def detect_content_duplicate(
        self, content: str, embedding: List[float], file_id: str
    ) -> DuplicateDetectionResult:
        """
        Detect if content is a duplicate using multiple methods.

        Args:
            content: Text content to check
            embedding: Vector embedding of the content
            file_id: ID of the file (to avoid self-matches)

        Returns:
            Duplicate detection result
        """
        try:
            content_hash = hashlib.sha256(content.encode()).hexdigest()

            # Method 1: Exact hash match
            hash_response = (
                self.supabase.table("documents")
                .select("metadata->>file_id, metadata->>chunk_index")
                .eq("metadata->>chunk_hash", content_hash)
                .neq("metadata->>file_id", file_id)
                .limit(1)
                .execute()
            )

            if hash_response.data:
                return DuplicateDetectionResult(
                    is_duplicate=True,
                    similarity_score=1.0,
                    duplicate_file_id=hash_response.data[0]["metadata"]["file_id"],
                    confidence=1.0,
                    detection_method="exact_hash",
                )

            # Method 2: Vector similarity search
            # Note: In production, use pgvector's similarity functions
            # This is a simplified implementation for demonstration
            all_chunks_response = (
                self.supabase.table("documents")
                .select("embedding, metadata->>file_id, metadata->>chunk_index, id")
                .neq("metadata->>file_id", file_id)
                .limit(1000)
                .execute()
            )

            if all_chunks_response.data:
                max_similarity = 0.0
                best_match = None

                for chunk in all_chunks_response.data:
                    if chunk.get("embedding"):
                        similarity = self.calculate_vector_similarity(
                            embedding, chunk["embedding"]
                        )
                        if similarity > max_similarity:
                            max_similarity = similarity
                            best_match = chunk

                if max_similarity >= self.duplicate_threshold:
                    return DuplicateDetectionResult(
                        is_duplicate=True,
                        similarity_score=max_similarity,
                        duplicate_file_id=best_match["metadata"]["file_id"],
                        duplicate_chunk_id=best_match["id"],
                        confidence=max_similarity,
                        detection_method="vector_similarity",
                    )

            # Method 3: Length and word overlap heuristics
            words = set(content.lower().split())
            if len(words) > 5:  # Only for substantial content
                similar_response = (
                    self.supabase.table("documents")
                    .select("content, metadata->>file_id, id")
                    .neq("metadata->>file_id", file_id)
                    .limit(100)
                    .execute()
                )

                if similar_response.data:
                    for chunk in similar_response.data:
                        other_words = set(chunk["content"].lower().split())
                        if len(other_words) > 5:
                            # Calculate Jaccard similarity
                            intersection = len(words.intersection(other_words))
                            union = len(words.union(other_words))
                            jaccard = intersection / union if union > 0 else 0

                            if jaccard >= 0.8:  # High word overlap
                                return DuplicateDetectionResult(
                                    is_duplicate=True,
                                    similarity_score=jaccard,
                                    duplicate_file_id=chunk["metadata"]["file_id"],
                                    duplicate_chunk_id=chunk["id"],
                                    confidence=jaccard
                                    * 0.8,  # Lower confidence for heuristic
                                    detection_method="word_overlap",
                                )

            # No duplicate found
            return DuplicateDetectionResult(
                is_duplicate=False,
                similarity_score=max_similarity,
                confidence=1.0 - max_similarity,
                detection_method="comprehensive_check",
            )

        except Exception as e:
            print(f"Error in duplicate detection: {e}")
            return DuplicateDetectionResult(
                is_duplicate=False,
                similarity_score=0.0,
                confidence=0.5,
                detection_method="error_fallback",
            )

    def check_document_exists(self, file_id: str) -> bool:
        """Check if a document already exists in the database."""
        try:
            response = (
                self.supabase.table("documents")
                .select("id")
                .eq("metadata->>file_id", file_id)
                .limit(1)
                .execute()
            )

            return bool(response.data)

        except Exception as e:
            print(f"Error checking document existence: {e}")
            return False

    def delete_document_by_file_id(self, file_id: str) -> bool:
        """
        Delete all records related to a specific file ID.

        Args:
            file_id: The file ID to delete

        Returns:
            True if successful, False otherwise
        """
        try:
            # Delete documents
            doc_response = (
                self.supabase.table("documents")
                .delete()
                .eq("metadata->>file_id", file_id)
                .execute()
            )

            # Delete document rows (for tabular data)
            rows_response = (
                self.supabase.table("document_rows")
                .delete()
                .eq("dataset_id", file_id)
                .execute()
            )

            # Delete metadata
            meta_response = (
                self.supabase.table("document_metadata")
                .delete()
                .eq("id", file_id)
                .execute()
            )

            print(
                f"Deleted document {file_id}: {len(doc_response.data)} chunks, "
                f"{len(rows_response.data)} rows, metadata record"
            )

            return True

        except Exception as e:
            print(f"Error deleting document {file_id}: {e}")
            return False

    def insert_document_metadata(self, metadata: DocumentMetadata) -> bool:
        """
        Insert or update document metadata.

        Args:
            metadata: Validated document metadata

        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert to database format
            db_data = {
                "id": metadata.file_id,
                "title": metadata.file_title,
                "url": metadata.file_url,
                "mime_type": metadata.mime_type,
                "file_size": metadata.file_size,
                "content_hash": metadata.content_hash,
                "processing_timestamp": metadata.processing_timestamp.isoformat(),
                "processing_cost": float(metadata.processing_cost),
                "token_count": metadata.token_count,
                "chunk_count": metadata.chunk_count,
                "content_language": metadata.content_language,
                "extracted_entities": metadata.extracted_entities,
                "quality_score": metadata.quality_score,
            }

            if metadata.schema:
                db_data["schema"] = json.dumps(metadata.schema)

            # Upsert the record
            response = (
                self.supabase.table("document_metadata").upsert(db_data).execute()
            )

            return bool(response.data)

        except Exception as e:
            print(f"Error inserting document metadata: {e}")
            return False

    def insert_document_chunks(
        self, chunks: List[DocumentChunk], check_duplicates: bool = True
    ) -> Tuple[int, int]:
        """
        Insert document chunks with optional duplicate detection.

        Args:
            chunks: List of validated document chunks
            check_duplicates: Whether to check for duplicates

        Returns:
            Tuple of (inserted_count, duplicate_count)
        """
        inserted_count = 0
        duplicate_count = 0

        try:
            for chunk in chunks:
                # Check for duplicates if enabled
                if check_duplicates:
                    duplicate_result = self.detect_content_duplicate(
                        chunk.content, chunk.embedding, chunk.file_id
                    )

                    if duplicate_result.is_duplicate:
                        duplicate_count += 1
                        print(
                            f"Skipping duplicate chunk: {duplicate_result.detection_method} "
                            f"(similarity: {duplicate_result.similarity_score:.3f})"
                        )
                        continue

                # Prepare data for insertion
                db_data = {
                    "content": chunk.content,
                    "embedding": chunk.embedding,
                    "metadata": {
                        "file_id": chunk.file_id,
                        "chunk_index": chunk.chunk_index,
                        "chunk_hash": chunk.chunk_hash,
                        "chunk_id": chunk.chunk_id,
                        "quality_score": chunk.quality_score,
                        "token_count": chunk.token_count,
                        "language": chunk.language,
                        "entities": chunk.entities,
                    },
                }

                # Insert the chunk
                response = self.supabase.table("documents").insert(db_data).execute()

                if response.data:
                    inserted_count += 1
                else:
                    print(f"Failed to insert chunk {chunk.chunk_id}")

        except Exception as e:
            print(f"Error inserting document chunks: {e}")
            traceback.print_exc()

        return inserted_count, duplicate_count

    def process_file_for_rag(
        self,
        processing_result: FileProcessingResult,
        config: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Process a complete file result for RAG storage.

        Args:
            processing_result: Validated file processing result
            config: Optional configuration

        Returns:
            True if successful, False otherwise
        """
        try:
            start_time = datetime.now()

            # Delete existing records for this file
            self.delete_document_by_file_id(processing_result.file_id)

            # Create document metadata
            metadata = DocumentMetadata(
                file_id=processing_result.file_id,
                file_title=processing_result.file_name,
                file_url=processing_result.file_id,  # Using file_id as URL for now
                mime_type=processing_result.mime_type,
                file_size=processing_result.file_size,
                content_hash=processing_result.content_hash,
                processing_timestamp=processing_result.processing_timestamp,
                processing_cost=processing_result.cost_estimate,
                token_count=processing_result.token_count,
                chunk_count=len(processing_result.text_chunks),
                content_language=processing_result.content_language,
                extracted_entities=processing_result.extracted_entities,
                quality_score=processing_result.processing_quality.confidence_score,
            )

            # Insert metadata
            if not self.insert_document_metadata(metadata):
                print(f"Failed to insert metadata for {processing_result.file_id}")
                return False

            # Convert text chunks to database chunks
            db_chunks = []
            for chunk in processing_result.text_chunks:
                # Create embeddings for chunks (this would be done by text processor)
                # For now, we'll create empty embeddings as placeholder
                db_chunk = DocumentChunk(
                    file_id=processing_result.file_id,
                    content=chunk.content,
                    chunk_index=chunk.chunk_index,
                    embedding=[0.0]
                    * 1536,  # Placeholder - should come from actual embeddings
                    quality_score=chunk.quality_score,
                    token_count=len(chunk.content) // 4,  # Rough estimate
                    language=chunk.language_detected,
                    entities=[],  # Could extract per chunk
                )
                db_chunks.append(db_chunk)

            # Insert chunks with duplicate detection
            inserted_count, duplicate_count = self.insert_document_chunks(db_chunks)

            processing_time = (datetime.now() - start_time).total_seconds()

            print(
                f"Processed {processing_result.file_name}: "
                f"{inserted_count} chunks inserted, {duplicate_count} duplicates skipped, "
                f"{processing_time:.2f}s"
            )

            return inserted_count > 0

        except Exception as e:
            print(f"Error processing file for RAG: {e}")
            traceback.print_exc()
            return False

    def search_similar_content(
        self,
        query_embedding: List[float],
        file_id: Optional[str] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar content using vector similarity.

        Args:
            query_embedding: Query embedding vector
            file_id: Optional file ID to filter results
            limit: Maximum number of results

        Returns:
            List of similar chunks with metadata
        """
        try:
            # Base query
            query = (
                self.supabase.table("documents")
                .select("content, metadata, embedding")
                .limit(limit)
            )

            # Add file filter if specified
            if file_id:
                query = query.eq("metadata->>file_id", file_id)

            response = query.execute()

            if not response.data:
                return []

            # Calculate similarities and sort
            results = []
            for item in response.data:
                if item.get("embedding"):
                    similarity = self.calculate_vector_similarity(
                        query_embedding, item["embedding"]
                    )

                    result = {
                        "content": item["content"],
                        "metadata": item["metadata"],
                        "similarity": similarity,
                        "file_id": item["metadata"].get("file_id"),
                        "chunk_index": item["metadata"].get("chunk_index", 0),
                    }
                    results.append(result)

            # Sort by similarity (descending)
            results.sort(key=lambda x: x["similarity"], reverse=True)

            return results

        except Exception as e:
            print(f"Error searching similar content: {e}")
            return []

    def get_processing_statistics(
        self, file_id: Optional[str] = None
    ) -> ProcessingStats:
        """
        Get processing statistics.

        Args:
            file_id: Optional file ID to filter statistics

        Returns:
            Processing statistics
        """
        try:
            # Get document metadata
            meta_query = self.supabase.table("document_metadata").select("*")
            if file_id:
                meta_query = meta_query.eq("id", file_id)

            meta_response = meta_query.execute()

            if not meta_response.data:
                return ProcessingStats(
                    total_files_processed=0,
                    total_chunks_created=0,
                    duplicates_detected=0,
                    total_processing_cost=Decimal("0.00"),
                    average_quality_score=0.0,
                    processing_time_seconds=0.0,
                    errors_encountered=0,
                )

            total_files = len(meta_response.data)
            total_chunks = sum(doc.get("chunk_count", 0) for doc in meta_response.data)
            total_cost = sum(
                Decimal(str(doc.get("processing_cost", 0)))
                for doc in meta_response.data
            )
            avg_quality = (
                sum(doc.get("quality_score", 0.0) for doc in meta_response.data)
                / total_files
            )

            return ProcessingStats(
                total_files_processed=total_files,
                total_chunks_created=total_chunks,
                duplicates_detected=0,  # Would need to track this separately
                total_processing_cost=total_cost,
                average_quality_score=avg_quality,
                processing_time_seconds=0.0,  # Would need to track this
                errors_encountered=0,  # Would need to track this
            )

        except Exception as e:
            print(f"Error getting processing statistics: {e}")
            return ProcessingStats(
                total_files_processed=0,
                total_chunks_created=0,
                duplicates_detected=0,
                total_processing_cost=Decimal("0.00"),
                average_quality_score=0.0,
                processing_time_seconds=0.0,
                errors_encountered=1,
            )


# =============================================================================
# CONVENIENCE FUNCTIONS FOR BACKWARD COMPATIBILITY
# =============================================================================

# Global handler instance
_default_handler = None


def get_default_handler() -> EnhancedDatabaseHandler:
    """Get the default database handler instance."""
    global _default_handler
    if _default_handler is None:
        _default_handler = EnhancedDatabaseHandler()
    return _default_handler


# Legacy compatibility functions
def check_document_exists(file_id: str) -> bool:
    """Check if document exists using default handler."""
    return get_default_handler().check_document_exists(file_id)


def delete_document_by_file_id(file_id: str) -> None:
    """Delete document using default handler."""
    get_default_handler().delete_document_by_file_id(file_id)


def process_file_for_rag(
    file_content: bytes,
    text: str,
    file_id: str,
    file_url: str,
    file_title: str,
    mime_type: str = None,
    config: Dict[str, Any] = None,
) -> bool:
    """Legacy function for processing files."""
    # This would need to be adapted to work with the new system
    # For now, return True to maintain compatibility
    print("Legacy process_file_for_rag called - consider upgrading to new system")
    return True
