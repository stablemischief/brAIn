"""
Duplicate Detection Engine for brAIn v2.0 RAG Pipeline

This module provides advanced duplicate detection using multiple methods including
content hashing, vector similarity, and intelligent heuristics.

Author: BMad Team
"""

import hashlib
import os
from typing import List, Dict, Any, Optional, Tuple, Set
from decimal import Decimal
import numpy as np
from dataclasses import dataclass
from datetime import datetime

# Pydantic imports
from pydantic import BaseModel, Field

# Database integration
try:
    from .database_handler import EnhancedDatabaseHandler

    DATABASE_AVAILABLE = True
except ImportError:
    DATABASE_AVAILABLE = False
    print("Warning: Database handler not available for duplicate detection")

# =============================================================================
# PYDANTIC MODELS
# =============================================================================


class DuplicateMatch(BaseModel):
    """Represents a duplicate match."""

    original_id: str = Field(description="ID of original document")
    duplicate_id: str = Field(description="ID of duplicate document")
    similarity_score: float = Field(ge=0.0, le=1.0, description="Similarity score")
    detection_method: str = Field(description="Method used to detect duplicate")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in match")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )


class DeduplicationResult(BaseModel):
    """Result of deduplication process."""

    total_documents: int = Field(ge=0, description="Total documents processed")
    unique_documents: int = Field(ge=0, description="Number of unique documents")
    duplicate_groups: List[List[str]] = Field(
        default_factory=list, description="Groups of duplicates"
    )
    duplicates_removed: int = Field(ge=0, description="Number of duplicates removed")
    processing_time: float = Field(ge=0.0, description="Processing time in seconds")
    cost_savings: Decimal = Field(
        ge=0, description="Estimated cost savings from deduplication"
    )


class DuplicateDetectionConfig(BaseModel):
    """Configuration for duplicate detection."""

    enable_hash_detection: bool = Field(
        default=True, description="Enable exact hash matching"
    )
    enable_vector_similarity: bool = Field(
        default=True, description="Enable vector similarity"
    )
    enable_fuzzy_matching: bool = Field(
        default=True, description="Enable fuzzy text matching"
    )
    similarity_threshold: float = Field(
        default=0.95, ge=0.0, le=1.0, description="Similarity threshold"
    )
    fuzzy_threshold: float = Field(
        default=0.90, ge=0.0, le=1.0, description="Fuzzy matching threshold"
    )
    min_content_length: int = Field(
        default=50, ge=1, description="Minimum content length to check"
    )
    batch_size: int = Field(
        default=100, ge=1, le=1000, description="Batch size for processing"
    )
    enable_content_normalization: bool = Field(
        default=True, description="Enable content normalization"
    )


# =============================================================================
# DUPLICATE DETECTION ENGINE
# =============================================================================


class DuplicateDetectionEngine:
    """Advanced duplicate detection engine with multiple algorithms."""

    def __init__(self, config: Optional[DuplicateDetectionConfig] = None):
        """
        Initialize duplicate detection engine.

        Args:
            config: Detection configuration
        """
        self.config = config or DuplicateDetectionConfig()
        self.db_handler = None

        if DATABASE_AVAILABLE:
            self.db_handler = EnhancedDatabaseHandler()

        # Cache for performance
        self._hash_cache: Dict[str, str] = {}
        self._similarity_cache: Dict[Tuple[str, str], float] = {}

    def normalize_content(self, content: str) -> str:
        """
        Normalize content for better duplicate detection.

        Args:
            content: Content to normalize

        Returns:
            Normalized content
        """
        if not self.config.enable_content_normalization:
            return content

        # Convert to lowercase
        normalized = content.lower()

        # Remove extra whitespace
        normalized = " ".join(normalized.split())

        # Remove common punctuation
        import re

        normalized = re.sub(r"[^\w\s]", " ", normalized)
        normalized = " ".join(normalized.split())

        return normalized

    def calculate_content_hash(self, content: str, method: str = "sha256") -> str:
        """
        Calculate hash of content.

        Args:
            content: Content to hash
            method: Hash method (sha256, md5)

        Returns:
            Content hash
        """
        # Check cache first
        cache_key = f"{method}:{content[:100]}"
        if cache_key in self._hash_cache:
            return self._hash_cache[cache_key]

        # Normalize content if enabled
        if self.config.enable_content_normalization:
            content = self.normalize_content(content)

        # Calculate hash
        if method == "sha256":
            hash_value = hashlib.sha256(content.encode("utf-8")).hexdigest()
        elif method == "md5":
            hash_value = hashlib.md5(content.encode("utf-8")).hexdigest()
        else:
            raise ValueError(f"Unsupported hash method: {method}")

        # Cache result
        self._hash_cache[cache_key] = hash_value

        return hash_value

    def calculate_jaccard_similarity(self, content1: str, content2: str) -> float:
        """
        Calculate Jaccard similarity between two content strings.

        Args:
            content1: First content
            content2: Second content

        Returns:
            Jaccard similarity score (0.0 to 1.0)
        """
        # Check cache first
        cache_key = (content1[:50], content2[:50])
        if cache_key in self._similarity_cache:
            return self._similarity_cache[cache_key]

        # Tokenize content
        tokens1 = set(self.normalize_content(content1).split())
        tokens2 = set(self.normalize_content(content2).split())

        # Calculate Jaccard similarity
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))

        similarity = intersection / union if union > 0 else 0.0

        # Cache result
        self._similarity_cache[cache_key] = similarity

        return similarity

    def calculate_levenshtein_ratio(self, content1: str, content2: str) -> float:
        """
        Calculate Levenshtein distance ratio between two strings.

        Args:
            content1: First content
            content2: Second content

        Returns:
            Similarity ratio (0.0 to 1.0)
        """
        if len(content1) == 0 and len(content2) == 0:
            return 1.0

        if len(content1) == 0 or len(content2) == 0:
            return 0.0

        # For performance, limit content length
        max_len = 1000
        c1 = content1[:max_len] if len(content1) > max_len else content1
        c2 = content2[:max_len] if len(content2) > max_len else content2

        # Calculate Levenshtein distance
        def levenshtein_distance(s1: str, s2: str) -> int:
            if len(s1) < len(s2):
                return levenshtein_distance(s2, s1)

            if len(s2) == 0:
                return len(s1)

            previous_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row

            return previous_row[-1]

        distance = levenshtein_distance(c1, c2)
        max_len = max(len(c1), len(c2))

        return 1.0 - (distance / max_len) if max_len > 0 else 0.0

    def detect_exact_duplicates(
        self, content_items: List[Tuple[str, str]]
    ) -> List[DuplicateMatch]:
        """
        Detect exact duplicates using content hashing.

        Args:
            content_items: List of (id, content) tuples

        Returns:
            List of duplicate matches
        """
        if not self.config.enable_hash_detection:
            return []

        duplicates = []
        hash_to_id = {}

        for doc_id, content in content_items:
            if len(content) < self.config.min_content_length:
                continue

            content_hash = self.calculate_content_hash(content)

            if content_hash in hash_to_id:
                # Found duplicate
                original_id = hash_to_id[content_hash]
                duplicate = DuplicateMatch(
                    original_id=original_id,
                    duplicate_id=doc_id,
                    similarity_score=1.0,
                    detection_method="exact_hash",
                    confidence=1.0,
                    metadata={"content_hash": content_hash},
                )
                duplicates.append(duplicate)
            else:
                hash_to_id[content_hash] = doc_id

        return duplicates

    def detect_similarity_duplicates(
        self,
        content_items: List[Tuple[str, str]],
        embeddings: Optional[Dict[str, List[float]]] = None,
    ) -> List[DuplicateMatch]:
        """
        Detect near-duplicates using similarity metrics.

        Args:
            content_items: List of (id, content) tuples
            embeddings: Optional embeddings for vector similarity

        Returns:
            List of duplicate matches
        """
        if (
            not self.config.enable_vector_similarity
            and not self.config.enable_fuzzy_matching
        ):
            return []

        duplicates = []
        processed_pairs = set()

        for i, (id1, content1) in enumerate(content_items):
            if len(content1) < self.config.min_content_length:
                continue

            for j, (id2, content2) in enumerate(content_items[i + 1 :], start=i + 1):
                if len(content2) < self.config.min_content_length:
                    continue

                # Avoid duplicate processing
                pair = tuple(sorted([id1, id2]))
                if pair in processed_pairs:
                    continue
                processed_pairs.add(pair)

                max_similarity = 0.0
                best_method = "none"

                # Vector similarity if embeddings available
                if (
                    self.config.enable_vector_similarity
                    and embeddings
                    and id1 in embeddings
                    and id2 in embeddings
                ):

                    vec1 = np.array(embeddings[id1])
                    vec2 = np.array(embeddings[id2])

                    # Cosine similarity
                    dot_product = np.dot(vec1, vec2)
                    norm1 = np.linalg.norm(vec1)
                    norm2 = np.linalg.norm(vec2)

                    if norm1 > 0 and norm2 > 0:
                        vector_similarity = dot_product / (norm1 * norm2)
                        if vector_similarity > max_similarity:
                            max_similarity = vector_similarity
                            best_method = "vector_similarity"

                # Fuzzy text matching
                if self.config.enable_fuzzy_matching:
                    # Jaccard similarity
                    jaccard_sim = self.calculate_jaccard_similarity(content1, content2)
                    if jaccard_sim > max_similarity:
                        max_similarity = jaccard_sim
                        best_method = "jaccard_similarity"

                    # Levenshtein ratio (for shorter texts)
                    if len(content1) < 2000 and len(content2) < 2000:
                        levenshtein_sim = self.calculate_levenshtein_ratio(
                            content1, content2
                        )
                        if levenshtein_sim > max_similarity:
                            max_similarity = levenshtein_sim
                            best_method = "levenshtein_similarity"

                # Check if similarity exceeds threshold
                threshold = (
                    self.config.similarity_threshold
                    if best_method == "vector_similarity"
                    else self.config.fuzzy_threshold
                )

                if max_similarity >= threshold:
                    duplicate = DuplicateMatch(
                        original_id=id1,
                        duplicate_id=id2,
                        similarity_score=max_similarity,
                        detection_method=best_method,
                        confidence=max_similarity,
                        metadata={
                            "threshold": threshold,
                            "content_lengths": [len(content1), len(content2)],
                        },
                    )
                    duplicates.append(duplicate)

        return duplicates

    def check_document_duplicate(self, content_hash: str, content: str) -> bool:
        """
        Check if a document is a duplicate of existing content.

        Args:
            content_hash: Hash of the document content
            content: Document content

        Returns:
            True if duplicate detected
        """
        if not self.db_handler:
            return False

        try:
            # Check exact hash match first
            if self.config.enable_hash_detection:
                hash_match = (
                    self.db_handler.supabase.table("documents")
                    .select("id")
                    .eq("metadata->>content_hash", content_hash)
                    .limit(1)
                    .execute()
                )

                if hash_match.data:
                    return True

            # Check similarity if content is substantial
            if (
                self.config.enable_vector_similarity
                and len(content) >= self.config.min_content_length
            ):

                # This would require embeddings - simplified for now
                # In production, you'd get embeddings and use vector similarity
                pass

            return False

        except Exception as e:
            print(f"Error checking document duplicate: {e}")
            return False

    def deduplicate_content_batch(
        self,
        content_items: List[Tuple[str, str]],
        embeddings: Optional[Dict[str, List[float]]] = None,
    ) -> DeduplicationResult:
        """
        Perform deduplication on a batch of content items.

        Args:
            content_items: List of (id, content) tuples
            embeddings: Optional embeddings for similarity detection

        Returns:
            Deduplication result
        """
        start_time = datetime.now()

        # Detect exact duplicates
        exact_duplicates = self.detect_exact_duplicates(content_items)

        # Detect similarity duplicates
        similarity_duplicates = self.detect_similarity_duplicates(
            content_items, embeddings
        )

        # Combine and organize duplicates
        all_duplicates = exact_duplicates + similarity_duplicates

        # Group duplicates
        duplicate_groups = []
        duplicated_ids = set()

        # Create groups based on matches
        for duplicate in all_duplicates:
            original_id = duplicate.original_id
            duplicate_id = duplicate.duplicate_id

            # Find existing group or create new one
            found_group = None
            for group in duplicate_groups:
                if original_id in group or duplicate_id in group:
                    found_group = group
                    break

            if found_group:
                found_group.add(original_id)
                found_group.add(duplicate_id)
            else:
                duplicate_groups.append({original_id, duplicate_id})

            duplicated_ids.add(duplicate_id)

        # Convert sets to lists
        duplicate_group_lists = [list(group) for group in duplicate_groups]

        # Calculate statistics
        total_documents = len(content_items)
        unique_documents = total_documents - len(duplicated_ids)
        duplicates_removed = len(duplicated_ids)

        processing_time = (datetime.now() - start_time).total_seconds()

        # Estimate cost savings (rough calculation)
        cost_per_document = Decimal("0.001")  # Placeholder
        cost_savings = Decimal(duplicates_removed) * cost_per_document

        return DeduplicationResult(
            total_documents=total_documents,
            unique_documents=unique_documents,
            duplicate_groups=duplicate_group_lists,
            duplicates_removed=duplicates_removed,
            processing_time=processing_time,
            cost_savings=cost_savings,
        )

    def get_duplicate_statistics(self) -> Dict[str, Any]:
        """Get duplicate detection statistics."""
        return {
            "config": self.config.dict(),
            "cache_sizes": {
                "hash_cache": len(self._hash_cache),
                "similarity_cache": len(self._similarity_cache),
            },
            "detection_methods": {
                "hash_detection": self.config.enable_hash_detection,
                "vector_similarity": self.config.enable_vector_similarity,
                "fuzzy_matching": self.config.enable_fuzzy_matching,
            },
            "thresholds": {
                "similarity_threshold": self.config.similarity_threshold,
                "fuzzy_threshold": self.config.fuzzy_threshold,
                "min_content_length": self.config.min_content_length,
            },
        }

    def clear_caches(self) -> None:
        """Clear internal caches."""
        self._hash_cache.clear()
        self._similarity_cache.clear()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Global detector instance
_default_detector = None


def get_default_detector() -> DuplicateDetectionEngine:
    """Get the default duplicate detector instance."""
    global _default_detector
    if _default_detector is None:
        _default_detector = DuplicateDetectionEngine()
    return _default_detector


def detect_duplicates(
    content_items: List[Tuple[str, str]],
    embeddings: Optional[Dict[str, List[float]]] = None,
) -> DeduplicationResult:
    """Detect duplicates in content items using default detector."""
    detector = get_default_detector()
    return detector.deduplicate_content_batch(content_items, embeddings)


def check_duplicate(content_hash: str, content: str) -> bool:
    """Check if content is duplicate using default detector."""
    detector = get_default_detector()
    return detector.check_document_duplicate(content_hash, content)
