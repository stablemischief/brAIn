"""
Document suggestion system for related content discovery.

This module provides intelligent document recommendations based on content similarity,
user behavior, search context, and collaborative filtering to help users discover
relevant related documents.
"""

import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter
from uuid import UUID
import asyncio
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .hybrid_search import SearchResult
from .context_ranking import UserContext, DocumentMetrics
from .search_history import SearchQuery, SearchInteraction, InteractionType


class SuggestionType(Enum):
    """Types of document suggestions"""
    CONTENT_SIMILAR = "content_similar"
    USER_BEHAVIOR = "user_behavior"
    COLLABORATIVE = "collaborative"
    TRENDING = "trending"
    FOLLOW_UP = "follow_up"
    TOPIC_RELATED = "topic_related"
    AUTHOR_RELATED = "author_related"


class SuggestionReason(Enum):
    """Reasons for suggesting a document"""
    SIMILAR_CONTENT = "Users who viewed this also found similar content helpful"
    USERS_ALSO_VIEWED = "Users who viewed this also viewed"
    TRENDING_NOW = "Trending in your areas of interest"
    RELATED_TOPIC = "Related to your current search topic"
    SAME_AUTHOR = "More from the same author"
    FOLLOW_UP_READING = "Recommended follow-up reading"
    BASED_ON_HISTORY = "Based on your search history"


@dataclass
class DocumentSuggestion:
    """A single document suggestion with context"""
    document_id: str
    title: str
    content_preview: str
    suggestion_type: SuggestionType
    reason: SuggestionReason
    confidence_score: float
    relevance_score: float
    
    # Additional context
    metadata: Dict[str, Any] = field(default_factory=dict)
    similar_to: Optional[str] = None  # Document ID this is similar to
    explanation: str = ""
    
    # Ranking factors
    content_similarity: float = 0.0
    user_interest_score: float = 0.0
    popularity_score: float = 0.0
    recency_score: float = 0.0


@dataclass
class SuggestionConfig:
    """Configuration for document suggestions"""
    # Number of suggestions per type
    max_content_similar: int = 5
    max_collaborative: int = 3
    max_trending: int = 3
    max_follow_up: int = 2
    
    # Similarity thresholds
    content_similarity_threshold: float = 0.3
    user_behavior_threshold: float = 0.2
    
    # Time windows
    trending_window: timedelta = timedelta(days=7)
    recent_interaction_window: timedelta = timedelta(days=30)
    
    # Scoring weights
    content_weight: float = 0.4
    popularity_weight: float = 0.3
    recency_weight: float = 0.2
    personalization_weight: float = 0.1
    
    # Filtering
    min_confidence_score: float = 0.2
    exclude_already_viewed: bool = True
    diversify_suggestions: bool = True


class DocumentSuggestionEngine:
    """
    Advanced document suggestion engine that provides personalized
    recommendations using multiple recommendation strategies.
    """
    
    def __init__(self, config: Optional[SuggestionConfig] = None):
        self.config = config or SuggestionConfig()
        
        # Caches and indices
        self._document_cache: Dict[str, Dict[str, Any]] = {}
        self._content_vectors: Dict[str, np.ndarray] = {}
        self._user_document_matrix: Dict[str, Dict[str, float]] = {}
        self._document_similarity_cache: Dict[Tuple[str, str], float] = {}
        
        # TF-IDF vectorizer for content similarity
        self._tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2),
            max_df=0.95,
            min_df=0.02
        )
        
        # Statistics tracking
        self._document_stats: Dict[str, DocumentMetrics] = {}
        self._trending_cache: Dict[str, List[str]] = {}
        self._last_trending_update: Optional[datetime] = None
    
    async def get_suggestions(
        self,
        current_document_id: str,
        user_context: Optional[UserContext] = None,
        search_history: Optional[List[SearchQuery]] = None,
        exclude_documents: Optional[Set[str]] = None
    ) -> List[DocumentSuggestion]:
        """
        Get comprehensive document suggestions for a user.
        
        Args:
            current_document_id: The document user is currently viewing
            user_context: User context for personalization
            search_history: Recent search history
            exclude_documents: Documents to exclude from suggestions
            
        Returns:
            List of ranked document suggestions
        """
        exclude_documents = exclude_documents or set()
        exclude_documents.add(current_document_id)  # Don't suggest current document
        
        all_suggestions = []
        
        # 1. Content-based similar documents
        content_suggestions = await self._get_content_similar_suggestions(
            current_document_id, exclude_documents
        )
        all_suggestions.extend(content_suggestions)
        
        # 2. Collaborative filtering suggestions
        if user_context:
            collab_suggestions = await self._get_collaborative_suggestions(
                current_document_id, user_context, exclude_documents
            )
            all_suggestions.extend(collab_suggestions)
        
        # 3. Trending documents
        trending_suggestions = await self._get_trending_suggestions(
            current_document_id, user_context, exclude_documents
        )
        all_suggestions.extend(trending_suggestions)
        
        # 4. Follow-up reading suggestions
        if search_history:
            followup_suggestions = await self._get_followup_suggestions(
                current_document_id, search_history, exclude_documents
            )
            all_suggestions.extend(followup_suggestions)
        
        # 5. Topic-based suggestions
        topic_suggestions = await self._get_topic_based_suggestions(
            current_document_id, user_context, exclude_documents
        )
        all_suggestions.extend(topic_suggestions)
        
        # 6. Author-based suggestions
        author_suggestions = await self._get_author_based_suggestions(
            current_document_id, exclude_documents
        )
        all_suggestions.extend(author_suggestions)
        
        # Filter and rank suggestions
        filtered_suggestions = self._filter_suggestions(all_suggestions)
        ranked_suggestions = self._rank_suggestions(filtered_suggestions, user_context)
        
        # Diversify if configured
        if self.config.diversify_suggestions:
            ranked_suggestions = self._diversify_suggestions(ranked_suggestions)
        
        return ranked_suggestions[:20]  # Return top 20 suggestions
    
    async def _get_content_similar_suggestions(
        self,
        document_id: str,
        exclude_documents: Set[str]
    ) -> List[DocumentSuggestion]:
        """Find documents with similar content"""
        suggestions = []
        
        # Get document content
        document = await self._get_document(document_id)
        if not document:
            return suggestions
        
        # Calculate content similarities
        similar_docs = await self._find_similar_documents(
            document_id, 
            self.config.max_content_similar * 2  # Get extra to account for filtering
        )
        
        for similar_doc_id, similarity_score in similar_docs:
            if similar_doc_id in exclude_documents:
                continue
            
            similar_document = await self._get_document(similar_doc_id)
            if not similar_document:
                continue
            
            suggestion = DocumentSuggestion(
                document_id=similar_doc_id,
                title=similar_document.get('title', 'Untitled'),
                content_preview=self._generate_preview(similar_document.get('content', '')),
                suggestion_type=SuggestionType.CONTENT_SIMILAR,
                reason=SuggestionReason.SIMILAR_CONTENT,
                confidence_score=similarity_score,
                relevance_score=similarity_score,
                similar_to=document_id,
                content_similarity=similarity_score,
                explanation=f"Similar content to '{document.get('title', 'current document')}'"
            )
            
            suggestions.append(suggestion)
            
            if len(suggestions) >= self.config.max_content_similar:
                break
        
        return suggestions
    
    async def _get_collaborative_suggestions(
        self,
        document_id: str,
        user_context: UserContext,
        exclude_documents: Set[str]
    ) -> List[DocumentSuggestion]:
        """Find documents using collaborative filtering"""
        suggestions = []
        
        if not user_context.user_id:
            return suggestions
        
        # Find users with similar behavior
        similar_users = await self._find_similar_users(user_context.user_id)
        
        # Get documents these similar users liked
        candidate_docs = defaultdict(float)
        
        for similar_user_id, similarity_score in similar_users[:10]:
            user_docs = await self._get_user_document_interactions(similar_user_id)
            
            for doc_id, interaction_score in user_docs.items():
                if doc_id in exclude_documents:
                    continue
                
                # Weight by user similarity and interaction strength
                weighted_score = similarity_score * interaction_score
                candidate_docs[doc_id] += weighted_score
        
        # Sort by collaborative score
        sorted_docs = sorted(candidate_docs.items(), key=lambda x: x[1], reverse=True)
        
        for doc_id, collab_score in sorted_docs[:self.config.max_collaborative]:
            document = await self._get_document(doc_id)
            if not document:
                continue
            
            suggestion = DocumentSuggestion(
                document_id=doc_id,
                title=document.get('title', 'Untitled'),
                content_preview=self._generate_preview(document.get('content', '')),
                suggestion_type=SuggestionType.COLLABORATIVE,
                reason=SuggestionReason.USERS_ALSO_VIEWED,
                confidence_score=min(1.0, collab_score),
                relevance_score=min(1.0, collab_score),
                user_interest_score=collab_score,
                explanation="Users with similar interests also viewed this"
            )
            
            suggestions.append(suggestion)
        
        return suggestions
    
    async def _get_trending_suggestions(
        self,
        document_id: str,
        user_context: Optional[UserContext],
        exclude_documents: Set[str]
    ) -> List[DocumentSuggestion]:
        """Find trending documents"""
        suggestions = []
        
        # Update trending cache if needed
        await self._update_trending_cache()
        
        # Get current document topics for contextualized trending
        current_doc = await self._get_document(document_id)
        current_topics = current_doc.get('topics', []) if current_doc else []
        
        # Get trending documents, prioritizing those in similar topics
        trending_docs = self._trending_cache.get('global', [])
        
        topic_trending = []
        if current_topics:
            for topic in current_topics:
                topic_trending.extend(self._trending_cache.get(topic, []))
        
        # Combine and prioritize topic-specific trending
        combined_trending = topic_trending + trending_docs
        
        seen = set()
        for doc_id in combined_trending:
            if doc_id in exclude_documents or doc_id in seen:
                continue
            
            seen.add(doc_id)
            document = await self._get_document(doc_id)
            if not document:
                continue
            
            # Calculate trending score based on recent activity
            trending_score = await self._calculate_trending_score(doc_id)
            
            suggestion = DocumentSuggestion(
                document_id=doc_id,
                title=document.get('title', 'Untitled'),
                content_preview=self._generate_preview(document.get('content', '')),
                suggestion_type=SuggestionType.TRENDING,
                reason=SuggestionReason.TRENDING_NOW,
                confidence_score=trending_score,
                relevance_score=trending_score,
                popularity_score=trending_score,
                explanation="Trending in your areas of interest"
            )
            
            suggestions.append(suggestion)
            
            if len(suggestions) >= self.config.max_trending:
                break
        
        return suggestions
    
    async def _get_followup_suggestions(
        self,
        document_id: str,
        search_history: List[SearchQuery],
        exclude_documents: Set[str]
    ) -> List[DocumentSuggestion]:
        """Find follow-up reading suggestions based on search history"""
        suggestions = []
        
        # Analyze search progression to understand information needs
        recent_queries = search_history[-10:]  # Last 10 queries
        
        # Extract topics and intent from recent searches
        search_topics = []
        search_keywords = []
        
        for query in recent_queries:
            search_keywords.extend(query.query_text.lower().split())
            # Extract topics from query analysis if available
            if query.query_analysis and hasattr(query.query_analysis, 'topics'):
                search_topics.extend(query.query_analysis.topics)
        
        # Find documents that address follow-up topics
        followup_candidates = await self._find_followup_documents(
            search_keywords, search_topics, exclude_documents
        )
        
        for doc_id, followup_score in followup_candidates[:self.config.max_follow_up]:
            document = await self._get_document(doc_id)
            if not document:
                continue
            
            suggestion = DocumentSuggestion(
                document_id=doc_id,
                title=document.get('title', 'Untitled'),
                content_preview=self._generate_preview(document.get('content', '')),
                suggestion_type=SuggestionType.FOLLOW_UP,
                reason=SuggestionReason.FOLLOW_UP_READING,
                confidence_score=followup_score,
                relevance_score=followup_score,
                explanation="Recommended based on your recent searches"
            )
            
            suggestions.append(suggestion)
        
        return suggestions
    
    async def _get_topic_based_suggestions(
        self,
        document_id: str,
        user_context: Optional[UserContext],
        exclude_documents: Set[str]
    ) -> List[DocumentSuggestion]:
        """Find documents related to similar topics"""
        suggestions = []
        
        current_doc = await self._get_document(document_id)
        if not current_doc:
            return suggestions
        
        current_topics = current_doc.get('topics', [])
        if not current_topics:
            return suggestions
        
        # Find documents with overlapping topics
        topic_docs = await self._find_documents_by_topics(current_topics, exclude_documents)
        
        for doc_id, topic_overlap_score in topic_docs[:5]:
            document = await self._get_document(doc_id)
            if not document:
                continue
            
            suggestion = DocumentSuggestion(
                document_id=doc_id,
                title=document.get('title', 'Untitled'),
                content_preview=self._generate_preview(document.get('content', '')),
                suggestion_type=SuggestionType.TOPIC_RELATED,
                reason=SuggestionReason.RELATED_TOPIC,
                confidence_score=topic_overlap_score,
                relevance_score=topic_overlap_score,
                explanation=f"Related topics: {', '.join(current_topics[:3])}"
            )
            
            suggestions.append(suggestion)
        
        return suggestions
    
    async def _get_author_based_suggestions(
        self,
        document_id: str,
        exclude_documents: Set[str]
    ) -> List[DocumentSuggestion]:
        """Find more documents from the same author"""
        suggestions = []
        
        current_doc = await self._get_document(document_id)
        if not current_doc:
            return suggestions
        
        author = current_doc.get('author')
        if not author:
            return suggestions
        
        # Find other documents by the same author
        author_docs = await self._find_documents_by_author(author, exclude_documents)
        
        for doc_id in author_docs[:3]:  # Limit author suggestions
            document = await self._get_document(doc_id)
            if not document:
                continue
            
            suggestion = DocumentSuggestion(
                document_id=doc_id,
                title=document.get('title', 'Untitled'),
                content_preview=self._generate_preview(document.get('content', '')),
                suggestion_type=SuggestionType.AUTHOR_RELATED,
                reason=SuggestionReason.SAME_AUTHOR,
                confidence_score=0.7,  # Moderate confidence for author-based
                relevance_score=0.6,
                explanation=f"More from {author}"
            )
            
            suggestions.append(suggestion)
        
        return suggestions
    
    async def _find_similar_documents(
        self,
        document_id: str,
        limit: int = 10
    ) -> List[Tuple[str, float]]:
        """Find documents similar to the given document using content similarity"""
        # This is a simplified version - in reality, you'd use proper vector similarity
        similar_docs = []
        
        # For demonstration, return mock similar documents
        # In a real implementation, this would:
        # 1. Get document embedding/TF-IDF vector
        # 2. Calculate cosine similarity with all other documents
        # 3. Return top K most similar
        
        # Mock implementation
        all_doc_ids = list(self._document_cache.keys())
        for other_id in all_doc_ids:
            if other_id != document_id:
                # Mock similarity score
                similarity = 0.8 - (abs(hash(document_id) - hash(other_id)) % 100) / 200
                if similarity > self.config.content_similarity_threshold:
                    similar_docs.append((other_id, similarity))
        
        # Sort by similarity
        similar_docs.sort(key=lambda x: x[1], reverse=True)
        
        return similar_docs[:limit]
    
    async def _find_similar_users(self, user_id: str) -> List[Tuple[str, float]]:
        """Find users with similar document interaction patterns"""
        # Mock implementation - in reality, this would use collaborative filtering
        similar_users = [
            ("user2", 0.8),
            ("user3", 0.7),
            ("user4", 0.6)
        ]
        return similar_users
    
    async def _get_user_document_interactions(self, user_id: str) -> Dict[str, float]:
        """Get user's document interaction scores"""
        # Mock implementation
        interactions = {
            "doc1": 0.9,
            "doc2": 0.8,
            "doc3": 0.7
        }
        return interactions
    
    async def _update_trending_cache(self):
        """Update the trending documents cache"""
        now = datetime.utcnow()
        
        # Update only if cache is stale
        if (self._last_trending_update and 
            now - self._last_trending_update < timedelta(hours=1)):
            return
        
        # Mock trending calculation
        # In reality, this would analyze recent document interactions
        trending_docs = ["doc1", "doc2", "doc3", "doc4", "doc5"]
        
        self._trending_cache['global'] = trending_docs
        self._last_trending_update = now
    
    async def _calculate_trending_score(self, document_id: str) -> float:
        """Calculate trending score for a document"""
        # Mock implementation - would calculate based on recent activity
        return 0.7
    
    async def _find_followup_documents(
        self,
        keywords: List[str],
        topics: List[str],
        exclude_documents: Set[str]
    ) -> List[Tuple[str, float]]:
        """Find documents that serve as good follow-up reading"""
        # Mock implementation
        followup_docs = [("doc1", 0.8), ("doc2", 0.7)]
        return [(doc_id, score) for doc_id, score in followup_docs 
                if doc_id not in exclude_documents]
    
    async def _find_documents_by_topics(
        self,
        topics: List[str],
        exclude_documents: Set[str]
    ) -> List[Tuple[str, float]]:
        """Find documents with overlapping topics"""
        # Mock implementation
        topic_docs = [("doc1", 0.9), ("doc2", 0.8)]
        return [(doc_id, score) for doc_id, score in topic_docs 
                if doc_id not in exclude_documents]
    
    async def _find_documents_by_author(
        self,
        author: str,
        exclude_documents: Set[str]
    ) -> List[str]:
        """Find documents by the same author"""
        # Mock implementation
        author_docs = ["doc1", "doc2", "doc3"]
        return [doc_id for doc_id in author_docs if doc_id not in exclude_documents]
    
    async def _get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get document metadata and content"""
        # Mock document cache - in reality, this would query the database
        if document_id not in self._document_cache:
            # Mock document data
            self._document_cache[document_id] = {
                'title': f'Document {document_id}',
                'content': f'Content for document {document_id}',
                'author': 'John Doe',
                'topics': ['AI', 'Machine Learning'],
                'created_at': datetime.utcnow().isoformat()
            }
        
        return self._document_cache.get(document_id)
    
    def _generate_preview(self, content: str, max_length: int = 200) -> str:
        """Generate a preview of document content"""
        if len(content) <= max_length:
            return content
        
        # Find the last complete sentence within the limit
        preview = content[:max_length]
        last_sentence = preview.rfind('.')
        
        if last_sentence > max_length // 2:  # If we found a reasonable sentence break
            return preview[:last_sentence + 1]
        else:
            return preview + "..."
    
    def _filter_suggestions(self, suggestions: List[DocumentSuggestion]) -> List[DocumentSuggestion]:
        """Filter suggestions based on quality thresholds"""
        return [
            s for s in suggestions 
            if s.confidence_score >= self.config.min_confidence_score
        ]
    
    def _rank_suggestions(
        self,
        suggestions: List[DocumentSuggestion],
        user_context: Optional[UserContext]
    ) -> List[DocumentSuggestion]:
        """Rank suggestions using composite scoring"""
        for suggestion in suggestions:
            # Calculate composite score
            score = (
                suggestion.content_similarity * self.config.content_weight +
                suggestion.popularity_score * self.config.popularity_weight +
                suggestion.recency_score * self.config.recency_weight +
                suggestion.user_interest_score * self.config.personalization_weight
            )
            
            # Update relevance score
            suggestion.relevance_score = min(1.0, score)
        
        # Sort by relevance score
        suggestions.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return suggestions
    
    def _diversify_suggestions(
        self,
        suggestions: List[DocumentSuggestion]
    ) -> List[DocumentSuggestion]:
        """Diversify suggestions to avoid redundancy"""
        diversified = []
        seen_types = set()
        type_counts = defaultdict(int)
        
        # Ensure diversity across suggestion types
        for suggestion in suggestions:
            suggestion_type = suggestion.suggestion_type
            
            # Limit each type to avoid domination
            type_limit = {
                SuggestionType.CONTENT_SIMILAR: 3,
                SuggestionType.COLLABORATIVE: 2,
                SuggestionType.TRENDING: 2,
                SuggestionType.FOLLOW_UP: 2,
                SuggestionType.TOPIC_RELATED: 2,
                SuggestionType.AUTHOR_RELATED: 1
            }.get(suggestion_type, 2)
            
            if type_counts[suggestion_type] < type_limit:
                diversified.append(suggestion)
                type_counts[suggestion_type] += 1
        
        return diversified
    
    async def track_suggestion_interaction(
        self,
        suggestion_id: str,
        document_id: str,
        user_id: str,
        interaction_type: InteractionType,
        suggestion_type: SuggestionType
    ):
        """Track user interactions with suggestions for learning"""
        # In a real implementation, this would:
        # 1. Store the interaction in the database
        # 2. Update suggestion quality metrics
        # 3. Improve future suggestion algorithms
        
        print(f"Tracked suggestion interaction: {user_id} {interaction_type.value} on {document_id} via {suggestion_type.value}")
    
    def get_suggestion_explanation(
        self,
        suggestion: DocumentSuggestion
    ) -> Dict[str, Any]:
        """Get detailed explanation for why a document was suggested"""
        return {
            "suggestion_id": suggestion.document_id,
            "type": suggestion.suggestion_type.value,
            "reason": suggestion.reason.value,
            "confidence": suggestion.confidence_score,
            "factors": {
                "content_similarity": suggestion.content_similarity,
                "user_interest": suggestion.user_interest_score,
                "popularity": suggestion.popularity_score,
                "recency": suggestion.recency_score
            },
            "explanation": suggestion.explanation
        }