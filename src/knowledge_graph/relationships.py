"""
Relationship detection module for knowledge graph construction.
Detects and scores relationships between entities using multiple approaches.
"""

import logging
import re
from typing import Dict, List, Set, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import asyncio
from collections import defaultdict, Counter
import math

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import openai
from pydantic import BaseModel, Field

from .entities import ExtractedEntity, EntityType, EntityMention

logger = logging.getLogger(__name__)


class RelationshipType(str, Enum):
    """Types of relationships between entities"""
    MENTIONS = "mentions"
    RELATED_TO = "related_to"
    CONTAINS = "contains"
    PART_OF = "part_of"
    CREATED_BY = "created_by"
    ASSOCIATED_WITH = "associated_with"
    LOCATED_IN = "located_in"
    OCCURRED_ON = "occurred_on"
    DEPENDS_ON = "depends_on"
    INFLUENCES = "influences"
    SIMILAR_TO = "similar_to"
    DERIVED_FROM = "derived_from"
    CUSTOM = "custom"


class RelationshipDetectionMethod(str, Enum):
    """Methods for detecting relationships"""
    COOCCURRENCE = "cooccurrence"
    CONTEXT_SIMILARITY = "context_similarity"
    PATTERN_MATCHING = "pattern_matching"
    EMBEDDING_SIMILARITY = "embedding_similarity"
    DEPENDENCY_PARSING = "dependency_parsing"
    AI_INFERENCE = "ai_inference"


@dataclass
class RelationshipEvidence:
    """Evidence supporting a relationship"""
    text_snippet: str
    start_pos: int
    end_pos: int
    confidence: float
    method: RelationshipDetectionMethod
    properties: Dict = None

    def __post_init__(self):
        if self.properties is None:
            self.properties = {}


class DetectedRelationship(BaseModel):
    """Detected relationship between two entities"""
    source_entity: str = Field(description="Source entity name")
    target_entity: str = Field(description="Target entity name")
    relationship_type: RelationshipType = Field(description="Type of relationship")
    relationship_name: Optional[str] = Field(None, description="Specific relationship name")
    strength: float = Field(description="Relationship strength score (0-1)")
    confidence: float = Field(description="Detection confidence (0-1)")
    evidence: List[RelationshipEvidence] = Field(default_factory=list, description="Supporting evidence")
    description: Optional[str] = Field(None, description="Human-readable description")
    properties: Dict = Field(default_factory=dict, description="Additional properties")
    bidirectional: bool = Field(False, description="Whether relationship works both ways")


class RelationshipDetectionConfig(BaseModel):
    """Configuration for relationship detection"""
    
    # Detection methods
    enable_cooccurrence: bool = True
    enable_context_similarity: bool = True
    enable_pattern_matching: bool = True
    enable_embedding_similarity: bool = True
    enable_dependency_parsing: bool = False
    enable_ai_inference: bool = False
    
    # Cooccurrence settings
    cooccurrence_window: int = 100  # Character distance for cooccurrence
    min_cooccurrence_count: int = 2
    
    # Context similarity settings
    context_window: int = 200  # Characters around entity for context
    min_context_similarity: float = 0.3
    
    # Pattern matching
    enable_custom_patterns: bool = True
    
    # Embedding similarity
    min_embedding_similarity: float = 0.7
    embedding_model: str = "text-embedding-ada-002"
    
    # AI inference
    ai_model: str = "gpt-3.5-turbo"
    ai_max_tokens: int = 500
    
    # Filtering
    min_relationship_strength: float = 0.2
    min_confidence: float = 0.3
    max_relationships_per_entity: int = 50
    
    # Relationship type preferences
    preferred_types: List[RelationshipType] = Field(default_factory=lambda: [
        RelationshipType.RELATED_TO, RelationshipType.PART_OF, 
        RelationshipType.ASSOCIATED_WITH, RelationshipType.MENTIONS
    ])


class RelationshipDetector:
    """Advanced relationship detection system"""
    
    def __init__(self, config: RelationshipDetectionConfig = None):
        self.config = config or RelationshipDetectionConfig()
        
        # Relationship pattern rules
        self.relationship_patterns = {
            RelationshipType.PART_OF: [
                r'(\w+)\s+(?:is\s+)?(?:a\s+)?part\s+of\s+(\w+)',
                r'(\w+)\s+(?:belongs\s+to|is\s+in)\s+(\w+)',
                r'(\w+)\s+component\s+of\s+(\w+)',
            ],
            RelationshipType.CREATED_BY: [
                r'(\w+)\s+(?:created|developed|built|designed)\s+by\s+(\w+)',
                r'(\w+)\s+is\s+the\s+creator\s+of\s+(\w+)',
                r'(\w+)\s+founded\s+(\w+)',
            ],
            RelationshipType.LOCATED_IN: [
                r'(\w+)\s+(?:located|situated|based)\s+in\s+(\w+)',
                r'(\w+)\s+(?:at|in)\s+(\w+)',
            ],
            RelationshipType.DEPENDS_ON: [
                r'(\w+)\s+(?:depends\s+on|relies\s+on|requires)\s+(\w+)',
                r'(\w+)\s+needs\s+(\w+)',
            ],
            RelationshipType.INFLUENCES: [
                r'(\w+)\s+(?:influences|affects|impacts)\s+(\w+)',
                r'(\w+)\s+is\s+influenced\s+by\s+(\w+)',
            ]
        }
        
        # Initialize caches
        self._embedding_cache = {}
        self._tfidf_vectorizer = None
    
    async def detect_relationships(
        self, 
        entities: List[ExtractedEntity], 
        text: str,
        document_id: Optional[str] = None
    ) -> List[DetectedRelationship]:
        """
        Detect relationships between entities in text.
        
        Args:
            entities: List of extracted entities
            text: Original text containing the entities
            document_id: Optional document identifier
            
        Returns:
            List of detected relationships
        """
        if len(entities) < 2:
            return []
        
        try:
            relationships = []
            
            # Generate entity pairs for analysis
            entity_pairs = []
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    entity_pairs.append((entities[i], entities[j]))
            
            logger.info(f"Analyzing {len(entity_pairs)} entity pairs for relationships")
            
            # Detect relationships using different methods
            for source_entity, target_entity in entity_pairs:
                pair_relationships = await self._detect_pair_relationships(
                    source_entity, target_entity, text
                )
                relationships.extend(pair_relationships)
            
            # Consolidate and filter relationships
            consolidated = await self._consolidate_relationships(relationships)
            filtered = self._filter_relationships(consolidated)
            
            logger.info(f"Detected {len(filtered)} relationships from {len(relationships)} candidates")
            return filtered
        
        except Exception as e:
            logger.error(f"Error detecting relationships: {e}")
            return []
    
    async def _detect_pair_relationships(
        self, 
        source: ExtractedEntity, 
        target: ExtractedEntity, 
        text: str
    ) -> List[DetectedRelationship]:
        """Detect relationships between a specific pair of entities"""
        relationships = []
        
        try:
            # Skip if entities are too similar (likely the same entity)
            if self._entities_too_similar(source, target):
                return relationships
            
            # Method 1: Cooccurrence analysis
            if self.config.enable_cooccurrence:
                cooc_rel = await self._detect_cooccurrence_relationship(source, target, text)
                if cooc_rel:
                    relationships.append(cooc_rel)
            
            # Method 2: Context similarity
            if self.config.enable_context_similarity:
                context_rel = await self._detect_context_similarity_relationship(source, target, text)
                if context_rel:
                    relationships.append(context_rel)
            
            # Method 3: Pattern matching
            if self.config.enable_pattern_matching:
                pattern_rels = await self._detect_pattern_relationships(source, target, text)
                relationships.extend(pattern_rels)
            
            # Method 4: Embedding similarity
            if self.config.enable_embedding_similarity:
                embed_rel = await self._detect_embedding_relationship(source, target, text)
                if embed_rel:
                    relationships.append(embed_rel)
            
            # Method 5: AI inference (if enabled)
            if self.config.enable_ai_inference:
                ai_rels = await self._detect_ai_relationships(source, target, text)
                relationships.extend(ai_rels)
        
        except Exception as e:
            logger.error(f"Error detecting pair relationships: {e}")
        
        return relationships
    
    async def _detect_cooccurrence_relationship(
        self, 
        source: ExtractedEntity, 
        target: ExtractedEntity, 
        text: str
    ) -> Optional[DetectedRelationship]:
        """Detect relationship based on entity cooccurrence"""
        
        # Find all mention pairs within the cooccurrence window
        cooccurrences = []
        
        for source_mention in source.mentions:
            for target_mention in target.mentions:
                distance = abs(source_mention.start_pos - target_mention.start_pos)
                
                if distance <= self.config.cooccurrence_window:
                    # Get context around both entities
                    start_pos = min(source_mention.start_pos, target_mention.start_pos)
                    end_pos = max(source_mention.end_pos, target_mention.end_pos)
                    context_start = max(0, start_pos - 50)
                    context_end = min(len(text), end_pos + 50)
                    context = text[context_start:context_end]
                    
                    cooccurrences.append({
                        'distance': distance,
                        'context': context,
                        'start_pos': start_pos,
                        'end_pos': end_pos
                    })
        
        if len(cooccurrences) < self.config.min_cooccurrence_count:
            return None
        
        # Calculate relationship strength based on cooccurrence frequency and proximity
        avg_distance = np.mean([c['distance'] for c in cooccurrences])
        frequency_score = min(len(cooccurrences) / 10.0, 1.0)  # Normalize by max expected frequency
        proximity_score = 1.0 - (avg_distance / self.config.cooccurrence_window)
        strength = (frequency_score + proximity_score) / 2.0
        
        # Create evidence
        evidence = []
        for cooc in cooccurrences[:5]:  # Limit evidence items
            evidence.append(RelationshipEvidence(
                text_snippet=cooc['context'],
                start_pos=cooc['start_pos'],
                end_pos=cooc['end_pos'],
                confidence=proximity_score,
                method=RelationshipDetectionMethod.COOCCURRENCE,
                properties={
                    'distance': cooc['distance'],
                    'cooccurrence_count': len(cooccurrences)
                }
            ))
        
        return DetectedRelationship(
            source_entity=source.name,
            target_entity=target.name,
            relationship_type=RelationshipType.MENTIONS,
            relationship_name="co_occurs_with",
            strength=strength,
            confidence=0.7,
            evidence=evidence,
            description=f"{source.name} frequently co-occurs with {target.name}",
            properties={
                'cooccurrence_count': len(cooccurrences),
                'avg_distance': avg_distance,
                'method': 'cooccurrence'
            },
            bidirectional=True
        )
    
    async def _detect_context_similarity_relationship(
        self, 
        source: ExtractedEntity, 
        target: ExtractedEntity, 
        text: str
    ) -> Optional[DetectedRelationship]:
        """Detect relationship based on context similarity"""
        
        # Extract contexts around entity mentions
        source_contexts = []
        target_contexts = []
        
        for mention in source.mentions:
            start = max(0, mention.start_pos - self.config.context_window // 2)
            end = min(len(text), mention.end_pos + self.config.context_window // 2)
            context = text[start:end].replace(mention.text, "[ENTITY]")
            source_contexts.append(context)
        
        for mention in target.mentions:
            start = max(0, mention.start_pos - self.config.context_window // 2)
            end = min(len(text), mention.end_pos + self.config.context_window // 2)
            context = text[start:end].replace(mention.text, "[ENTITY]")
            target_contexts.append(context)
        
        if not source_contexts or not target_contexts:
            return None
        
        # Calculate TF-IDF similarity between contexts
        all_contexts = source_contexts + target_contexts
        
        if self._tfidf_vectorizer is None:
            self._tfidf_vectorizer = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
        
        try:
            tfidf_matrix = self._tfidf_vectorizer.fit_transform(all_contexts)
            
            # Calculate similarity between source and target contexts
            source_vectors = tfidf_matrix[:len(source_contexts)]
            target_vectors = tfidf_matrix[len(source_contexts):]
            
            similarity_scores = []
            for i in range(source_vectors.shape[0]):
                for j in range(target_vectors.shape[0]):
                    sim = cosine_similarity(
                        source_vectors[i:i+1], 
                        target_vectors[j:j+1]
                    )[0, 0]
                    similarity_scores.append(sim)
            
            max_similarity = max(similarity_scores)
            avg_similarity = np.mean(similarity_scores)
            
            if max_similarity < self.config.min_context_similarity:
                return None
            
            # Create relationship
            evidence = [RelationshipEvidence(
                text_snippet=f"Context similarity: {max_similarity:.3f}",
                start_pos=0,
                end_pos=0,
                confidence=max_similarity,
                method=RelationshipDetectionMethod.CONTEXT_SIMILARITY,
                properties={
                    'max_similarity': max_similarity,
                    'avg_similarity': avg_similarity,
                    'context_comparisons': len(similarity_scores)
                }
            )]
            
            return DetectedRelationship(
                source_entity=source.name,
                target_entity=target.name,
                relationship_type=RelationshipType.SIMILAR_TO,
                relationship_name="similar_context",
                strength=avg_similarity,
                confidence=max_similarity,
                evidence=evidence,
                description=f"{source.name} appears in similar contexts to {target.name}",
                properties={
                    'context_similarity': max_similarity,
                    'method': 'context_similarity'
                },
                bidirectional=True
            )
        
        except Exception as e:
            logger.error(f"Error in context similarity calculation: {e}")
            return None
    
    async def _detect_pattern_relationships(
        self, 
        source: ExtractedEntity, 
        target: ExtractedEntity, 
        text: str
    ) -> List[DetectedRelationship]:
        """Detect relationships using predefined patterns"""
        relationships = []
        
        # Create entity name variations for pattern matching
        source_names = [source.name] + source.aliases
        target_names = [target.name] + target.aliases
        
        for rel_type, patterns in self.relationship_patterns.items():
            for pattern in patterns:
                for source_name in source_names:
                    for target_name in target_names:
                        # Try pattern with source->target
                        matches = list(re.finditer(
                            pattern.format(source_name, target_name), 
                            text, 
                            re.IGNORECASE
                        ))
                        
                        for match in matches:
                            evidence = [RelationshipEvidence(
                                text_snippet=match.group(),
                                start_pos=match.start(),
                                end_pos=match.end(),
                                confidence=0.9,  # High confidence for pattern matches
                                method=RelationshipDetectionMethod.PATTERN_MATCHING,
                                properties={
                                    'pattern': pattern,
                                    'match_groups': match.groups()
                                }
                            )]
                            
                            relationship = DetectedRelationship(
                                source_entity=source.name,
                                target_entity=target.name,
                                relationship_type=rel_type,
                                relationship_name=rel_type.value,
                                strength=0.8,
                                confidence=0.9,
                                evidence=evidence,
                                description=f"Pattern-based relationship: {match.group()}",
                                properties={
                                    'pattern_match': True,
                                    'method': 'pattern_matching',
                                    'pattern_type': rel_type.value
                                }
                            )
                            
                            relationships.append(relationship)
                        
                        # Try pattern with target->source (reverse)
                        matches = list(re.finditer(
                            pattern.format(target_name, source_name), 
                            text, 
                            re.IGNORECASE
                        ))
                        
                        for match in matches:
                            evidence = [RelationshipEvidence(
                                text_snippet=match.group(),
                                start_pos=match.start(),
                                end_pos=match.end(),
                                confidence=0.9,
                                method=RelationshipDetectionMethod.PATTERN_MATCHING,
                                properties={
                                    'pattern': pattern,
                                    'match_groups': match.groups(),
                                    'reversed': True
                                }
                            )]
                            
                            relationship = DetectedRelationship(
                                source_entity=target.name,  # Reversed
                                target_entity=source.name,
                                relationship_type=rel_type,
                                relationship_name=rel_type.value,
                                strength=0.8,
                                confidence=0.9,
                                evidence=evidence,
                                description=f"Pattern-based relationship: {match.group()}",
                                properties={
                                    'pattern_match': True,
                                    'method': 'pattern_matching',
                                    'pattern_type': rel_type.value,
                                    'reversed': True
                                }
                            )
                            
                            relationships.append(relationship)
        
        return relationships
    
    async def _detect_embedding_relationship(
        self, 
        source: ExtractedEntity, 
        target: ExtractedEntity, 
        text: str
    ) -> Optional[DetectedRelationship]:
        """Detect relationship using embedding similarity"""
        
        try:
            # Get or compute embeddings for entity names
            source_embedding = await self._get_embedding(source.name)
            target_embedding = await self._get_embedding(target.name)
            
            if source_embedding is None or target_embedding is None:
                return None
            
            # Calculate cosine similarity
            similarity = cosine_similarity(
                [source_embedding], [target_embedding]
            )[0, 0]
            
            if similarity < self.config.min_embedding_similarity:
                return None
            
            evidence = [RelationshipEvidence(
                text_snippet=f"Embedding similarity: {similarity:.3f}",
                start_pos=0,
                end_pos=0,
                confidence=similarity,
                method=RelationshipDetectionMethod.EMBEDDING_SIMILARITY,
                properties={
                    'embedding_similarity': similarity,
                    'embedding_model': self.config.embedding_model
                }
            )]
            
            return DetectedRelationship(
                source_entity=source.name,
                target_entity=target.name,
                relationship_type=RelationshipType.SIMILAR_TO,
                relationship_name="embedding_similar",
                strength=similarity,
                confidence=similarity,
                evidence=evidence,
                description=f"{source.name} is semantically similar to {target.name}",
                properties={
                    'embedding_similarity': similarity,
                    'method': 'embedding_similarity'
                },
                bidirectional=True
            )
        
        except Exception as e:
            logger.error(f"Error in embedding similarity calculation: {e}")
            return None
    
    async def _detect_ai_relationships(
        self, 
        source: ExtractedEntity, 
        target: ExtractedEntity, 
        text: str
    ) -> List[DetectedRelationship]:
        """Detect relationships using AI inference"""
        relationships = []
        
        if not self.config.enable_ai_inference:
            return relationships
        
        try:
            # Create prompt for AI relationship detection
            prompt = f"""Analyze the relationship between "{source.name}" and "{target.name}" in the following text:

{text[:2000]}  # Limit text length

Identify any relationships between these entities. For each relationship, provide:
1. Relationship type (mentions, related_to, part_of, created_by, etc.)
2. Confidence score (0-1)
3. Brief explanation
4. Relevant text snippet

Format as JSON array: [{{"type": "related_to", "confidence": 0.8, "explanation": "...", "snippet": "..."}}]"""

            response = await openai.ChatCompletion.acreate(
                model=self.config.ai_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.config.ai_max_tokens,
                temperature=0.1
            )
            
            # Parse AI response (simplified - would need robust JSON parsing)
            # This is a placeholder for AI relationship detection
            
        except Exception as e:
            logger.error(f"Error in AI relationship detection: {e}")
        
        return relationships
    
    async def _get_embedding(self, text: str) -> Optional[np.ndarray]:
        """Get embedding for text (with caching)"""
        if text in self._embedding_cache:
            return self._embedding_cache[text]
        
        try:
            # Placeholder for embedding generation
            # In practice, would use OpenAI API or local embedding model
            embedding = np.random.rand(1536)  # Placeholder
            self._embedding_cache[text] = embedding
            return embedding
        
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    async def _consolidate_relationships(
        self, 
        relationships: List[DetectedRelationship]
    ) -> List[DetectedRelationship]:
        """Consolidate duplicate or similar relationships"""
        if not relationships:
            return []
        
        # Group relationships by entity pair and type
        groups = defaultdict(list)
        
        for rel in relationships:
            # Create consistent key for entity pair
            entities = tuple(sorted([rel.source_entity, rel.target_entity]))
            key = (entities, rel.relationship_type)
            groups[key].append(rel)
        
        consolidated = []
        
        for (entities, rel_type), group in groups.items():
            if len(group) == 1:
                consolidated.append(group[0])
            else:
                # Merge multiple relationships of same type
                merged = await self._merge_relationships(group)
                consolidated.append(merged)
        
        return consolidated
    
    async def _merge_relationships(
        self, 
        relationships: List[DetectedRelationship]
    ) -> DetectedRelationship:
        """Merge multiple relationships into one"""
        if len(relationships) == 1:
            return relationships[0]
        
        # Use the relationship with highest confidence as base
        base_rel = max(relationships, key=lambda r: r.confidence)
        
        # Combine evidence from all relationships
        all_evidence = []
        for rel in relationships:
            all_evidence.extend(rel.evidence)
        
        # Calculate combined strength and confidence
        strengths = [r.strength for r in relationships]
        confidences = [r.confidence for r in relationships]
        
        combined_strength = np.mean(strengths)
        combined_confidence = max(confidences)  # Use maximum confidence
        
        # Combine properties
        combined_properties = base_rel.properties.copy()
        combined_properties.update({
            'merged_count': len(relationships),
            'detection_methods': list(set(
                evidence.method.value for rel in relationships for evidence in rel.evidence
            )),
            'strength_range': [min(strengths), max(strengths)],
            'confidence_range': [min(confidences), max(confidences)]
        })
        
        return DetectedRelationship(
            source_entity=base_rel.source_entity,
            target_entity=base_rel.target_entity,
            relationship_type=base_rel.relationship_type,
            relationship_name=base_rel.relationship_name,
            strength=combined_strength,
            confidence=combined_confidence,
            evidence=all_evidence,
            description=f"Merged relationship based on {len(relationships)} detections",
            properties=combined_properties,
            bidirectional=base_rel.bidirectional
        )
    
    def _filter_relationships(
        self, 
        relationships: List[DetectedRelationship]
    ) -> List[DetectedRelationship]:
        """Filter relationships based on configuration"""
        filtered = []
        
        # Group relationships by entity to enforce limits
        entity_counts = defaultdict(int)
        
        # Sort by confidence first
        sorted_relationships = sorted(relationships, key=lambda r: r.confidence, reverse=True)
        
        for relationship in sorted_relationships:
            # Filter by strength and confidence
            if relationship.strength < self.config.min_relationship_strength:
                continue
            
            if relationship.confidence < self.config.min_confidence:
                continue
            
            # Filter by preferred types
            if (self.config.preferred_types and 
                relationship.relationship_type not in self.config.preferred_types):
                continue
            
            # Enforce per-entity limits
            source_count = entity_counts[relationship.source_entity]
            target_count = entity_counts[relationship.target_entity]
            
            if (source_count >= self.config.max_relationships_per_entity or
                target_count >= self.config.max_relationships_per_entity):
                continue
            
            # Add to filtered list and update counts
            filtered.append(relationship)
            entity_counts[relationship.source_entity] += 1
            entity_counts[relationship.target_entity] += 1
        
        return filtered
    
    def _entities_too_similar(
        self, 
        entity1: ExtractedEntity, 
        entity2: ExtractedEntity
    ) -> bool:
        """Check if entities are too similar (likely the same entity)"""
        
        # Check exact name match
        if entity1.name.lower() == entity2.name.lower():
            return True
        
        # Check if one name is contained in the other
        name1 = entity1.name.lower()
        name2 = entity2.name.lower()
        
        if name1 in name2 or name2 in name1:
            return True
        
        # Check aliases
        all_names1 = {entity1.name.lower()} | {alias.lower() for alias in entity1.aliases}
        all_names2 = {entity2.name.lower()} | {alias.lower() for alias in entity2.aliases}
        
        if all_names1 & all_names2:  # Set intersection
            return True
        
        return False


# Global detector instance
_detector: Optional[RelationshipDetector] = None


def get_relationship_detector() -> RelationshipDetector:
    """Get global relationship detector instance"""
    global _detector
    if _detector is None:
        _detector = RelationshipDetector()
    return _detector


def set_relationship_detector_config(config: RelationshipDetectionConfig):
    """Set configuration for global relationship detector"""
    global _detector
    _detector = RelationshipDetector(config)