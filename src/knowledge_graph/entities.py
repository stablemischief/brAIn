"""
Entity extraction module for knowledge graph construction.
Extracts named entities and concepts from text using multiple approaches.
"""

import logging
import re
from typing import Dict, List, Set, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import asyncio

import openai
from pydantic import BaseModel, Field
import spacy
from spacy import displacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)

try:
    nltk.data.find('chunkers/maxent_ne_chunker')
except LookupError:
    nltk.download('maxent_ne_chunker', quiet=True)

try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words', quiet=True)

logger = logging.getLogger(__name__)


class EntityType(str, Enum):
    """Entity type classification"""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    CONCEPT = "concept"
    TECHNOLOGY = "technology"
    PRODUCT = "product"
    EVENT = "event"
    TOPIC = "topic"
    KEYWORD = "keyword"
    DATE = "date"
    NUMBER = "number"
    CUSTOM = "custom"


class ExtractionMethod(str, Enum):
    """Entity extraction method"""
    SPACY_NER = "spacy_ner"
    NLTK_NER = "nltk_ner"
    OPENAI_NER = "openai_ner"
    TFIDF_KEYWORDS = "tfidf_keywords"
    REGEX_PATTERNS = "regex_patterns"
    CUSTOM_RULES = "custom_rules"


@dataclass
class EntityMention:
    """Represents a single entity mention in text"""
    text: str
    start_pos: int
    end_pos: int
    entity_type: EntityType
    confidence: float
    context: str
    method: ExtractionMethod
    properties: Dict = None

    def __post_init__(self):
        if self.properties is None:
            self.properties = {}


class ExtractedEntity(BaseModel):
    """Pydantic model for extracted entities"""
    name: str = Field(description="Canonical name of the entity")
    entity_type: EntityType = Field(description="Type classification of the entity")
    mentions: List[EntityMention] = Field(default_factory=list, description="All mentions of this entity")
    confidence: float = Field(default=0.0, description="Overall confidence score")
    description: Optional[str] = Field(None, description="Generated description of the entity")
    properties: Dict = Field(default_factory=dict, description="Additional entity properties")
    aliases: List[str] = Field(default_factory=list, description="Alternative names for this entity")


class EntityExtractionConfig(BaseModel):
    """Configuration for entity extraction"""
    
    # Method selection
    enable_spacy: bool = True
    enable_nltk: bool = True
    enable_openai: bool = False
    enable_tfidf: bool = True
    enable_regex: bool = True
    enable_custom_rules: bool = True
    
    # Spacy model
    spacy_model: str = "en_core_web_sm"
    
    # OpenAI settings
    openai_model: str = "gpt-3.5-turbo"
    openai_max_tokens: int = 1000
    
    # TFIDF settings
    tfidf_max_features: int = 100
    tfidf_min_df: int = 2
    tfidf_ngram_range: Tuple[int, int] = (1, 3)
    
    # Filtering settings
    min_confidence: float = 0.3
    min_entity_length: int = 2
    max_entity_length: int = 100
    stopword_filter: bool = True
    
    # Entity type preferences
    preferred_types: List[EntityType] = Field(default_factory=lambda: [
        EntityType.PERSON, EntityType.ORGANIZATION, EntityType.LOCATION, 
        EntityType.TECHNOLOGY, EntityType.CONCEPT
    ])


class EntityExtractor:
    """Advanced entity extraction system using multiple NLP approaches"""
    
    def __init__(self, config: EntityExtractionConfig = None):
        self.config = config or EntityExtractionConfig()
        
        # Initialize NLP models
        self.nlp = None
        self.stopwords = set(stopwords.words('english'))
        
        # Regex patterns for specific entities
        self.regex_patterns = {
            EntityType.DATE: [
                r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
                r'\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b',
                r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'
            ],
            EntityType.NUMBER: [
                r'\b\d+(?:,\d{3})*(?:\.\d+)?\b',
                r'\b\d+(?:\.\d+)?(?:[kKmMbBtT])\b'
            ],
            EntityType.TECHNOLOGY: [
                r'\b(?:API|SDK|AI|ML|NLP|GPU|CPU|RAM|SSD|HDD|HTTP|HTTPS|REST|GraphQL|JSON|XML|SQL|NoSQL)\b',
                r'\b(?:Python|JavaScript|TypeScript|Java|C\+\+|C#|Go|Rust|Swift|Kotlin)\b',
                r'\b(?:React|Vue|Angular|Django|Flask|FastAPI|Express|Spring|Laravel)\b'
            ]
        }
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize NLP models and dependencies"""
        try:
            if self.config.enable_spacy:
                try:
                    self.nlp = spacy.load(self.config.spacy_model)
                    logger.info(f"Loaded spaCy model: {self.config.spacy_model}")
                except OSError:
                    logger.warning(f"spaCy model {self.config.spacy_model} not found. Install with: python -m spacy download {self.config.spacy_model}")
                    self.config.enable_spacy = False
            
        except Exception as e:
            logger.error(f"Error initializing NLP models: {e}")
    
    async def extract_entities(
        self, 
        text: str, 
        document_id: Optional[str] = None
    ) -> List[ExtractedEntity]:
        """
        Extract entities from text using multiple methods.
        
        Args:
            text: Input text to extract entities from
            document_id: Optional document identifier
            
        Returns:
            List of extracted entities with confidence scores
        """
        if not text or not text.strip():
            return []
        
        try:
            # Extract entities using different methods
            all_mentions = []
            
            if self.config.enable_spacy and self.nlp:
                spacy_mentions = await self._extract_with_spacy(text)
                all_mentions.extend(spacy_mentions)
            
            if self.config.enable_nltk:
                nltk_mentions = await self._extract_with_nltk(text)
                all_mentions.extend(nltk_mentions)
            
            if self.config.enable_openai:
                openai_mentions = await self._extract_with_openai(text)
                all_mentions.extend(openai_mentions)
            
            if self.config.enable_tfidf:
                tfidf_mentions = await self._extract_with_tfidf(text)
                all_mentions.extend(tfidf_mentions)
            
            if self.config.enable_regex:
                regex_mentions = await self._extract_with_regex(text)
                all_mentions.extend(regex_mentions)
            
            if self.config.enable_custom_rules:
                custom_mentions = await self._extract_with_custom_rules(text)
                all_mentions.extend(custom_mentions)
            
            # Consolidate and clean entities
            entities = await self._consolidate_entities(all_mentions)
            
            # Filter by confidence and preferences
            filtered_entities = self._filter_entities(entities)
            
            logger.info(f"Extracted {len(filtered_entities)} entities from text ({len(all_mentions)} mentions)")
            return filtered_entities
        
        except Exception as e:
            logger.error(f"Error during entity extraction: {e}")
            return []
    
    async def _extract_with_spacy(self, text: str) -> List[EntityMention]:
        """Extract entities using spaCy NER"""
        mentions = []
        
        try:
            doc = self.nlp(text)
            
            for ent in doc.ents:
                entity_type = self._map_spacy_label_to_type(ent.label_)
                confidence = self._calculate_spacy_confidence(ent)
                
                # Get context around the entity
                start_idx = max(0, ent.start - 5)
                end_idx = min(len(doc), ent.end + 5)
                context = doc[start_idx:end_idx].text
                
                mention = EntityMention(
                    text=ent.text,
                    start_pos=ent.start_char,
                    end_pos=ent.end_char,
                    entity_type=entity_type,
                    confidence=confidence,
                    context=context,
                    method=ExtractionMethod.SPACY_NER,
                    properties={
                        "spacy_label": ent.label_,
                        "spacy_description": spacy.explain(ent.label_)
                    }
                )
                mentions.append(mention)
        
        except Exception as e:
            logger.error(f"Error in spaCy extraction: {e}")
        
        return mentions
    
    async def _extract_with_nltk(self, text: str) -> List[EntityMention]:
        """Extract entities using NLTK NER"""
        mentions = []
        
        try:
            sentences = sent_tokenize(text)
            
            for sent in sentences:
                tokens = word_tokenize(sent)
                pos_tags = pos_tag(tokens)
                chunks = ne_chunk(pos_tags, binary=False)
                
                current_pos = 0
                for chunk in chunks:
                    if hasattr(chunk, 'label'):
                        # Named entity chunk
                        entity_text = ' '.join([token for token, pos in chunk.leaves()])
                        entity_type = self._map_nltk_label_to_type(chunk.label())
                        
                        # Find position in original text
                        start_pos = text.find(entity_text, current_pos)
                        if start_pos != -1:
                            end_pos = start_pos + len(entity_text)
                            
                            # Get context
                            context_start = max(0, start_pos - 50)
                            context_end = min(len(text), end_pos + 50)
                            context = text[context_start:context_end]
                            
                            mention = EntityMention(
                                text=entity_text,
                                start_pos=start_pos,
                                end_pos=end_pos,
                                entity_type=entity_type,
                                confidence=0.7,  # NLTK doesn't provide confidence
                                context=context,
                                method=ExtractionMethod.NLTK_NER,
                                properties={
                                    "nltk_label": chunk.label(),
                                    "pos_tags": [pos for token, pos in chunk.leaves()]
                                }
                            )
                            mentions.append(mention)
                            current_pos = end_pos
        
        except Exception as e:
            logger.error(f"Error in NLTK extraction: {e}")
        
        return mentions
    
    async def _extract_with_openai(self, text: str) -> List[EntityMention]:
        """Extract entities using OpenAI GPT models"""
        mentions = []
        
        if not self.config.enable_openai:
            return mentions
        
        try:
            prompt = f"""Extract named entities from the following text. For each entity, provide:
1. The entity text
2. The entity type (person, organization, location, concept, technology, product, event, topic)
3. A confidence score (0-1)
4. Start and end positions in the text

Text: {text[:2000]}  # Limit text length

Format as JSON array:
[{{"text": "entity", "type": "person", "confidence": 0.9, "start": 0, "end": 6}}]"""

            response = await openai.ChatCompletion.acreate(
                model=self.config.openai_model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.config.openai_max_tokens,
                temperature=0.1
            )
            
            # Parse response (simplified - would need robust JSON parsing)
            # This is a placeholder for OpenAI entity extraction
            
        except Exception as e:
            logger.error(f"Error in OpenAI extraction: {e}")
        
        return mentions
    
    async def _extract_with_tfidf(self, text: str) -> List[EntityMention]:
        """Extract keywords using TF-IDF"""
        mentions = []
        
        try:
            # Preprocess text
            sentences = sent_tokenize(text)
            
            if len(sentences) < 2:
                return mentions
            
            # Initialize TF-IDF vectorizer
            vectorizer = TfidfVectorizer(
                max_features=self.config.tfidf_max_features,
                min_df=self.config.tfidf_min_df,
                ngram_range=self.config.tfidf_ngram_range,
                stop_words='english'
            )
            
            # Fit and transform
            tfidf_matrix = vectorizer.fit_transform(sentences)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get top scoring terms
            scores = tfidf_matrix.mean(axis=0).A1
            top_indices = scores.argsort()[-20:][::-1]  # Top 20 terms
            
            for idx in top_indices:
                term = feature_names[idx]
                score = scores[idx]
                
                if score > 0.1:  # Minimum TF-IDF threshold
                    # Find all occurrences in text
                    for match in re.finditer(re.escape(term), text, re.IGNORECASE):
                        start_pos = match.start()
                        end_pos = match.end()
                        
                        # Get context
                        context_start = max(0, start_pos - 50)
                        context_end = min(len(text), end_pos + 50)
                        context = text[context_start:context_end]
                        
                        mention = EntityMention(
                            text=term,
                            start_pos=start_pos,
                            end_pos=end_pos,
                            entity_type=EntityType.KEYWORD,
                            confidence=min(score * 2, 1.0),  # Scale TF-IDF score
                            context=context,
                            method=ExtractionMethod.TFIDF_KEYWORDS,
                            properties={
                                "tfidf_score": score,
                                "ngram_length": len(term.split())
                            }
                        )
                        mentions.append(mention)
        
        except Exception as e:
            logger.error(f"Error in TF-IDF extraction: {e}")
        
        return mentions
    
    async def _extract_with_regex(self, text: str) -> List[EntityMention]:
        """Extract entities using regex patterns"""
        mentions = []
        
        try:
            for entity_type, patterns in self.regex_patterns.items():
                for pattern in patterns:
                    for match in re.finditer(pattern, text, re.IGNORECASE):
                        start_pos = match.start()
                        end_pos = match.end()
                        matched_text = match.group()
                        
                        # Get context
                        context_start = max(0, start_pos - 50)
                        context_end = min(len(text), end_pos + 50)
                        context = text[context_start:context_end]
                        
                        mention = EntityMention(
                            text=matched_text,
                            start_pos=start_pos,
                            end_pos=end_pos,
                            entity_type=entity_type,
                            confidence=0.8,  # High confidence for regex matches
                            context=context,
                            method=ExtractionMethod.REGEX_PATTERNS,
                            properties={
                                "pattern": pattern,
                                "match_groups": match.groups()
                            }
                        )
                        mentions.append(mention)
        
        except Exception as e:
            logger.error(f"Error in regex extraction: {e}")
        
        return mentions
    
    async def _extract_with_custom_rules(self, text: str) -> List[EntityMention]:
        """Extract entities using custom rules"""
        mentions = []
        
        try:
            # Custom rule: Capitalize words that might be proper nouns
            words = word_tokenize(text)
            pos_tags = pos_tag(words)
            
            current_pos = 0
            for word, pos in pos_tags:
                if pos in ['NNP', 'NNPS'] and len(word) >= 3:  # Proper nouns
                    # Find position in text
                    start_pos = text.find(word, current_pos)
                    if start_pos != -1:
                        end_pos = start_pos + len(word)
                        
                        # Get context
                        context_start = max(0, start_pos - 50)
                        context_end = min(len(text), end_pos + 50)
                        context = text[context_start:context_end]
                        
                        # Determine entity type based on context and POS
                        entity_type = self._infer_entity_type_from_context(word, context, pos)
                        
                        mention = EntityMention(
                            text=word,
                            start_pos=start_pos,
                            end_pos=end_pos,
                            entity_type=entity_type,
                            confidence=0.6,
                            context=context,
                            method=ExtractionMethod.CUSTOM_RULES,
                            properties={
                                "pos_tag": pos,
                                "inferred_type": entity_type.value
                            }
                        )
                        mentions.append(mention)
                        current_pos = end_pos
        
        except Exception as e:
            logger.error(f"Error in custom rules extraction: {e}")
        
        return mentions
    
    async def _consolidate_entities(self, mentions: List[EntityMention]) -> List[ExtractedEntity]:
        """Consolidate overlapping mentions into entities"""
        if not mentions:
            return []
        
        # Group mentions by normalized text
        entity_groups = {}
        
        for mention in mentions:
            normalized_text = mention.text.lower().strip()
            
            # Skip very short or very long entities
            if (len(normalized_text) < self.config.min_entity_length or 
                len(normalized_text) > self.config.max_entity_length):
                continue
            
            # Skip stopwords if filtering enabled
            if (self.config.stopword_filter and 
                normalized_text in self.stopwords):
                continue
            
            # Group by normalized text
            if normalized_text not in entity_groups:
                entity_groups[normalized_text] = []
            entity_groups[normalized_text].append(mention)
        
        # Create consolidated entities
        entities = []
        for normalized_text, group_mentions in entity_groups.items():
            if not group_mentions:
                continue
            
            # Find canonical name (most frequent original form)
            name_counts = {}
            for mention in group_mentions:
                name = mention.text.strip()
                name_counts[name] = name_counts.get(name, 0) + 1
            
            canonical_name = max(name_counts.keys(), key=name_counts.get)
            
            # Determine best entity type (by confidence)
            type_confidences = {}
            for mention in group_mentions:
                if mention.entity_type not in type_confidences:
                    type_confidences[mention.entity_type] = []
                type_confidences[mention.entity_type].append(mention.confidence)
            
            best_type = max(type_confidences.keys(), 
                           key=lambda t: np.mean(type_confidences[t]))
            
            # Calculate overall confidence
            all_confidences = [m.confidence for m in group_mentions]
            overall_confidence = np.mean(all_confidences)
            
            # Collect aliases
            aliases = list(set(m.text.strip() for m in group_mentions 
                             if m.text.strip() != canonical_name))
            
            # Create entity
            entity = ExtractedEntity(
                name=canonical_name,
                entity_type=best_type,
                mentions=group_mentions,
                confidence=overall_confidence,
                aliases=aliases,
                properties={
                    "mention_count": len(group_mentions),
                    "extraction_methods": list(set(m.method.value for m in group_mentions)),
                    "type_distribution": {t.value: len(confidences) 
                                        for t, confidences in type_confidences.items()}
                }
            )
            
            entities.append(entity)
        
        # Sort by confidence
        entities.sort(key=lambda e: e.confidence, reverse=True)
        return entities
    
    def _filter_entities(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Filter entities based on configuration preferences"""
        filtered = []
        
        for entity in entities:
            # Filter by confidence
            if entity.confidence < self.config.min_confidence:
                continue
            
            # Filter by preferred types
            if (self.config.preferred_types and 
                entity.entity_type not in self.config.preferred_types):
                continue
            
            filtered.append(entity)
        
        return filtered
    
    def _map_spacy_label_to_type(self, spacy_label: str) -> EntityType:
        """Map spaCy entity labels to our EntityType enum"""
        mapping = {
            "PERSON": EntityType.PERSON,
            "ORG": EntityType.ORGANIZATION,
            "GPE": EntityType.LOCATION,  # Geopolitical entity
            "LOC": EntityType.LOCATION,
            "DATE": EntityType.DATE,
            "TIME": EntityType.DATE,
            "MONEY": EntityType.NUMBER,
            "PERCENT": EntityType.NUMBER,
            "QUANTITY": EntityType.NUMBER,
            "CARDINAL": EntityType.NUMBER,
            "PRODUCT": EntityType.PRODUCT,
            "EVENT": EntityType.EVENT,
            "WORK_OF_ART": EntityType.CONCEPT,
            "LAW": EntityType.CONCEPT,
            "LANGUAGE": EntityType.CONCEPT,
        }
        
        return mapping.get(spacy_label, EntityType.CONCEPT)
    
    def _map_nltk_label_to_type(self, nltk_label: str) -> EntityType:
        """Map NLTK entity labels to our EntityType enum"""
        mapping = {
            "PERSON": EntityType.PERSON,
            "ORGANIZATION": EntityType.ORGANIZATION,
            "GPE": EntityType.LOCATION,
            "LOCATION": EntityType.LOCATION,
            "FACILITY": EntityType.LOCATION,
        }
        
        return mapping.get(nltk_label, EntityType.CONCEPT)
    
    def _calculate_spacy_confidence(self, ent) -> float:
        """Calculate confidence score for spaCy entities"""
        # spaCy doesn't provide direct confidence scores
        # Use heuristics based on entity properties
        base_confidence = 0.8
        
        # Adjust based on entity length
        if len(ent.text) < 3:
            base_confidence -= 0.2
        elif len(ent.text) > 20:
            base_confidence -= 0.1
        
        # Adjust based on label certainty
        if ent.label_ in ["PERSON", "ORG", "GPE"]:
            base_confidence += 0.1
        
        return max(0.1, min(1.0, base_confidence))
    
    def _infer_entity_type_from_context(
        self, 
        word: str, 
        context: str, 
        pos_tag: str
    ) -> EntityType:
        """Infer entity type from context and linguistic features"""
        
        # Technology keywords
        tech_indicators = ['api', 'sdk', 'framework', 'library', 'algorithm', 'system']
        if any(indicator in context.lower() for indicator in tech_indicators):
            return EntityType.TECHNOLOGY
        
        # Organization indicators
        org_indicators = ['company', 'corporation', 'inc', 'ltd', 'organization']
        if any(indicator in context.lower() for indicator in org_indicators):
            return EntityType.ORGANIZATION
        
        # Person indicators
        person_indicators = ['mr', 'mrs', 'dr', 'prof', 'said', 'told']
        if any(indicator in context.lower() for indicator in person_indicators):
            return EntityType.PERSON
        
        # Location indicators
        location_indicators = ['city', 'country', 'state', 'located', 'in']
        if any(indicator in context.lower() for indicator in location_indicators):
            return EntityType.LOCATION
        
        # Default based on POS tag
        if pos_tag in ['NNP', 'NNPS']:
            return EntityType.CONCEPT
        
        return EntityType.KEYWORD


# Global extractor instance
_extractor: Optional[EntityExtractor] = None


def get_entity_extractor() -> EntityExtractor:
    """Get global entity extractor instance"""
    global _extractor
    if _extractor is None:
        _extractor = EntityExtractor()
    return _extractor


def set_entity_extractor_config(config: EntityExtractionConfig):
    """Set configuration for global entity extractor"""
    global _extractor
    _extractor = EntityExtractor(config)