"""
Quality Assessment Engine for brAIn v2.0 RAG Pipeline

This module provides comprehensive quality assessment for document processing,
text extraction, and overall pipeline performance with AI-powered analysis.

Author: BMad Team
"""

import os
import re
import statistics
from typing import List, Dict, Any, Optional, Tuple, Union
from decimal import Decimal
from datetime import datetime
from dataclasses import dataclass
import numpy as np

# Pydantic imports
from pydantic import BaseModel, Field, validator

# Local imports
from .text_processor import FileProcessingResult, TextChunk, ProcessingQuality

# Language detection (optional)
try:
    import langdetect
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False

# =============================================================================
# PYDANTIC MODELS FOR QUALITY ASSESSMENT
# =============================================================================

class ContentQualityMetrics(BaseModel):
    """Metrics for content quality assessment."""
    readability_score: float = Field(ge=0.0, le=100.0, description="Readability score (0-100)")
    coherence_score: float = Field(ge=0.0, le=1.0, description="Content coherence score")
    information_density: float = Field(ge=0.0, le=1.0, description="Information density score")
    language_consistency: float = Field(ge=0.0, le=1.0, description="Language consistency score")
    entity_coverage: float = Field(ge=0.0, le=1.0, description="Entity coverage score")
    structure_quality: float = Field(ge=0.0, le=1.0, description="Document structure quality")
    completeness_score: float = Field(ge=0.0, le=1.0, description="Content completeness score")

class ExtractionQualityMetrics(BaseModel):
    """Metrics for text extraction quality."""
    extraction_accuracy: float = Field(ge=0.0, le=1.0, description="Extraction accuracy estimate")
    text_preservation: float = Field(ge=0.0, le=1.0, description="Text preservation quality")
    formatting_retention: float = Field(ge=0.0, le=1.0, description="Formatting retention score")
    error_rate: float = Field(ge=0.0, le=1.0, description="Estimated error rate")
    method_reliability: float = Field(ge=0.0, le=1.0, description="Extraction method reliability")
    
class ProcessingQualityMetrics(BaseModel):
    """Metrics for overall processing quality."""
    chunking_quality: float = Field(ge=0.0, le=1.0, description="Text chunking quality")
    embedding_reliability: float = Field(ge=0.0, le=1.0, description="Embedding generation quality")
    metadata_completeness: float = Field(ge=0.0, le=1.0, description="Metadata completeness")
    cost_efficiency: float = Field(ge=0.0, le=1.0, description="Cost efficiency score")
    processing_speed: float = Field(ge=0.0, le=1.0, description="Processing speed score")

class QualityAssessmentResult(BaseModel):
    """Complete quality assessment result."""
    overall_quality_score: float = Field(ge=0.0, le=1.0, description="Overall quality score")
    content_metrics: ContentQualityMetrics
    extraction_metrics: ExtractionQualityMetrics
    processing_metrics: ProcessingQualityMetrics
    quality_grade: str = Field(description="Quality grade (A, B, C, D, F)")
    recommendations: List[str] = Field(default_factory=list, description="Quality improvement recommendations")
    issues_detected: List[str] = Field(default_factory=list, description="Issues found during assessment")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in assessment")
    assessment_timestamp: datetime = Field(default_factory=datetime.utcnow)

class QualityThresholds(BaseModel):
    """Quality thresholds for different grades."""
    excellent_threshold: float = Field(default=0.90, ge=0.0, le=1.0)
    good_threshold: float = Field(default=0.75, ge=0.0, le=1.0)
    acceptable_threshold: float = Field(default=0.60, ge=0.0, le=1.0)
    poor_threshold: float = Field(default=0.40, ge=0.0, le=1.0)

# =============================================================================
# QUALITY ASSESSMENT ENGINE
# =============================================================================

class QualityAssessmentEngine:
    """Advanced quality assessment engine for document processing."""
    
    def __init__(self, thresholds: Optional[QualityThresholds] = None):
        """
        Initialize quality assessment engine.
        
        Args:
            thresholds: Quality thresholds for grading
        """
        self.thresholds = thresholds or QualityThresholds()
        
        # Common word lists for analysis
        self.common_english_words = {
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have',
            'i', 'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you',
            'do', 'at', 'this', 'but', 'his', 'by', 'from'
        }
        
        # Stop words for content analysis
        self.stop_words = {
            'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
            'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him',
            'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its',
            'itself', 'they', 'them', 'their', 'theirs', 'themselves'
        }
    
    def calculate_readability_score(self, text: str) -> float:
        """
        Calculate readability score using Flesch Reading Ease formula.
        
        Args:
            text: Text to analyze
            
        Returns:
            Readability score (0-100, higher is more readable)
        """
        if not text or len(text) < 10:
            return 0.0
        
        try:
            # Count sentences, words, and syllables
            sentences = len(re.findall(r'[.!?]+', text))
            words = len(text.split())
            
            if sentences == 0 or words == 0:
                return 0.0
            
            # Estimate syllables (simplified)
            def count_syllables(word):
                word = word.lower()
                if len(word) <= 3:
                    return 1
                vowels = 'aeiouy'
                syllable_count = 0
                prev_was_vowel = False
                
                for char in word:
                    if char in vowels:
                        if not prev_was_vowel:
                            syllable_count += 1
                        prev_was_vowel = True
                    else:
                        prev_was_vowel = False
                
                # Handle silent 'e'
                if word.endswith('e'):
                    syllable_count -= 1
                
                return max(1, syllable_count)
            
            syllables = sum(count_syllables(word) for word in text.split())
            
            # Flesch Reading Ease formula
            score = 206.835 - (1.015 * (words / sentences)) - (84.6 * (syllables / words))
            
            # Clamp to 0-100 range
            return max(0.0, min(100.0, score))
            
        except Exception as e:
            print(f"Error calculating readability: {e}")
            return 50.0  # Default middle score
    
    def calculate_coherence_score(self, text: str) -> float:
        """
        Calculate text coherence using various heuristics.
        
        Args:
            text: Text to analyze
            
        Returns:
            Coherence score (0.0 to 1.0)
        """
        if not text or len(text) < 50:
            return 0.0
        
        try:
            sentences = text.split('.')
            sentences = [s.strip() for s in sentences if s.strip()]
            
            if len(sentences) < 2:
                return 0.5
            
            coherence_score = 0.8  # Base score
            
            # Check for transition words
            transition_words = {
                'however', 'therefore', 'furthermore', 'moreover', 'additionally',
                'consequently', 'meanwhile', 'similarly', 'likewise', 'nevertheless',
                'although', 'because', 'since', 'thus', 'hence'
            }
            
            text_lower = text.lower()
            transition_count = sum(1 for word in transition_words if word in text_lower)
            transition_density = transition_count / len(sentences)
            
            # Adjust score based on transition density
            if transition_density > 0.1:
                coherence_score += 0.1
            elif transition_density < 0.05:
                coherence_score -= 0.1
            
            # Check sentence length variation
            sentence_lengths = [len(s.split()) for s in sentences if s]
            if sentence_lengths:
                length_variance = statistics.variance(sentence_lengths) if len(sentence_lengths) > 1 else 0
                avg_length = statistics.mean(sentence_lengths)
                
                # Good coherence has moderate sentence length variation
                if avg_length > 5 and length_variance > 0:
                    coherence_score += 0.05
            
            # Check for repetitive content
            words = text_lower.split()
            unique_words = set(words)
            word_diversity = len(unique_words) / len(words) if words else 0
            
            if word_diversity < 0.3:  # Too repetitive
                coherence_score -= 0.2
            elif word_diversity > 0.6:  # Good diversity
                coherence_score += 0.1
            
            return max(0.0, min(1.0, coherence_score))
            
        except Exception as e:
            print(f"Error calculating coherence: {e}")
            return 0.5
    
    def calculate_information_density(self, text: str) -> float:
        """
        Calculate information density of text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Information density score (0.0 to 1.0)
        """
        if not text or len(text) < 10:
            return 0.0
        
        try:
            words = text.lower().split()
            
            if not words:
                return 0.0
            
            # Count non-stop words
            content_words = [word for word in words if word not in self.stop_words]
            content_ratio = len(content_words) / len(words)
            
            # Count unique content words
            unique_content_words = set(content_words)
            uniqueness_ratio = len(unique_content_words) / len(words)
            
            # Look for numbers, dates, names (high information content)
            info_patterns = [
                r'\d+',  # Numbers
                r'\d{4}',  # Years
                r'[A-Z][a-z]+\s+[A-Z][a-z]+',  # Proper names
                r'[A-Z]{2,}',  # Acronyms
                r'\$\d+',  # Money
                r'\d+%'  # Percentages
            ]
            
            info_content_count = 0
            for pattern in info_patterns:
                info_content_count += len(re.findall(pattern, text))
            
            info_density = info_content_count / len(words)
            
            # Combine metrics
            density_score = (content_ratio * 0.4 + uniqueness_ratio * 0.4 + 
                           min(info_density * 10, 1.0) * 0.2)
            
            return max(0.0, min(1.0, density_score))
            
        except Exception as e:
            print(f"Error calculating information density: {e}")
            return 0.5
    
    def calculate_language_consistency(self, text: str) -> float:
        """
        Calculate language consistency score.
        
        Args:
            text: Text to analyze
            
        Returns:
            Language consistency score (0.0 to 1.0)
        """
        if not text or len(text) < 100:
            return 1.0  # Assume consistent for short text
        
        try:
            if not LANGDETECT_AVAILABLE:
                # Fallback: check for English patterns
                english_words_count = sum(1 for word in text.lower().split() 
                                        if word in self.common_english_words)
                total_words = len(text.split())
                english_ratio = english_words_count / total_words if total_words > 0 else 0
                return min(1.0, english_ratio * 2)  # Scale up
            
            # Split text into chunks and detect language for each
            chunk_size = 200
            chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
            
            if len(chunks) < 2:
                return 1.0
            
            languages = []
            for chunk in chunks:
                if len(chunk) > 50:
                    try:
                        lang = langdetect.detect(chunk)
                        languages.append(lang)
                    except:
                        pass
            
            if not languages:
                return 0.8  # Default if detection fails
            
            # Calculate consistency
            most_common_lang = max(set(languages), key=languages.count)
            consistency = languages.count(most_common_lang) / len(languages)
            
            return consistency
            
        except Exception as e:
            print(f"Error calculating language consistency: {e}")
            return 0.8
    
    def assess_content_quality(self, text: str, entities: List[str] = None) -> ContentQualityMetrics:
        """
        Assess overall content quality.
        
        Args:
            text: Text to assess
            entities: Optional list of extracted entities
            
        Returns:
            Content quality metrics
        """
        entities = entities or []
        
        readability_score = self.calculate_readability_score(text)
        coherence_score = self.calculate_coherence_score(text)
        information_density = self.calculate_information_density(text)
        language_consistency = self.calculate_language_consistency(text)
        
        # Entity coverage (how well entities are represented)
        entity_coverage = 0.8  # Default
        if entities and text:
            # Calculate what portion of text contains entity references
            entity_mentions = 0
            for entity in entities:
                if entity.startswith(('EMAIL:', 'URL:', 'DATE:')):
                    entity_value = entity.split(':', 1)[1]
                    if entity_value.lower() in text.lower():
                        entity_mentions += 1
            
            entity_coverage = min(1.0, entity_mentions / max(len(entities), 1))
        
        # Structure quality (paragraphs, formatting)
        structure_quality = 0.7  # Default
        if '\n\n' in text:  # Has paragraphs
            structure_quality += 0.2
        if any(marker in text for marker in ['1.', '2.', '-', '*']):  # Has lists
            structure_quality += 0.1
        
        structure_quality = min(1.0, structure_quality)
        
        # Completeness (does text seem complete?)
        completeness_score = 0.8  # Default
        if text.strip().endswith(('.', '!', '?')):  # Proper ending
            completeness_score += 0.1
        if len(text) > 100:  # Substantial content
            completeness_score += 0.1
        
        completeness_score = min(1.0, completeness_score)
        
        return ContentQualityMetrics(
            readability_score=readability_score,
            coherence_score=coherence_score,
            information_density=information_density,
            language_consistency=language_consistency,
            entity_coverage=entity_coverage,
            structure_quality=structure_quality,
            completeness_score=completeness_score
        )
    
    def assess_extraction_quality(self, processing_quality: ProcessingQuality, 
                                file_size: int, extracted_length: int) -> ExtractionQualityMetrics:
        """
        Assess text extraction quality.
        
        Args:
            processing_quality: Processing quality from text processor
            file_size: Original file size
            extracted_length: Length of extracted text
            
        Returns:
            Extraction quality metrics
        """
        # Use existing processing quality as base
        extraction_accuracy = processing_quality.estimated_accuracy
        
        # Text preservation based on extraction ratio
        if file_size > 0:
            extraction_ratio = extracted_length / file_size
            if 0.01 <= extraction_ratio <= 0.5:  # Reasonable range
                text_preservation = 0.9
            elif extraction_ratio > 0.5:  # Very high extraction
                text_preservation = 0.95
            else:  # Very low extraction
                text_preservation = 0.5
        else:
            text_preservation = 0.7
        
        # Formatting retention based on extraction method
        method_scores = {
            'docx_xml': 0.9,
            'pdf_pdfplumber': 0.8,
            'pdf_pypdf': 0.7,
            'xlsx_openpyxl': 0.95,
            'csv_native': 0.98,
            'txt_direct': 0.99,
            'fallback': 0.5
        }
        
        formatting_retention = method_scores.get(
            processing_quality.extraction_method, 0.7
        )
        
        # Error rate based on warnings and errors
        error_count = len(processing_quality.errors) + len(processing_quality.warnings) * 0.5
        error_rate = min(1.0, error_count * 0.1)  # Scale errors to rate
        
        # Method reliability
        method_reliability = method_scores.get(
            processing_quality.extraction_method, 0.7
        )
        
        return ExtractionQualityMetrics(
            extraction_accuracy=extraction_accuracy,
            text_preservation=text_preservation,
            formatting_retention=formatting_retention,
            error_rate=error_rate,
            method_reliability=method_reliability
        )
    
    def assess_processing_quality(self, chunks: List[TextChunk], 
                                processing_time: float, 
                                estimated_cost: Decimal) -> ProcessingQualityMetrics:
        """
        Assess overall processing quality.
        
        Args:
            chunks: Text chunks created
            processing_time: Time taken to process
            estimated_cost: Estimated processing cost
            
        Returns:
            Processing quality metrics
        """
        # Chunking quality
        if not chunks:
            chunking_quality = 0.0
        else:
            # Check chunk size distribution
            chunk_sizes = [len(chunk.content) for chunk in chunks]
            avg_size = statistics.mean(chunk_sizes)
            size_variance = statistics.variance(chunk_sizes) if len(chunk_sizes) > 1 else 0
            
            # Good chunking has reasonable sizes and low variance
            if 100 <= avg_size <= 800 and size_variance < avg_size:
                chunking_quality = 0.9
            else:
                chunking_quality = 0.6
            
            # Quality of individual chunks
            avg_chunk_quality = statistics.mean([chunk.quality_score for chunk in chunks])
            chunking_quality = (chunking_quality + avg_chunk_quality) / 2
        
        # Embedding reliability (placeholder - would need actual embedding analysis)
        embedding_reliability = 0.85
        
        # Metadata completeness (check what metadata we have)
        metadata_completeness = 0.8  # Base score
        
        # Cost efficiency (lower cost per token is better)
        cost_efficiency = 0.8  # Default
        if estimated_cost > 0:
            # This would need actual cost benchmarks
            cost_efficiency = min(1.0, 1.0 / (float(estimated_cost) * 1000 + 1))
        
        # Processing speed (faster is better, but quality matters more)
        processing_speed = 0.8  # Default
        if processing_time > 0 and chunks:
            chunks_per_second = len(chunks) / processing_time
            if chunks_per_second > 2:
                processing_speed = 0.9
            elif chunks_per_second < 0.5:
                processing_speed = 0.6
        
        return ProcessingQualityMetrics(
            chunking_quality=chunking_quality,
            embedding_reliability=embedding_reliability,
            metadata_completeness=metadata_completeness,
            cost_efficiency=cost_efficiency,
            processing_speed=processing_speed
        )
    
    def grade_quality_score(self, score: float) -> str:
        """
        Convert quality score to letter grade.
        
        Args:
            score: Quality score (0.0 to 1.0)
            
        Returns:
            Letter grade (A, B, C, D, F)
        """
        if score >= self.thresholds.excellent_threshold:
            return "A"
        elif score >= self.thresholds.good_threshold:
            return "B"
        elif score >= self.thresholds.acceptable_threshold:
            return "C"
        elif score >= self.thresholds.poor_threshold:
            return "D"
        else:
            return "F"
    
    def generate_recommendations(self, content_metrics: ContentQualityMetrics,
                               extraction_metrics: ExtractionQualityMetrics,
                               processing_metrics: ProcessingQualityMetrics) -> List[str]:
        """
        Generate quality improvement recommendations.
        
        Args:
            content_metrics: Content quality metrics
            extraction_metrics: Extraction quality metrics
            processing_metrics: Processing quality metrics
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Content recommendations
        if content_metrics.readability_score < 30:
            recommendations.append("Content has low readability - consider using simpler language")
        
        if content_metrics.coherence_score < 0.6:
            recommendations.append("Content lacks coherence - add transition words and logical flow")
        
        if content_metrics.information_density < 0.4:
            recommendations.append("Content has low information density - consider adding more specific details")
        
        if content_metrics.language_consistency < 0.8:
            recommendations.append("Content has mixed languages - ensure consistent language use")
        
        # Extraction recommendations
        if extraction_metrics.extraction_accuracy < 0.7:
            recommendations.append("Extraction accuracy is low - consider alternative extraction methods")
        
        if extraction_metrics.text_preservation < 0.6:
            recommendations.append("Text preservation is poor - check for format-specific extraction issues")
        
        if extraction_metrics.error_rate > 0.3:
            recommendations.append("High error rate detected - review extraction process for issues")
        
        # Processing recommendations
        if processing_metrics.chunking_quality < 0.6:
            recommendations.append("Chunking quality is poor - adjust chunk size or overlap settings")
        
        if processing_metrics.cost_efficiency < 0.5:
            recommendations.append("Processing cost is high - consider optimizing token usage")
        
        if processing_metrics.processing_speed < 0.5:
            recommendations.append("Processing speed is slow - consider performance optimizations")
        
        return recommendations
    
    def assess_processing_result(self, result: FileProcessingResult) -> QualityAssessmentResult:
        """
        Perform complete quality assessment on a processing result.
        
        Args:
            result: File processing result to assess
            
        Returns:
            Complete quality assessment
        """
        # Assess different quality aspects
        content_metrics = self.assess_content_quality(
            result.extracted_text, result.extracted_entities
        )
        
        extraction_metrics = self.assess_extraction_quality(
            result.processing_quality, result.file_size, len(result.extracted_text)
        )
        
        processing_metrics = self.assess_processing_quality(
            result.text_chunks, result.processing_time, result.cost_estimate
        )
        
        # Calculate overall quality score
        overall_score = (
            content_metrics.readability_score / 100.0 * 0.15 +
            content_metrics.coherence_score * 0.15 +
            content_metrics.information_density * 0.10 +
            content_metrics.language_consistency * 0.05 +
            content_metrics.entity_coverage * 0.05 +
            content_metrics.structure_quality * 0.05 +
            content_metrics.completeness_score * 0.05 +
            extraction_metrics.extraction_accuracy * 0.20 +
            extraction_metrics.text_preservation * 0.10 +
            extraction_metrics.formatting_retention * 0.05 +
            (1.0 - extraction_metrics.error_rate) * 0.05 +
            processing_metrics.chunking_quality * 0.15 +
            processing_metrics.embedding_reliability * 0.10 +
            processing_metrics.metadata_completeness * 0.05 +
            processing_metrics.cost_efficiency * 0.05 +
            processing_metrics.processing_speed * 0.05
        )
        
        overall_score = max(0.0, min(1.0, overall_score))
        
        # Generate grade and recommendations
        quality_grade = self.grade_quality_score(overall_score)
        recommendations = self.generate_recommendations(
            content_metrics, extraction_metrics, processing_metrics
        )
        
        # Detect issues
        issues_detected = []
        if extraction_metrics.error_rate > 0.2:
            issues_detected.append("High extraction error rate")
        if processing_metrics.chunking_quality < 0.5:
            issues_detected.append("Poor chunking quality")
        if content_metrics.coherence_score < 0.5:
            issues_detected.append("Low content coherence")
        
        # Confidence based on data quality
        confidence = min(1.0, (
            result.processing_quality.confidence_score * 0.5 +
            (1.0 - extraction_metrics.error_rate) * 0.3 +
            processing_metrics.metadata_completeness * 0.2
        ))
        
        return QualityAssessmentResult(
            overall_quality_score=overall_score,
            content_metrics=content_metrics,
            extraction_metrics=extraction_metrics,
            processing_metrics=processing_metrics,
            quality_grade=quality_grade,
            recommendations=recommendations,
            issues_detected=issues_detected,
            confidence=confidence
        )

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Global assessor instance
_default_assessor = None

def get_default_assessor() -> QualityAssessmentEngine:
    """Get the default quality assessor instance."""
    global _default_assessor
    if _default_assessor is None:
        _default_assessor = QualityAssessmentEngine()
    return _default_assessor

def assess_quality(result: FileProcessingResult) -> QualityAssessmentResult:
    """Assess quality using default assessor."""
    assessor = get_default_assessor()
    return assessor.assess_processing_result(result)

def quick_quality_check(text: str) -> float:
    """Quick quality check returning overall score."""
    assessor = get_default_assessor()
    content_metrics = assessor.assess_content_quality(text)
    
    # Simple overall score calculation
    return (
        content_metrics.readability_score / 100.0 * 0.3 +
        content_metrics.coherence_score * 0.3 +
        content_metrics.information_density * 0.2 +
        content_metrics.language_consistency * 0.2
    )