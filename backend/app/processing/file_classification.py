"""
AI-Powered File Type Classification for brAIn v2.0

This module provides advanced file type detection using content analysis,
machine learning patterns, and AI-powered classification beyond MIME types.

Author: BMad Team
"""

import hashlib
import mimetypes
import os
import struct
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple

# Core imports
from pydantic import BaseModel, Field
import magic  # python-magic for file type detection

# AI imports
import anthropic
import openai
from langfuse import Langfuse

# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class FileSignature(BaseModel):
    """File signature analysis result."""
    magic_bytes: str = Field(description="Hex representation of magic bytes")
    mime_type: str = Field(description="MIME type from magic bytes")
    file_extension: str = Field(description="Detected file extension")
    confidence: float = Field(ge=0.0, le=1.0, description="Detection confidence")

class ContentAnalysis(BaseModel):
    """Content-based file analysis result."""
    language_detected: Optional[str] = Field(None, description="Detected language")
    structure_type: str = Field(description="Document structure type")
    content_patterns: List[str] = Field(default_factory=list, description="Detected content patterns")
    complexity_score: float = Field(ge=0.0, le=1.0, description="Content complexity")
    metadata_richness: float = Field(ge=0.0, le=1.0, description="Metadata richness score")

class FileTypeAnalysis(BaseModel):
    """Comprehensive file type analysis result."""
    detected_format: str = Field(description="Final detected format")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Overall confidence")
    detection_method: str = Field(description="Primary detection method used")
    signature_analysis: FileSignature
    content_analysis: ContentAnalysis
    ai_analysis: Dict[str, Any] = Field(default_factory=dict, description="AI analysis results")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    processing_recommendations: List[str] = Field(default_factory=list)

# =============================================================================
# FILE CLASSIFICATION PATTERNS
# =============================================================================

class FilePatternDatabase:
    """Database of file patterns for intelligent classification."""
    
    # Magic byte signatures for advanced detection
    MAGIC_SIGNATURES = {
        # PDF variants
        b'\x25\x50\x44\x46': 'pdf',
        b'\x25\x21\x50\x53': 'postscript',
        
        # Office documents
        b'\x50\x4B\x03\x04': 'office_zip',  # DOCX, XLSX, PPTX
        b'\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1': 'office_ole',  # DOC, XLS, PPT
        
        # Images
        b'\xFF\xD8\xFF': 'jpeg',
        b'\x89\x50\x4E\x47\x0D\x0A\x1A\x0A': 'png',
        b'\x47\x49\x46\x38': 'gif',
        
        # Archives
        b'\x50\x4B\x03\x04': 'zip',
        b'\x52\x61\x72\x21\x1A\x07': 'rar',
        b'\x1F\x8B\x08': 'gzip',
        
        # Executables
        b'\x4D\x5A': 'exe',
        b'\x7F\x45\x4C\x46': 'elf',
        
        # Media
        b'\x49\x44\x33': 'mp3',
        b'\xFF\xFB': 'mp3_alt',
        b'\x00\x00\x00\x18\x66\x74\x79\x70': 'mp4',
    }
    
    # Content patterns for text-based analysis
    CONTENT_PATTERNS = {
        'json': [r'^\s*\{.*\}\s*$', r'"[^"]*"\s*:\s*'],
        'xml': [r'<\?xml\s+version', r'<[^>]+>[^<]*</[^>]+>'],
        'html': [r'<!DOCTYPE\s+html', r'<html[^>]*>', r'<head>', r'<body>'],
        'css': [r'[^{]*\{[^}]*\}', r'@media\s+', r'@import\s+'],
        'javascript': [r'function\s+\w+\s*\(', r'var\s+\w+\s*=', r'console\.log\s*\('],
        'python': [r'def\s+\w+\s*\(', r'import\s+\w+', r'if\s+__name__\s*==\s*["\']__main__["\']'],
        'sql': [r'SELECT\s+.*\s+FROM', r'INSERT\s+INTO', r'CREATE\s+TABLE'],
        'yaml': [r'^[^:\s]+:\s*$', r'^\s*-\s+'],
        'markdown': [r'^#+\s+', r'\*\*[^*]+\*\*', r'\[[^\]]+\]\([^)]+\)'],
        'log': [r'\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}', r'\[ERROR\]', r'\[INFO\]', r'\[DEBUG\]']
    }
    
    # Domain-specific patterns
    DOMAIN_PATTERNS = {
        'legal': [
            r'WHEREAS,?', r'THEREFORE,?', r'IN WITNESS WHEREOF',
            r'Agreement', r'Contract', r'Plaintiff', r'Defendant'
        ],
        'medical': [
            r'Patient\s+ID', r'Diagnosis:', r'Treatment:', r'Medication:',
            r'mg/kg', r'ml/hr', r'Patient presents with'
        ],
        'financial': [
            r'\$[\d,]+\.?\d*', r'Account\s+Number', r'Balance:',
            r'Credit', r'Debit', r'Invoice', r'Statement'
        ],
        'technical': [
            r'API', r'endpoint', r'function\(', r'class\s+\w+',
            r'// Comment', r'/* Comment', r'TODO:', r'FIXME:'
        ],
        'academic': [
            r'Abstract:', r'Keywords:', r'References?:',
            r'Figure\s+\d+', r'Table\s+\d+', r'et al\.', r'doi:'
        ]
    }

# =============================================================================
# AI FILE CLASSIFIER
# =============================================================================

class AIFileClassifier:
    """
    AI-powered file classifier with content analysis and pattern recognition.
    """
    
    def __init__(
        self,
        ai_client: Optional[Union[anthropic.Anthropic, openai.OpenAI]] = None,
        langfuse_client: Optional[Langfuse] = None
    ):
        """
        Initialize AI file classifier.
        
        Args:
            ai_client: AI client for intelligent analysis
            langfuse_client: Langfuse for monitoring
        """
        self.ai_client = ai_client
        self.langfuse = langfuse_client
        self.pattern_db = FilePatternDatabase()
        
        # Initialize magic library for file type detection
        try:
            self.magic = magic.Magic(mime=True)
            self.magic_available = True
        except Exception:
            self.magic = None
            self.magic_available = False
            print("Warning: python-magic not available, using basic detection")
        
        # Classification statistics
        self.stats = {
            "total_classifications": 0,
            "successful_ai_analysis": 0,
            "pattern_matches": 0,
            "signature_detections": 0
        }
    
    async def classify_with_content_analysis(
        self,
        file_path: Union[str, Path],
        domain_context: Optional[str] = None
    ) -> FileTypeAnalysis:
        """
        Classify file using comprehensive content analysis.
        
        Args:
            file_path: Path to file to classify
            domain_context: Domain context for specialized detection
            
        Returns:
            Comprehensive file type analysis
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        try:
            # Step 1: Signature-based analysis
            signature_analysis = await self._analyze_file_signature(file_path)
            
            # Step 2: Content-based analysis
            content_analysis = await self._analyze_file_content(file_path, domain_context)
            
            # Step 3: AI-powered analysis (if available)
            ai_analysis = {}
            if self.ai_client:
                ai_analysis = await self._ai_content_analysis(file_path, content_analysis)
            
            # Step 4: Combine results for final classification
            final_classification = await self._combine_analysis_results(
                signature_analysis, content_analysis, ai_analysis, domain_context
            )
            
            # Step 5: Generate processing recommendations
            recommendations = await self._generate_processing_recommendations(
                final_classification, signature_analysis, content_analysis
            )
            
            # Create comprehensive result
            result = FileTypeAnalysis(
                detected_format=final_classification["format"],
                confidence_score=final_classification["confidence"],
                detection_method=final_classification["method"],
                signature_analysis=signature_analysis,
                content_analysis=content_analysis,
                ai_analysis=ai_analysis,
                metadata={
                    "file_size": file_path.stat().st_size,
                    "file_extension": file_path.suffix.lower(),
                    "mime_type_guess": mimetypes.guess_type(file_path)[0],
                    "domain_context": domain_context
                },
                processing_recommendations=recommendations
            )
            
            # Update statistics
            self.stats["total_classifications"] += 1
            if ai_analysis:
                self.stats["successful_ai_analysis"] += 1
            if content_analysis.content_patterns:
                self.stats["pattern_matches"] += 1
            
            return result
            
        except Exception as e:
            # Return basic classification on error
            basic_mime = mimetypes.guess_type(file_path)[0] or "application/octet-stream"
            return FileTypeAnalysis(
                detected_format=basic_mime.split('/')[-1],
                confidence_score=0.3,
                detection_method="fallback",
                signature_analysis=FileSignature(
                    magic_bytes="",
                    mime_type=basic_mime,
                    file_extension=file_path.suffix.lower(),
                    confidence=0.3
                ),
                content_analysis=ContentAnalysis(
                    structure_type="unknown",
                    complexity_score=0.0,
                    metadata_richness=0.0
                ),
                metadata={"error": str(e)}
            )
    
    async def _analyze_file_signature(self, file_path: Path) -> FileSignature:
        """Analyze file signature using magic bytes."""
        try:
            # Read first 1024 bytes for signature analysis
            with open(file_path, 'rb') as f:
                header = f.read(1024)
            
            # Get magic bytes hex representation
            magic_hex = header[:16].hex()
            
            # Check against known signatures
            detected_format = None
            confidence = 0.5
            
            for signature, format_name in self.pattern_db.MAGIC_SIGNATURES.items():
                if header.startswith(signature):
                    detected_format = format_name
                    confidence = 0.9
                    break
            
            # Use python-magic if available
            mime_type = "application/octet-stream"
            if self.magic_available:
                try:
                    mime_type = self.magic.from_file(str(file_path))
                    if not detected_format:
                        detected_format = mime_type.split('/')[-1]
                        confidence = 0.8
                except Exception:
                    pass
            
            # Fallback to file extension
            if not detected_format:
                detected_format = file_path.suffix.lower().lstrip('.')
                confidence = 0.4
            
            return FileSignature(
                magic_bytes=magic_hex,
                mime_type=mime_type,
                file_extension=detected_format,
                confidence=confidence
            )
            
        except Exception as e:
            return FileSignature(
                magic_bytes="",
                mime_type="application/octet-stream",
                file_extension=file_path.suffix.lower().lstrip('.'),
                confidence=0.1
            )
    
    async def _analyze_file_content(
        self, 
        file_path: Path, 
        domain_context: Optional[str]
    ) -> ContentAnalysis:
        """Analyze file content for patterns and structure."""
        try:
            # Try to read as text
            content = ""
            encoding_used = "utf-8"
            
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read(8192)  # Read first 8KB
                        encoding_used = encoding
                        break
                except UnicodeDecodeError:
                    continue
            
            if not content:
                # Binary file
                return ContentAnalysis(
                    structure_type="binary",
                    complexity_score=0.0,
                    metadata_richness=0.0
                )
            
            # Analyze content patterns
            detected_patterns = []
            structure_type = "text"
            
            # Check for specific content patterns
            import re
            for pattern_type, patterns in self.pattern_db.CONTENT_PATTERNS.items():
                for pattern in patterns:
                    if re.search(pattern, content, re.IGNORECASE | re.MULTILINE):
                        detected_patterns.append(pattern_type)
                        structure_type = pattern_type
                        break
                if detected_patterns:
                    break
            
            # Check domain-specific patterns if context provided
            if domain_context and domain_context in self.pattern_db.DOMAIN_PATTERNS:
                domain_patterns = self.pattern_db.DOMAIN_PATTERNS[domain_context]
                for pattern in domain_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        detected_patterns.append(f"domain_{domain_context}")
                        break
            
            # Calculate complexity and metadata richness
            complexity_score = self._calculate_content_complexity(content)
            metadata_richness = self._calculate_metadata_richness(content, detected_patterns)
            
            # Language detection (basic)
            language_detected = self._detect_language_basic(content)
            
            return ContentAnalysis(
                language_detected=language_detected,
                structure_type=structure_type,
                content_patterns=detected_patterns,
                complexity_score=complexity_score,
                metadata_richness=metadata_richness
            )
            
        except Exception as e:
            return ContentAnalysis(
                structure_type="unknown",
                complexity_score=0.0,
                metadata_richness=0.0
            )
    
    async def _ai_content_analysis(
        self, 
        file_path: Path, 
        content_analysis: ContentAnalysis
    ) -> Dict[str, Any]:
        """Use AI for advanced content analysis."""
        if not self.ai_client:
            return {}
        
        try:
            # Read sample content for AI analysis
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                sample_content = f.read(2048)  # 2KB sample
            
            # Create AI prompt for classification
            prompt = f"""
            Analyze this file content sample and provide classification insights:
            
            File: {file_path.name}
            Content Sample:
            {sample_content[:1500]}
            
            Please analyze and return:
            1. Most likely document type/format
            2. Content category (technical, legal, medical, academic, etc.)
            3. Processing complexity (simple, moderate, complex)
            4. Key characteristics observed
            5. Confidence level (0-100)
            
            Format response as JSON.
            """
            
            # Call AI service based on client type
            if isinstance(self.ai_client, anthropic.Anthropic):
                response = await self._call_anthropic_analysis(prompt)
            elif isinstance(self.ai_client, openai.OpenAI):
                response = await self._call_openai_analysis(prompt)
            else:
                response = {}
            
            return response
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _call_anthropic_analysis(self, prompt: str) -> Dict[str, Any]:
        """Call Anthropic for content analysis."""
        try:
            message = self.ai_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=500,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse response
            response_text = message.content[0].text
            
            # Try to extract JSON from response
            import json
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
            # If no JSON found, parse manually
            return {
                "document_type": "text",
                "category": "general",
                "complexity": "moderate",
                "confidence": 70,
                "raw_response": response_text
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _call_openai_analysis(self, prompt: str) -> Dict[str, Any]:
        """Call OpenAI for content analysis."""
        try:
            response = self.ai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0
            )
            
            response_text = response.choices[0].message.content
            
            # Try to extract JSON from response
            import json
            import re
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
            return {
                "document_type": "text",
                "category": "general",
                "complexity": "moderate",
                "confidence": 70,
                "raw_response": response_text
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    async def _combine_analysis_results(
        self,
        signature_analysis: FileSignature,
        content_analysis: ContentAnalysis,
        ai_analysis: Dict[str, Any],
        domain_context: Optional[str]
    ) -> Dict[str, Any]:
        """Combine all analysis results for final classification."""
        
        # Weight different detection methods
        signature_weight = 0.4
        content_weight = 0.4
        ai_weight = 0.2
        
        # Determine final format
        candidates = []
        
        # Add signature-based candidate
        candidates.append({
            "format": signature_analysis.file_extension,
            "confidence": signature_analysis.confidence * signature_weight,
            "method": "signature"
        })
        
        # Add content-based candidate
        if content_analysis.content_patterns:
            best_pattern = content_analysis.content_patterns[0]
            candidates.append({
                "format": best_pattern,
                "confidence": (len(content_analysis.content_patterns) / 3) * content_weight,
                "method": "content_pattern"
            })
        
        # Add AI-based candidate if available
        if ai_analysis and "document_type" in ai_analysis:
            ai_confidence = ai_analysis.get("confidence", 70) / 100.0
            candidates.append({
                "format": ai_analysis["document_type"],
                "confidence": ai_confidence * ai_weight,
                "method": "ai_analysis"
            })
        
        # Select best candidate
        if candidates:
            best_candidate = max(candidates, key=lambda x: x["confidence"])
            return best_candidate
        
        # Fallback
        return {
            "format": signature_analysis.file_extension,
            "confidence": 0.3,
            "method": "fallback"
        }
    
    async def _generate_processing_recommendations(
        self,
        classification: Dict[str, Any],
        signature_analysis: FileSignature,
        content_analysis: ContentAnalysis
    ) -> List[str]:
        """Generate processing recommendations based on analysis."""
        recommendations = []
        
        # Confidence-based recommendations
        if classification["confidence"] < 0.5:
            recommendations.append("Low confidence - consider manual format specification")
        
        # Content complexity recommendations
        if content_analysis.complexity_score > 0.8:
            recommendations.append("High complexity content - use quality-focused processing")
        elif content_analysis.complexity_score < 0.3:
            recommendations.append("Simple content - speed-optimized processing suitable")
        
        # Format-specific recommendations
        format_name = classification["format"]
        if format_name in ["pdf", "docx", "doc"]:
            recommendations.append("Structured document - enable metadata extraction")
        elif format_name in ["txt", "md", "log"]:
            recommendations.append("Plain text - enable advanced text analysis")
        elif format_name in ["json", "xml", "yaml"]:
            recommendations.append("Structured data - enable parsing validation")
        
        return recommendations
    
    def _calculate_content_complexity(self, content: str) -> float:
        """Calculate content complexity score."""
        if not content:
            return 0.0
        
        import re
        
        factors = {
            "special_chars": len(re.findall(r'[^\w\s]', content)) / len(content),
            "numbers": len(re.findall(r'\d+', content)) / max(len(content.split()), 1),
            "capitals": len(re.findall(r'[A-Z]', content)) / len(content),
            "line_variance": self._calculate_line_length_variance(content),
            "vocabulary_richness": self._calculate_vocabulary_richness(content)
        }
        
        # Weighted average
        complexity = (
            factors["special_chars"] * 0.2 +
            factors["numbers"] * 0.15 +
            factors["capitals"] * 0.15 +
            factors["line_variance"] * 0.25 +
            factors["vocabulary_richness"] * 0.25
        )
        
        return min(complexity, 1.0)
    
    def _calculate_metadata_richness(self, content: str, patterns: List[str]) -> float:
        """Calculate metadata richness score."""
        if not content:
            return 0.0
        
        richness_factors = 0.0
        
        # Pattern diversity
        richness_factors += min(len(patterns) / 5.0, 1.0) * 0.4
        
        # Structural elements (headers, lists, etc.)
        import re
        structural_elements = len(re.findall(r'^#+\s|^\*\s|^\d+\.\s|^-\s', content, re.MULTILINE))
        richness_factors += min(structural_elements / 20.0, 1.0) * 0.3
        
        # Formatting indicators
        formatting_indicators = len(re.findall(r'\*\*[^*]+\*\*|__[^_]+__|`[^`]+`', content))
        richness_factors += min(formatting_indicators / 10.0, 1.0) * 0.3
        
        return min(richness_factors, 1.0)
    
    def _calculate_line_length_variance(self, content: str) -> float:
        """Calculate variance in line lengths."""
        lines = content.split('\n')
        if len(lines) < 2:
            return 0.0
        
        lengths = [len(line) for line in lines]
        mean_length = sum(lengths) / len(lengths)
        variance = sum((x - mean_length) ** 2 for x in lengths) / len(lengths)
        
        # Normalize to 0-1 scale
        return min(variance / 1000.0, 1.0)
    
    def _calculate_vocabulary_richness(self, content: str) -> float:
        """Calculate vocabulary richness (unique words / total words)."""
        words = content.lower().split()
        if not words:
            return 0.0
        
        unique_words = len(set(words))
        total_words = len(words)
        
        return unique_words / total_words
    
    def _detect_language_basic(self, content: str) -> Optional[str]:
        """Basic language detection."""
        # Very simple language detection based on common patterns
        import re
        
        # English indicators
        english_patterns = r'\b(the|and|or|but|in|on|at|to|for|of|with|by)\b'
        english_matches = len(re.findall(english_patterns, content, re.IGNORECASE))
        
        # If we have enough English indicators, assume English
        if english_matches > 5:
            return "english"
        
        # Could add more sophisticated detection here
        return None
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get classifier health status."""
        return {
            "magic_available": self.magic_available,
            "ai_client_available": self.ai_client is not None,
            "statistics": self.stats.copy(),
            "pattern_database_size": {
                "magic_signatures": len(self.pattern_db.MAGIC_SIGNATURES),
                "content_patterns": len(self.pattern_db.CONTENT_PATTERNS),
                "domain_patterns": len(self.pattern_db.DOMAIN_PATTERNS)
            }
        }

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Global classifier instance
_default_classifier = None

def get_default_classifier() -> AIFileClassifier:
    """Get default AI file classifier."""
    global _default_classifier
    if _default_classifier is None:
        _default_classifier = AIFileClassifier()
    return _default_classifier

async def classify_file_ai(
    file_path: Union[str, Path],
    domain_context: Optional[str] = None
) -> FileTypeAnalysis:
    """Classify file using AI with default classifier."""
    classifier = get_default_classifier()
    return await classifier.classify_with_content_analysis(file_path, domain_context)