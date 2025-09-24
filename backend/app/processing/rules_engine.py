"""
Custom Extraction Rules Engine for brAIn v2.0

This module provides a flexible rules engine for custom document extraction patterns,
allowing users to define domain-specific extraction logic and content parsing rules.

Author: BMad Team
"""

import re
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, Tuple

# Core imports
from pydantic import BaseModel, Field, field_validator
from enum import Enum

# =============================================================================
# PYDANTIC MODELS
# =============================================================================


class RuleType(str, Enum):
    """Types of extraction rules."""

    REGEX = "regex"
    XPATH = "xpath"
    JSON_PATH = "json_path"
    PATTERN_MATCH = "pattern_match"
    CONDITIONAL = "conditional"
    TRANSFORMATION = "transformation"
    VALIDATION = "validation"


class RulePriority(str, Enum):
    """Rule execution priority levels."""

    CRITICAL = "critical"  # Must execute first
    HIGH = "high"  # High priority
    NORMAL = "normal"  # Standard priority
    LOW = "low"  # Low priority
    CLEANUP = "cleanup"  # Post-processing cleanup


class CustomRule(BaseModel):
    """Definition of a custom extraction rule."""

    rule_id: str = Field(description="Unique rule identifier")
    name: str = Field(description="Human-readable rule name")
    description: str = Field(description="Rule description and purpose")
    rule_type: RuleType = Field(description="Type of rule")
    priority: RulePriority = Field(default=RulePriority.NORMAL)
    pattern: str = Field(description="Rule pattern (regex, xpath, etc.)")
    replacement: Optional[str] = Field(None, description="Replacement pattern")
    conditions: Dict[str, Any] = Field(
        default_factory=dict, description="Rule conditions"
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Rule metadata")
    enabled: bool = Field(default=True, description="Whether rule is active")
    created_at: datetime = Field(default_factory=datetime.now)


class RuleExecutionResult(BaseModel):
    """Result of rule execution."""

    rule_id: str = Field(description="ID of executed rule")
    success: bool = Field(description="Whether rule executed successfully")
    matches_found: int = Field(ge=0, description="Number of matches found")
    extracted_content: List[str] = Field(
        default_factory=list, description="Extracted content"
    )
    transformed_content: Optional[str] = Field(None, description="Transformed content")
    execution_time: float = Field(ge=0.0, description="Execution time in seconds")
    error_message: Optional[str] = Field(None, description="Error if execution failed")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Execution metadata"
    )


class RuleSet(BaseModel):
    """Collection of related extraction rules."""

    ruleset_id: str = Field(description="Unique ruleset identifier")
    name: str = Field(description="Ruleset name")
    description: str = Field(description="Ruleset description")
    rules: List[CustomRule] = Field(default_factory=list, description="Rules in set")
    domain: Optional[str] = Field(
        None, description="Domain context (legal, medical, etc.)"
    )
    file_types: List[str] = Field(
        default_factory=list, description="Supported file types"
    )
    version: str = Field(default="1.0", description="Ruleset version")
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Ruleset metadata"
    )


class ExtractionJob(BaseModel):
    """Extraction job using custom rules."""

    job_id: str = Field(description="Unique job identifier")
    file_path: str = Field(description="Path to file being processed")
    ruleset: RuleSet = Field(description="Ruleset to apply")
    content: str = Field(description="Content to process")
    execution_results: List[RuleExecutionResult] = Field(default_factory=list)
    overall_success: bool = Field(default=True)
    total_execution_time: float = Field(ge=0.0, description="Total processing time")
    created_at: datetime = Field(default_factory=datetime.now)


# =============================================================================
# RULES ENGINE
# =============================================================================


class ExtractionRulesEngine:
    """
    Flexible rules engine for custom document extraction patterns.
    Supports multiple rule types and execution strategies.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize rules engine.

        Args:
            config: Configuration options for the engine
        """
        self.config = config or {}

        # Built-in rule patterns for common use cases
        self.builtin_patterns = self._initialize_builtin_patterns()

        # Rule execution statistics
        self.stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "total_matches": 0,
            "rules_registered": 0,
        }

        # Rule registry
        self.rule_registry: Dict[str, CustomRule] = {}
        self.ruleset_registry: Dict[str, RuleSet] = {}

    def register_rule(self, rule: CustomRule) -> None:
        """Register a custom rule."""
        self.rule_registry[rule.rule_id] = rule
        self.stats["rules_registered"] += 1

    def register_ruleset(self, ruleset: RuleSet) -> None:
        """Register a ruleset."""
        self.ruleset_registry[ruleset.ruleset_id] = ruleset
        for rule in ruleset.rules:
            self.register_rule(rule)

    async def compile_rules(
        self, rule_definitions: List[Union[str, Dict[str, Any], CustomRule]]
    ) -> RuleSet:
        """
        Compile rule definitions into an executable ruleset.

        Args:
            rule_definitions: List of rule definitions (strings, dicts, or CustomRule objects)

        Returns:
            Compiled ruleset ready for execution
        """
        rules = []

        for i, rule_def in enumerate(rule_definitions):
            try:
                if isinstance(rule_def, str):
                    # String rule - treat as regex pattern
                    rule = CustomRule(
                        rule_id=f"rule_{i}",
                        name=f"Pattern Rule {i+1}",
                        description=f"Regex pattern: {rule_def[:50]}...",
                        rule_type=RuleType.REGEX,
                        pattern=rule_def,
                        replacement=None,
                    )
                elif isinstance(rule_def, dict):
                    # Dictionary rule definition
                    rule = CustomRule(**rule_def)
                elif isinstance(rule_def, CustomRule):
                    # Already a CustomRule object
                    rule = rule_def
                else:
                    continue

                rules.append(rule)

            except Exception as e:
                print(f"Warning: Failed to compile rule {i}: {e}")
                continue

        # Create ruleset
        ruleset = RuleSet(
            ruleset_id=f"compiled_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            name="Compiled Ruleset",
            description=f"Compiled from {len(rule_definitions)} rule definitions",
            rules=rules,
        )

        return ruleset

    async def extract_with_rules(
        self,
        file_path: Union[str, Path],
        ruleset: RuleSet,
        optimization_strategy: Any = None,
    ) -> Dict[str, Any]:
        """
        Extract content using custom rules.

        Args:
            file_path: Path to file to process
            ruleset: Rules to apply
            optimization_strategy: Optional optimization strategy

        Returns:
            Extraction results
        """
        start_time = datetime.now()
        file_path = Path(file_path)

        try:
            # Read file content
            content = await self._read_file_content(file_path)

            # Create extraction job
            job = ExtractionJob(
                job_id=f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                file_path=str(file_path),
                ruleset=ruleset,
                content=content,
            )

            # Execute rules in priority order
            execution_results = await self._execute_ruleset(job, content)

            # Process results
            final_content = await self._process_execution_results(
                execution_results, content
            )

            # Calculate metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            total_matches = sum(result.matches_found for result in execution_results)

            # Update statistics
            self.stats["total_executions"] += 1
            self.stats["successful_executions"] += 1
            self.stats["total_matches"] += total_matches

            return {
                "content": final_content,
                "token_count": len(final_content.split()),
                "cost": 0.0,  # Rules-based extraction has no AI cost
                "preview": final_content[:500] if final_content else "",
                "extraction_method": "rules_engine",
                "rules_applied": len(execution_results),
                "matches_found": total_matches,
                "processing_time": processing_time,
                "execution_results": [result.dict() for result in execution_results],
            }

        except Exception as e:
            self.stats["failed_executions"] += 1
            raise RuntimeError(f"Rules extraction failed: {str(e)}")

    async def _read_file_content(self, file_path: Path) -> str:
        """Read file content with encoding detection."""
        encodings = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]

        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue

        # If all encodings fail, read as binary and decode with errors
        with open(file_path, "rb") as f:
            content = f.read()
            return content.decode("utf-8", errors="replace")

    async def _execute_ruleset(
        self, job: ExtractionJob, content: str
    ) -> List[RuleExecutionResult]:
        """Execute all rules in a ruleset."""
        results = []

        # Sort rules by priority
        sorted_rules = sorted(
            job.ruleset.rules,
            key=lambda r: self._get_priority_order(r.priority),
            reverse=True,
        )

        current_content = content

        for rule in sorted_rules:
            if not rule.enabled:
                continue

            try:
                result = await self._execute_single_rule(rule, current_content)
                results.append(result)

                # Update content if rule produced transformation
                if result.transformed_content is not None:
                    current_content = result.transformed_content

            except Exception as e:
                error_result = RuleExecutionResult(
                    rule_id=rule.rule_id,
                    success=False,
                    matches_found=0,
                    execution_time=0.0,
                    error_message=str(e),
                    transformed_content=None,
                )
                results.append(error_result)

        return results

    async def _execute_single_rule(
        self, rule: CustomRule, content: str
    ) -> RuleExecutionResult:
        """Execute a single extraction rule."""
        start_time = datetime.now()

        try:
            if rule.rule_type == RuleType.REGEX:
                return await self._execute_regex_rule(rule, content, start_time)
            elif rule.rule_type == RuleType.PATTERN_MATCH:
                return await self._execute_pattern_rule(rule, content, start_time)
            elif rule.rule_type == RuleType.CONDITIONAL:
                return await self._execute_conditional_rule(rule, content, start_time)
            elif rule.rule_type == RuleType.TRANSFORMATION:
                return await self._execute_transformation_rule(
                    rule, content, start_time
                )
            elif rule.rule_type == RuleType.VALIDATION:
                return await self._execute_validation_rule(rule, content, start_time)
            elif rule.rule_type == RuleType.JSON_PATH:
                return await self._execute_json_path_rule(rule, content, start_time)
            elif rule.rule_type == RuleType.XPATH:
                return await self._execute_xpath_rule(rule, content, start_time)
            else:
                raise ValueError(f"Unknown rule type: {rule.rule_type}")

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return RuleExecutionResult(
                rule_id=rule.rule_id,
                success=False,
                matches_found=0,
                execution_time=execution_time,
                error_message=str(e),
            )

    async def _execute_regex_rule(
        self, rule: CustomRule, content: str, start_time: datetime
    ) -> RuleExecutionResult:
        """Execute a regex-based extraction rule."""
        try:
            # Compile regex pattern
            flags = re.IGNORECASE | re.MULTILINE | re.DOTALL
            pattern = re.compile(rule.pattern, flags)

            # Find all matches
            matches = pattern.findall(content)

            # Apply replacement if specified
            transformed_content = None
            if rule.replacement is not None:
                transformed_content = pattern.sub(rule.replacement, content)

            execution_time = (datetime.now() - start_time).total_seconds()

            return RuleExecutionResult(
                rule_id=rule.rule_id,
                success=True,
                matches_found=len(matches),
                extracted_content=[str(match) for match in matches],
                transformed_content=transformed_content,
                execution_time=execution_time,
                metadata={
                    "pattern": rule.pattern,
                    "flags": "IGNORECASE|MULTILINE|DOTALL",
                },
            )

        except re.error as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return RuleExecutionResult(
                rule_id=rule.rule_id,
                success=False,
                matches_found=0,
                execution_time=execution_time,
                error_message=f"Invalid regex pattern: {str(e)}",
            )

    async def _execute_pattern_rule(
        self, rule: CustomRule, content: str, start_time: datetime
    ) -> RuleExecutionResult:
        """Execute a pattern matching rule."""
        extracted_content = []

        # Built-in patterns
        if rule.pattern in self.builtin_patterns:
            pattern_func = self.builtin_patterns[rule.pattern]
            matches = pattern_func(content)
            extracted_content = [str(match) for match in matches]
        else:
            # Custom pattern logic could be implemented here
            extracted_content = []

        execution_time = (datetime.now() - start_time).total_seconds()

        return RuleExecutionResult(
            rule_id=rule.rule_id,
            success=True,
            matches_found=len(extracted_content),
            extracted_content=extracted_content,
            execution_time=execution_time,
        )

    async def _execute_conditional_rule(
        self, rule: CustomRule, content: str, start_time: datetime
    ) -> RuleExecutionResult:
        """Execute a conditional rule."""
        conditions = rule.conditions
        extracted_content = []

        # Simple condition evaluation
        condition_met = True

        # Check content length condition
        if "min_length" in conditions:
            condition_met &= len(content) >= conditions["min_length"]

        if "max_length" in conditions:
            condition_met &= len(content) <= conditions["max_length"]

        # Check pattern existence
        if "contains" in conditions:
            for pattern in conditions["contains"]:
                condition_met &= pattern in content

        if "not_contains" in conditions:
            for pattern in conditions["not_contains"]:
                condition_met &= pattern not in content

        # If condition is met, extract using the pattern
        if condition_met and rule.pattern:
            try:
                pattern = re.compile(rule.pattern, re.IGNORECASE | re.MULTILINE)
                matches = pattern.findall(content)
                extracted_content = [str(match) for match in matches]
            except re.error:
                pass

        execution_time = (datetime.now() - start_time).total_seconds()

        return RuleExecutionResult(
            rule_id=rule.rule_id,
            success=True,
            matches_found=len(extracted_content),
            extracted_content=extracted_content,
            execution_time=execution_time,
            metadata={"condition_met": condition_met, "conditions": conditions},
        )

    async def _execute_transformation_rule(
        self, rule: CustomRule, content: str, start_time: datetime
    ) -> RuleExecutionResult:
        """Execute a content transformation rule."""
        try:
            transformed_content = content

            # Apply transformation based on pattern
            if rule.pattern == "normalize_whitespace":
                transformed_content = re.sub(r"\s+", " ", content).strip()
            elif rule.pattern == "remove_empty_lines":
                transformed_content = "\n".join(
                    line for line in content.split("\n") if line.strip()
                )
            elif rule.pattern == "lowercase":
                transformed_content = content.lower()
            elif rule.pattern == "uppercase":
                transformed_content = content.upper()
            elif rule.pattern == "remove_special_chars":
                transformed_content = re.sub(r"[^\w\s]", "", content)
            elif rule.pattern.startswith("replace:"):
                # Format: "replace:old_text:new_text"
                parts = rule.pattern.split(":", 2)
                if len(parts) == 3:
                    _, old_text, new_text = parts
                    transformed_content = content.replace(old_text, new_text)
            else:
                # Treat as regex replacement
                if rule.replacement:
                    pattern = re.compile(rule.pattern, re.IGNORECASE | re.MULTILINE)
                    transformed_content = pattern.sub(rule.replacement, content)

            execution_time = (datetime.now() - start_time).total_seconds()

            return RuleExecutionResult(
                rule_id=rule.rule_id,
                success=True,
                matches_found=1 if transformed_content != content else 0,
                transformed_content=transformed_content,
                execution_time=execution_time,
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return RuleExecutionResult(
                rule_id=rule.rule_id,
                success=False,
                matches_found=0,
                execution_time=execution_time,
                error_message=str(e),
            )

    async def _execute_validation_rule(
        self, rule: CustomRule, content: str, start_time: datetime
    ) -> RuleExecutionResult:
        """Execute a validation rule."""
        validation_passed = True
        validation_messages = []

        try:
            # Length validation
            if "min_length" in rule.conditions:
                if len(content) < rule.conditions["min_length"]:
                    validation_passed = False
                    validation_messages.append(
                        f"Content too short: {len(content)} < {rule.conditions['min_length']}"
                    )

            if "max_length" in rule.conditions:
                if len(content) > rule.conditions["max_length"]:
                    validation_passed = False
                    validation_messages.append(
                        f"Content too long: {len(content)} > {rule.conditions['max_length']}"
                    )

            # Pattern validation
            if rule.pattern:
                pattern = re.compile(rule.pattern, re.IGNORECASE)
                if not pattern.search(content):
                    validation_passed = False
                    validation_messages.append(
                        f"Required pattern not found: {rule.pattern}"
                    )

            execution_time = (datetime.now() - start_time).total_seconds()

            return RuleExecutionResult(
                rule_id=rule.rule_id,
                success=True,
                matches_found=1 if validation_passed else 0,
                extracted_content=(
                    validation_messages
                    if not validation_passed
                    else ["Validation passed"]
                ),
                execution_time=execution_time,
                metadata={"validation_passed": validation_passed},
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return RuleExecutionResult(
                rule_id=rule.rule_id,
                success=False,
                matches_found=0,
                execution_time=execution_time,
                error_message=str(e),
            )

    async def _execute_json_path_rule(
        self, rule: CustomRule, content: str, start_time: datetime
    ) -> RuleExecutionResult:
        """Execute a JSONPath extraction rule."""
        try:
            # Parse JSON content
            json_data = json.loads(content)

            # Simple JSONPath implementation for basic paths
            path_parts = rule.pattern.strip("$.").split(".")
            current_data = json_data
            extracted_content = []

            try:
                for part in path_parts:
                    if "[" in part and "]" in part:
                        # Array access
                        key, index_str = part.split("[", 1)
                        index = int(index_str.rstrip("]"))
                        if key:
                            current_data = current_data[key][index]
                        else:
                            current_data = current_data[index]
                    else:
                        # Object access
                        current_data = current_data[part]

                if isinstance(current_data, list):
                    extracted_content = [str(item) for item in current_data]
                else:
                    extracted_content = [str(current_data)]

            except (KeyError, IndexError, TypeError):
                extracted_content = []

            execution_time = (datetime.now() - start_time).total_seconds()

            return RuleExecutionResult(
                rule_id=rule.rule_id,
                success=True,
                matches_found=len(extracted_content),
                extracted_content=extracted_content,
                execution_time=execution_time,
            )

        except json.JSONDecodeError as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return RuleExecutionResult(
                rule_id=rule.rule_id,
                success=False,
                matches_found=0,
                execution_time=execution_time,
                error_message=f"Invalid JSON content: {str(e)}",
            )

    async def _execute_xpath_rule(
        self, rule: CustomRule, content: str, start_time: datetime
    ) -> RuleExecutionResult:
        """Execute an XPath extraction rule."""
        try:
            # For XPath, we'd need an XML parser like lxml
            # This is a simplified implementation
            execution_time = (datetime.now() - start_time).total_seconds()

            return RuleExecutionResult(
                rule_id=rule.rule_id,
                success=False,
                matches_found=0,
                execution_time=execution_time,
                error_message="XPath rules require lxml library (not implemented in this version)",
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return RuleExecutionResult(
                rule_id=rule.rule_id,
                success=False,
                matches_found=0,
                execution_time=execution_time,
                error_message=str(e),
            )

    async def _process_execution_results(
        self, results: List[RuleExecutionResult], original_content: str
    ) -> str:
        """Process rule execution results to produce final content."""
        # Start with original content
        final_content = original_content

        # Apply transformations in order
        for result in results:
            if result.success and result.transformed_content is not None:
                final_content = result.transformed_content

        # If no transformations, collect all extracted content
        if final_content == original_content:
            extracted_parts = []
            for result in results:
                if result.success and result.extracted_content:
                    extracted_parts.extend(result.extracted_content)

            if extracted_parts:
                final_content = "\n".join(extracted_parts)

        return final_content

    def _get_priority_order(self, priority: RulePriority) -> int:
        """Get numeric priority order for sorting."""
        priority_map = {
            RulePriority.CRITICAL: 5,
            RulePriority.HIGH: 4,
            RulePriority.NORMAL: 3,
            RulePriority.LOW: 2,
            RulePriority.CLEANUP: 1,
        }
        return priority_map.get(priority, 3)

    def _initialize_builtin_patterns(self) -> Dict[str, Callable]:
        """Initialize built-in extraction patterns."""

        def extract_emails(content: str) -> List[str]:
            email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
            return re.findall(email_pattern, content)

        def extract_phone_numbers(content: str) -> List[str]:
            phone_pattern = r"\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b"
            return re.findall(phone_pattern, content)

        def extract_urls(content: str) -> List[str]:
            url_pattern = r"https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?"
            return re.findall(url_pattern, content)

        def extract_dates(content: str) -> List[str]:
            date_patterns = [
                r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b",
                r"\b\d{4}[/-]\d{1,2}[/-]\d{1,2}\b",
                r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},?\s+\d{4}\b",
            ]
            dates = []
            for pattern in date_patterns:
                dates.extend(re.findall(pattern, content, re.IGNORECASE))
            return dates

        def extract_money(content: str) -> List[str]:
            money_pattern = r"\$[\d,]+\.?\d{0,2}"
            return re.findall(money_pattern, content)

        def extract_numbers(content: str) -> List[str]:
            number_pattern = r"\b\d+(?:\.\d+)?\b"
            return re.findall(number_pattern, content)

        return {
            "emails": extract_emails,
            "phone_numbers": extract_phone_numbers,
            "urls": extract_urls,
            "dates": extract_dates,
            "money": extract_money,
            "numbers": extract_numbers,
        }

    def create_domain_ruleset(self, domain: str) -> RuleSet:
        """Create a pre-configured ruleset for a specific domain."""
        if domain == "legal":
            return self._create_legal_ruleset()
        elif domain == "medical":
            return self._create_medical_ruleset()
        elif domain == "financial":
            return self._create_financial_ruleset()
        elif domain == "technical":
            return self._create_technical_ruleset()
        else:
            return self._create_generic_ruleset()

    def _create_legal_ruleset(self) -> RuleSet:
        """Create ruleset for legal documents."""
        rules = [
            CustomRule(
                rule_id="legal_parties",
                name="Extract Legal Parties",
                description="Extract plaintiff and defendant names",
                rule_type=RuleType.REGEX,
                priority=RulePriority.HIGH,
                pattern=r"(?:Plaintiff|Defendant):\s*([A-Z][a-zA-Z\s]+)",
            ),
            CustomRule(
                rule_id="legal_dates",
                name="Extract Legal Dates",
                description="Extract important dates in legal documents",
                rule_type=RuleType.PATTERN_MATCH,
                pattern="dates",
            ),
            CustomRule(
                rule_id="legal_clauses",
                name="Extract Legal Clauses",
                description="Extract WHEREAS, THEREFORE clauses",
                rule_type=RuleType.REGEX,
                pattern=r"(WHEREAS,?[^.]+\.)",
            ),
        ]

        return RuleSet(
            ruleset_id="legal_standard",
            name="Standard Legal Document Extraction",
            description="Extract key information from legal documents",
            rules=rules,
            domain="legal",
            file_types=["pdf", "doc", "docx", "txt"],
        )

    def _create_medical_ruleset(self) -> RuleSet:
        """Create ruleset for medical documents."""
        rules = [
            CustomRule(
                rule_id="patient_id",
                name="Extract Patient ID",
                description="Extract patient identification numbers",
                rule_type=RuleType.REGEX,
                pattern=r"Patient\s+ID:?\s*([A-Z0-9-]+)",
            ),
            CustomRule(
                rule_id="diagnosis",
                name="Extract Diagnosis",
                description="Extract diagnosis information",
                rule_type=RuleType.REGEX,
                pattern=r"Diagnosis:?\s*([A-Z][^.]+\.)",
            ),
            CustomRule(
                rule_id="medication",
                name="Extract Medications",
                description="Extract medication information",
                rule_type=RuleType.REGEX,
                pattern=r"(?:mg/kg|ml/hr|[A-Z][a-z]+\s+\d+\s*mg)",
            ),
        ]

        return RuleSet(
            ruleset_id="medical_standard",
            name="Standard Medical Document Extraction",
            description="Extract key information from medical documents",
            rules=rules,
            domain="medical",
            file_types=["pdf", "doc", "docx", "txt"],
        )

    def _create_financial_ruleset(self) -> RuleSet:
        """Create ruleset for financial documents."""
        rules = [
            CustomRule(
                rule_id="account_numbers",
                name="Extract Account Numbers",
                description="Extract account numbers",
                rule_type=RuleType.REGEX,
                pattern=r"Account\s+(?:Number|#):?\s*([0-9-]+)",
            ),
            CustomRule(
                rule_id="monetary_amounts",
                name="Extract Money Amounts",
                description="Extract monetary values",
                rule_type=RuleType.PATTERN_MATCH,
                pattern="money",
            ),
            CustomRule(
                rule_id="financial_dates",
                name="Extract Financial Dates",
                description="Extract dates from financial documents",
                rule_type=RuleType.PATTERN_MATCH,
                pattern="dates",
            ),
        ]

        return RuleSet(
            ruleset_id="financial_standard",
            name="Standard Financial Document Extraction",
            description="Extract key information from financial documents",
            rules=rules,
            domain="financial",
            file_types=["pdf", "doc", "docx", "txt", "csv"],
        )

    def _create_technical_ruleset(self) -> RuleSet:
        """Create ruleset for technical documents."""
        rules = [
            CustomRule(
                rule_id="api_endpoints",
                name="Extract API Endpoints",
                description="Extract API endpoint definitions",
                rule_type=RuleType.REGEX,
                pattern=r"(?:GET|POST|PUT|DELETE|PATCH)\s+(/[^\s]+)",
            ),
            CustomRule(
                rule_id="code_comments",
                name="Extract Code Comments",
                description="Extract comments from code",
                rule_type=RuleType.REGEX,
                pattern=r"(?://.*$|/\*.*?\*/)",
            ),
            CustomRule(
                rule_id="function_definitions",
                name="Extract Function Definitions",
                description="Extract function definitions",
                rule_type=RuleType.REGEX,
                pattern=r"(?:def|function)\s+(\w+)\s*\([^)]*\)",
            ),
        ]

        return RuleSet(
            ruleset_id="technical_standard",
            name="Standard Technical Document Extraction",
            description="Extract key information from technical documents",
            rules=rules,
            domain="technical",
            file_types=["txt", "md", "py", "js", "java", "cpp"],
        )

    def _create_generic_ruleset(self) -> RuleSet:
        """Create generic ruleset for general documents."""
        rules = [
            CustomRule(
                rule_id="emails",
                name="Extract Email Addresses",
                description="Extract email addresses",
                rule_type=RuleType.PATTERN_MATCH,
                pattern="emails",
            ),
            CustomRule(
                rule_id="phone_numbers",
                name="Extract Phone Numbers",
                description="Extract phone numbers",
                rule_type=RuleType.PATTERN_MATCH,
                pattern="phone_numbers",
            ),
            CustomRule(
                rule_id="urls",
                name="Extract URLs",
                description="Extract web URLs",
                rule_type=RuleType.PATTERN_MATCH,
                pattern="urls",
            ),
            CustomRule(
                rule_id="normalize_whitespace",
                name="Normalize Whitespace",
                description="Clean up excessive whitespace",
                rule_type=RuleType.TRANSFORMATION,
                priority=RulePriority.CLEANUP,
                pattern="normalize_whitespace",
            ),
        ]

        return RuleSet(
            ruleset_id="generic_standard",
            name="Generic Document Extraction",
            description="Extract common elements from any document",
            rules=rules,
            domain="generic",
            file_types=["txt", "doc", "docx", "pdf"],
        )

    def get_health_status(self) -> Dict[str, Any]:
        """Get rules engine health status."""
        return {
            "rules_registered": len(self.rule_registry),
            "rulesets_registered": len(self.ruleset_registry),
            "builtin_patterns": len(self.builtin_patterns),
            "statistics": self.stats.copy(),
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

# Global rules engine instance
_default_engine = None


def get_default_rules_engine() -> ExtractionRulesEngine:
    """Get default rules engine instance."""
    global _default_engine
    if _default_engine is None:
        _default_engine = ExtractionRulesEngine()
    return _default_engine


async def extract_with_custom_rules(
    file_path: Union[str, Path],
    rule_definitions: List[Union[str, Dict[str, Any], CustomRule]],
    domain: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract content using custom rules."""
    engine = get_default_rules_engine()

    # Use domain ruleset if specified, otherwise compile custom rules
    if domain:
        ruleset = engine.create_domain_ruleset(domain)
    else:
        ruleset = await engine.compile_rules(rule_definitions)

    return await engine.extract_with_rules(file_path, ruleset)
