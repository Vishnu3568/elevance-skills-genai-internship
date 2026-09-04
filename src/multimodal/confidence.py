"""Multimodal Evidence Confidence Assessment (Phase 5 — Day 28 Step 2).

Provides deterministic confidence assessment across multimodal evidence items
(VisualEvidenceItem, UnifiedEvidenceItem, reasoning outputs, and validation reports).
Evaluates individual item confidence, computes aggregate metrics, and explicitly
distinguishes between available, missing, and malformed confidence without inventing
data or exposing raw image byte payloads.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .context import MultimodalContext, UnifiedEvidenceItem
from .evidence_validator import EvidenceValidationResult
from .models import ModalityType, MultimodalResponse, VisualEvidenceItem


# -----------------------------------------------------------------------------
# Confidence Status Classification
# -----------------------------------------------------------------------------

class ConfidenceStatus(str, Enum):
    """Categorical classification of confidence evaluation."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    INVALID = "invalid"
    MIXED = "mixed"


# -----------------------------------------------------------------------------
# Single Evidence Item Confidence Record
# -----------------------------------------------------------------------------

@dataclass
class ItemConfidence:
    """Detailed confidence evaluation for a single evidence item."""

    item_index: int
    status: ConfidenceStatus
    is_valid: bool
    confidence_score: Optional[float] = None
    raw_confidence: Any = None
    error_message: Optional[str] = None
    description_snippet: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize item confidence record to dictionary without raw bytes."""
        raw_val = self.raw_confidence
        if isinstance(raw_val, (bytes, bytearray)):
            raw_val = f"<bytes: {len(raw_val)}>"

        return {
            "item_index": self.item_index,
            "status": self.status.value,
            "is_valid": self.is_valid,
            "confidence_score": self.confidence_score,
            "raw_confidence": raw_val,
            "error_message": self.error_message,
            "description_snippet": self.description_snippet,
        }


# -----------------------------------------------------------------------------
# Aggregate Confidence Assessment Report
# -----------------------------------------------------------------------------

@dataclass
class ConfidenceAssessmentResult:
    """Structured report returned after evaluating multimodal evidence confidence."""

    status: ConfidenceStatus
    is_confident: bool
    aggregate_confidence: Optional[float]
    min_confidence: Optional[float]
    max_confidence: Optional[float]
    threshold: float
    total_items: int
    valid_confidence_count: int
    unavailable_confidence_count: int
    invalid_confidence_count: int
    item_confidences: List[ItemConfidence] = field(default_factory=list)
    confidence_distribution: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize assessment report cleanly to a JSON-safe dictionary."""
        return {
            "status": self.status.value,
            "is_confident": self.is_confident,
            "aggregate_confidence": self.aggregate_confidence,
            "min_confidence": self.min_confidence,
            "max_confidence": self.max_confidence,
            "threshold": self.threshold,
            "total_items": self.total_items,
            "valid_confidence_count": self.valid_confidence_count,
            "unavailable_confidence_count": self.unavailable_confidence_count,
            "invalid_confidence_count": self.invalid_confidence_count,
            "item_confidences": [ic.to_dict() for ic in self.item_confidences],
            "confidence_distribution": dict(self.confidence_distribution),
            "warnings": list(self.warnings),
            "metadata": _clean_metadata_dict(self.metadata),
        }


# -----------------------------------------------------------------------------
# Metadata Cleansing Helper (Zero Raw Byte Leakage)
# -----------------------------------------------------------------------------

def _clean_metadata_dict(data: Any) -> Any:
    """Recursively sanitize metadata dictionaries to purge raw bytes or buffers."""
    if isinstance(data, (bytes, bytearray)):
        return f"<bytes: {len(data)}>"
    if isinstance(data, dict):
        cleaned: Dict[str, Any] = {}
        for k, v in data.items():
            if k in ("data", "base64_str", "artifact_data"):
                continue
            cleaned[k] = _clean_metadata_dict(v)
        return cleaned
    if isinstance(data, (list, tuple)):
        return [_clean_metadata_dict(v) for v in data]
    return data


# -----------------------------------------------------------------------------
# Multimodal Confidence Assessor Engine
# -----------------------------------------------------------------------------

class MultimodalConfidenceAssessor:
    """Evaluates and aggregates confidence scores associated with multimodal evidence."""

    def __init__(
        self,
        threshold: float = 0.7,
        aggregation_method: str = "mean",
    ) -> None:
        """Initialize the confidence assessor.

        Args:
            threshold: Confidence threshold required to consider evidence sufficiently confident.
            aggregation_method: Strategy for computing aggregate score ("mean", "min", or "harmonic").
        """
        if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
            raise TypeError(f"threshold must be numeric, got {type(threshold).__name__}")
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError(f"threshold ({threshold}) must be in range [0.0, 1.0].")

        valid_methods = {"mean", "min", "harmonic"}
        if aggregation_method not in valid_methods:
            raise ValueError(f"Unsupported aggregation_method '{aggregation_method}'. Choose from {valid_methods}.")

        self.threshold = float(threshold)
        self.aggregation_method = aggregation_method

    def assess_item(self, item: Any, item_index: int = 0) -> ItemConfidence:
        """Assess the confidence of a single evidence item.

        Args:
            item: Candidate evidence item (VisualEvidenceItem, UnifiedEvidenceItem, dict, etc.).
            item_index: Positional index of the item.

        Returns:
            ItemConfidence: Detailed record of item confidence status and validated score.
        """
        raw_conf: Any = None
        desc_snippet = ""

        # Extract confidence and description based on item type
        if isinstance(item, (VisualEvidenceItem, UnifiedEvidenceItem)):
            raw_conf = item.confidence
            desc_snippet = item.description[:60] if item.description else ""
        elif isinstance(item, dict):
            raw_conf = item.get("confidence")
            desc_val = item.get("description", "")
            desc_snippet = str(desc_val)[:60] if desc_val else ""
        elif hasattr(item, "confidence"):
            raw_conf = getattr(item, "confidence")
            desc = getattr(item, "description", "")
            desc_snippet = str(desc)[:60] if desc else ""
        else:
            return ItemConfidence(
                item_index=item_index,
                status=ConfidenceStatus.INVALID,
                is_valid=False,
                raw_confidence=repr(item),
                error_message=f"Item {item_index}: Unsupported object type '{type(item).__name__}'.",
                description_snippet=repr(item)[:60],
            )

        # 1. Check if confidence is missing / None
        if raw_conf is None:
            return ItemConfidence(
                item_index=item_index,
                status=ConfidenceStatus.UNAVAILABLE,
                is_valid=False,
                confidence_score=None,
                raw_confidence=None,
                error_message=f"Item {item_index}: Confidence score is missing (None).",
                description_snippet=desc_snippet,
            )

        # 2. Check if boolean (bool is subclass of int in Python)
        if isinstance(raw_conf, bool):
            return ItemConfidence(
                item_index=item_index,
                status=ConfidenceStatus.INVALID,
                is_valid=False,
                confidence_score=None,
                raw_confidence=raw_conf,
                error_message=f"Item {item_index}: Confidence score cannot be boolean ({raw_conf}).",
                description_snippet=desc_snippet,
            )

        # 3. Check if numeric
        if not isinstance(raw_conf, (int, float)):
            return ItemConfidence(
                item_index=item_index,
                status=ConfidenceStatus.INVALID,
                is_valid=False,
                confidence_score=None,
                raw_confidence=raw_conf,
                error_message=f"Item {item_index}: Confidence score must be numeric, got {type(raw_conf).__name__}.",
                description_snippet=desc_snippet,
            )

        # 4. Check for NaN or Inf
        float_conf = float(raw_conf)
        if math.isnan(float_conf) or math.isinf(float_conf):
            return ItemConfidence(
                item_index=item_index,
                status=ConfidenceStatus.INVALID,
                is_valid=False,
                confidence_score=None,
                raw_confidence=raw_conf,
                error_message=f"Item {item_index}: Confidence score cannot be NaN or Infinite.",
                description_snippet=desc_snippet,
            )

        # 5. Check range bounds [0.0, 1.0]
        if float_conf < 0.0 or float_conf > 1.0:
            return ItemConfidence(
                item_index=item_index,
                status=ConfidenceStatus.INVALID,
                is_valid=False,
                confidence_score=None,
                raw_confidence=raw_conf,
                error_message=f"Item {item_index}: Confidence score ({float_conf}) out of bounds [0.0, 1.0].",
                description_snippet=desc_snippet,
            )

        # Successfully validated
        return ItemConfidence(
            item_index=item_index,
            status=ConfidenceStatus.AVAILABLE,
            is_valid=True,
            confidence_score=round(float_conf, 4),
            raw_confidence=raw_conf,
            error_message=None,
            description_snippet=desc_snippet,
        )

    def assess(
        self,
        evidence: Optional[Union[List[Any], MultimodalContext, MultimodalResponse, EvidenceValidationResult, Any]],
    ) -> ConfidenceAssessmentResult:
        """Assess the confidence for an evidence collection or container.

        Args:
            evidence: Evidence items list, MultimodalContext, MultimodalResponse, or EvidenceValidationResult.

        Returns:
            ConfidenceAssessmentResult: Aggregate assessment and per-item evaluations.
        """
        # 1. Handle None or empty collection
        if evidence is None:
            return ConfidenceAssessmentResult(
                status=ConfidenceStatus.UNAVAILABLE,
                is_confident=False,
                aggregate_confidence=None,
                min_confidence=None,
                max_confidence=None,
                threshold=self.threshold,
                total_items=0,
                valid_confidence_count=0,
                unavailable_confidence_count=0,
                invalid_confidence_count=0,
                warnings=["No evidence provided (evidence is None)."],
            )

        # 2. Extract items from high-level containers
        raw_items: List[Any] = []
        container_name = type(evidence).__name__

        if isinstance(evidence, EvidenceValidationResult):
            raw_items = list(evidence.valid_evidence) + list(evidence.invalid_evidence)
            if not raw_items and evidence.total_items == 0:
                raw_items = []
        elif isinstance(evidence, MultimodalResponse):
            raw_items = list(evidence.visual_evidence)
        elif isinstance(evidence, MultimodalContext):
            raw_items = list(evidence.evidence_items)
        elif hasattr(evidence, "combined_evidence") and isinstance(evidence.combined_evidence, list):
            raw_items = list(evidence.combined_evidence)
        elif hasattr(evidence, "visual_evidence") and isinstance(evidence.visual_evidence, list):
            raw_items = list(evidence.visual_evidence)
        elif isinstance(evidence, list):
            raw_items = list(evidence)
        elif isinstance(evidence, dict):
            if "evidence_items" in evidence and isinstance(evidence["evidence_items"], list):
                raw_items = list(evidence["evidence_items"])
            elif "visual_evidence" in evidence and isinstance(evidence["visual_evidence"], list):
                raw_items = list(evidence["visual_evidence"])
            else:
                raw_items = [evidence]
        else:
            raw_items = [evidence]

        total_items = len(raw_items)
        if total_items == 0:
            return ConfidenceAssessmentResult(
                status=ConfidenceStatus.UNAVAILABLE,
                is_confident=False,
                aggregate_confidence=None,
                min_confidence=None,
                max_confidence=None,
                threshold=self.threshold,
                total_items=0,
                valid_confidence_count=0,
                unavailable_confidence_count=0,
                invalid_confidence_count=0,
                warnings=["Evidence collection is empty."],
                metadata={"container": container_name},
            )

        # 3. Assess each item
        item_confidences: List[ItemConfidence] = []
        valid_scores: List[float] = []
        unavailable_count = 0
        invalid_count = 0
        warnings: List[str] = []

        dist: Dict[str, int] = {"high": 0, "medium": 0, "low": 0}

        for idx, item in enumerate(raw_items):
            ic = self.assess_item(item, item_index=idx)
            item_confidences.append(ic)

            if ic.status == ConfidenceStatus.AVAILABLE and ic.confidence_score is not None:
                valid_scores.append(ic.confidence_score)
                if ic.confidence_score >= 0.8:
                    dist["high"] += 1
                elif ic.confidence_score >= 0.5:
                    dist["medium"] += 1
                else:
                    dist["low"] += 1
            elif ic.status == ConfidenceStatus.UNAVAILABLE:
                unavailable_count += 1
                if ic.error_message:
                    warnings.append(ic.error_message)
            elif ic.status == ConfidenceStatus.INVALID:
                invalid_count += 1
                if ic.error_message:
                    warnings.append(ic.error_message)

        valid_count = len(valid_scores)

        # 4. Compute aggregate metrics
        aggregate_conf: Optional[float] = None
        min_conf: Optional[float] = None
        max_conf: Optional[float] = None

        if valid_count > 0:
            min_conf = round(min(valid_scores), 4)
            max_conf = round(max(valid_scores), 4)

            if self.aggregation_method == "min":
                aggregate_conf = min_conf
            elif self.aggregation_method == "harmonic":
                # Guard against division by zero with small epsilon
                eps = 1e-6
                aggregate_conf = round(valid_count / sum(1.0 / (s + eps) for s in valid_scores), 4)
            else:  # mean (default)
                aggregate_conf = round(sum(valid_scores) / valid_count, 4)

        # 5. Determine overall status
        if valid_count == total_items:
            status = ConfidenceStatus.AVAILABLE
        elif valid_count == 0 and invalid_count > 0:
            status = ConfidenceStatus.INVALID
        elif valid_count == 0 and unavailable_count > 0:
            status = ConfidenceStatus.UNAVAILABLE
        else:
            status = ConfidenceStatus.MIXED

        is_confident = aggregate_conf is not None and aggregate_conf >= self.threshold

        return ConfidenceAssessmentResult(
            status=status,
            is_confident=is_confident,
            aggregate_confidence=aggregate_conf,
            min_confidence=min_conf,
            max_confidence=max_conf,
            threshold=self.threshold,
            total_items=total_items,
            valid_confidence_count=valid_count,
            unavailable_confidence_count=unavailable_count,
            invalid_confidence_count=invalid_count,
            item_confidences=item_confidences,
            confidence_distribution=dist,
            warnings=warnings,
            metadata={"container": container_name, "aggregation_method": self.aggregation_method},
        )


# -----------------------------------------------------------------------------
# Convenience Functional Interface
# -----------------------------------------------------------------------------

def assess_multimodal_confidence(
    evidence: Optional[Union[List[Any], MultimodalContext, MultimodalResponse, EvidenceValidationResult, Any]],
    threshold: float = 0.7,
    aggregation_method: str = "mean",
) -> ConfidenceAssessmentResult:
    """Assess multimodal evidence confidence with sensible defaults.

    Args:
        evidence: Evidence items list, MultimodalContext, MultimodalResponse, or dict.
        threshold: Confidence threshold for sufficiency.
        aggregation_method: Calculation strategy ("mean", "min", "harmonic").

    Returns:
        ConfidenceAssessmentResult: Structured confidence report.
    """
    assessor = MultimodalConfidenceAssessor(threshold=threshold, aggregation_method=aggregation_method)
    return assessor.assess(evidence)
