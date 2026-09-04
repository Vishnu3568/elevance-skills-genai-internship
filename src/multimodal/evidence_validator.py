"""Multimodal Evidence Validation Component (Phase 5 — Day 28 Step 1).

Provides robust, deterministic structural validation for multimodal evidence items
(VisualEvidenceItem, UnifiedEvidenceItem, reasoning outputs, and context containers).
Distinguishes between valid, invalid, and absent evidence without raising exceptions
for normal validation failures, while strictly preventing raw image byte leakage.
"""

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .context import EvidenceProvenance, MultimodalContext, UnifiedEvidenceItem
from .models import ModalityType, MultimodalResponse, VisualEvidenceItem


# -----------------------------------------------------------------------------
# Evidence Status Enumeration
# -----------------------------------------------------------------------------

class EvidenceStatus(str, Enum):
    """Categorical classification of evidence validation outcome."""

    VALID = "valid"
    INVALID = "invalid"
    NO_EVIDENCE = "no_evidence"
    PARTIAL = "partial"


# -----------------------------------------------------------------------------
# Single Evidence Item Validation Detail
# -----------------------------------------------------------------------------

@dataclass
class EvidenceItemValidation:
    """Detailed validation evaluation for an individual evidence item."""

    item_index: int
    is_valid: bool
    description: str = ""
    modality: Optional[str] = None
    region_label: Optional[str] = None
    confidence: Optional[float] = None
    source_type: Optional[str] = None
    error_messages: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize item validation to dictionary without raw bytes."""
        return {
            "item_index": self.item_index,
            "is_valid": self.is_valid,
            "description": self.description,
            "modality": self.modality,
            "region_label": self.region_label,
            "confidence": self.confidence,
            "source_type": self.source_type,
            "error_messages": list(self.error_messages),
            "metadata": _clean_metadata_dict(self.metadata),
        }


# -----------------------------------------------------------------------------
# Aggregate Evidence Validation Result
# -----------------------------------------------------------------------------

@dataclass
class EvidenceValidationResult:
    """Structured report returned after evaluating multimodal evidence sufficiency."""

    status: EvidenceStatus
    is_valid: bool
    total_items: int
    valid_items_count: int
    invalid_items_count: int
    valid_evidence: List[Any] = field(default_factory=list)
    invalid_evidence: List[Dict[str, Any]] = field(default_factory=list)
    item_validations: List[EvidenceItemValidation] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    source_summary: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize validation result cleanly to a JSON-safe dictionary."""
        serialized_valid: List[Any] = []
        for ev in self.valid_evidence:
            if hasattr(ev, "to_dict"):
                serialized_valid.append(ev.to_dict())
            elif isinstance(ev, dict):
                serialized_valid.append(_clean_metadata_dict(ev))
            else:
                serialized_valid.append(str(ev))

        return {
            "status": self.status.value,
            "is_valid": self.is_valid,
            "total_items": self.total_items,
            "valid_items_count": self.valid_items_count,
            "invalid_items_count": self.invalid_items_count,
            "valid_evidence": serialized_valid,
            "invalid_evidence": list(self.invalid_evidence),
            "item_validations": [iv.to_dict() for iv in self.item_validations],
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "source_summary": _clean_metadata_dict(self.source_summary),
        }


# -----------------------------------------------------------------------------
# Metadata Cleansing Helper (Guarantees Zero Raw Byte Leakage)
# -----------------------------------------------------------------------------

def _clean_metadata_dict(data: Any) -> Any:
    """Recursively sanitize metadata dictionaries to purge raw bytes or image buffers."""
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
# Evidence Validator Engine
# -----------------------------------------------------------------------------

class MultimodalEvidenceValidator:
    """Validates structural correctness and usability of multimodal evidence items."""

    def __init__(self, min_confidence: float = 0.0) -> None:
        """Initialize the validator.

        Args:
            min_confidence: Optional minimum confidence threshold for evidence validity.
        """
        if not isinstance(min_confidence, (int, float)) or isinstance(min_confidence, bool):
            raise TypeError(f"min_confidence must be a float, got {type(min_confidence).__name__}")
        if min_confidence < 0.0 or min_confidence > 1.0:
            raise ValueError(f"min_confidence ({min_confidence}) must be in range [0.0, 1.0].")

        self.min_confidence = float(min_confidence)

    def validate_item(self, item: Any, item_index: int = 0) -> EvidenceItemValidation:
        """Validate an individual evidence item and return an EvidenceItemValidation record.

        Args:
            item: Candidate evidence object (VisualEvidenceItem, UnifiedEvidenceItem, or dict).
            item_index: Positional index of the item.

        Returns:
            EvidenceItemValidation: Structured validation outcome for this specific item.
        """
        errors: List[str] = []
        raw_meta: Dict[str, Any] = {}
        description = ""
        modality_str: Optional[str] = None
        region_label: Optional[str] = None
        confidence: Optional[float] = None
        source_type: Optional[str] = None

        # 1. Type validation of the container
        if isinstance(item, VisualEvidenceItem):
            description = item.description
            region_label = item.region_label
            confidence = item.confidence
            modality_str = ModalityType.IMAGE_ONLY.value
            source_type = "visual_evidence"
        elif isinstance(item, UnifiedEvidenceItem):
            description = item.description
            region_label = item.region_label
            confidence = item.confidence
            modality_str = item.modality.value if isinstance(item.modality, ModalityType) else str(item.modality)
            source_type = item.source_type
            raw_meta = dict(item.metadata)
            if item.provenance is not None:
                raw_meta["provenance"] = item.provenance.to_dict()
        elif isinstance(item, dict):
            raw_meta = dict(item)
            desc_val = item.get("description")
            if isinstance(desc_val, str):
                description = desc_val
            else:
                errors.append(f"Item {item_index}: 'description' must be a string, got {type(desc_val).__name__}")

            region_label = item.get("region_label")
            if region_label is not None and not isinstance(region_label, str):
                errors.append(f"Item {item_index}: 'region_label' must be a string or None, got {type(region_label).__name__}")

            conf_val = item.get("confidence")
            if conf_val is not None:
                if not isinstance(conf_val, (int, float)) or isinstance(conf_val, bool):
                    errors.append(f"Item {item_index}: 'confidence' must be numeric, got {type(conf_val).__name__}")
                elif conf_val < 0.0 or conf_val > 1.0:
                    errors.append(f"Item {item_index}: 'confidence' ({conf_val}) must be between 0.0 and 1.0")
                else:
                    confidence = float(conf_val)
            else:
                confidence = 1.0

            mod_val = item.get("modality")
            if mod_val is not None:
                if isinstance(mod_val, ModalityType):
                    modality_str = mod_val.value
                elif isinstance(mod_val, str):
                    modality_str = mod_val
                else:
                    errors.append(f"Item {item_index}: 'modality' must be a ModalityType or string, got {type(mod_val).__name__}")

            src_val = item.get("source_type")
            if src_val is not None:
                if isinstance(src_val, str):
                    source_type = src_val
                else:
                    errors.append(f"Item {item_index}: 'source_type' must be a string, got {type(src_val).__name__}")
        else:
            errors.append(
                f"Item {item_index}: Unsupported evidence type '{type(item).__name__}'. "
                "Expected VisualEvidenceItem, UnifiedEvidenceItem, or dict."
            )
            return EvidenceItemValidation(
                item_index=item_index,
                is_valid=False,
                error_messages=errors,
            )

        # 2. Validate description content
        if not description or not description.strip():
            if not any("description" in err for err in errors):
                errors.append(f"Item {item_index}: Missing or empty 'description'.")

        # 3. Validate confidence bounds against threshold
        if confidence is not None and confidence < self.min_confidence:
            errors.append(
                f"Item {item_index}: Confidence ({confidence:.3f}) below threshold ({self.min_confidence:.3f})."
            )

        is_valid = len(errors) == 0

        return EvidenceItemValidation(
            item_index=item_index,
            is_valid=is_valid,
            description=description.strip() if description else "",
            modality=modality_str,
            region_label=region_label,
            confidence=confidence,
            source_type=source_type,
            error_messages=errors,
            metadata=_clean_metadata_dict(raw_meta),
        )

    def validate_evidence(
        self,
        evidence: Optional[Union[List[Any], MultimodalContext, MultimodalResponse, Any]],
    ) -> EvidenceValidationResult:
        """Validate an evidence container, context, response, or raw evidence collection.

        Args:
            evidence: Evidence items list, MultimodalContext, MultimodalResponse, or reasoning result.

        Returns:
            EvidenceValidationResult: Comprehensive validation outcome.
        """
        # 1. Handle None or empty collection
        if evidence is None:
            return EvidenceValidationResult(
                status=EvidenceStatus.NO_EVIDENCE,
                is_valid=False,
                total_items=0,
                valid_items_count=0,
                invalid_items_count=0,
                warnings=["No evidence provided (evidence is None)."],
            )

        # 2. Extract evidence items from high-level contracts if applicable
        raw_items: List[Any] = []
        source_summary: Dict[str, Any] = {}

        if isinstance(evidence, MultimodalResponse):
            raw_items = list(evidence.visual_evidence)
            source_summary["container"] = "MultimodalResponse"
            source_summary["session_id"] = evidence.session_id
            source_summary["modality"] = evidence.modality.value if isinstance(evidence.modality, ModalityType) else str(evidence.modality)
        elif isinstance(evidence, MultimodalContext):
            raw_items = list(evidence.evidence_items)
            source_summary["container"] = "MultimodalContext"
            source_summary["session_id"] = evidence.session_id
            source_summary["modality"] = evidence.modality.value if isinstance(evidence.modality, ModalityType) else str(evidence.modality)
        elif hasattr(evidence, "combined_evidence") and isinstance(evidence.combined_evidence, list):
            # ReasoningResult / MultimodalReasoningResult
            raw_items = list(evidence.combined_evidence)
            source_summary["container"] = type(evidence).__name__
            source_summary["session_id"] = getattr(evidence, "session_id", "unknown")
        elif hasattr(evidence, "visual_evidence") and isinstance(evidence.visual_evidence, list):
            raw_items = list(evidence.visual_evidence)
            source_summary["container"] = type(evidence).__name__
        elif isinstance(evidence, list):
            raw_items = list(evidence)
            source_summary["container"] = "list"
        elif isinstance(evidence, dict):
            # Dict containing an evidence collection or a single evidence item
            if "evidence_items" in evidence and isinstance(evidence["evidence_items"], list):
                raw_items = list(evidence["evidence_items"])
            elif "visual_evidence" in evidence and isinstance(evidence["visual_evidence"], list):
                raw_items = list(evidence["visual_evidence"])
            elif "description" in evidence:
                raw_items = [evidence]
            else:
                raw_items = []
            source_summary["container"] = "dict"
        else:
            # Single object or unsupported type
            raw_items = [evidence]
            source_summary["container"] = type(evidence).__name__

        total_items = len(raw_items)
        if total_items == 0:
            return EvidenceValidationResult(
                status=EvidenceStatus.NO_EVIDENCE,
                is_valid=False,
                total_items=0,
                valid_items_count=0,
                invalid_items_count=0,
                warnings=["Evidence collection is empty."],
                source_summary=source_summary,
            )

        # 3. Validate each item
        valid_evidence: List[Any] = []
        invalid_evidence: List[Dict[str, Any]] = []
        item_validations: List[EvidenceItemValidation] = []
        all_errors: List[str] = []

        for idx, item in enumerate(raw_items):
            item_val = self.validate_item(item, item_index=idx)
            item_validations.append(item_val)

            if item_val.is_valid:
                valid_evidence.append(item)
            else:
                all_errors.extend(item_val.error_messages)
                invalid_dict: Dict[str, Any] = {
                    "item_index": idx,
                    "errors": list(item_val.error_messages),
                }
                if hasattr(item, "to_dict"):
                    invalid_dict["item_data"] = item.to_dict()
                elif isinstance(item, dict):
                    invalid_dict["item_data"] = _clean_metadata_dict(item)
                else:
                    invalid_dict["item_repr"] = repr(item)
                invalid_evidence.append(invalid_dict)

        valid_count = len(valid_evidence)
        invalid_count = len(invalid_evidence)

        # 4. Determine aggregate status
        if invalid_count == 0 and valid_count > 0:
            status = EvidenceStatus.VALID
            overall_valid = True
        elif valid_count == 0 and invalid_count > 0:
            status = EvidenceStatus.INVALID
            overall_valid = False
        else:
            status = EvidenceStatus.PARTIAL
            overall_valid = False

        return EvidenceValidationResult(
            status=status,
            is_valid=overall_valid,
            total_items=total_items,
            valid_items_count=valid_count,
            invalid_items_count=invalid_count,
            valid_evidence=valid_evidence,
            invalid_evidence=invalid_evidence,
            item_validations=item_validations,
            errors=all_errors,
            source_summary=source_summary,
        )


# -----------------------------------------------------------------------------
# Convenience Functional Interface
# -----------------------------------------------------------------------------

def validate_multimodal_evidence(
    evidence: Optional[Union[List[Any], MultimodalContext, MultimodalResponse, Any]],
    min_confidence: float = 0.0,
) -> EvidenceValidationResult:
    """Validate multimodal evidence with sensible defaults.

    Args:
        evidence: Evidence items list, MultimodalContext, MultimodalResponse, or dict.
        min_confidence: Optional minimum confidence threshold.

    Returns:
        EvidenceValidationResult: Structured validation report.
    """
    validator = MultimodalEvidenceValidator(min_confidence=min_confidence)
    return validator.validate_evidence(evidence)
