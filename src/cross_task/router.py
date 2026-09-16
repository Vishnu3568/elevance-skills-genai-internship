"""Domain and Modality Router for Cross-Task Integration (Day 36).

Provides deterministic, explainable routing across specialist domains (Customer Support,
Medical Q&A, Scientific Literature, and Multimodal Assistant) with cross-cutting
multilingual language identification and query alignment.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .models import DomainType, UnifiedRequest

try:
    from src.multilingual.detector import LanguageDetector
    from src.multilingual.models import LanguageIdentificationResult, SupportedLanguage
except ImportError:
    from multilingual.detector import LanguageDetector  # type: ignore
    from multilingual.models import LanguageIdentificationResult, SupportedLanguage  # type: ignore


# -----------------------------------------------------------------------------
# Domain Keyword Lexicons
# -----------------------------------------------------------------------------

MEDICAL_KEYWORDS = {
    "cancer", "symptom", "symptoms", "disease", "treatment", "diagnosis", "dosage",
    "medicine", "medication", "fever", "pain", "cardiac", "heart", "syndrome",
    "infection", "surgery", "diabetes", "blood pressure", "hypertension", "headache",
    "tumor", "clinical", "pathology", "therapy", "hospital", "doctor", "physician",
    "biopsy", "chemotherapy", "radiation", "side effect", "side effects", "prognosis",
    "prevention", "causes", "cure", "chronic", "acute", "breast cancer", "lung cancer",
}

SCIENTIFIC_KEYWORDS = {
    "arxiv", "paper", "papers", "transformer", "transformers", "attention mechanism",
    "attention", "neural network", "deep learning", "diffusion", "diffusion model",
    "tree-lstm", "lstm", "benchmark", "datasets", "empirical", "ablation", "gan",
    "gans", "bert", "gpt", "knn-lm", "memorization", "rhetorical", "language model",
    "language models", "citation", "citations", "co-occurrence", "cs.ai", "cs.lg",
    "cs.cl", "cs.cv", "stat.ml",
}

CUSTOMER_KEYWORDS = {
    "course", "courses", "fee", "fees", "refund", "refunds", "enroll", "enrollment",
    "certificate", "certification", "validity", "prerequisite", "prerequisites",
    "lms", "batch", "batches", "payment", "policy", "policies", "syllabus",
    "duration", "schedule", "login", "access", "account", "support", "discount",
    "internship", "placement", "job", "career", "trainer", "instructor", "mentor",
}

# Cross-lingual translated domain cues (Spanish, French, German, Hindi transliterations)
MULTILINGUAL_DOMAIN_CUES: Dict[str, Dict[str, str]] = {
    "es": {
        "curso": "course",
        "precio": "fee",
        "costo": "fee",
        "reembolso": "refund",
        "certificado": "certificate",
        "duracion": "duration",
        "cancer": "cancer",
        "sintoma": "symptom",
        "tratamiento": "treatment",
        "dolor": "pain",
        "medico": "doctor",
        "articulo": "paper",
        "atencion": "attention",
    },
    "fr": {
        "cours": "course",
        "frais": "fee",
        "remboursement": "refund",
        "certificat": "certificate",
        "duree": "duration",
        "cancer": "cancer",
        "symptome": "symptom",
        "traitement": "treatment",
        "douleur": "pain",
        "medecin": "doctor",
        "papier": "paper",
        "attention": "attention",
    },
    "hi": {
        "course": "course",
        "fees": "fee",
        "certificate": "certificate",
        "refund": "refund",
        "duration": "duration",
        "paisa": "fee",
        "bimari": "disease",
        "ilaj": "treatment",
        "dard": "pain",
    },
}


@dataclass
class RoutingDecision:
    """Explaining result of the cross-task domain routing."""
    domain: DomainType
    detected_language: str
    detected_language_name: str
    confidence: float
    reasoning: str
    is_cross_lingual: bool = False
    language_result: Optional[LanguageIdentificationResult] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize routing decision."""
        return {
            "domain": self.domain.value,
            "detected_language": self.detected_language,
            "detected_language_name": self.detected_language_name,
            "confidence": round(self.confidence, 4),
            "reasoning": self.reasoning,
            "is_cross_lingual": self.is_cross_lingual,
            "metadata": dict(self.metadata),
        }


class CrossTaskRouter:
    """Deterministic, explainable cross-task router with cross-cutting multilingual awareness."""

    def __init__(self, language_detector: Optional[LanguageDetector] = None) -> None:
        """Initialize router with language detection subsystem."""
        self.language_detector = (
            language_detector if language_detector is not None else LanguageDetector()
        )

    def route(self, request: UnifiedRequest) -> RoutingDecision:
        """Determine optimal domain and execution plan for an incoming UnifiedRequest.

        Routing Hierarchy:
        1. Modality Inspection: Image present -> MULTIMODAL.
        2. Domain Override: Explicit caller override respected if valid.
        3. Cross-Cutting Language Identification: Identify language using LanguageDetector.
        4. Cross-Lingual Keyword Alignment: Map non-English domain terms.
        5. Domain Intent Scoring: Evaluate Medical, Scientific, and Customer Support signals.
        6. Deterministic Resolution: Dispatches to the highest-scoring domain specialist.
        """
        # 1. Modality Inspection
        if request.image is not None:
            lang_id = self.language_detector.detect(request.query if request.query else "image analysis")
            return RoutingDecision(
                domain=DomainType.MULTIMODAL,
                detected_language=lang_id.language,
                detected_language_name=lang_id.language_name,
                confidence=1.0,
                reasoning="Input contains an ImageArtifact; routed to Multimodal Assistant Service.",
                is_cross_lingual=lang_id.language != SupportedLanguage.ENGLISH.value,
                language_result=lang_id,
            )

        # 2. Explicit Domain Override
        if request.domain_override:
            norm_override = request.domain_override.lower()
            for d in DomainType:
                if d.value == norm_override:
                    lang_id = self.language_detector.detect(request.query)
                    return RoutingDecision(
                        domain=d,
                        detected_language=lang_id.language,
                        detected_language_name=lang_id.language_name,
                        confidence=1.0,
                        reasoning=f"Explicit caller domain override applied: '{d.value}'.",
                        is_cross_lingual=lang_id.language != SupportedLanguage.ENGLISH.value,
                        language_result=lang_id,
                    )

        clean_query = request.query.strip().lower()
        if not clean_query:
            return RoutingDecision(
                domain=DomainType.CUSTOMER_SUPPORT,
                detected_language="en",
                detected_language_name="English",
                confidence=0.5,
                reasoning="Empty query string; routed to default Customer Support fallback.",
                is_cross_lingual=False,
            )

        # 3. Language Identification (Cross-Cutting Layer)
        lang_id = self.language_detector.detect(clean_query)
        detected_lang = lang_id.language
        is_non_english = detected_lang != SupportedLanguage.ENGLISH.value

        # 4. Tokenization and Cross-Lingual Word Expansion
        tokens = set(re.findall(r"\b\w+\b", clean_query))
        expanded_tokens = set(tokens)
        if detected_lang in MULTILINGUAL_DOMAIN_CUES:
            for word in tokens:
                if word in MULTILINGUAL_DOMAIN_CUES[detected_lang]:
                    expanded_tokens.add(MULTILINGUAL_DOMAIN_CUES[detected_lang][word])

        # 5. Check for standalone arXiv ID
        if bool(re.match(r"^\s*\d{4}\.\d{4,5}(?:v\d+)?\s*$", clean_query)):
            return RoutingDecision(
                domain=DomainType.SCIENTIFIC,
                detected_language=detected_lang,
                detected_language_name=lang_id.language_name,
                confidence=1.0,
                reasoning="Query matches standalone arXiv ID pattern; routed to Scientific Literature Service.",
                is_cross_lingual=is_non_english,
                language_result=lang_id,
            )

        # 6. Domain Scoring
        medical_hits = sum(1 for kw in MEDICAL_KEYWORDS if kw in clean_query or kw in expanded_tokens)
        scientific_hits = sum(1 for kw in SCIENTIFIC_KEYWORDS if kw in clean_query or kw in expanded_tokens)
        customer_hits = sum(1 for kw in CUSTOMER_KEYWORDS if kw in clean_query or kw in expanded_tokens)

        # 7. Deterministic Decision Logic
        if medical_hits > 0 and medical_hits >= scientific_hits and medical_hits >= customer_hits:
            confidence = min(0.99, 0.75 + (medical_hits * 0.1))
            return RoutingDecision(
                domain=DomainType.MEDICAL,
                detected_language=detected_lang,
                detected_language_name=lang_id.language_name,
                confidence=confidence,
                reasoning=f"Matched {medical_hits} clinical/medical domain signals in query.",
                is_cross_lingual=is_non_english,
                language_result=lang_id,
                metadata={"matched_signals": medical_hits},
            )

        if scientific_hits > 0 and scientific_hits > customer_hits:
            confidence = min(0.99, 0.75 + (scientific_hits * 0.1))
            return RoutingDecision(
                domain=DomainType.SCIENTIFIC,
                detected_language=detected_lang,
                detected_language_name=lang_id.language_name,
                confidence=confidence,
                reasoning=f"Matched {scientific_hits} scientific literature/AI research domain signals in query.",
                is_cross_lingual=is_non_english,
                language_result=lang_id,
                metadata={"matched_signals": scientific_hits},
            )

        if customer_hits > 0:
            confidence = min(0.99, 0.80 + (customer_hits * 0.08))
            return RoutingDecision(
                domain=DomainType.CUSTOMER_SUPPORT,
                detected_language=detected_lang,
                detected_language_name=lang_id.language_name,
                confidence=confidence,
                reasoning=f"Matched {customer_hits} ed-tech customer service domain signals in query.",
                is_cross_lingual=is_non_english,
                language_result=lang_id,
                metadata={"matched_signals": customer_hits},
            )

        # 8. Default Domain Gating (General Customer FAQ with standard confidence)
        return RoutingDecision(
            domain=DomainType.CUSTOMER_SUPPORT,
            detected_language=detected_lang,
            detected_language_name=lang_id.language_name,
            confidence=0.70,
            reasoning="Defaulted to Customer Support FAQ service based on standard conversational query pattern.",
            is_cross_lingual=is_non_english,
            language_result=lang_id,
        )
