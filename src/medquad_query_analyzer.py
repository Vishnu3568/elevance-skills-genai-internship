"""Medical Query Analyzer Module for MedQuAD Medical Q&A Pipeline.

Analyzes raw medical user queries to extract clinical entities (symptoms, diseases, treatments),
infer question intent mapped to MedQuAD question types, and link topics to UMLS CUIs.
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple, Any


@dataclass
class MedicalEntity:
    """Represents a medical entity extracted from a user query."""
    text: str
    category: str  # "DISEASE", "SYMPTOM", "TREATMENT"
    cui: Optional[str] = None
    confidence: float = 1.0


@dataclass
class MedicalQueryAnalysis:
    """Structured representation of an analyzed medical user query."""
    raw_query: str
    clean_query: str
    intent: str
    raw_qtype_match: Optional[str] = None
    entities: List[MedicalEntity] = field(default_factory=list)
    primary_topic: Optional[str] = None
    matched_cuis: List[str] = field(default_factory=list)


# Base deterministic lexicons for clinical concepts
DEFAULT_SYMPTOM_LEXICON: Set[str] = {
    "shortness of breath", "blurred vision", "weight loss", "sore throat",
    "chest pain", "joint pain", "muscle aches", "abdominal pain",
    "pain", "fever", "nausea", "fatigue", "cough", "swelling", "weakness",
    "rash", "dizziness", "headache", "vomiting", "itching", "bleeding",
    "diarrhea", "chills", "congestion", "thirst", "irritability", "lump"
}

DEFAULT_TREATMENT_LEXICON: Set[str] = {
    "radiation therapy", "hormone therapy", "targeted therapy", "biological therapy",
    "lumpectomy", "mastectomy", "chemotherapy", "radiation", "surgery",
    "treatment", "treatments", "therapy", "medication", "medications", "medicine",
    "insulin", "antibiotics", "dosage", "dose", "dialysis", "transplant",
    "prescription", "vaccine", "vaccination", "cure", "drug", "drugs"
}

# Intent keyword patterns ordered by priority
INTENT_PATTERNS: List[Tuple[str, Optional[str], List[str]]] = [
    # (Intent, raw_qtype_match, [regex patterns])
    (
        "SAFETY_ADVERSE",
        "side effects",
        [
            r"\bside\s+effects?\b", r"\badverse\s+effects?\b", r"\ballergic\s+reactions?\b",
            r"\bcontraindications?\b", r"\bdrug\s+interactions?\b", r"\boverdose\b",
            r"\bemergency\b", r"\bwarning\b", r"\bdangerous\b", r"\btoxicity\b"
        ]
    ),
    (
        "MEDICATION_USAGE",
        "usage",
        [
            r"\bhow\s+to\s+take\b", r"\bhow\s+much\b", r"\bdosage\b", r"\bdose\b",
            r"\bhow\s+to\s+use\b", r"\busage\b", r"\bindications?\b", r"\bprescribed\b",
            r"\bbrand\s+names?\b", r"\bhow\s+to\s+store\b", r"\bmissed\s+dose\b", r"\bforget\s+a\s+dose\b"
        ]
    ),
    (
        "TREATMENT",
        "treatment",
        [
            r"\btreatments?\b", r"\bhow\s+to\s+treat\b", r"\bhow\s+is\s+.+\s+treated\b",
            r"\bcure\b", r"\bcures\b", r"\btherap(y|ies)\b", r"\bsurger(y|ies)\b",
            r"\bchemotherap(y|ies)\b", r"\bmedications?\s+for\b", r"\bmanagement\b",
            r"\bhow\s+to\s+manage\b", r"\boptions\b"
        ]
    ),
    (
        "DIAGNOSIS",
        "exams and tests",
        [
            r"\bdiagnos(is|ed|e)\b", r"\bhow\s+is\s+.+\s+diagnosed\b", r"\btests?\b",
            r"\bexams?\b", r"\bscreening\b", r"\bscan\b", r"\bdetected\b",
            r"\bstages?\b", r"\bprognosis\b", r"\boutlook\b", r"\bwhen\s+to\s+see\s+a\s+doctor\b"
        ]
    ),
    (
        "SYMPTOMS",
        "symptoms",
        [
            r"\bsymptoms?\b", r"\bsigns?\b", r"\bfeel\s+like\b", r"\bcomplications?\b",
            r"\bmanifestations?\b", r"\bhow\s+does\s+.+\s+feel\b", r"\bwarning\s+signs?\b"
        ]
    ),
    (
        "PREVENTION",
        "prevention",
        [
            r"\bprevent(ion|ed|ing|s)?\b", r"\bhow\s+to\s+prevent\b", r"\bavoid\b",
            r"\breduce\s+risk\b", r"\bprecautions?\b", r"\bvaccin(e|ation|ated)\b"
        ]
    ),
    (
        "CAUSES_GENETICS",
        "causes",
        [
            r"\bcauses?\b", r"\bwhat\s+causes\b", r"\bwhy\s+does\b", r"\betiolog(y|ies)\b",
            r"\bgenetics?\b", r"\binherit(ance|ed)?\b", r"\bgene\s+mutations?\b",
            r"\brisk\s+factors?\b", r"\bsusceptibility\b"
        ]
    ),
    (
        "GENERAL_INFORMATION",
        "information",
        [
            r"\bwhat\s+is\b", r"\bwhat\s+are\b", r"\btell\s+me\s+about\b",
            r"\boverview\b", r"\binformation\b", r"\bexplain\b", r"\bdetails?\b",
            r"\bdefinition\b", r"\bresearch\b"
        ]
    )
]


class MedicalVocabulary:
    """Holds disease/topic vocabulary, synonyms, and CUI mappings."""

    def __init__(self):
        # Maps normalized term -> (Canonical Focus, CUIs list, is_synonym flag)
        self._term_lookup: Dict[str, Tuple[str, List[str], bool]] = {}
        # Sorted list of terms by descending length for longest-phrase-first matching
        self._sorted_terms: List[str] = []

    def add_topic(self, focus: str, synonyms: Optional[List[str]] = None, cuis: Optional[List[str]] = None):
        """Add a medical topic and its synonyms to the lookup vocabulary."""
        clean_focus = focus.strip()
        if not clean_focus:
            return

        cui_list = [c.strip() for c in cuis] if cuis else []
        norm_focus = self._normalize_term(clean_focus)
        self._term_lookup[norm_focus] = (clean_focus, cui_list, False)

        if synonyms:
            for syn in synonyms:
                clean_syn = syn.strip()
                if clean_syn:
                    norm_syn = self._normalize_term(clean_syn)
                    self._term_lookup[norm_syn] = (clean_focus, cui_list, True)

        # Re-sort terms by descending word count and character length
        self._sorted_terms = sorted(
            self._term_lookup.keys(),
            key=lambda t: (len(t.split()), len(t)),
            reverse=True
        )

    def find_topics(self, text: str) -> List[Tuple[str, str, List[str], bool, int, int]]:
        """Find matching topics in text using word-boundary-aware matching.

        Returns:
            List of tuples: (matched_term, canonical_focus, cuis, is_synonym, start_pos, end_pos)
        """
        norm_text = self._normalize_term(text)
        matches = []
        matched_spans = []

        for term in self._sorted_terms:
            pattern = r'\b' + re.escape(term) + r'\b'
            for m in re.finditer(pattern, norm_text):
                start, end = m.start(), m.end()
                # Check for overlap with already matched longer span
                if any(s <= start and end <= e for s, e in matched_spans):
                    continue

                canonical_focus, cuis, is_synonym = self._term_lookup[term]
                matches.append((term, canonical_focus, cuis, is_synonym, start, end))
                matched_spans.append((start, end))

        return matches

    @staticmethod
    def _normalize_term(term: str) -> str:
        """Normalize whitespace and lowercase for reliable lookup."""
        return re.sub(r'\s+', ' ', term.strip().lower())

    @classmethod
    def from_records(cls, records: List[Any]) -> "MedicalVocabulary":
        """Construct a MedicalVocabulary from a list of MedicalQARecord objects."""
        vocab = cls()
        for rec in records:
            focus = getattr(rec, "focus", "")
            synonyms = getattr(rec, "synonyms", [])
            cuis = getattr(rec, "cuis", [])
            vocab.add_topic(focus=focus, synonyms=synonyms, cuis=cuis)
        return vocab


class MedicalQueryAnalyzer:
    """Analyzes medical queries for intent, topics, and clinical entities."""

    def __init__(
        self,
        vocabulary: Optional[MedicalVocabulary] = None,
        symptom_lexicon: Optional[Set[str]] = None,
        treatment_lexicon: Optional[Set[str]] = None
    ):
        self.vocabulary = vocabulary if vocabulary is not None else MedicalVocabulary()
        self.symptom_lexicon = sorted(
            symptom_lexicon if symptom_lexicon is not None else DEFAULT_SYMPTOM_LEXICON,
            key=lambda t: (len(t.split()), len(t)),
            reverse=True
        )
        self.treatment_lexicon = sorted(
            treatment_lexicon if treatment_lexicon is not None else DEFAULT_TREATMENT_LEXICON,
            key=lambda t: (len(t.split()), len(t)),
            reverse=True
        )

    def analyze(self, query: str) -> MedicalQueryAnalysis:
        """Process a raw user query into a structured MedicalQueryAnalysis payload.

        Args:
            query (str): The medical query string.

        Returns:
            MedicalQueryAnalysis: Structured analysis with entities, topic, and intent.

        Raises:
            TypeError: If query is not a string.
            ValueError: If query is empty or whitespace-only.
        """
        if not isinstance(query, str):
            raise TypeError(f"Expected query to be a string, got {type(query).__name__}")

        stripped_query = query.strip()
        if not stripped_query:
            raise ValueError("Query cannot be empty or whitespace-only.")

        # 1. Clean query normalization
        clean_query = re.sub(r'\s+', ' ', stripped_query).lower()

        entities: List[MedicalEntity] = []
        primary_topic: Optional[str] = None
        matched_cuis: List[str] = []
        detected_entity_spans: List[Tuple[int, int]] = []

        # 2. Disease / Topic Matching (from MedQuAD Vocabulary)
        topic_matches = self.vocabulary.find_topics(clean_query)
        if topic_matches:
            # First match is the longest/best match
            top_match = topic_matches[0]
            matched_term, canonical_focus, cuis, is_synonym, start, end = top_match
            primary_topic = canonical_focus
            matched_cuis = list(cuis)

            for term, focus, c_list, is_syn, s, e in topic_matches:
                conf = 0.90 if is_syn else 1.00
                cui_val = c_list[0] if c_list else None
                entities.append(
                    MedicalEntity(
                        text=term,
                        category="DISEASE",
                        cui=cui_val,
                        confidence=conf
                    )
                )
                detected_entity_spans.append((s, e))

        # 3. Symptom Entity Detection
        for symptom in self.symptom_lexicon:
            pattern = r'\b' + re.escape(symptom) + r'\b'
            for m in re.finditer(pattern, clean_query):
                s, e = m.start(), m.end()
                if not any(span_s <= s and e <= span_e for span_s, span_e in detected_entity_spans):
                    entities.append(
                        MedicalEntity(
                            text=symptom,
                            category="SYMPTOM",
                            cui=None,
                            confidence=0.85
                        )
                    )
                    detected_entity_spans.append((s, e))

        # 4. Treatment Entity Detection
        for treatment in self.treatment_lexicon:
            pattern = r'\b' + re.escape(treatment) + r'\b'
            for m in re.finditer(pattern, clean_query):
                s, e = m.start(), m.end()
                if not any(span_s <= s and e <= span_e for span_s, span_e in detected_entity_spans):
                    entities.append(
                        MedicalEntity(
                            text=treatment,
                            category="TREATMENT",
                            cui=None,
                            confidence=0.85
                        )
                    )
                    detected_entity_spans.append((s, e))

        # 5. Question Intent Detection (Deterministic Priority Hierarchy)
        inferred_intent = "GENERAL_INFORMATION"
        raw_qtype_match: Optional[str] = "information"

        intent_matched = False
        for intent_name, qtype_val, patterns in INTENT_PATTERNS:
            for pat in patterns:
                if re.search(pat, clean_query, re.IGNORECASE):
                    inferred_intent = intent_name
                    raw_qtype_match = qtype_val
                    intent_matched = True
                    break
            if intent_matched:
                break

        # If no pattern matched, fallback to general information with no specific qtype
        if not intent_matched:
            inferred_intent = "GENERAL_INFORMATION"
            raw_qtype_match = None

        return MedicalQueryAnalysis(
            raw_query=query,
            clean_query=clean_query,
            intent=inferred_intent,
            raw_qtype_match=raw_qtype_match,
            entities=entities,
            primary_topic=primary_topic,
            matched_cuis=matched_cuis
        )


if __name__ == "__main__":
    print("Medical Query Analyzer module initialized cleanly.")
