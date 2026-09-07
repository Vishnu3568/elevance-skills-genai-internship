"""Deterministic Language Identification Engine for Multilingual Chatbot (Phase 6).

Implements statistical character n-gram modeling, script analysis, and lexical profile
matching for high-accuracy language detection across English, Spanish, French, German, and Hindi.
"""

import math
import re
import unicodedata
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from src.multilingual.models import (
    LanguageCandidate,
    LanguageIdentificationResult,
    SupportedLanguage,
)


# -----------------------------------------------------------------------------
# Lexical and Script Profiles
# -----------------------------------------------------------------------------

# Devanagari script Unicode block range: U+0900 to U+097F
DEVANAGARI_RANGE: Tuple[int, int] = (0x0900, 0x097F)

# Characteristic stopwords per supported language (lowercase)
LANGUAGE_STOPWORDS: Dict[str, Set[str]] = {
    SupportedLanguage.ENGLISH.value: {
        "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
        "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
        "this", "but", "his", "by", "from", "they", "we", "say", "her",
        "she", "or", "an", "will", "my", "one", "all", "would", "there",
        "their", "what", "so", "up", "out", "if", "about", "who", "get",
        "which", "go", "me", "when", "make", "can", "like", "time", "no",
        "just", "him", "know", "take", "people", "into", "year", "your",
        "good", "some", "could", "them", "see", "other", "than", "then",
        "now", "look", "only", "come", "its", "over", "think", "also",
        "back", "after", "use", "two", "how", "our", "work", "first",
        "well", "way", "even", "new", "want", "because", "any", "these",
        "give", "day", "most", "us", "is", "are", "was", "were", "been",
        "has", "had", "does", "did", "please", "help", "hello", "hi",
    },
    SupportedLanguage.SPANISH.value: {
        "de", "la", "que", "el", "en", "y", "a", "los", "del", "se",
        "las", "por", "un", "para", "con", "no", "una", "su", "al", "lo",
        "como", "más", "pero", "sus", "le", "ya", "o", "este", "sí",
        "porque", "esta", "entre", "cuando", "muy", "sin", "sobre", "también",
        "me", "hasta", "hay", "donde", "quien", "desde", "todo", "nos",
        "durante", "todos", "uno", "les", "ni", "contra", "otros", "ese",
        "eso", "ante", "ellos", "e", "esto", "mí", "antes", "algunos",
        "qué", "unos", "yo", "otro", "otras", "otra", "él", "tanto",
        "esa", "estos", "mucho", "quienes", "nada", "muchos", "cual",
        "poco", "ella", "estar", "estas", "algunas", "algo", "nosotros",
        "hola", "gracias", "por favor", "buenos", "días", "tardes", "ayuda",
        "cómo", "estás", "está", "son", "es", "fue", "ser", "tiene", "hacer",
    },
    SupportedLanguage.FRENCH.value: {
        "de", "la", "le", "et", "les", "des", "en", "un", "du", "une",
        "que", "est", "pour", "qui", "dans", "a", "par", "plus", "pas",
        "au", "sur", "ne", "ce", "il", "sont", "se", "avec", "son",
        "ou", "ont", "ses", "mais", "aux", "nous", "sa", "cette", "comme",
        "on", "tout", "aussi", "ils", "leur", "bien", "y", "deux",
        "même", "fait", "été", "si", "sans", "faire", "peut", "ces",
        "donc", "après", "avoir", "d'un", "d'une", "l'on", "qu'il", "qu'elle",
        "bonjour", "merci", "s'il", "vous", "plaît", "comment", "allez", "salut",
        "très", "quel", "quelle", "quelles", "quels", "pourquoi", "quand",
        "suis", "êtes", "sommes", "était", "avoir", "être", "ici", "là",
    },
    SupportedLanguage.GERMAN.value: {
        "der", "die", "und", "in", "den", "von", "zu", "das", "mit", "sich",
        "des", "auf", "für", "ist", "im", "dem", "nicht", "ein", "eine",
        "als", "auch", "es", "an", "werden", "aus", "er", "hat", "dass",
        "sie", "nach", "wird", "bei", "einer", "um", "am", "sind", "noch",
        "wie", "einem", "über", "einen", "so", "zum", "war", "haben", "nur",
        "oder", "aber", "vor", "zur", "bis", "mehr", "durch", "man", "sein",
        "wurde", "sei", "prozent", "hatte", "kann", "gegen", "vom", "können",
        "schon", "wenn", "habe", "seine", "ihre", "unter", "wir", "sollen",
        "hallo", "guten", "tag", "morgen", "bitte", "danke", "hilfe",
        "warum", "wann", "welche", "welcher", "welches", "ich", "du", "ihr",
    },
    SupportedLanguage.HINDI.value: {
        "के", "का", "एक", "की", "है", "में", "से", "को", "और", "पर",
        "इस", "होता", "कि", "जो", "कर", "मे", "गया", "भी", "हैं",
        "था", "थे", "थी", "नहीं", "यह", "वह", "तो", "ही", "या",
        "अपने", "किया", "दिया", "लिया", "करना", "होने", "रहा", "रहे",
        "रही", "बात", "लिए", "हुए", "हुआ", "हुई", "जब", "तब",
        "कहा", "जाता", "सकते", "सकता", "सकती", "नमस्ते", "धन्यवाद",
        "कृपया", "मदद", "कैसे", "क्या", "कहाँ", "कौन", "क्यों", "आप",
        "हम", "मैं", "तुम", "मेरा", "मेरी", "आपका", "आपकी",
    },
}

# Diacritical/script markers that strongly signal particular languages
LANGUAGE_DIACRITICS: Dict[str, Set[str]] = {
    SupportedLanguage.SPANISH.value: {"ñ", "¿", "¡", "á", "é", "í", "ó", "ú", "ü"},
    SupportedLanguage.FRENCH.value: {"é", "è", "ê", "ë", "à", "â", "ç", "î", "ï", "ô", "ù", "û", "ü", "œ", "æ"},
    SupportedLanguage.GERMAN.value: {"ä", "ö", "ü", "ß"},
}

# High-frequency character trigrams for Latin-script languages
CHARACTER_TRIGRAM_PROFILES: Dict[str, Set[str]] = {
    SupportedLanguage.ENGLISH.value: {
        "the", "and", "ing", "ion", "tio", "ent", "ati", "for", "her", "ter",
        "hat", "tha", "ere", "ate", "his", "con", "res", "ver", "all", "ons",
        "nth", "int", "est", "sta", "ith", "wit", "thi", "oth", "pro", "ear",
    },
    SupportedLanguage.SPANISH.value: {
        "que", "del", "con", "par", "ara", "ión", "los", "ent", "est", "ien",
        "aci", "cio", "com", "por", "nte", "ado", "las", "ida", "ció", "ode",
        "tra", "res", "una", "cio", "cia", "tad", "mos", "una", "est", "tod",
    },
    SupportedLanguage.FRENCH.value: {
        "les", "des", "que", "est", "ent", "ion", "our", "pou", "dan", "ans",
        "une", "qui", "par", "ont", "com", "men", "ous", "ett", "tte", "eme",
        "tre", "tio", "ati", "rai", "ais", "ait", "ant", "son", "ell", "lle",
    },
    SupportedLanguage.GERMAN.value: {
        "der", "ein", "ich", "und", "die", "den", "sch", "che", "cht", "ung",
        "gen", "nde", "ter", "ber", "ver", "ten", "das", "isc", "sch", "lic",
        "och", "ige", "hei", "eit", "abe", "ach", "auf", "mit", "ste", "ach",
    },
}


# -----------------------------------------------------------------------------
# Language Detector Implementation
# -----------------------------------------------------------------------------

class LanguageDetector:
    """Production-grade deterministic language identification component."""

    def __init__(
        self,
        confidence_threshold: float = 0.50,
        custom_provider: Optional[Callable[[str], Optional[LanguageIdentificationResult]]] = None,
    ) -> None:
        """Initialize the language detector.

        Args:
            confidence_threshold (float): Minimum confidence to mark detection reliable.
            custom_provider (Optional[Callable]): Optional external provider hook.
        """
        if not (0.0 <= confidence_threshold <= 1.0):
            raise ValueError(f"Confidence threshold {confidence_threshold} must be in [0.0, 1.0].")
        self.confidence_threshold = confidence_threshold
        self.custom_provider = custom_provider

    def _is_devanagari_char(self, char: str) -> bool:
        """Check if a character falls within the Devanagari Unicode block."""
        code_point = ord(char)
        return DEVANAGARI_RANGE[0] <= code_point <= DEVANAGARI_RANGE[1]

    def _extract_script(self, text: str) -> str:
        """Determine primary script family of the given text."""
        devanagari_count = sum(1 for c in text if self._is_devanagari_char(c))
        alpha_count = sum(1 for c in text if c.isalpha())

        if alpha_count == 0:
            return "None"
        if (devanagari_count / alpha_count) >= 0.30:
            return "Devanagari"
        return "Latin"

    def _extract_words(self, text: str) -> List[str]:
        """Tokenize text into lowercase word tokens."""
        # Use regex to extract unicode word tokens
        return [w.lower() for w in re.findall(r"[\w']+", text, flags=re.UNICODE) if w]

    def _extract_trigrams(self, text: str) -> List[str]:
        """Extract sliding character trigrams from text."""
        cleaned = re.sub(r"\s+", " ", text.lower().strip())
        if len(cleaned) < 3:
            return []
        return [cleaned[i:i + 3] for i in range(len(cleaned) - 2)]

    def detect(self, text: str) -> LanguageIdentificationResult:
        """Identify the language of the provided input text.

        Args:
            text (str): Input query text.

        Returns:
            LanguageIdentificationResult: Structured identification result.
        """
        if text is None or not isinstance(text, str):
            return LanguageIdentificationResult(
                text="" if text is None else str(text),
                language=SupportedLanguage.UNKNOWN.value,
                confidence=0.0,
                is_supported=False,
                language_name="Unknown",
                is_reliable=False,
                detected_script="None",
                metadata={"reason": "Non-string or None input"},
            )

        stripped = text.strip()
        if not stripped:
            return LanguageIdentificationResult(
                text=text,
                language=SupportedLanguage.UNKNOWN.value,
                confidence=0.0,
                is_supported=False,
                language_name="Unknown",
                is_reliable=False,
                detected_script="None",
                metadata={"reason": "Empty or whitespace-only input"},
            )

        # Check if text contains any alphabetic characters
        alpha_chars = [c for c in stripped if c.isalpha()]
        if not alpha_chars:
            return LanguageIdentificationResult(
                text=text,
                language=SupportedLanguage.UNKNOWN.value,
                confidence=0.0,
                is_supported=False,
                language_name="Unknown",
                is_reliable=False,
                detected_script="Non-Alphabetic",
                metadata={"reason": "No alphabetic characters in text"},
            )

        # If custom provider is configured, attempt invocation first
        if self.custom_provider is not None:
            try:
                external_result = self.custom_provider(text)
                if external_result is not None:
                    return external_result
            except Exception as e:
                # Fall back gracefully to internal statistical detector
                pass

        # 1. Script Analysis
        script = self._extract_script(stripped)
        if script == "Devanagari":
            devanagari_count = sum(1 for c in stripped if self._is_devanagari_char(c))
            confidence = min(0.99, max(0.70, devanagari_count / len(alpha_chars)))
            return LanguageIdentificationResult(
                text=text,
                language=SupportedLanguage.HINDI.value,
                confidence=confidence,
                is_supported=True,
                language_name=SupportedLanguage.get_language_name(SupportedLanguage.HINDI.value),
                candidates=[
                    LanguageCandidate(
                        language=SupportedLanguage.HINDI.value,
                        confidence=confidence,
                        language_name="Hindi",
                    )
                ],
                is_reliable=confidence >= self.confidence_threshold,
                detected_script="Devanagari",
                metadata={"script": "Devanagari", "devanagari_ratio": round(devanagari_count / len(alpha_chars), 3)},
            )

        # 2. Latin-Script Language Analysis (English, Spanish, French, German)
        words = self._extract_words(stripped)
        trigrams = self._extract_trigrams(stripped)
        scores: Dict[str, float] = {
            SupportedLanguage.ENGLISH.value: 0.0,
            SupportedLanguage.SPANISH.value: 0.0,
            SupportedLanguage.FRENCH.value: 0.0,
            SupportedLanguage.GERMAN.value: 0.0,
        }

        # Diacritics matching (High-precision signal)
        lower_text = stripped.lower()
        for lang_code, diacritics in LANGUAGE_DIACRITICS.items():
            for d in diacritics:
                if d in lower_text:
                    scores[lang_code] += 2.5

        # Stopwords matching (High-precision signal)
        for word in words:
            for lang_code, stopwords in LANGUAGE_STOPWORDS.items():
                if lang_code in scores and word in stopwords:
                    scores[lang_code] += 1.8

        # Trigram matching (Robust background signal)
        if trigrams:
            for tri in trigrams:
                for lang_code, tri_profile in CHARACTER_TRIGRAM_PROFILES.items():
                    if tri in tri_profile:
                        scores[lang_code] += 0.35

        # Normalize scores to probabilities
        total_score = sum(scores.values())
        candidates: List[LanguageCandidate] = []

        if total_score > 0.0:
            for lang_code, raw_score in scores.items():
                prob = raw_score / total_score
                # Scale probability smoothly based on evidence strength
                confidence_multiplier = min(1.0, math.sqrt(total_score / 2.0))
                calibrated_conf = min(0.99, max(0.10, prob * confidence_multiplier))
                candidates.append(
                    LanguageCandidate(
                        language=lang_code,
                        confidence=calibrated_conf,
                        language_name=SupportedLanguage.get_language_name(lang_code),
                    )
                )
            candidates.sort(key=lambda c: c.confidence, reverse=True)
            top_candidate = candidates[0]
            top_lang = top_candidate.language
            top_conf = top_candidate.confidence
            is_reliable = top_conf >= self.confidence_threshold and (
                len(candidates) == 1 or (top_conf - candidates[1].confidence) >= 0.05
            )
        else:
            # Insufficient lexical or n-gram evidence
            top_lang = SupportedLanguage.UNKNOWN.value
            top_conf = 0.0
            is_reliable = False

        return LanguageIdentificationResult(
            text=text,
            language=top_lang,
            confidence=top_conf,
            is_supported=top_lang in SupportedLanguage.get_supported_codes(),
            language_name=SupportedLanguage.get_language_name(top_lang),
            candidates=candidates,
            is_reliable=is_reliable,
            detected_script=script,
            metadata={
                "word_count": len(words),
                "trigram_count": len(trigrams),
                "raw_scores": {k: round(v, 2) for k, v in scores.items()},
            },
        )
