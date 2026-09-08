"""Deterministic Multilingual Intent Detection Engine for Phase 6.

Classifies customer service domain queries into canonical intents across English,
Spanish, French, German, and Hindi without external translation APIs or models.
"""

import re
import unicodedata
from typing import Any, Dict, List, Optional, Set, Tuple

from src.multilingual.detector import LanguageDetector
from src.multilingual.models import (
    MultilingualIntent,
    MultilingualIntentResult,
    SupportedLanguage,
)


# -----------------------------------------------------------------------------
# Intent Pattern Profiles per Supported Language
# -----------------------------------------------------------------------------

INTENT_LEXICON: Dict[str, Dict[str, List[str]]] = {
    # -------------------------------------------------------------------------
    # 1. English (en)
    # -------------------------------------------------------------------------
    SupportedLanguage.ENGLISH.value: {
        MultilingualIntent.GREETING.value: [
            r"\b(?:hello|hi|hey|good\s+morning|good\s+afternoon|good\s+evening|greetings)\b",
        ],
        MultilingualIntent.REFUND_POLICY.value: [
            r"\b(?:refund|refunds|money\s+back|return\s+policy|cancel(?:lation)?|get\s+my\s+money\s+back)\b",
        ],
        MultilingualIntent.PREREQUISITES.value: [
            r"\b(?:prerequisite|prerequisites|prior\s+experience|no\s+coding|programming\s+background|beginner|non-technical|4gb\s+ram|system\s+requirements?|laptop\s+requirements?)\b",
        ],
        MultilingualIntent.COURSE_DETAILS.value: [
            r"\b(?:duration|how\s+long|lifetime\s+access|syllabus|curriculum|topics?\s+covered|schedule|self-paced|datasets?\s+used|toy\s+datasets?|course\s+upgrades?)\b",
        ],
        MultilingualIntent.CAREER_ASSISTANCE.value: [
            r"\b(?:job\s+assistance|guarantee\s+(?:me\s+)?a\s+job|placement|resume|virtual\s+internship|cv|interview\s+prep|recruiters?|career\s+support)\b",
        ],
        MultilingualIntent.SUPPORT_CONTACT.value: [
            r"\b(?:contact\s+(?:the\s+)?instructors?|discord(?:\s+community|\s+server)?|reach\s+out|ask\s+questions?|doubts?\s+support|email\s+support|customer\s+care)\b",
        ],
        MultilingualIntent.PAYMENT_PRICING.value: [
            r"\b(?:emi(?:\s+options?)?|installments?|pricing|cost|price|fee|discounts?|payment\s+options?)\b",
        ],
        MultilingualIntent.TECHNICAL_SUPPORT.value: [
            r"\b(?:error|bug|spill\s+error|install\s+power\s+pivot|excel\s+formula|power\s+bi\s+(?:on|in)\s+mac|virtual\s+machine|code\s+(?:is\s+)?not\s+working)\b",
        ],
    },

    # -------------------------------------------------------------------------
    # 2. Spanish (es)
    # -------------------------------------------------------------------------
    SupportedLanguage.SPANISH.value: {
        MultilingualIntent.GREETING.value: [
            r"\b(?:hola|buenos\s+d[ií]as|buenas\s+tardes|buenas\s+noches|saludos|qu[eé]\s+tal)\b",
        ],
        MultilingualIntent.REFUND_POLICY.value: [
            r"\b(?:reembolso|reembolsos|devoluci[oó]n|devolver|cancelar|cancelaci[oó]n|recuperar\s+mi\s+dinero)\b",
        ],
        MultilingualIntent.PREREQUISITES.value: [
            r"\b(?:requisitos?|prerrequisitos?|experiencia\s+previa|sin\s+programaci[oó]n|principiante|no\s+t[eé]cnico|requisitos?\s+del\s+sistema|laptop|memoria\s+ram)\b",
        ],
        MultilingualIntent.COURSE_DETAILS.value: [
            r"\b(?:duraci[oó]n|cu[aá]nto\s+dura|acceso\s+de\s+por\s+vida|acceso\s+vitalicio|plan\s+de\s+estudios|temario|horario|a\s+su\s+propio\s+ritmo|conjunto\s+de\s+datos)\b",
        ],
        MultilingualIntent.CAREER_ASSISTANCE.value: [
            r"\b(?:asistencia\s+laboral|garant[ií]a\s+de\s+empleo|bolsa\s+de\s+trabajo|curr[ií]culum|curr[ií]culo|pasant[ií]a|pr[aá]cticas|entrevista|apoyo\s+profesional)\b",
        ],
        MultilingualIntent.SUPPORT_CONTACT.value: [
            r"\b(?:contactar\s+(?:a\s+los\s+)?instructores|discord|contactar|atenci[oó]n\s+al\s+cliente|resolver\s+dudas|comunidad|correo\s+de\s+soporte)\b",
        ],
        MultilingualIntent.PAYMENT_PRICING.value: [
            r"\b(?:cuotas|plazos|opci[oó]n\s+de\s+emi|pago|precio|costo|coste|tarifa|descuento|opciones?\s+de\s+pago)\b",
        ],
        MultilingualIntent.TECHNICAL_SUPPORT.value: [
            r"\b(?:error|fallo|instalar\s+power\s+pivot|f[oó]rmula\s+de\s+excel|power\s+bi\s+en\s+mac|m[aá]quina\s+virtual|c[oó]digo\s+no\s+funciona)\b",
        ],
    },

    # -------------------------------------------------------------------------
    # 3. French (fr)
    # -------------------------------------------------------------------------
    SupportedLanguage.FRENCH.value: {
        MultilingualIntent.GREETING.value: [
            r"\b(?:bonjour|salut|bonsoir|coucou|bienvenue)\b",
        ],
        MultilingualIntent.REFUND_POLICY.value: [
            r"\b(?:remboursement|remboursements|rembourser|annuler|annulation|r[eé]cup[eé]rer\s+mon\s+argent)\b",
        ],
        MultilingualIntent.PREREQUISITES.value: [
            r"\b(?:pr[eé]requis|conditions?\s+pr[eé]alables?|exp[eé]rience\s+pr[eé]alable|sans\s+programmation|d[eé]butant|non\s+technique|configuration\s+requise|ordinateur|m[eé]moire\s+ram)\b",
        ],
        MultilingualIntent.COURSE_DETAILS.value: [
            r"\b(?:dur[eé]e|combien\s+de\s+temps|acc[eè]s\s+[aà]\s+vie|programme|syllabus|sujets\s+abord[eé]s|emploi\s+du\s+temps|[aà]\s+votre\s+rythme|jeu\s+de\s+donn[eé]es)\b",
        ],
        MultilingualIntent.CAREER_ASSISTANCE.value: [
            r"\b(?:aide\s+[aà]\s+l'emploi|garantie\s+d'emploi|placement|curriculum\s+vitae|cv|stage|pr[eé]paration\s+aux\s+entretiens|recruteurs?|soutien\s+de\s+carri[eè]re)\b",
        ],
        MultilingualIntent.SUPPORT_CONTACT.value: [
            r"\b(?:contacter\s+(?:les\s+)?instructeurs|discord|contacter|service\s+client|poser\s+des\s+questions|support\s+des\s+doutes|communaut[eé])\b",
        ],
        MultilingualIntent.PAYMENT_PRICING.value: [
            r"\b(?:[eé]ch[eé]ances|paiement\s+[eé]chelonn[eé]|option\s+emi|paiement|prix|co[uû]t|frais|r[eé]duction|options?\s+de\s+paiement)\b",
        ],
        MultilingualIntent.TECHNICAL_SUPPORT.value: [
            r"\b(?:erreur|bogue|installer\s+power\s+pivot|formule\s+excel|power\s+bi\s+sur\s+mac|machine\s+virtuelle|code\s+ne\s+fonctionne\s+pas)\b",
        ],
    },

    # -------------------------------------------------------------------------
    # 4. German (de)
    # -------------------------------------------------------------------------
    SupportedLanguage.GERMAN.value: {
        MultilingualIntent.GREETING.value: [
            r"\b(?:hallo|guten\s+tag|guten\s+morgen|guten\s+abend|gr[uü][sß]\s+gott|servus|moin)\b",
        ],
        MultilingualIntent.REFUND_POLICY.value: [
            r"\b(?:r[uü]ckerstattung|erstatten|zur[uü]ckgeben|stornieren|stornierung|geld\s+zur[uü]ck)\b",
        ],
        MultilingualIntent.PREREQUISITES.value: [
            r"\b(?:voraussetzungen?|vorbedingungen?|vorerfahrung|ohne\s+programmiererfahrung|anf[aä]nger|nicht-technisch|systemanforderungen?|laptop|arbeitsspeicher|ram)\b",
        ],
        MultilingualIntent.COURSE_DETAILS.value: [
            r"\b(?:dauer|wie\s+lange|lebenslanger\s+zugang|lehrplan|kursplan|behandelte\s+themen|zeitplan|im\s+eigenen\s+tempo|datensatz|kursaktualisierungen?)\b",
        ],
        MultilingualIntent.CAREER_ASSISTANCE.value: [
            r"\b(?:berufsunterst[uü]tzung|jobgarantie|stellenvermittlung|lebenslauf|praktikum|vorstellungsgespr[aä]ch|recruiter|karrierehilfe)\b",
        ],
        MultilingualIntent.SUPPORT_CONTACT.value: [
            r"\b(?:dozenten\s+kontaktieren|discord|kontakt|kundenservice|fragen\s+stellen|hilfe\s+bei\s+fragen|community|support)\b",
        ],
        MultilingualIntent.PAYMENT_PRICING.value: [
            r"\b(?:ratenzahlung|monatliche\s+raten|emi-option|zahlung|preis|kosten|geb[uü]hr|rabatt|zahlungsoptionen?)\b",
        ],
        MultilingualIntent.TECHNICAL_SUPPORT.value: [
            r"\b(?:fehler|bug|power\s+pivot\s+installieren|excel-formel|power\s+bi\s+auf\s+mac|virtuelle\s+maschine|code\s+funktioniert\s+nicht)\b",
        ],
    },

    # -------------------------------------------------------------------------
    # 5. Hindi (hi)
    # -------------------------------------------------------------------------
    SupportedLanguage.HINDI.value: {
        MultilingualIntent.GREETING.value: [
            r"(?:नमस्ते|नमस्कार|प्रणाम|हैलो|हेलो|सुप्रभात)",
        ],
        MultilingualIntent.REFUND_POLICY.value: [
            r"(?:रिफंड|वापसी|पैसे\s+वापस|रद्द|कैंसल|रिफंड\s+नीति|धनवापसी)",
        ],
        MultilingualIntent.PREREQUISITES.value: [
            r"(?:शुरुआती|योग्यता|कोडिंग\s+अनुभव|प्रोग्रामिंग\s+ज्ञान|गैर-तकनीकी|लैपटॉप|रैम|ज़रूरत|पूर्व\s+अनुभव)",
        ],
        MultilingualIntent.COURSE_DETAILS.value: [
            r"(?:अवधि|कितना\s+समय|लाइफटाइम\s+एक्सेस|आजीवन\s+पहुंच|पाठ्यक्रम|सिलेबस|विषय|शेड्यूल|डेटासेट)",
        ],
        MultilingualIntent.CAREER_ASSISTANCE.value: [
            r"(?:नौकरी|जॉब|इंटर्नशिप|रिज्यूमे|प्लेसमेंट|साक्षात्कार|कैरियर\s+सपोर्ट|जॉब\s+गारंटी)",
        ],
        MultilingualIntent.SUPPORT_CONTACT.value: [
            r"(?:संपर्क|शिक्षक\s+से\s+संपर्क|डिस्कॉर्ड|सपोर्ट|सवाल|संदेह|कम्युनिटी|सहायता)",
        ],
        MultilingualIntent.PAYMENT_PRICING.value: [
            r"(?:ईएमआई|किस्त|कीमत|फीस|भुगतान|लागत|छूट|किस्तों\s+में)",
        ],
        MultilingualIntent.TECHNICAL_SUPPORT.value: [
            r"(?:त्रुटि|बग|एरर|फार्मूला|कोड\s+काम\s+नहीं\s+कर\s+रहा|इंस्टॉल|मैक\s+पर\s+पॉवर\s+बीआई)",
        ],
    },
}


# -----------------------------------------------------------------------------
# Multilingual Intent Classifier Engine
# -----------------------------------------------------------------------------

class MultilingualIntentClassifier:
    """Production-grade deterministic multilingual intent classifier."""

    def __init__(
        self,
        language_detector: Optional[LanguageDetector] = None,
        confidence_threshold: float = 0.50,
    ) -> None:
        """Initialize the intent classifier.

        Args:
            language_detector (Optional[LanguageDetector]): Injected or default detector.
            confidence_threshold (float): Minimum confidence threshold for classification.
        """
        if not (0.0 <= confidence_threshold <= 1.0):
            raise ValueError(f"Confidence threshold {confidence_threshold} must be within [0.0, 1.0].")
        self.detector = language_detector if language_detector is not None else LanguageDetector()
        self.confidence_threshold = confidence_threshold

    def _normalize(self, text: str) -> str:
        """Normalize unicode text and lowercase."""
        return unicodedata.normalize("NFKC", text).lower().strip()

    def classify(
        self,
        text: str,
        language: Optional[str] = None,
    ) -> MultilingualIntentResult:
        """Classify user query into canonical customer-service domain intent.

        Args:
            text (str): Query text.
            language (Optional[str]): Language code if already identified.

        Returns:
            MultilingualIntentResult: Structured intent classification result.
        """
        if text is None or not isinstance(text, str):
            return MultilingualIntentResult(
                text="" if text is None else str(text),
                intent=MultilingualIntent.UNKNOWN.value,
                confidence=0.0,
                language=SupportedLanguage.UNKNOWN.value,
                is_recognized=False,
                matched_keywords=[],
                metadata={"reason": "Non-string or None input"},
            )

        stripped = text.strip()
        if not stripped:
            return MultilingualIntentResult(
                text=text,
                intent=MultilingualIntent.UNKNOWN.value,
                confidence=0.0,
                language=SupportedLanguage.UNKNOWN.value,
                is_recognized=False,
                matched_keywords=[],
                metadata={"reason": "Empty or whitespace-only input"},
            )

        # Check for alphabetic characters
        alpha_chars = [c for c in stripped if c.isalpha()]
        if not alpha_chars:
            return MultilingualIntentResult(
                text=text,
                intent=MultilingualIntent.UNKNOWN.value,
                confidence=0.0,
                language=SupportedLanguage.UNKNOWN.value,
                is_recognized=False,
                matched_keywords=[],
                metadata={"reason": "No alphabetic characters in query"},
            )

        # Determine effective language if not provided
        effective_lang = language
        if not effective_lang or effective_lang == SupportedLanguage.UNKNOWN.value:
            det_result = self.detector.detect(stripped)
            effective_lang = det_result.language if det_result.is_supported else SupportedLanguage.ENGLISH.value

        norm_query = self._normalize(stripped)

        # Match intent patterns for effective language (and fallback cross-language if needed)
        candidate_scores: Dict[str, Tuple[float, List[str]]] = {}

        # 1. Primary language pattern evaluation
        lang_profiles = [effective_lang]
        # Include English fallback for cross-lingual terms (like 'discord', 'mac', '4gb ram')
        if effective_lang != SupportedLanguage.ENGLISH.value and SupportedLanguage.ENGLISH.value not in lang_profiles:
            lang_profiles.append(SupportedLanguage.ENGLISH.value)

        for lang in lang_profiles:
            intents_for_lang = INTENT_LEXICON.get(lang, {})
            for intent_name, patterns in intents_for_lang.items():
                matched_terms: List[str] = []
                for pattern in patterns:
                    matches = re.findall(pattern, norm_query, flags=re.IGNORECASE | re.UNICODE)
                    if matches:
                        for m in matches:
                            if isinstance(m, str):
                                matched_terms.append(m)
                            elif isinstance(m, tuple):
                                matched_terms.extend([x for x in m if x])

                if matched_terms:
                    # Weight score based on number of matches and match length
                    match_score = 0.80 + min(0.18, len(matched_terms) * 0.05)
                    if intent_name not in candidate_scores or match_score > candidate_scores[intent_name][0]:
                        candidate_scores[intent_name] = (match_score, matched_terms)

        if candidate_scores:
            # Sort candidates by score descending
            sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1][0], reverse=True)
            best_intent, (best_score, matched_kw) = sorted_candidates[0]

            # Ensure confidence is strictly within [0.0, 1.0]
            confidence = min(1.0, max(0.0, best_score))
            is_recognized = confidence >= self.confidence_threshold

            return MultilingualIntentResult(
                text=text,
                intent=best_intent,
                confidence=confidence,
                language=effective_lang,
                is_recognized=is_recognized,
                matched_keywords=matched_kw,
                metadata={
                    "total_intent_matches": len(sorted_candidates),
                    "all_candidates": {k: round(v[0], 3) for k, v in sorted_candidates},
                },
            )

        # 2. If no specialized intent matched but text has valid linguistic content -> GENERAL_INQUIRY
        return MultilingualIntentResult(
            text=text,
            intent=MultilingualIntent.GENERAL_INQUIRY.value,
            confidence=0.60,
            language=effective_lang,
            is_recognized=True,
            matched_keywords=[],
            metadata={"reason": "Grounded fallback general query"},
        )
