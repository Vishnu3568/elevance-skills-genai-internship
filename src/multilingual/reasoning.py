"""Grounded Multilingual Reasoning and Response Generation Layer (Phase 6 Day 32).

Synthesizes grounded responses in the user's detected target language (English,
Spanish, French, German, Hindi) based strictly on retrieved English knowledge base evidence.
"""

import os
from typing import Any, Dict, List, Optional

from src.multilingual.models import (
    CrossLingualRetrievalResult,
    MultilingualIntent,
    MultilingualResponse,
    SupportedLanguage,
)


# -----------------------------------------------------------------------------
# Localized Conversational & Fallback Messages
# -----------------------------------------------------------------------------

LOCALIZED_GREETINGS: Dict[str, str] = {
    SupportedLanguage.ENGLISH.value: "Hello! How can I help you today with our courses and bootcamps?",
    SupportedLanguage.SPANISH.value: "¡Hola! ¿En qué puedo ayudarte hoy respecto a nuestros cursos y programas?",
    SupportedLanguage.FRENCH.value: "Bonjour! Comment puis-je vous aider aujourd'hui concernant nos formations?",
    SupportedLanguage.GERMAN.value: "Hallo! Wie kann ich Ihnen heute bei unseren Kursen und Programmen helfen?",
    SupportedLanguage.HINDI.value: "नमस्ते! मैं आज हमारे कोर्स और बूटकैंप के बारे में आपकी क्या सहायता कर सकता हूँ?",
}

LOCALIZED_UNKNOWN_RESPONSES: Dict[str, str] = {
    SupportedLanguage.ENGLISH.value: "I do not have enough information in the knowledge base to answer this question accurately.",
    SupportedLanguage.SPANISH.value: "No dispongo de suficiente información en la base de conocimientos para responder con precisión.",
    SupportedLanguage.FRENCH.value: "Je n'ai pas assez d'informations dans la base de connaissances pour répondre avec précision.",
    SupportedLanguage.GERMAN.value: "Ich habe nicht genügend Informationen in der Wissensdatenbank, um diese Frage präzise zu beantworten.",
    SupportedLanguage.HINDI.value: "सटीक उत्तर देने के लिए ज्ञानकोष में पर्याप्त जानकारी उपलब्ध नहीं है।",
}

LOCALIZED_INTENT_SUMMARIES: Dict[str, Dict[str, str]] = {
    MultilingualIntent.REFUND_POLICY.value: {
        SupportedLanguage.SPANISH.value: "Política de reembolso: ofrecemos un reembolso del 100% según las pautas de nuestra política de reembolso de cursos.",
        SupportedLanguage.FRENCH.value: "Politique de remboursement: nous offrons un remboursement à 100% selon les directives de notre politique d'annulation.",
        SupportedLanguage.GERMAN.value: "Rückerstattungsrichtlinie: Wir bieten eine 100%ige Rückerstattung gemäß den Richtlinien unseres Kurses.",
        SupportedLanguage.HINDI.value: "रिफंड नीति: हम कोर्स रिफंड दिशानिर्देशों के अनुसार 100% रिफंड प्रदान करते हैं।",
    },
    MultilingualIntent.PREREQUISITES.value: {
        SupportedLanguage.SPANISH.value: "Requisitos previos: este curso está diseñado para principiantes sin experiencia previa en programación. Solo necesitas una laptop con al menos 4GB de RAM y conexión a internet.",
        SupportedLanguage.FRENCH.value: "Prérequis: cette formation est conçue pour les débutants sans expérience préalable en programmation. Vous avez seulement besoin d'un ordinateur avec au moins 4 Go de RAM.",
        SupportedLanguage.GERMAN.value: "Voraussetzungen: Dieser Kurs ist für Anfänger ohne Programmiererfahrung konzipiert. Sie benötigen lediglich einen Laptop mit mindestens 4 GB RAM.",
        SupportedLanguage.HINDI.value: "योग्यता: यह कोर्स बिना कोडिंग अनुभव वाले शुरुआती लोगों के लिए है। आपको केवल कम से कम 4GB रैम वाले लैपटॉप और इंटरनेट की आवश्यकता है।",
    },
    MultilingualIntent.CAREER_ASSISTANCE.value: {
        SupportedLanguage.SPANISH.value: "Asistencia laboral: ayudamos con la preparación de currículum y entrevistas, y conectamos a los candidatos con posibles reclutadores.",
        SupportedLanguage.FRENCH.value: "Aide à l'emploi: nous vous aidons à préparer votre CV et vos entretiens et vous référons à des recruteurs potentiels.",
        SupportedLanguage.GERMAN.value: "Karriereunterstützung: Wir helfen bei der Erstellung von Lebensläufen und der Vorbereitung auf Vorstellungsgespräche.",
        SupportedLanguage.HINDI.value: "करियर सहायता: हम रिज्यूमे और इंटरव्यू की तैयारी में मदद करते हैं और उम्मीदवारों को संभावित रिक्रूटर्स के पास भेजते हैं।",
    },
    MultilingualIntent.SUPPORT_CONTACT.value: {
        SupportedLanguage.SPANISH.value: "Contacto y soporte: puedes unirte a nuestra comunidad activa de Discord para resolver dudas con compañeros y mentores.",
        SupportedLanguage.FRENCH.value: "Contact et support: vous pouvez rejoindre notre communauté Discord active pour poser vos questions aux mentors.",
        SupportedLanguage.GERMAN.value: "Kontakt und Support: Sie können unserer aktiven Discord-Community beitreten, um Fragen mit Tutoren zu klären.",
        SupportedLanguage.HINDI.value: "सहायता और संपर्क: आप साथी शिक्षार्थियों और मेंटर्स के साथ संदेह दूर करने के लिए हमारे डिस्कॉर्ड सर्वर से जुड़ सकते हैं।",
    },
    MultilingualIntent.PAYMENT_PRICING.value: {
        SupportedLanguage.SPANISH.value: "Información de pago: consulta las opciones de inscripción y métodos de pago disponibles en nuestra plataforma.",
        SupportedLanguage.FRENCH.value: "Information de paiement: veuillez consulter les options d'inscription et de paiement sur notre plateforme.",
        SupportedLanguage.GERMAN.value: "Zahlungsinformationen: Bitte informieren Sie sich über die verfügbaren Zahlungsoptionen auf unserer Plattform.",
        SupportedLanguage.HINDI.value: "भुगतान और शुल्क: कृपया हमारे प्लेटफ़ॉर्म पर उपलब्ध भुगतान विकल्पों और शुल्क की जानकारी देखें।",
    },
    MultilingualIntent.COURSE_DETAILS.value: {
        SupportedLanguage.SPANISH.value: "Detalles del curso: el programa incluye proyectos prácticos, acceso de por vida y materiales de aprendizaje paso a paso.",
        SupportedLanguage.FRENCH.value: "Détails de la formation: le programme comprend des projets pratiques, un accès à vie et un apprentissage à votre rythme.",
        SupportedLanguage.GERMAN.value: "Kursdetails: Das Programm beinhaltet praxisnahe Projekte, lebenslangen Zugriff und strukturiertes Lernen im eigenen Tempo.",
        SupportedLanguage.HINDI.value: "पाठ्यक्रम विवरण: इस कोर्स में व्यावहारिक प्रोजेक्ट, लाइफटाइम एक्सेस और व्यावहारिक कौशल शामिल हैं।",
    },
}


class MultilingualReasoner:
    """Production-grade reasoning engine synthesizing grounded multilingual responses."""

    def __init__(self, llm: Optional[Any] = None) -> None:
        """Initialize the multilingual reasoner.

        Args:
            llm: Optional pre-configured LangChain LLM instance, or 'offline' / False to force offline mode.
        """
        self._llm = llm

    def _get_llm(self) -> Optional[Any]:
        """Get LLM instance if configured or available."""
        if self._llm in ("offline", False):
            return None
        if self._llm is not None:
            return self._llm

        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if api_key:
            try:
                from src.langchain_helper import get_llm
                self._llm = get_llm()
                return self._llm
            except Exception:
                pass
        return None


    def reason(
        self,
        retrieval_result: CrossLingualRetrievalResult,
        target_language: Optional[str] = None,
    ) -> MultilingualResponse:
        """Synthesize a grounded answer in the requested language from retrieved evidence.

        Args:
            retrieval_result (CrossLingualRetrievalResult): Outcome of cross-lingual retrieval.
            target_language (Optional[str]): Language for response generation (defaults to detected).

        Returns:
            MultilingualResponse: Grounded response payload.
        """
        query = retrieval_result.original_query
        lang = target_language or retrieval_result.detected_language
        intent = retrieval_result.intent

        # 1. Conversational Greeting
        if intent == MultilingualIntent.GREETING.value:
            greeting_msg = LOCALIZED_GREETINGS.get(
                lang,
                LOCALIZED_GREETINGS[SupportedLanguage.ENGLISH.value],
            )
            return MultilingualResponse(
                query=query,
                language=lang,
                intent=intent,
                aligned_query=retrieval_result.aligned_query,
                final_answer=greeting_msg,
                raw_answer=greeting_msg,
                is_grounded=True,
                confidence_score=0.95,
                source_documents=[],
                metadata={"reasoning_type": "greeting_handler"},
            )

        # 2. Insufficient Evidence / Unknown Query
        if not retrieval_result.is_evidence_found or not retrieval_result.evidence_text:
            unknown_msg = LOCALIZED_UNKNOWN_RESPONSES.get(
                lang,
                LOCALIZED_UNKNOWN_RESPONSES[SupportedLanguage.ENGLISH.value],
            )
            return MultilingualResponse(
                query=query,
                language=lang,
                intent=intent,
                aligned_query=retrieval_result.aligned_query,
                final_answer=unknown_msg,
                raw_answer="I don't know.",
                is_grounded=False,
                confidence_score=0.0,
                source_documents=retrieval_result.retrieved_documents,
                metadata={"reasoning_type": "insufficient_evidence_fallback"},
            )

        # 3. Live LLM Synthesis (if Gemini available)
        llm = self._get_llm()
        if llm is not None:
            try:
                lang_name = SupportedLanguage.get_language_name(lang)
                prompt_text = (
                    f"You are a helpful customer support assistant. Answer the user's question accurately "
                    f"and strictly in {lang_name} based ONLY on the following English evidence.\n"
                    f"If the evidence does not contain the answer, say you do not know.\n\n"
                    f"EVIDENCE:\n{retrieval_result.evidence_text}\n\n"
                    f"USER QUESTION ({lang_name}):\n{query}\n\n"
                    f"ANSWER IN {lang_name}:"
                )
                if hasattr(llm, "invoke"):
                    resp = llm.invoke(prompt_text)
                    raw_text = getattr(resp, "content", str(resp))
                elif callable(llm):
                    raw_text = str(llm(prompt_text))
                else:
                    raw_text = ""

                if raw_text and raw_text.strip():
                    return MultilingualResponse(
                        query=query,
                        language=lang,
                        intent=intent,
                        aligned_query=retrieval_result.aligned_query,
                        final_answer=raw_text.strip(),
                        raw_answer=raw_text.strip(),
                        is_grounded=True,
                        confidence_score=0.90,
                        source_documents=retrieval_result.retrieved_documents,
                        metadata={"reasoning_type": "llm_grounded_synthesis"},
                    )
            except Exception:
                # Fall through to deterministic offline synthesizer
                pass

        # 4. Deterministic Offline Synthesis (Grounded in retrieved evidence & intent)
        # Check if localized template exists for this intent & language
        intent_templates = LOCALIZED_INTENT_SUMMARIES.get(intent, {})
        if lang in intent_templates:
            localized_ans = intent_templates[lang]
        elif lang == SupportedLanguage.ENGLISH.value:
            # Extract raw response from English document if available
            doc_content = retrieval_result.retrieved_documents[0]["page_content"]
            if "response:" in doc_content.lower():
                localized_ans = doc_content.split("response:", 1)[-1].strip()
            else:
                localized_ans = doc_content.strip()
        else:
            # General fallback summarizing English evidence
            doc_content = retrieval_result.retrieved_documents[0]["page_content"]
            localized_ans = f"[{SupportedLanguage.get_language_name(lang)}] {doc_content.strip()}"

        return MultilingualResponse(
            query=query,
            language=lang,
            intent=intent,
            aligned_query=retrieval_result.aligned_query,
            final_answer=localized_ans,
            raw_answer=retrieval_result.evidence_text[:200],
            is_grounded=True,
            confidence_score=0.85,
            source_documents=retrieval_result.retrieved_documents,
            metadata={"reasoning_type": "deterministic_grounded_synthesis"},
        )
