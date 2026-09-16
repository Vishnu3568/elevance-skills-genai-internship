"""Cross-Task Orchestrator for ElevanceSkills-GenAI-Internship (Day 36).

Orchestrates cross-domain query dispatch, multimodal request handling, and cross-cutting
multilingual intelligence across Tasks 1–6 with zero spaghetti coupling.
"""

import time
from typing import Any, Dict, Optional

from .adapters import (
    adapt_chatbot_response,
    adapt_medical_response,
    adapt_multilingual_response,
    adapt_multimodal_response,
    adapt_scientific_response,
)
from .models import (
    DomainType,
    UnifiedRequest,
    UnifiedResponse,
)
from .router import CrossTaskRouter, RoutingDecision


class CrossTaskOrchestrator:
    """Central orchestrator coordinating specialist domain services and cross-cutting layers."""

    def __init__(
        self,
        router: Optional[CrossTaskRouter] = None,
        chatbot_service: Optional[Any] = None,
        medical_qa_service: Optional[Any] = None,
        scientific_expert_service: Optional[Any] = None,
        multimodal_service: Optional[Any] = None,
        multilingual_service: Optional[Any] = None,
    ) -> None:
        """Initialize CrossTaskOrchestrator with optional injected services."""
        self.router = router if router is not None else CrossTaskRouter()
        self._chatbot_service = chatbot_service
        self._medical_qa_service = medical_qa_service
        self._scientific_expert_service = scientific_expert_service
        self._multimodal_service = multimodal_service
        self._multilingual_service = multilingual_service

    # -------------------------------------------------------------------------
    # Lazy Service Initializers (Preserves Startup Speed and Injection Support)
    # -------------------------------------------------------------------------

    def get_chatbot_service(self) -> Any:
        """Retrieve or lazily initialize Customer Service chatbot."""
        if self._chatbot_service is None:
            try:
                from src.chatbot_service import ChatbotService
                self._chatbot_service = ChatbotService()
            except ImportError:
                from chatbot_service import ChatbotService  # type: ignore
                self._chatbot_service = ChatbotService()
        return self._chatbot_service

    def get_medical_qa_service(self) -> Any:
        """Retrieve or lazily initialize Medical Q&A service."""
        if self._medical_qa_service is None:
            try:
                from src.langchain_helper import get_instructor_embeddings
                from src.medical_qa_service import MedicalQAService
                from langchain_community.vectorstores import FAISS
            except ImportError:
                from langchain_helper import get_instructor_embeddings  # type: ignore
                from medical_qa_service import MedicalQAService  # type: ignore
                from langchain.vectorstores import FAISS  # type: ignore

            try:
                vector_db = FAISS.load_local("faiss_index_medical", get_instructor_embeddings())
            except Exception:
                vector_db = None
            self._medical_qa_service = MedicalQAService(vector_db=vector_db)
        return self._medical_qa_service

    def get_scientific_expert_service(self) -> Any:
        """Retrieve or lazily initialize Scientific Expert service."""
        if self._scientific_expert_service is None:
            try:
                from src.langchain_helper import get_instructor_embeddings, get_llm
                from src.scientific_kb import (
                    ScientificExpertService,
                    ScientificGenerator,
                    ScientificRetriever,
                    load_scientific_vector_store,
                )
            except ImportError:
                from langchain_helper import get_instructor_embeddings, get_llm  # type: ignore
                from scientific_kb import (  # type: ignore
                    ScientificExpertService,
                    ScientificGenerator,
                    ScientificRetriever,
                    load_scientific_vector_store,
                )

            try:
                vector_store = load_scientific_vector_store("faiss_index_scientific", embeddings=get_instructor_embeddings())
                retriever = ScientificRetriever(vector_store=vector_store, default_k=3)
                try:
                    generator = ScientificGenerator(llm=get_llm())
                except Exception:
                    generator = ScientificGenerator(llm=lambda p: "Evidence retrieved successfully.")
                self._scientific_expert_service = ScientificExpertService(retriever=retriever, generator=generator)
            except Exception:
                self._scientific_expert_service = None
        return self._scientific_expert_service

    def get_multimodal_service(self) -> Any:
        """Retrieve or lazily initialize Multimodal Assistant service."""
        if self._multimodal_service is None:
            try:
                from src.multimodal.service import MultimodalAssistantService
                self._multimodal_service = MultimodalAssistantService()
            except ImportError:
                from multimodal.service import MultimodalAssistantService  # type: ignore
                self._multimodal_service = MultimodalAssistantService()
        return self._multimodal_service

    def get_multilingual_service(self) -> Any:
        """Retrieve or lazily initialize Multilingual cross-lingual service."""
        if self._multilingual_service is None:
            try:
                from src.multilingual.service import MultilingualService
                self._multilingual_service = MultilingualService()
            except ImportError:
                from multilingual.service import MultilingualService  # type: ignore
                self._multilingual_service = MultilingualService()
        return self._multilingual_service

    # -------------------------------------------------------------------------
    # Core Orchestration Dispatch
    # -------------------------------------------------------------------------

    def dispatch(self, request: UnifiedRequest) -> UnifiedResponse:
        """Dispatch UnifiedRequest through routing, specialist execution, and adaptation."""
        start_time = time.perf_counter()

        # 1. Evaluate Routing and Language
        decision: RoutingDecision = self.router.route(request)

        try:
            # 2. Multimodal Routing Branch
            if decision.domain == DomainType.MULTIMODAL:
                try:
                    from src.multimodal.models import MultimodalRequest
                except ImportError:
                    from multimodal.models import MultimodalRequest  # type: ignore

                mm_req = MultimodalRequest(
                    query=request.query if request.query else "",
                    images=[request.image] if request.image is not None else [],
                    session_id=request.session_id,
                    temperature=request.temperature,
                )
                mm_service = self.get_multimodal_service()
                mm_resp = mm_service.process_request(mm_req, session_id=request.session_id)
                execution_time_ms = (time.perf_counter() - start_time) * 1000.0

                unified_resp = adapt_multimodal_response(
                    resp=mm_resp,
                    query=request.query,
                    detected_language=decision.detected_language,
                    detected_language_name=decision.detected_language_name,
                    session_id=request.session_id,
                    execution_time_ms=execution_time_ms,
                )
                unified_resp.metadata["routing_decision"] = decision.to_dict()
                return unified_resp

            # 3. Medical Domain Branch
            if decision.domain == DomainType.MEDICAL:
                med_service = self.get_medical_qa_service()
                med_resp = med_service.process_query(request.query)
                execution_time_ms = (time.perf_counter() - start_time) * 1000.0

                unified_resp = adapt_medical_response(
                    resp=med_resp,
                    query=request.query,
                    detected_language=decision.detected_language,
                    detected_language_name=decision.detected_language_name,
                    session_id=request.session_id,
                    execution_time_ms=execution_time_ms,
                )
                unified_resp.metadata["routing_decision"] = decision.to_dict()
                return unified_resp

            # 4. Scientific Domain Branch
            if decision.domain == DomainType.SCIENTIFIC:
                sci_service = self.get_scientific_expert_service()
                sci_resp = sci_service.process_query(request.query)
                execution_time_ms = (time.perf_counter() - start_time) * 1000.0

                unified_resp = adapt_scientific_response(
                    resp=sci_resp,
                    query=request.query,
                    detected_language=decision.detected_language,
                    detected_language_name=decision.detected_language_name,
                    session_id=request.session_id,
                    execution_time_ms=execution_time_ms,
                )
                unified_resp.metadata["routing_decision"] = decision.to_dict()
                return unified_resp

            # 5. Customer Support / Multilingual Cross-Lingual Branch
            if decision.is_cross_lingual and decision.detected_language != "en":
                try:
                    from src.multilingual.models import MultilingualTextRequest
                except ImportError:
                    from multilingual.models import MultilingualTextRequest  # type: ignore

                ml_req = MultilingualTextRequest(
                    text=request.query,
                    session_id=request.session_id,
                    forced_language=decision.detected_language,
                )
                ml_service = self.get_multilingual_service()
                ml_resp = ml_service.process_text_request(ml_req, session_id=request.session_id)
                execution_time_ms = (time.perf_counter() - start_time) * 1000.0

                unified_resp = adapt_multilingual_response(
                    resp=ml_resp,
                    query=request.query,
                    domain=DomainType.CUSTOMER_SUPPORT.value,
                    session_id=request.session_id,
                    execution_time_ms=execution_time_ms,
                )
                unified_resp.metadata["routing_decision"] = decision.to_dict()
                return unified_resp

            # Standard Customer Support Pipeline (Tasks 1 & 2)
            cb_service = self.get_chatbot_service()
            cb_resp = cb_service.process_query(request.query)
            execution_time_ms = (time.perf_counter() - start_time) * 1000.0

            unified_resp = adapt_chatbot_response(
                resp=cb_resp,
                query=request.query,
                detected_language=decision.detected_language,
                detected_language_name=decision.detected_language_name,
                session_id=request.session_id,
                execution_time_ms=execution_time_ms,
            )
            unified_resp.metadata["routing_decision"] = decision.to_dict()
            return unified_resp

        except Exception as e:
            execution_time_ms = (time.perf_counter() - start_time) * 1000.0
            return UnifiedResponse(
                query=request.query,
                final_text_response=(
                    "An error occurred while coordinating your request across specialist services. "
                    f"Details: {str(e)}"
                ),
                domain=decision.domain.value,
                detected_language=decision.detected_language,
                detected_language_name=decision.detected_language_name,
                confidence_score=0.0,
                confidence_tier="INSUFFICIENT",
                is_grounded=False,
                warning_message=f"Orchestration exception: {str(e)}",
                execution_time_ms=execution_time_ms,
                session_id=request.session_id,
                metadata={"error": str(e), "routing_decision": decision.to_dict()},
            )
