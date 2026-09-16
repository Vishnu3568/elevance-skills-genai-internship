"""Cross-Task High-Level Service Facade (Day 36).

Provides the unified programmatic entry point for full-platform query execution,
connecting applications and interfaces cleanly to the CrossTaskOrchestrator.
"""

from typing import Any, Dict, Optional

from .models import UnifiedRequest, UnifiedResponse
from .orchestrator import CrossTaskOrchestrator


class CrossTaskService:
    """High-level service interface coordinating cross-task interactions."""

    def __init__(self, orchestrator: Optional[CrossTaskOrchestrator] = None) -> None:
        """Initialize the cross-task service."""
        self.orchestrator = orchestrator if orchestrator is not None else CrossTaskOrchestrator()

    def process(
        self,
        query: str,
        image: Optional[Any] = None,
        session_id: str = "default_session",
        domain_override: Optional[str] = None,
        language_hint: Optional[str] = None,
        temperature: float = 0.1,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> UnifiedResponse:
        """Process a query through the unified cross-task architecture.

        Args:
            query (str): The natural language query.
            image (Optional[Any]): Optional ImageArtifact for multimodal requests.
            session_id (str): Conversation session tracking ID.
            domain_override (Optional[str]): Optional manual domain pin.
            language_hint (Optional[str]): Optional language hint.
            temperature (float): Generation temperature.
            metadata (Optional[Dict[str, Any]]): Optional request metadata.

        Returns:
            UnifiedResponse: Standardized response payload.
        """
        request = UnifiedRequest(
            query=query,
            image=image,
            session_id=session_id,
            language_hint=language_hint,
            domain_override=domain_override,
            temperature=temperature,
            metadata=metadata if metadata is not None else {},
        )
        return self.process_request(request)

    def process_request(self, request: UnifiedRequest) -> UnifiedResponse:
        """Process an already instantiated UnifiedRequest.

        Args:
            request (UnifiedRequest): Unified request data model.

        Returns:
            UnifiedResponse: Standardized response payload.
        """
        if not isinstance(request, UnifiedRequest):
            raise TypeError(f"Expected UnifiedRequest instance, got {type(request).__name__}")
        return self.orchestrator.dispatch(request)
