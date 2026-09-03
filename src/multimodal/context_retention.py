"""Multimodal Context Retention Layer (Phase 5 — Day 27 Step 2).

Provides retrieval and structured formatting of recent multimodal conversation history
from active MultimodalConversationSession instances. Preserves dialogue flow,
textual queries, assistant answers, and visual metadata summaries in chronological
order without duplicating raw byte payloads or leaking cross-session state.
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Union

from .conversation import (
    ConversationTurn,
    MultimodalConversationSession,
    MultimodalSessionManager,
)
from .models import ModalityType


# -----------------------------------------------------------------------------
# Retained Turn View Contract
# -----------------------------------------------------------------------------

@dataclass
class RetainedTurnView:
    """Compact, byte-free view of a single dialogue turn for reasoning consumption."""

    turn_index: int
    modality: ModalityType
    user_query: str
    assistant_answer: str
    scene_description: Optional[str] = None
    detected_objects: List[str] = field(default_factory=list)
    visible_text: List[str] = field(default_factory=list)
    visual_attributes: Dict[str, Any] = field(default_factory=dict)
    image_metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Serialize turn view to dictionary without raw image bytes."""
        return {
            "turn_index": self.turn_index,
            "modality": self.modality.value if isinstance(self.modality, ModalityType) else str(self.modality),
            "user_query": self.user_query,
            "assistant_answer": self.assistant_answer,
            "scene_description": self.scene_description,
            "detected_objects": list(self.detected_objects),
            "visible_text": list(self.visible_text),
            "visual_attributes": dict(self.visual_attributes),
            "image_metadata": dict(self.image_metadata),
            "created_at": self.created_at,
        }


# -----------------------------------------------------------------------------
# Retained Conversation Context Container
# -----------------------------------------------------------------------------

@dataclass
class RetainedConversationContext:
    """Structured representation of recent dialogue context retrieved from a session."""

    session_id: str
    retained_turns: List[RetainedTurnView] = field(default_factory=list)
    total_session_turns: int = 0
    window_size: int = 0
    has_history: bool = False

    @property
    def turn_count(self) -> int:
        """Number of turns currently retained in this context window."""
        return len(self.retained_turns)

    def get_dialogue_history_text(self) -> str:
        """Format the chronological dialogue history into clean textual lines."""
        if not self.retained_turns:
            return "No previous conversation history."

        blocks: List[str] = []
        for turn in self.retained_turns:
            lines: List[str] = [
                f"Turn {turn.turn_index + 1} [{turn.modality.value.upper()}]:",
                f"User: {turn.user_query if turn.user_query else '[Image Upload]'}",
            ]
            if turn.scene_description or turn.detected_objects:
                details: List[str] = []
                if turn.scene_description:
                    details.append(f"Scene: {turn.scene_description}")
                if turn.detected_objects:
                    details.append(f"Detected: {', '.join(turn.detected_objects)}")
                lines.append(f"Visual Context: {' | '.join(details)}")

            lines.append(f"Assistant: {turn.assistant_answer}")
            blocks.append("\n".join(lines))

        return "\n\n".join(blocks)

    def get_last_turn(self) -> Optional[RetainedTurnView]:
        """Return the most recent turn in the retained window, or None."""
        return self.retained_turns[-1] if self.retained_turns else None

    def get_all_detected_objects(self) -> List[str]:
        """Return all unique detected objects mentioned across retained turns."""
        seen = set()
        result: List[str] = []
        for turn in self.retained_turns:
            for obj in turn.detected_objects:
                if obj not in seen:
                    seen.add(obj)
                    result.append(obj)
        return result

    def get_last_visual_summary(self) -> Optional[Dict[str, Any]]:
        """Return the most recent visual context summary from retained turns, if any."""
        for turn in reversed(self.retained_turns):
            if turn.scene_description or turn.detected_objects or turn.image_metadata:
                return {
                    "turn_index": turn.turn_index,
                    "scene_description": turn.scene_description,
                    "detected_objects": list(turn.detected_objects),
                    "visible_text": list(turn.visible_text),
                    "visual_attributes": dict(turn.visual_attributes),
                    "image_metadata": dict(turn.image_metadata),
                }
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize retained context to dictionary without raw bytes."""
        return {
            "session_id": self.session_id,
            "has_history": self.has_history,
            "turn_count": self.turn_count,
            "total_session_turns": self.total_session_turns,
            "window_size": self.window_size,
            "retained_turns": [turn.to_dict() for turn in self.retained_turns],
        }


# -----------------------------------------------------------------------------
# Multimodal Context Retriever Service
# -----------------------------------------------------------------------------

class MultimodalContextRetriever:
    """Extracts, filters, and formats recent conversation context from sessions."""

    def __init__(self, default_window_size: int = 3) -> None:
        """Initialize retriever with default history window size.

        Args:
            default_window_size: Number of recent turns to retrieve by default.
        """
        if not isinstance(default_window_size, int) or isinstance(default_window_size, bool):
            raise TypeError(f"default_window_size must be an int, got {type(default_window_size).__name__}")
        if default_window_size <= 0:
            raise ValueError(f"default_window_size must be positive, got {default_window_size}")

        self.default_window_size = default_window_size

    def retrieve_context(
        self,
        session: Optional[MultimodalConversationSession],
        window_size: Optional[int] = None,
    ) -> RetainedConversationContext:
        """Retrieve recent conversation turns from an active session.

        Args:
            session: Active conversation session, or None.
            window_size: Optional custom turn window (defaults to self.default_window_size).

        Returns:
            RetainedConversationContext: Structured, chronologically ordered context.
        """
        if session is None:
            eff_window = window_size if window_size is not None else self.default_window_size
            return RetainedConversationContext(
                session_id="",
                retained_turns=[],
                total_session_turns=0,
                window_size=eff_window,
                has_history=False,
            )

        if not isinstance(session, MultimodalConversationSession):
            raise TypeError(
                f"Expected MultimodalConversationSession, got {type(session).__name__}"
            )

        eff_window = window_size if window_size is not None else self.default_window_size
        if not isinstance(eff_window, int) or isinstance(eff_window, bool) or eff_window <= 0:
            raise ValueError(f"window_size must be a positive integer, got {eff_window}")

        turns = session.get_turns()
        if not turns:
            return RetainedConversationContext(
                session_id=session.session_id,
                retained_turns=[],
                total_session_turns=session.total_turns_count,
                window_size=eff_window,
                has_history=False,
            )

        # Slice the most recent eff_window turns, preserving chronological order
        recent_turns = turns[-eff_window:] if len(turns) > eff_window else list(turns)

        turn_views: List[RetainedTurnView] = []
        for t in recent_turns:
            scene_desc = None
            detected_objs: List[str] = []
            vis_text: List[str] = []
            vis_attrs: Dict[str, Any] = {}
            img_meta: Dict[str, Any] = {}

            if t.context_summary is not None:
                scene_desc = t.context_summary.scene_description
                detected_objs = list(t.context_summary.detected_objects)
                vis_text = list(t.context_summary.visible_text)
                vis_attrs = dict(t.context_summary.visual_attributes)
                img_meta = dict(t.context_summary.image_metadata)

            view = RetainedTurnView(
                turn_index=t.turn_index,
                modality=t.modality,
                user_query=t.user_query,
                assistant_answer=t.assistant_response.answer,
                scene_description=scene_desc,
                detected_objects=detected_objs,
                visible_text=vis_text,
                visual_attributes=vis_attrs,
                image_metadata=img_meta,
                created_at=t.created_at,
            )
            turn_views.append(view)

        return RetainedConversationContext(
            session_id=session.session_id,
            retained_turns=turn_views,
            total_session_turns=session.total_turns_count,
            window_size=eff_window,
            has_history=len(turn_views) > 0,
        )

    def retrieve_context_from_manager(
        self,
        manager: MultimodalSessionManager,
        session_id: str,
        window_size: Optional[int] = None,
    ) -> RetainedConversationContext:
        """Retrieve recent context for a session managed by MultimodalSessionManager.

        Args:
            manager: Active session manager registry.
            session_id: Identifier of the target session.
            window_size: Optional custom turn window.

        Returns:
            RetainedConversationContext: Structured context for the requested session.
        """
        if not isinstance(manager, MultimodalSessionManager):
            raise TypeError(f"Expected MultimodalSessionManager, got {type(manager).__name__}")

        session = manager.get_session(session_id)
        if session is None:
            eff_window = window_size if window_size is not None else self.default_window_size
            return RetainedConversationContext(
                session_id=session_id if isinstance(session_id, str) else "",
                retained_turns=[],
                total_session_turns=0,
                window_size=eff_window,
                has_history=False,
            )

        return self.retrieve_context(session, window_size=window_size)


# -----------------------------------------------------------------------------
# Convenience Functional Interface
# -----------------------------------------------------------------------------

def retrieve_conversation_context(
    session: Optional[MultimodalConversationSession],
    window_size: int = 3,
) -> RetainedConversationContext:
    """Retrieve recent conversation context with sensible defaults.

    Args:
        session: Active conversation session, or None.
        window_size: Number of recent turns to retrieve (default 3).

    Returns:
        RetainedConversationContext: Bounded, chronologically ordered context.
    """
    retriever = MultimodalContextRetriever(default_window_size=window_size)
    return retriever.retrieve_context(session, window_size=window_size)
