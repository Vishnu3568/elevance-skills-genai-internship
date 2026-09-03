"""Multimodal Conversation and Session State Management (Phase 5 — Day 27 Step 1).

Provides explicit, validated, and bounded conversation session tracking for the
Multimodal AI Assistant. Preserves chronological multi-turn dialogue history
(user queries, modalities, context references, and assistant responses) while
strictly avoiding raw image byte duplication and isolating session state.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from typing import Any, Dict, List, Optional, Union

from .context import MultimodalContext
from .models import ModalityType, MultimodalResponse, validate_multimodal_response


# -----------------------------------------------------------------------------
# Lightweight Context Summary Model (No Raw Bytes)
# -----------------------------------------------------------------------------

@dataclass
class MultimodalContextSummary:
    """Lightweight metadata-only reference of multimodal context for conversation history.

    Explicitly avoids storing raw byte payloads (such as ImageArtifact.data or base64 blobs).
    """

    scene_description: Optional[str] = None
    detected_objects: List[str] = field(default_factory=list)
    visible_text: List[str] = field(default_factory=list)
    visual_attributes: Dict[str, Any] = field(default_factory=dict)
    image_metadata: Dict[str, Any] = field(default_factory=dict)
    text_entities: List[str] = field(default_factory=list)
    evidence_count: int = 0

    @classmethod
    def from_context(
        cls,
        context: Optional[Union[MultimodalContext, "MultimodalContextSummary", Dict[str, Any]]],
    ) -> Optional["MultimodalContextSummary"]:
        """Extract a clean, byte-free summary from a MultimodalContext or dict."""
        if context is None:
            return None

        if isinstance(context, MultimodalContextSummary):
            return context

        if isinstance(context, MultimodalContext):
            img_meta: Dict[str, Any] = {}
            if context.visual_context.artifact is not None:
                art = context.visual_context.artifact
                img_meta = {
                    "format": art.format,
                    "mime_type": art.mime_type,
                    "width": art.width,
                    "height": art.height,
                    "file_name": art.file_name,
                    "aspect_ratio": round(art.width / art.height, 4) if art.height > 0 else 1.0,
                }

            return cls(
                scene_description=context.visual_context.scene_description,
                detected_objects=list(context.visual_context.detected_objects),
                visible_text=list(context.visual_context.visible_text),
                visual_attributes=dict(context.visual_context.visual_attributes),
                image_metadata=img_meta,
                text_entities=list(context.text_context.extracted_entities),
                evidence_count=len(context.evidence_items),
            )

        if isinstance(context, dict):
            return cls(
                scene_description=context.get("scene_description"),
                detected_objects=list(context.get("detected_objects", [])),
                visible_text=list(context.get("visible_text", [])),
                visual_attributes=dict(context.get("visual_attributes", {})),
                image_metadata=dict(context.get("image_metadata", {})),
                text_entities=list(context.get("text_entities", [])),
                evidence_count=int(context.get("evidence_count", 0)),
            )

        raise TypeError(f"Cannot build MultimodalContextSummary from {type(context).__name__}")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize context summary to dictionary."""
        return {
            "scene_description": self.scene_description,
            "detected_objects": list(self.detected_objects),
            "visible_text": list(self.visible_text),
            "visual_attributes": dict(self.visual_attributes),
            "image_metadata": dict(self.image_metadata),
            "text_entities": list(self.text_entities),
            "evidence_count": self.evidence_count,
        }


# -----------------------------------------------------------------------------
# Conversation Turn Model
# -----------------------------------------------------------------------------

@dataclass
class ConversationTurn:
    """Represents a single conversational interaction in a multimodal session."""

    turn_index: int
    session_id: str
    modality: ModalityType
    user_query: str
    assistant_response: MultimodalResponse
    context_summary: Optional[MultimodalContextSummary] = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def __post_init__(self) -> None:
        """Validate conversation turn invariants."""
        validate_conversation_turn(self)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize conversation turn to dictionary without raw byte payloads."""
        return {
            "turn_index": self.turn_index,
            "session_id": self.session_id,
            "modality": self.modality.value if isinstance(self.modality, ModalityType) else str(self.modality),
            "user_query": self.user_query,
            "assistant_response": self.assistant_response.to_dict(),
            "context_summary": self.context_summary.to_dict() if self.context_summary else None,
            "created_at": self.created_at,
        }


def validate_conversation_turn(turn: ConversationTurn) -> None:
    """Validate conversation turn schema and invariants."""
    if not isinstance(turn, ConversationTurn):
        raise TypeError(f"Expected ConversationTurn instance, got {type(turn).__name__}")

    if not isinstance(turn.turn_index, int) or isinstance(turn.turn_index, bool):
        raise TypeError(f"turn_index must be an integer, got {type(turn.turn_index).__name__}")
    if turn.turn_index < 0:
        raise ValueError(f"turn_index cannot be negative, got {turn.turn_index}")

    if not isinstance(turn.session_id, str):
        raise TypeError(f"session_id must be a string, got {type(turn.session_id).__name__}")
    if not turn.session_id.strip():
        raise ValueError("session_id cannot be empty or whitespace-only.")

    if not isinstance(turn.modality, ModalityType):
        raise TypeError(f"modality must be a ModalityType, got {type(turn.modality).__name__}")

    if not isinstance(turn.user_query, str):
        raise TypeError(f"user_query must be a string, got {type(turn.user_query).__name__}")

    if not isinstance(turn.assistant_response, MultimodalResponse):
        raise TypeError(
            f"assistant_response must be a MultimodalResponse instance, got {type(turn.assistant_response).__name__}"
        )
    validate_multimodal_response(turn.assistant_response)

    if turn.context_summary is not None and not isinstance(turn.context_summary, MultimodalContextSummary):
        raise TypeError(
            f"context_summary must be a MultimodalContextSummary or None, got {type(turn.context_summary).__name__}"
        )


# -----------------------------------------------------------------------------
# Multimodal Conversation Session
# -----------------------------------------------------------------------------

class MultimodalConversationSession:
    """Manages an ordered, bounded sequence of multimodal conversation turns for a single session."""

    def __init__(
        self,
        session_id: str = "default_multimodal_session",
        max_history_turns: int = 5,
    ) -> None:
        """Initialize the conversation session.

        Args:
            session_id: Unique identifier for this conversation session.
            max_history_turns: Maximum number of conversation turns to retain (FIFO bounded).

        Raises:
            TypeError: If arguments are of incorrect types.
            ValueError: If session_id is empty or max_history_turns <= 0.
        """
        if not isinstance(session_id, str):
            raise TypeError(f"session_id must be a string, got {type(session_id).__name__}")
        clean_session_id = session_id.strip()
        if not clean_session_id:
            raise ValueError("session_id cannot be empty or whitespace-only.")

        if not isinstance(max_history_turns, int) or isinstance(max_history_turns, bool):
            raise TypeError(f"max_history_turns must be an integer, got {type(max_history_turns).__name__}")
        if max_history_turns <= 0:
            raise ValueError(f"max_history_turns must be greater than zero, got {max_history_turns}.")

        self.session_id = clean_session_id
        self.max_history_turns = max_history_turns
        self.turns: List[ConversationTurn] = []
        self._total_turns_count = 0

    def add_turn(
        self,
        query: str,
        response: MultimodalResponse,
        modality: Optional[ModalityType] = None,
        context: Optional[Union[MultimodalContext, MultimodalContextSummary, Dict[str, Any]]] = None,
    ) -> ConversationTurn:
        """Add a completed interaction turn to the conversation history.

        Args:
            query: User's query text (or empty string for image-only interactions).
            response: Completed assistant response contract.
            modality: Interaction modality (defaults to response.modality if None).
            context: Optional context reference to capture structured summary.

        Returns:
            ConversationTurn: The created and validated turn.
        """
        if not isinstance(query, str):
            raise TypeError(f"query must be a string, got {type(query).__name__}")

        if not isinstance(response, MultimodalResponse):
            raise TypeError(f"response must be a MultimodalResponse, got {type(response).__name__}")

        resolved_modality = modality if modality is not None else response.modality
        if not isinstance(resolved_modality, ModalityType):
            raise TypeError(f"modality must be a ModalityType, got {type(resolved_modality).__name__}")

        summary = MultimodalContextSummary.from_context(context)

        turn = ConversationTurn(
            turn_index=self._total_turns_count,
            session_id=self.session_id,
            modality=resolved_modality,
            user_query=query,
            assistant_response=response,
            context_summary=summary,
        )

        self.turns.append(turn)
        self._total_turns_count += 1
        self._enforce_bounds()
        return turn

    def _enforce_bounds(self) -> None:
        """Enforce maximum history length by pruning oldest turns FIFO."""
        if len(self.turns) > self.max_history_turns:
            excess = len(self.turns) - self.max_history_turns
            self.turns = self.turns[excess:]

    def get_turns(self) -> List[ConversationTurn]:
        """Return a shallow copy of the active turns in chronological order."""
        return list(self.turns)

    def get_turn_count(self) -> int:
        """Return the number of currently retained turns in memory."""
        return len(self.turns)

    @property
    def total_turns_count(self) -> int:
        """Return the lifetime count of all turns added to this session."""
        return self._total_turns_count

    def get_last_turn(self) -> Optional[ConversationTurn]:
        """Return the most recent turn, or None if history is empty."""
        return self.turns[-1] if self.turns else None

    def get_history_summary(self) -> str:
        """Generate a human-readable textual summary of dialogue turns."""
        if not self.turns:
            return "No previous conversation history."

        lines: List[str] = [f"--- Conversation Session: {self.session_id} ---"]
        for turn in self.turns:
            lines.append(
                f"Turn {turn.turn_index + 1} [{turn.modality.value.upper()}]: "
                f"User: '{turn.user_query}' -> Assistant: {turn.assistant_response.answer[:80]}"
                f"{'...' if len(turn.assistant_response.answer) > 80 else ''}"
            )
        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all stored turns in the session while preserving session metadata."""
        self.turns.clear()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize session state and turns to dictionary without raw bytes."""
        return {
            "session_id": self.session_id,
            "max_history_turns": self.max_history_turns,
            "retained_turn_count": len(self.turns),
            "total_turn_count": self._total_turns_count,
            "turns": [turn.to_dict() for turn in self.turns],
        }


# -----------------------------------------------------------------------------
# Session Manager (Multi-Session Isolation)
# -----------------------------------------------------------------------------

class MultimodalSessionManager:
    """Manages multiple isolated MultimodalConversationSession instances."""

    def __init__(self, default_max_history_turns: int = 5) -> None:
        """Initialize the session manager.

        Args:
            default_max_history_turns: Default history capacity for newly spawned sessions.
        """
        if not isinstance(default_max_history_turns, int) or isinstance(default_max_history_turns, bool):
            raise TypeError(
                f"default_max_history_turns must be an int, got {type(default_max_history_turns).__name__}"
            )
        if default_max_history_turns <= 0:
            raise ValueError("default_max_history_turns must be greater than zero.")

        self.default_max_history_turns = default_max_history_turns
        self._sessions: Dict[str, MultimodalConversationSession] = {}

    def get_or_create_session(
        self,
        session_id: str,
        max_history_turns: Optional[int] = None,
    ) -> MultimodalConversationSession:
        """Retrieve an existing session or initialize a new isolated session."""
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("session_id must be a non-empty string.")

        clean_id = session_id.strip()
        if clean_id not in self._sessions:
            limit = max_history_turns if max_history_turns is not None else self.default_max_history_turns
            self._sessions[clean_id] = MultimodalConversationSession(
                session_id=clean_id,
                max_history_turns=limit,
            )
        return self._sessions[clean_id]

    def get_session(self, session_id: str) -> Optional[MultimodalConversationSession]:
        """Return the session if it exists, otherwise None."""
        if not isinstance(session_id, str):
            return None
        return self._sessions.get(session_id.strip())

    def has_session(self, session_id: str) -> bool:
        """Check if a session exists."""
        if not isinstance(session_id, str):
            return False
        return session_id.strip() in self._sessions

    def delete_session(self, session_id: str) -> bool:
        """Delete a session by ID. Returns True if deleted, False if not found."""
        clean_id = session_id.strip() if isinstance(session_id, str) else ""
        if clean_id in self._sessions:
            del self._sessions[clean_id]
            return True
        return False

    def clear_all(self) -> None:
        """Clear all active sessions."""
        self._sessions.clear()

    def active_session_count(self) -> int:
        """Return the number of active sessions currently tracked."""
        return len(self._sessions)

    def list_session_ids(self) -> List[str]:
        """Return a list of all active session IDs."""
        return list(self._sessions.keys())
