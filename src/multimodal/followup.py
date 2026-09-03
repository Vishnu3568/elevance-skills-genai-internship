"""Multimodal Follow-Up Questions and Context Resolution (Phase 5 — Day 27 Step 3).

Provides deterministic resolution of conversational follow-up questions, anaphoric
pronouns ("it", "this", "that"), visual references ("the previous image", "the diagram"),
and entity/component references against recent RetainedConversationContext.
"""

from dataclasses import asdict, dataclass, field
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from .context_retention import (
    MultimodalContextRetriever,
    RetainedConversationContext,
    RetainedTurnView,
)
from .conversation import MultimodalConversationSession, MultimodalSessionManager
from .models import ModalityType


# -----------------------------------------------------------------------------
# Regex Reference Patterns
# -----------------------------------------------------------------------------

# Visual references
VISUAL_REF_PATTERN = re.compile(
    r"\b(this\s+image|the\s+previous\s+image|the\s+last\s+image|the\s+image|this\s+diagram|"
    r"the\s+diagram|this\s+chart|the\s+chart|the\s+flowchart|this\s+flowchart|"
    r"the\s+figure|this\s+figure|the\s+schematic|this\s+schematic)\b",
    re.IGNORECASE,
)

# Prior answer / statement references
ANSWER_REF_PATTERN = re.compile(
    r"\b(the\s+previous\s+answer|the\s+last\s+answer|your\s+previous\s+answer|"
    r"what\s+you\s+(?:said|mentioned|explained|described)|"
    r"you\s+(?:said|mentioned|explained|described))\b",
    re.IGNORECASE,
)

# Component / entity references
COMPONENT_REF_PATTERN = re.compile(
    r"\b(the\s+(?:[a-zA-Z0-9_\-]+\s+)?component(?:\s+(?:mentioned|described)(?:\s+earlier)?)?|"
    r"the\s+component\s+you\s+mentioned|the\s+(?:[a-zA-Z0-9_\-]+\s+)?element|"
    r"the\s+(?:[a-zA-Z0-9_\-]+\s+)?block)\b",
    re.IGNORECASE,
)

# Anaphoric pronouns (standalone "it", "its", "this", "that")
PRONOUN_IT_PATTERN = re.compile(r"\b([Ii]t)\b")
PRONOUN_ITS_PATTERN = re.compile(r"\b([Ii]ts)\b")
PRONOUN_THIS_PATTERN = re.compile(r"\b([Tt]his)\b(?!\s+(?:image|diagram|chart|flowchart|figure|schematic|component|element|block))")
PRONOUN_THAT_PATTERN = re.compile(r"\b([Tt]hat)\b")

# Short follow-up phrases
SHORT_FOLLOWUP_STARTS = (
    "why?",
    "why",
    "how?",
    "how",
    "how so?",
    "how come?",
    "how does it work?",
    "what does it do?",
    "what about it?",
    "can you elaborate?",
    "tell me more",
    "explain more",
    "why is that?",
)


# -----------------------------------------------------------------------------
# Follow-Up Resolution Contract
# -----------------------------------------------------------------------------

@dataclass
class FollowUpResolution:
    """Structured result of resolving a conversational follow-up query."""

    raw_query: str
    resolved_query: str
    is_followup: bool
    detected_references: List[str] = field(default_factory=list)
    referenced_modality: Optional[ModalityType] = None
    target_topic: Optional[str] = None
    retained_visual_summary: Optional[Dict[str, Any]] = None
    resolution_strategy: str = "pass_through"
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize follow-up resolution without raw byte payloads."""
        return {
            "raw_query": self.raw_query,
            "resolved_query": self.resolved_query,
            "is_followup": self.is_followup,
            "detected_references": list(self.detected_references),
            "referenced_modality": self.referenced_modality.value if self.referenced_modality else None,
            "target_topic": self.target_topic,
            "retained_visual_summary": dict(self.retained_visual_summary) if self.retained_visual_summary else None,
            "resolution_strategy": self.resolution_strategy,
            "confidence": self.confidence,
        }


# -----------------------------------------------------------------------------
# Topic Extraction Utilities
# -----------------------------------------------------------------------------

def extract_focal_topic_from_context(
    context: RetainedConversationContext,
) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[ModalityType]]:
    """Extract candidate focal topic, visual summary, and modality from history."""
    if not context.has_history or not context.retained_turns:
        return None, None, None

    last_turn = context.get_last_turn()
    if last_turn is None:
        return None, None, None

    visual_summary = context.get_last_visual_summary()
    modality = last_turn.modality

    # 1. Check if scene description provides a clear descriptive topic
    if last_turn.scene_description:
        desc = last_turn.scene_description.strip()
        # Clean leading words like "Technical flowchart depicting..." or "Diagram of a..."
        clean_desc = re.sub(
            r"^(?:a\s+|an\s+|the\s+)?(?:technical\s+)?(?:flowchart|diagram|chart|image|picture|schematic)?\s*(?:depicting|showing|illustrating|of)?\s*",
            "",
            desc,
            flags=re.IGNORECASE,
        ).strip()
        clean_desc = re.sub(r"^(?:a\s+|an\s+|the\s+)", "", clean_desc, flags=re.IGNORECASE).strip()
        if clean_desc:
            return clean_desc, visual_summary, modality

    # 2. Check assistant response for explicit presentation statements
    # e.g. "The image shows a neural-network architecture."
    answer_text = last_turn.assistant_answer
    match = re.search(
        r"(?:shows|depicts|illustrates|presents|explains|discusses)\s+(?:a\s+|an\s+|the\s+)?([a-zA-Z0-9_\-\s]+?)(?:\.|\,|$|\n)",
        answer_text,
        re.IGNORECASE,
    )
    if match:
        extracted = match.group(1).strip()
        extracted = re.sub(r"^(?:a\s+|an\s+|the\s+)", "", extracted, flags=re.IGNORECASE).strip()
        if extracted and len(extracted.split()) <= 6:
            return extracted, visual_summary, modality

    # 3. Check user query in last turn
    user_q = last_turn.user_query.strip()
    if user_q:
        clean_q = re.sub(
            r"^(?:what is|explain|describe|tell me about|how does|summarize|can you explain)\s+(?:a\s+|an\s+|the\s+)?",
            "",
            user_q,
            flags=re.IGNORECASE,
        ).strip()
        clean_q = re.sub(r"^(?:a\s+|an\s+|the\s+)", "", clean_q, flags=re.IGNORECASE).strip()
        # Avoid generic phrases like "shown in this image"
        if clean_q and not re.search(r"\b(shown in this image|this image|the image|this diagram)\b", clean_q, re.IGNORECASE):
            return clean_q.rstrip("?").strip(), visual_summary, modality

    # 4. Fallback to scene description or first sentence of assistant answer
    if last_turn.scene_description:
        fallback_desc = re.sub(r"^(?:a\s+|an\s+|the\s+)", "", last_turn.scene_description.strip(), flags=re.IGNORECASE).strip()
        return fallback_desc, visual_summary, modality

    first_sentence = answer_text.split(".")[0].strip()
    first_sentence = re.sub(r"^(?:a\s+|an\s+|the\s+)", "", first_sentence, flags=re.IGNORECASE).strip()
    if first_sentence and len(first_sentence.split()) <= 8:
        return first_sentence, visual_summary, modality

    return "the discussed subject", visual_summary, modality


# -----------------------------------------------------------------------------
# Follow-Up Resolver Engine
# -----------------------------------------------------------------------------

class MultimodalFollowUpResolver:
    """Detects and resolves conversational follow-up queries using retained context."""

    def __init__(self, context_retriever: Optional[MultimodalContextRetriever] = None) -> None:
        """Initialize resolver with optional custom context retriever."""
        self.context_retriever = context_retriever or MultimodalContextRetriever()

    def resolve_followup(
        self,
        query: str,
        context: Optional[RetainedConversationContext] = None,
        session: Optional[MultimodalConversationSession] = None,
    ) -> FollowUpResolution:
        """Resolve a user query against recent conversation context.

        Args:
            query: The current raw user query string.
            context: Optional RetainedConversationContext instance.
            session: Optional MultimodalConversationSession (used if context is None).

        Returns:
            FollowUpResolution: Structured interpretation and resolved query text.
        """
        if not isinstance(query, str):
            raise TypeError(f"query must be a string, got {type(query).__name__}")

        stripped_query = query.strip()
        if not stripped_query:
            return FollowUpResolution(
                raw_query="",
                resolved_query="",
                is_followup=False,
                resolution_strategy="empty_query",
            )

        # Obtain retained context if session is provided and context is missing
        eff_context = context
        if eff_context is None and session is not None:
            eff_context = self.context_retriever.retrieve_context(session)

        # If no conversation context exists, return raw query as-is
        if eff_context is None or not eff_context.has_history or eff_context.turn_count == 0:
            return FollowUpResolution(
                raw_query=stripped_query,
                resolved_query=stripped_query,
                is_followup=False,
                resolution_strategy="no_history_fallback",
            )

        # Detect candidate references
        detected_refs: List[str] = []
        is_followup = False

        # Check visual references
        for m in VISUAL_REF_PATTERN.finditer(stripped_query):
            ref = m.group(1)
            detected_refs.append(ref)
            is_followup = True

        # Check answer references
        for m in ANSWER_REF_PATTERN.finditer(stripped_query):
            ref = m.group(1)
            detected_refs.append(ref)
            is_followup = True

        # Check component references
        for m in COMPONENT_REF_PATTERN.finditer(stripped_query):
            ref = m.group(1)
            detected_refs.append(ref)
            is_followup = True

        # Check pronouns
        for m in PRONOUN_IT_PATTERN.finditer(stripped_query):
            detected_refs.append(m.group(1))
            is_followup = True

        for m in PRONOUN_ITS_PATTERN.finditer(stripped_query):
            detected_refs.append(m.group(1))
            is_followup = True

        for m in PRONOUN_THIS_PATTERN.finditer(stripped_query):
            detected_refs.append(m.group(1))
            is_followup = True

        for m in PRONOUN_THAT_PATTERN.finditer(stripped_query):
            detected_refs.append(m.group(1))
            is_followup = True

        # Check short follow-up starts
        lower_q = stripped_query.lower()
        if not is_followup:
            for start in SHORT_FOLLOWUP_STARTS:
                if lower_q == start or lower_q.startswith(start + " "):
                    detected_refs.append(start)
                    is_followup = True
                    break

        # If no conversational dependency detected, return raw query
        if not is_followup:
            return FollowUpResolution(
                raw_query=stripped_query,
                resolved_query=stripped_query,
                is_followup=False,
                resolution_strategy="independent_query",
            )

        # Extract target topic and visual metadata from context
        topic, vis_summary, mod = extract_focal_topic_from_context(eff_context)
        resolved_topic = topic or "the previous subject"

        resolved_query = stripped_query
        strategy = "anaphora_resolution"

        # 1. Resolve visual references (e.g. "this image", "the previous image", "the diagram")
        if VISUAL_REF_PATTERN.search(resolved_query):
            strategy = "visual_reference_resolution"

            def _replace_vis(m: re.Match) -> str:
                match_str = m.group(1).lower()
                if "diagram" in match_str:
                    return f"the {resolved_topic} diagram"
                elif "chart" in match_str or "flowchart" in match_str:
                    return f"the {resolved_topic} flowchart"
                elif "schematic" in match_str:
                    return f"the {resolved_topic} schematic"
                return f"the previous image ({resolved_topic})"

            resolved_query = VISUAL_REF_PATTERN.sub(_replace_vis, resolved_query)

        # 2. Resolve component references (e.g. "What does the blue component do?")
        if COMPONENT_REF_PATTERN.search(resolved_query):
            strategy = "component_reference_binding"
            # If the query asks about a component and does not yet mention the topic
            if resolved_topic.lower() not in resolved_query.lower():
                # Augment: "What does the blue component do?" -> "What does the blue component do in the <topic>?"
                clean_end = resolved_query.rstrip("?").strip()
                resolved_query = f"{clean_end} in the {resolved_topic}?"

        # 3. Resolve prior answer references (e.g. "the previous answer")
        if ANSWER_REF_PATTERN.search(resolved_query):
            strategy = "answer_reference_binding"

            def _replace_ans(m: re.Match) -> str:
                return f"the previous answer regarding {resolved_topic}"

            resolved_query = ANSWER_REF_PATTERN.sub(_replace_ans, resolved_query)

        # 4. Resolve possessive "its" -> "the <topic>'s"
        if PRONOUN_ITS_PATTERN.search(resolved_query):
            # Check pattern like "What is its computational complexity?" -> "What is the computational complexity of the <topic>?"
            complex_match = re.search(r"^[Ww]hat is its\s+([a-zA-Z0-9_\-\s]+?)\??$", resolved_query)
            if complex_match:
                attr = complex_match.group(1).strip()
                resolved_query = f"What is the {attr} of {resolved_topic}?"
            else:
                resolved_query = PRONOUN_ITS_PATTERN.sub(f"the {resolved_topic}'s", resolved_query)

        # 5. Resolve "it"
        if PRONOUN_IT_PATTERN.search(resolved_query):
            # Case A: "How does it work?" -> "How does the <topic> work?"
            work_match = re.search(r"^[Hh]ow does it\s+([a-zA-Z0-9_\-\s]+?)\??$", resolved_query)
            if work_match:
                predicate = work_match.group(1).strip()
                resolved_query = f"How does the {resolved_topic} {predicate}?"
            # Case B: "What does it do?" -> "What does the <topic> do?"
            elif re.search(r"^[Ww]hat does it\s+([a-zA-Z0-9_\-\s]+?)\??$", resolved_query):
                do_match = re.search(r"^[Ww]hat does it\s+([a-zA-Z0-9_\-\s]+?)\??$", resolved_query)
                predicate = do_match.group(1).strip() if do_match else "do"
                resolved_query = f"What does the {resolved_topic} {predicate}?"
            # Case C: "Explain it in detail" -> "Explain the <topic> in detail"
            elif re.search(r"\b(explain|describe|clarify|elaborate on)\s+it\b", resolved_query, re.IGNORECASE):
                resolved_query = re.sub(
                    r"\b(explain|describe|clarify|elaborate on)\s+it\b",
                    rf"\1 the {resolved_topic}",
                    resolved_query,
                    flags=re.IGNORECASE,
                )
            else:
                resolved_query = PRONOUN_IT_PATTERN.sub(f"the {resolved_topic}", resolved_query)

        # 6. Resolve standalone "this" or "that"
        if PRONOUN_THIS_PATTERN.search(resolved_query):
            resolved_query = PRONOUN_THIS_PATTERN.sub(f"the {resolved_topic}", resolved_query)

        if PRONOUN_THAT_PATTERN.search(resolved_query):
            if lower_q in ("why is that?", "why is that"):
                resolved_query = f"Why is that regarding {resolved_topic}?"
            else:
                resolved_query = PRONOUN_THAT_PATTERN.sub(f"that regarding {resolved_topic}", resolved_query)

        # 7. Short follow-up augmentation if still unmodified
        if resolved_query == stripped_query and is_followup:
            clean_q = stripped_query.rstrip("?").strip()
            resolved_query = f"{clean_q} regarding {resolved_topic}?"

        # Clean double spaces or duplicate determiners
        resolved_query = re.sub(r"\s+", " ", resolved_query).strip()
        resolved_query = re.sub(r"\bthe\s+the\b", "the", resolved_query, flags=re.IGNORECASE)

        # Infer referenced modality
        ref_modality = mod
        if any(w in lower_q for w in ("image", "diagram", "chart", "component", "visual", "color")):
            ref_modality = ModalityType.TEXT_AND_IMAGE

        return FollowUpResolution(
            raw_query=stripped_query,
            resolved_query=resolved_query,
            is_followup=True,
            detected_references=detected_refs,
            referenced_modality=ref_modality,
            target_topic=resolved_topic,
            retained_visual_summary=vis_summary,
            resolution_strategy=strategy,
            confidence=1.0,
        )


# -----------------------------------------------------------------------------
# Convenience Functional Interface
# -----------------------------------------------------------------------------

def resolve_multimodal_followup(
    query: str,
    context: Optional[RetainedConversationContext] = None,
    session: Optional[MultimodalConversationSession] = None,
) -> FollowUpResolution:
    """Resolve a follow-up query against retained multimodal dialogue context.

    Args:
        query: Current user query.
        context: Optional retained conversation context.
        session: Optional conversation session.

    Returns:
        FollowUpResolution: Structured interpretation and standalone query string.
    """
    resolver = MultimodalFollowUpResolver()
    return resolver.resolve_followup(query, context=context, session=session)
