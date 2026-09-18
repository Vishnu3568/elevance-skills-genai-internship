"""Scientific answer generation layer for the Scientific Domain Expert Chatbot.

Provides grounded prompt engineering, open-source LLM abstraction, source
attribution, and structured answer synthesis over retrieved arXiv papers.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

try:
    # pyrefly: ignore [missing-import]
    from src.scientific_kb.retriever import (  # type: ignore
        ScientificRetrievalResult,
        format_retrieval_context,
    )
except ImportError:
    # pyrefly: ignore [missing-import]
    from scientific_kb.retriever import (  # type: ignore
        ScientificRetrievalResult,
        format_retrieval_context,
    )

SCIENTIFIC_QA_PROMPT_TEMPLATE = """You are an expert scientific AI research assistant specializing in Artificial Intelligence, Machine Learning, and Natural Language Processing.

Given the following scientific paper evidence, provide a thorough, accurate, and evidence-grounded explanation to answer the user's question.

RULES:
1. Base your answer strictly on the provided scientific papers evidence.
2. Clearly cite paper titles and arXiv IDs from the context when discussing their findings or architectures.
3. Explain technical concepts rigorously yet intuitively.
4. If the provided papers do not contain sufficient evidence to answer the question, explicitly state: "The provided scientific papers do not contain sufficient evidence to answer this question." Do not fabricate or invent unsupported scientific claims.

---
SCIENTIFIC EVIDENCE:
{context}
---

QUESTION:
{query}

GROUNDED SCIENTIFIC EXPLANATION:"""

INSUFFICIENT_EVIDENCE_PHRASE = "The provided scientific papers do not contain sufficient evidence to answer this question."

_UNSET = object()


class OpenSourceScientificLLM:
    """Open-source Hugging Face Seq2Seq LLM adapter for scientific explanation generation.

    Encapsulates the local open-source model (default: google/flan-t5-base) using Hugging Face
    transformers pipeline with deterministic parameters, token control, and lazy loading.
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        temperature: float = 0.1,
        max_length: int = 512,
    ):
        """Initialize open-source scientific LLM adapter.

        Args:
            model_name (str): Hugging Face model identifier (default: google/flan-t5-base).
            temperature (float): Generation temperature for deterministic output.
            max_length (int): Maximum sequence generation length.
        """
        self.model_name = model_name
        self.temperature = temperature
        self.max_length = max_length
        self._pipeline: Optional[Any] = None

    def _get_pipeline(self) -> Any:
        """Lazily initialize and return the Hugging Face text2text pipeline."""
        if self._pipeline is None:
            os.environ["TRANSFORMERS_NO_TF"] = "1"
            os.environ["USE_TF"] = "0"
            from transformers import pipeline

            self._pipeline = pipeline(
                "text2text-generation",
                model=self.model_name,
                max_length=self.max_length,
                do_sample=(self.temperature > 0.0),
            )
        return self._pipeline

    def invoke(self, prompt: str) -> str:
        """Invoke open-source model pipeline on input prompt.

        Args:
            prompt (str): Prompt text to generate explanation from.

        Returns:
            str: Generated explanation text.
        """
        pipe = self._get_pipeline()
        outputs = pipe(prompt)
        if isinstance(outputs, list) and outputs and isinstance(outputs[0], dict) and "generated_text" in outputs[0]:
            return str(outputs[0]["generated_text"]).strip()
        return str(outputs).strip()

    def __call__(self, prompt: str) -> str:
        """Callable protocol support."""
        return self.invoke(prompt)

    def predict(self, prompt: str) -> str:
        """Predict protocol support."""
        return self.invoke(prompt)

    @property
    def is_open_source(self) -> bool:
        """Identity flag confirming this is an open-source model."""
        return True


def get_open_source_scientific_llm(
    model_name: str = "google/flan-t5-base",
    temperature: float = 0.1,
    max_length: int = 512,
) -> OpenSourceScientificLLM:
    """Factory creating a lazily loaded OpenSourceScientificLLM instance.

    Args:
        model_name (str): Open-source model name (default: google/flan-t5-base).
        temperature (float): Sampling temperature.
        max_length (int): Maximum generation length.

    Returns:
        OpenSourceScientificLLM: Configured open-source LLM instance.
    """
    return OpenSourceScientificLLM(
        model_name=model_name,
        temperature=temperature,
        max_length=max_length,
    )


@dataclass
class ScientificAnswer:
    """Structured result containing generated scientific explanation and provenance metadata."""

    query: str
    answer: str
    sources: List[ScientificRetrievalResult] = field(default_factory=list)
    grounded: bool = True
    raw_response: str = ""


def build_scientific_prompt(query: str, context: str) -> str:
    """Construct the standardized scientific QA prompt.

    Args:
        query (str): User question.
        context (str): Formatted scientific paper evidence.

    Returns:
        str: Fully rendered prompt string.
    """
    return SCIENTIFIC_QA_PROMPT_TEMPLATE.format(
        context=context.strip(),
        query=query.strip(),
    )


class ScientificGenerator:
    """Generator orchestrating prompt formatting and open-source LLM synthesis."""

    def __init__(
        self,
        llm: Any = _UNSET,
        model_name: str = "google/flan-t5-base",
        temperature: float = 0.1,
    ):
        """Initialize the ScientificGenerator.

        Args:
            llm (Optional[Any]): Configurable LLM instance (callable or LangChain model).
                If unset, defaults to local open-source LLM (google/flan-t5-base).
            model_name (str): Open-source model identifier.
            temperature (float): Generation temperature for deterministic output.
        """
        self.model_name = model_name
        self.temperature = temperature

        if llm is _UNSET:
            self.llm = get_open_source_scientific_llm(
                model_name=model_name,
                temperature=temperature,
            )
        else:
            self.llm = llm

    @property
    def is_open_source(self) -> bool:
        """Check if active LLM backend is an open-source model."""
        if isinstance(self.llm, OpenSourceScientificLLM):
            return True
        if self.llm is not None and getattr(self.llm, "is_open_source", False):
            return True
        llm_type = type(self.llm).__name__.lower()
        return "gemini" not in llm_type and "google" not in llm_type

    def _invoke_llm(self, prompt: str) -> str:
        """Invoke the configured LLM handling various invocation protocols."""
        if self.llm is None:
            raise RuntimeError(
                "No LLM backend configured. Please supply a valid LLM instance to ScientificGenerator."
            )

        # 1. LangChain invoke / OpenSourceScientificLLM invoke
        if hasattr(self.llm, "invoke"):
            res = self.llm.invoke(prompt)
            content = getattr(res, "content", None)
            if content is not None:
                return str(content).strip()
            if isinstance(res, list) and res and isinstance(res[0], dict) and "generated_text" in res[0]:
                return str(res[0]["generated_text"]).strip()
            return str(res).strip()

        # 2. Callable pipeline or plain function
        if callable(self.llm):
            res = self.llm(prompt)
            if isinstance(res, list) and res and isinstance(res[0], dict) and "generated_text" in res[0]:
                return str(res[0]["generated_text"]).strip()
            return str(res).strip()

        # 3. Predict protocol
        if hasattr(self.llm, "predict"):
            res = self.llm.predict(prompt)
            return str(res).strip()

        raise TypeError(f"Unsupported LLM object type: {type(self.llm).__name__}")

    def generate_answer(
        self,
        query: str,
        retrieval_results: List[ScientificRetrievalResult],
    ) -> ScientificAnswer:
        """Generate an evidence-grounded scientific answer from retrieved paper context.

        Args:
            query (str): User question.
            retrieval_results (List[ScientificRetrievalResult]): List of retrieved paper results.

        Returns:
            ScientificAnswer: Structured response object with answer and sources.

        Raises:
            TypeError: If query is not a string.
            ValueError: If query is empty or whitespace-only.
        """
        if not isinstance(query, str):
            raise TypeError(f"Expected query to be a string, got {type(query).__name__}")

        stripped_query = query.strip()
        if not stripped_query:
            raise ValueError("Query cannot be empty or whitespace-only.")

        # Handle empty retrieval results deterministically without calling LLM
        if not retrieval_results:
            return ScientificAnswer(
                query=stripped_query,
                answer=INSUFFICIENT_EVIDENCE_PHRASE,
                sources=[],
                grounded=False,
                raw_response="No evidence provided.",
            )

        context_str = format_retrieval_context(retrieval_results)
        prompt = build_scientific_prompt(stripped_query, context_str)

        raw_output = self._invoke_llm(prompt)
        clean_answer = raw_output.strip()

        # Check if model acknowledged lack of sufficient evidence
        norm_ans = clean_answer.lower()
        is_ungrounded = (
            "not contain sufficient evidence" in norm_ans
            or "insufficient evidence" in norm_ans
            or norm_ans.startswith("i don't know")
            or norm_ans.startswith("i do not know")
        )

        return ScientificAnswer(
            query=stripped_query,
            answer=clean_answer,
            sources=list(retrieval_results),
            grounded=not is_ungrounded,
            raw_response=raw_output,
        )
