# 🏥 Task 2: MedQuAD Medical Q&A Assistant Guide

This document describes the architecture, setup, indexing pipeline, and operational workflow for **Task 2: Medical Q&A Assistant** (based on the National Institutes of Health MedQuAD dataset).

---

## 🛠️ 1. How to Build the Medical FAISS Index

The Medical Q&A Assistant reads from an isolated, metadata-rich FAISS vector database.

To build the production medical FAISS index from the MedQuAD XML dataset:

```bash
python src/medquad_indexer.py --xml_dir <path_to_medquad_xml_directory>
```

### Options:
- `--xml_dir` (required): Absolute or relative path to the folder containing raw MedQuAD `.xml` files.
- `--index_path` (optional): Output directory for the persisted FAISS vector index (defaults to `faiss_index_medical/`).

---

## 📂 2. Expected Vector Index Path

The application expects the persisted medical FAISS index to reside at:

```text
faiss_index_medical/
├── index.faiss
└── index.pkl
```

This directory is isolated from the customer service index (`faiss_index/`) and is ignored by Git to prevent committing large binary artifacts.

---

## 🚀 3. How to Launch the Medical Q&A UI

To launch the isolated Medical Q&A Streamlit application:

```bash
streamlit run src/medical_main.py
```

---

## 🔑 4. Required Environment Variables

Set your Google Gemini API key in the `.env` file at the root of the repository:

```bash
GOOGLE_API_KEY="your_api_key_here"
```

If `GOOGLE_API_KEY` is not present, the system operates in retrieval-only safety fallback mode without breaking.

---

## 🛡️ 5. Safety States & System Behavior

`MedicalQAService` categorizes every user query into one of 5 explicit safety states:

| Safety Status | Condition | System Action |
| :--- | :--- | :--- |
| **`GROUNDED`** | Evidence meets threshold ($\ge 0.50$) with recognized clinical topic/entities. | Invokes Gemini LLM with strict evidence-only prompt; displays answer, retrieval confidence tier, and NIH citations. |
| **`INSUFFICIENT_EVIDENCE`** | Medical query, but evidence similarity is weak ($<0.50$) or empty. | **Bypasses LLM**. Returns safe clinical disclaimer (*"I could not find sufficient grounded information..."*). |
| **`OUT_OF_DOMAIN`** | Non-medical question (no medical topic, entities, or intent). | **Bypasses LLM**. Returns scope boundary notice (*"This question does not appear to be related..."*). |
| **`RETRIEVAL_ERROR`** | Vector DB search raised an exception. | **Bypasses LLM**. Returns safe system error message without exposing Python stack traces. |
| **`GENERATION_ERROR`** | Evidence was sufficient, but LLM call failed. | Returns safe fallback message while preserving retrieved evidence and citations for diagnostics. |

---

## 🔒 6. Key Principles
- **Baseline Isolation**: Task 1 files (`src/main.py`, `src/chatbot_service.py`, `src/response_policy.py`, `src/langchain_helper.py`, `faiss_index/`) remain 100% untouched.
- **Context as DATA**: Medical evidence context is stuffed into prompts as reference DATA only, preventing prompt injection attacks.
- **Retrieval Confidence**: UI badges display *"Knowledge-base retrieval confidence"*, never claiming medical diagnosis certainty.
