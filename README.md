# Multi-Domain Generative AI & RAG Platform

[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/framework-Streamlit%20%7C%20LangChain-orange.svg)](https://streamlit.io/)
[![Embeddings](https://img.shields.io/badge/embeddings-HuggingFace%20MiniLM%20%7C%20Instructor-green.svg)](https://huggingface.co/)
[![Vector Store](https://img.shields.io/badge/vector%20store-FAISS-red.svg)](https://github.com/facebookresearch/faiss)
[![LLM](https://img.shields.io/badge/LLM-Google%20Gemini%20%7C%20FLAN--T5%20(Open--Source)-4285F4.svg)](https://huggingface.co/google/flan-t5-base)

An enterprise-grade, multi-domain Generative AI and Retrieval-Augmented Generation (RAG) platform.

The system evolves a single-domain customer support baseline into a decoupled, unified intelligence ecosystem spanning customer support with sentiment-conditioned response policies, live dynamic knowledge base management, NIH MedQuAD clinical Q&A with deterministic safety boundaries, arXiv AI/ML scientific domain expertise with open-source LLM explanation generation, multimodal document/image/audio intelligence, and cross-cutting multilingual capabilities.

---

## 📋 Table of Contents

1. [System Evolution & Engineering Progression](#system-evolution--engineering-progression)
2. [Architecture Overview](#architecture-overview)
3. [Core System Modules](#core-system-modules)
   - [Module 1: Sentiment Analysis](#module-1--sentiment-analysis)
   - [Module 2: Medical Clinical Q&A (NIH MedQuAD)](#module-2--medical-clinical-qa-nih-medquad)
   - [Module 3: Dynamic Knowledge Base & Ingestion Pipeline](#module-3--dynamic-knowledge-base--ingestion-pipeline)
   - [Module 4: Scientific Domain Expert (arXiv AI/ML)](#module-4--scientific-domain-expert-arxiv-aiml)
   - [Module 5: Multimodal Intelligence (Vision, Audio, Docs)](#module-5--multimodal-intelligence-vision-audio-docs)
   - [Module 6: Cross-Cutting Multilingual Engine](#module-6--cross-cutting-multilingual-engine)
4. [Cross-Task Integration & Unified Routing](#cross-task-integration--unified-routing)
5. [User Interfaces & Port Allocation](#user-interfaces--port-allocation)
6. [Datasets & Protected Production Assets](#datasets--protected-production-assets)
7. [Installation & Setup](#installation--setup)
8. [Running the Applications](#running-the-applications)
9. [Testing & Quality Assurance](#testing--quality-assurance)
10. [Static Analysis & Code Quality](#static-analysis--code-quality)
11. [Architectural Context & Design Notes](#architectural-context--design-notes)
12. [Repository Directory Structure](#repository-directory-structure)

---

## 🏛️ System Evolution & Engineering Progression

This repository represents a complete architectural evolution from a monolithic prototype to a modular multi-domain platform:

1. **Initial Baseline**: Began with a simple LangChain + Google PaLM prototype for customer service FAQ retrieval, as preserved in [`README.training.md`](file:///E:/Project%20Folder/NLP-chatbot-files/ElevanceSkills-GenAI-Internship/README.training.md).
2. **Domain Decoupling & Modernization**: Upgraded the core LLM foundation to Google Gemini, implemented HuggingFace sentence embeddings (`all-MiniLM-L6-v2`), and structured services into modular components under `src/`.
3. **Specialized Intelligence Capabilities**: Sequentially built and validated sentiment-aware response conditioning, clinical RAG with NIH MedQuAD, live dynamic knowledge base sync, scientific paper analysis over 100 arXiv AI/ML publications with open-source LLM generation, multi-format multimodal processing, and cross-lingual routing.
4. **Unified Orchestration**: Constructed a non-invasive cross-task routing layer (`src/cross_task/`) providing an intelligent unified interface while preserving all standalone applications and protected vector assets.

---

## 🏗️ Architecture Overview

The system is structured as an extensible micro-architecture with clear domain boundaries:

```mermaid
graph TD
    User([User Query / File Upload]) --> UI{Streamlit Frontends}
    
    UI -->|Port 8500| Unified[Unified Assistant / Orchestrator]
    UI -->|Port 8501| StandaloneCS[Customer Support UI]
    UI -->|Port 8502| StandaloneMed[Medical Q&A UI]
    UI -->|Port 8503| StandaloneMM[Multimodal UI]
    UI -->|Port 8504| StandaloneML[Multilingual UI]
    UI -->|Port 8505| StandaloneSci[Scientific Expert UI]

    Unified --> Router[Cross-Task Router]
    Unified --> SessionMgr[Session State Manager]
    Unified --> LangDetect[Cross-Cutting Language Detector]

    Router -->|EdTech FAQ + Sentiment| CS[Customer Support Service]
    Router -->|NIH MedQuAD Clinical| Med[Medical Q&A Service]
    Router -->|arXiv AI/ML Research| Sci[Scientific KB Service]
    Router -->|Vision / Audio / Docs| MM[Multimodal Service]

    CS --> SentimentMod[RoBERTa Sentiment Analyzer]
    CS --> DynamicKB[Dynamic KB Updater & Scheduler]
    CS --> FAISS_CS[(faiss_index / CS)]

    Med --> MedAnalyzer[Clinical Entity & Intent Analyzer]
    Med --> MedSafety[5 Safety State Verifier]
    Med --> FAISS_Med[(faiss_index_medical)]

    Sci --> SciPipeline[Search, Synthesizer & Concept Explainer]
    Sci --> FAISS_Sci[(faiss_index_scientific)]
    SciPipeline --> FlanT5[Open-Source FLAN-T5-Base LLM]

    MM --> GeminiMM[Gemini 1.5 Flash Multi-Modal Vision/Audio]

    LangDetect -.->|Cross-Cutting| CS
    LangDetect -.->|Cross-Cutting| Med
    LangDetect -.->|Cross-Cutting| Sci
```

---

## 🔬 Core System Modules

### Module 1 — Sentiment Analysis

- **Implementation**: [`src/sentiment_analyzer.py`](file:///E:/Project%20Folder/NLP-chatbot-files/ElevanceSkills-GenAI-Internship/src/sentiment_analyzer.py) and [`src/response_policy.py`](file:///E:/Project%20Folder/NLP-chatbot-files/ElevanceSkills-GenAI-Internship/src/response_policy.py).
- **Core Technology**: CardiffNLP RoBERTa model (`cardiffnlp/twitter-roberta-base-sentiment-latest`) with robust rule/lexicon-based fallback when transformer models are offline.
- **Architectural Role**: Operates as a behavioral response conditioning filter inside Customer Support rather than a distinct conversational domain.
- **Behavior**:
  - `NEGATIVE`: Prefixes an empathetic apology and customer-first validation before delivering the FAQ answer.
  - `POSITIVE`: Appends an encouraging appreciation closing note.
  - `NEUTRAL`: Delivers clean, unadorned FAQ responses.

### Module 2 — Medical Clinical Q&A (NIH MedQuAD)

- **Implementation**: [`src/medical_qa_service.py`](file:///E:/Project%20Folder/NLP-chatbot-files/ElevanceSkills-GenAI-Internship/src/medical_qa_service.py), [`src/medquad_retriever.py`](file:///E:/Project%20Folder/NLP-chatbot-files/ElevanceSkills-GenAI-Internship/src/medquad_retriever.py), [`src/medquad_query_analyzer.py`](file:///E:/Project%20Folder/NLP-chatbot-files/ElevanceSkills-GenAI-Internship/src/medquad_query_analyzer.py).
- **Core Technology**: 12 NIH collections parsed from XML, indexed using HuggingFace sentence transformers into isolated vector store `faiss_index_medical/`.
- **Safety States**: Every medical query is classified into one of five rigorous states:
  1. `GROUNDED`: Evidence similarity $\ge 0.50$ with recognized clinical entities. Generates grounded answer with confidence tier and NIH citations.
  2. `INSUFFICIENT_EVIDENCE`: Medical intent present but retrieval similarity $< 0.50$. Bypasses LLM; issues clinical safety disclaimer.
  3. `OUT_OF_DOMAIN`: Non-medical intent detected. Bypasses LLM; issues domain boundary guidance.
  4. `RETRIEVAL_ERROR`: Vector index lookup failure. Bypasses LLM; safely logs without leaking stack traces.
  5. `GENERATION_ERROR`: LLM failure despite valid evidence. Provides safe fallback while retaining retrieved reference data.
- **Documentation**: Detailed guide in [`docs/MEDICAL_QA_GUIDE.md`](file:///E:/Project%20Folder/NLP-chatbot-files/ElevanceSkills-GenAI-Internship/docs/MEDICAL_QA_GUIDE.md).

### Module 3 — Dynamic Knowledge Base & Ingestion Pipeline

- **Implementation**: [`src/knowledge_base/`](file:///E:/Project%20Folder/NLP-chatbot-files/ElevanceSkills-GenAI-Internship/src/knowledge_base/) (`kb_updater.py`, `scheduler.py`, `validator.py`, `atomic_writer.py`).
- **Capabilities**:
  - Thread-safe, atomic updates to [`dataset/knowledge_base.csv`](file:///E:/Project%20Folder/NLP-chatbot-files/ElevanceSkills-GenAI-Internship/dataset/knowledge_base.csv) with schema validation and deduplication.
  - Background synchronization via `KBUpdateScheduler` for zero-downtime knowledge base reloads.
  - Dedicated admin sync console embedded in [`src/main.py`](file:///E:/Project%20Folder/NLP-chatbot-files/ElevanceSkills-GenAI-Internship/src/main.py) for manual or automated FAQ refreshes.

### Module 4 — Scientific Domain Expert (arXiv AI/ML)

- **Implementation**: [`src/scientific_kb/`](file:///E:/Project%20Folder/NLP-chatbot-files/ElevanceSkills-GenAI-Internship/src/scientific_kb/) (`service.py`, `retriever.py`, `synthesizer.py`, `query_classifier.py`, `concept_explainer.py`, `generation.py`, `understanding.py`, `grounding.py`, `conversation.py`, `visualization.py`).
- **Core Technology**: 100 AI/ML research papers from arXiv indexed into `faiss_index_scientific/`.
- **Open-Source LLM Explanation Generation**: Uses the local open-source `google/flan-t5-base` model through [`OpenSourceScientificLLM`](file:///E:/Project%20Folder/NLP-chatbot-files/ElevanceSkills-GenAI-Internship/src/scientific_kb/generation.py) (`transformers.pipeline` for `text2text-generation` running on CPU via PyTorch). Preserves lazy loading: model weights (~990 MB) are automatically retrieved from Hugging Face Hub and cached locally upon first scientific inquiry and are **never committed** to Git.
- **Key Capabilities**:
  - Paper semantic search and metadata extraction (titles, authors, categories, publication dates, DOIs, URLs).
  - Information extraction and structured paper understanding (methodology, key contributions, empirical results, limitations).
  - Structured paper summarization with multi-section markdown reporting.
  - Multi-paper comparative analysis and thematic matrix synthesis.
  - Two-tiered concept explanations (rigorous technical & intuitive mental models) grounded in retrieved evidence.
  - Conversational follow-up and multi-turn context query condensation.
  - Interactive concept visualization (`visualization.py`).
  - Dedicated Streamlit user interface ([`src/scientific_main.py`](file:///E:/Project%20Folder/NLP-chatbot-files/ElevanceSkills-GenAI-Internship/src/scientific_main.py) on Port `8505`).

### Module 5 — Multimodal Intelligence (Vision, Audio, Docs)

- **Implementation**: [`src/multimodal/`](file:///E:/Project%20Folder/NLP-chatbot-files/ElevanceSkills-GenAI-Internship/src/multimodal/) (`service.py`, `file_processor.py`, `prompt_templates.py`).
- **Supported Media**:
  - **Images**: PNG, JPG, JPEG, WEBP (analyzed via Gemini 1.5 Flash Vision).
  - **Audio**: WAV, MP3 (transcribed and analyzed via Gemini audio capabilities).
  - **Documents**: PDF, TXT, DOCX (text extracted, chunked, and contextualized).
- **Safety & Limits**: Enforces 20MB file size limit, strict MIME type validation, temporary file cleanup, and defensive fallback error handling.

### Module 6 — Cross-Cutting Multilingual Engine

- **Implementation**: [`src/multilingual/`](file:///E:/Project%20Folder/NLP-chatbot-files/ElevanceSkills-GenAI-Internship/src/multilingual/) (`service.py`, `language_detector.py`, `prompts.py`).
- **Languages Supported**: English (`en`), Spanish (`es`), French (`fr`), German (`de`), Hindi (`hi`).
- **Cross-Cutting Role**:
  - Functions as an orthogonal platform capability rather than a standalone domain.
  - Performs non-English detection and translation for Customer Support.
  - Normalizes non-English queries into English for retrieval against English vector databases (MedQuAD and arXiv), ensuring domain evidence is accurately retrieved before returning localized responses with detected language metadata.

---

## 🔄 Cross-Task Integration & Unified Routing

The unified layer in [`src/cross_task/`](file:///E:/Project%20Folder/NLP-chatbot-files/ElevanceSkills-GenAI-Internship/src/cross_task/) brings all subsystems together:

- **`TaskRouter`** ([`router.py`](file:///E:/Project%20Folder/NLP-chatbot-files/ElevanceSkills-GenAI-Internship/src/cross_task/router.py)):
  - Analyzes queries using priority keywords, regex patterns, and domain markers.
  - Routes to `CUSTOMER_SUPPORT`, `MEDICAL_QA`, `SCIENTIFIC_RESEARCH`, or `MULTIMODAL`.
  - Supports explicit UI domain overrides or fully automatic intent routing.
- **`UnifiedOrchestrator`** ([`orchestrator.py`](file:///E:/Project%20Folder/NLP-chatbot-files/ElevanceSkills-GenAI-Internship/src/cross_task/orchestrator.py)):
  - Coordinates domain dispatching, query normalization, and response aggregation.
  - Invokes canonical domain interfaces (`ask()` for Scientific, `process_query()` for Customer Support and Medical, `process_request()` for Multimodal and Multilingual).
  - Tracks execution latency, confidence scores, and cross-cutting language detection.
- **`SessionManager`** ([`session_manager.py`](file:///E:/Project%20Folder/NLP-chatbot-files/ElevanceSkills-GenAI-Internship/src/cross_task/session_manager.py)):
  - Maintains conversation history across domain switches without data contamination.
  - Provides thread-safe session resets and message export.

---

## 🖥️ User Interfaces & Port Allocation

Each application can run standalone or through the unified portal. Each Streamlit UI runs on its designated port:

| Interface | Script Path | Default Port | Description |
| :--- | :--- | :--- | :--- |
| **Unified AI Assistant** | [`src/unified_main.py`](file:///E:/Project%20Folder/NLP-chatbot-files/ElevanceSkills-GenAI-Internship/src/unified_main.py) | **`8500`** | Central portal orchestrating all domains with auto/manual routing |
| **Customer Support UI** | [`src/main.py`](file:///E:/Project%20Folder/NLP-chatbot-files/ElevanceSkills-GenAI-Internship/src/main.py) | **`8501`** | EdTech FAQ bot with Sentiment conditioning & Dynamic KB sync |
| **Medical Q&A UI** | [`src/medical_main.py`](file:///E:/Project%20Folder/NLP-chatbot-files/ElevanceSkills-GenAI-Internship/src/medical_main.py) | **`8502`** | MedQuAD clinical Q&A with 5 safety states & NIH source badges |
| **Multimodal UI** | [`src/multimodal_main.py`](file:///E:/Project%20Folder/NLP-chatbot-files/ElevanceSkills-GenAI-Internship/src/multimodal_main.py) | **`8503`** | Image, audio, and document analysis assistant |
| **Multilingual UI** | [`src/multilingual_main.py`](file:///E:/Project%20Folder/NLP-chatbot-files/ElevanceSkills-GenAI-Internship/src/multilingual_main.py) | **`8504`** | Standalone 5-language conversational bot |
| **Scientific Research UI** | [`src/scientific_main.py`](file:///E:/Project%20Folder/NLP-chatbot-files/ElevanceSkills-GenAI-Internship/src/scientific_main.py) | **`8505`** | arXiv AI/ML research paper search, explanation & synthesis |

---

## 💾 Datasets & Protected Production Assets

> [!IMPORTANT]
> **Production Asset Protection Policy**: The datasets and persisted vector indexes below are immutable baseline assets. They must **never** be deleted, overwritten, or automatically re-indexed during routine execution or testing.

- **Customer Support Dataset**: [`dataset/knowledge_base.csv`](file:///E:/Project%20Folder/NLP-chatbot-files/ElevanceSkills-GenAI-Internship/dataset/knowledge_base.csv) (79 records).
- **Scientific Research Dataset**: [`dataset/arxiv_ai_ml_subset.jsonl`](file:///E:/Project%20Folder/NLP-chatbot-files/ElevanceSkills-GenAI-Internship/dataset/arxiv_ai_ml_subset.jsonl) (100 research papers).
- **Customer FAISS Index**: `faiss_index/` (Contains 76 embedded vectors matching baseline state).
- **Medical FAISS Index**: `faiss_index_medical/` (Contains 5 curated MedQuAD vectors with metadata).
- **Scientific FAISS Index**: `faiss_index_scientific/` (Contains 100 embedded arXiv research papers).

---

## ⚙️ Installation & Setup

### Prerequisites

- Python `3.11` (x64 recommended)
- Git

### 1. Clone the Repository

```bash
git clone https://github.com/Vishnu3568/elevance-skills-genai-internship.git
cd elevance-skills-genai-internship
```

### 2. Set Up Virtual Environment

```bash
# Windows
python -m venv chatbot
chatbot\Scripts\activate

# Linux / macOS
python3 -m venv chatbot
source chatbot/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

> [!NOTE]
> **Reproducible Dependency Alignment**: `requirements.txt` is fully reproducible across Linux and Windows. It pins modern modular LangChain packages (`langchain==1.3.14`, `langchain-community==0.4.2`, `langchain-core==1.5.3`, `langchain-classic==1.0.8`, `langchain-google-genai==4.3.2`), points to the official PyTorch CPU repository (`--extra-index-url https://download.pytorch.org/whl/cpu`) with `torch==2.13.0+cpu` for lightweight cloud deployment without CUDA bloat, and provides runtime dependencies for Hugging Face Transformers (`transformers==4.30.2`, `sentencepiece==0.2.2`, `tokenizers==0.13.3`, `safetensors==0.8.0`), vector retrieval (`sentence-transformers==2.2.2`, `InstructorEmbedding==1.0.1`, `faiss-cpu==1.7.4`), `streamlit==1.32.0`, `pandas==2.3.3`, and `pillow==10.4.0`.

### 4. Configure Environment Variables

Create a `.env` file in the project root (see [`.env.example`](file:///E:/Project%20Folder/NLP-chatbot-files/ElevanceSkills-GenAI-Internship/.env.example)):

```env
GOOGLE_API_KEY="your_google_gemini_api_key_here"
```

> [!IMPORTANT]
> **API Key Scope & Offline Boot**:
> - The Unified AI Assistant boots and initializes without requiring `GOOGLE_API_KEY`.
> - Task 4 (Scientific Domain Expert) runs completely offline and locally using the open-source `google/flan-t5-base` pipeline and does not require `GOOGLE_API_KEY`.
> - `GOOGLE_API_KEY` is required for Gemini-based workflows: Customer Support FAQ answers, Medical Q&A generation, Multimodal image/audio reasoning (Gemini 1.5 Flash), and Multilingual reasoning.

---

## 🚀 Running the Applications

Launch the Unified AI Assistant or any standalone application:

```bash
# 1. Launch the Unified Multi-Domain Assistant (Recommended)
streamlit run src/unified_main.py --server.port 8500

# 2. Launch Customer Support Chatbot (with Sentiment & Dynamic KB)
streamlit run src/main.py --server.port 8501

# 3. Launch Medical Q&A Assistant (MedQuAD)
streamlit run src/medical_main.py --server.port 8502

# 4. Launch Multimodal Assistant (Images / Audio / Documents)
streamlit run src/multimodal_main.py --server.port 8503

# 5. Launch Multilingual Chatbot (Standalone)
streamlit run src/multilingual_main.py --server.port 8504

# 6. Launch Scientific Research Assistant (arXiv)
streamlit run src/scientific_main.py --server.port 8505
```

---

## 🧪 Testing & Quality Assurance

The repository includes an extensive automated regression test suite covering all domains, cross-task integrations, and edge cases.

### Running the Full Test Suite

To run the full regression test suite with environment paths properly configured:

```bash
# Windows PowerShell
$env:PYTHONPATH="E:\Project Folder\NLP-chatbot-files\ElevanceSkills-GenAI-Internship\chatbot\Lib\site-packages;."; $env:NLTK_DISABLE_IMPORT_SECURITY="1"; $env:TRANSFORMERS_NO_TF="1"; $env:USE_TF="0"; pytest

# Linux / macOS
PYTHONPATH=chatbot/lib/python3.11/site-packages:. NLTK_DISABLE_IMPORT_SECURITY=1 TRANSFORMERS_NO_TF=1 USE_TF=0 pytest
```

### Running Specific Test Suites

```bash
# Run Cross-Task Integration test suites
pytest -k cross_task

# Run Medical Q&A test suites
pytest -k "medical or medquad"

# Run Multilingual test suites
pytest -k multilingual

# Run Multimodal test suites
pytest -k multimodal

# Run Scientific Research test suites
pytest -k scientific

# Run Customer Support & Dynamic KB test suites
pytest -k "customer or knowledge or dynamic or chatbot"
```

### Test Baseline Status

- **Total Discovered Tests**: **735 passed, 0 failed** across **65 test modules**.
- **Pass Rate**: **100% pass rate**.
- **Verification**: All 735 tests pass cleanly with zero failures and zero errors under the verified execution environment.

---

## 🔍 Static Analysis & Code Quality

Type checking and linting are configured via [`pyproject.toml`](file:///E:/Project%20Folder/NLP-chatbot-files/ElevanceSkills-GenAI-Internship/pyproject.toml) and [`pyrightconfig.json`](file:///E:/Project%20Folder/NLP-chatbot-files/ElevanceSkills-GenAI-Internship/pyrightconfig.json).

To run Pyright over the integration layer:

```bash
npx pyright src/cross_task src/unified_main.py
```

- **Result**: `0 errors, 0 warnings, 0 informations`.

---

## ⚠️ Architectural Context & Design Notes

1. **Customer Support Dataset & Index Alignment**: The Customer Support FAQ CSV (`dataset/knowledge_base.csv`) contains 79 entries (76 baseline entries + 3 dynamic updates), while the pre-built, protected baseline `faiss_index/` vector store contains 76 vectors. Isolation tests explicitly assert both invariants (79 CSV records, 76 baseline vectors) without destructive re-indexing.
2. **Medical/Scientific Multilingual Responses**: While Customer Support features direct target-language response generation, Medical and Scientific RAG systems normalize queries to English for retrieval accuracy against NIH and arXiv corpora and preserve clinical/scientific terminology in English while tagging the response with detected language metadata.
3. **Module Packaging**: `src/` contains [`src/__init__.py`](file:///E:/Project%20Folder/NLP-chatbot-files/ElevanceSkills-GenAI-Internship/src/__init__.py) as a package marker for consistent cross-module imports.

---

## 📁 Repository Directory Structure

```text
ElevanceSkills-GenAI-Internship/
├── dataset/                         # Protected production datasets
│   ├── knowledge_base.csv           # Customer Support FAQs (79 entries)
│   └── arxiv_ai_ml_subset.jsonl     # arXiv AI/ML research papers (100 papers)
├── docs/                            # Domain documentation
│   └── MEDICAL_QA_GUIDE.md          # NIH MedQuAD technical manual
├── faiss_index/                     # Protected Customer Support vector index (76 vectors)
├── faiss_index_medical/             # Protected MedQuAD medical vector index (5 vectors)
├── faiss_index_scientific/          # Protected arXiv scientific vector index (100 vectors)
├── src/                             # Application source code
│   ├── cross_task/                  # Unified cross-task routing & orchestration
│   │   ├── models.py                # Domain enums, request/response models
│   │   ├── router.py                # Intent routing & keyword classification
│   │   ├── orchestrator.py          # Central execution orchestrator
│   │   └── session_manager.py       # Multi-domain session state management
│   ├── knowledge_base/              # Dynamic KB management (Module 3)
│   │   ├── atomic_writer.py         # Thread-safe atomic file writing
│   │   ├── kb_updater.py            # Knowledge base sync service
│   │   ├── scheduler.py             # Periodic background sync scheduler
│   │   └── validator.py             # Schema & data integrity validation
│   ├── multilingual/                # Multilingual capability (Module 6)
│   │   ├── language_detector.py     # Script, ngram & stopword detection
│   │   ├── prompts.py               # Multilingual prompt templates
│   │   └── service.py               # Cross-lingual translation & query handling
│   ├── multimodal/                  # Multimodal processing (Module 5)
│   │   ├── file_processor.py        # MIME detection, text extraction & audio handling
│   │   ├── prompt_templates.py      # Vision & document prompts
│   │   └── service.py               # Gemini multimodal orchestration
│   ├── scientific_kb/               # Scientific domain expert (Module 4)
│   │   ├── concept_explainer.py     # Multi-level concept explainer
│   │   ├── conversation.py          # Conversational memory & query condensation
│   │   ├── generation.py            # Open-source FLAN-T5 LLM adapter & grounded prompts
│   │   ├── grounding.py             # Citation extraction & evidence validation
│   │   ├── query_classifier.py      # arXiv query intent classifier
│   │   ├── retriever.py             # Scientific FAISS retriever
│   │   ├── service.py               # End-to-end scientific service (ask API)
│   │   ├── synthesizer.py           # Multi-paper synthesis & comparison
│   │   ├── understanding.py         # Structured paper extraction & analysis
│   │   └── visualization.py         # Concept hierarchy & relation visualization
│   ├── chatbot_service.py           # Customer Support RAG service
│   ├── langchain_helper.py          # LangChain FAISS initialization
│   ├── main.py                      # Customer Support UI (Port 8501)
│   ├── medical_main.py              # Medical Q&A UI (Port 8502)
│   ├── medical_qa_service.py        # MedQuAD clinical service & safety states (Module 2)
│   ├── medquad_indexer.py           # MedQuAD XML parsing & indexing pipeline
│   ├── medquad_parser.py            # MedQuAD XML schema parser
│   ├── medquad_query_analyzer.py    # Clinical entity & question intent analyzer
│   ├── medquad_retriever.py         # Medical vector retriever
│   ├── multimodal_main.py           # Multimodal Assistant UI (Port 8503)
│   ├── multilingual_main.py         # Multilingual Assistant UI (Port 8504)
│   ├── response_policy.py           # Sentiment-conditioned response policies (Module 1)
│   ├── scientific_main.py           # Scientific Research UI (Port 8505)
│   ├── sentiment_analyzer.py        # CardiffNLP RoBERTa sentiment classifier (Module 1)
│   └── unified_main.py              # Unified AI Assistant Portal (Port 8500)
├── tests/                           # Automated test suite (735 tests across 65 modules)
├── .env.example                     # Environment template
├── pyproject.toml                   # Pytest & Pyright configuration
├── pyrightconfig.json               # Pyright type checker settings
├── README.md                        # Production system documentation
├── README.training.md               # Original baseline training documentation
└── requirements.txt                 # Pinned reproducible dependencies
```

---

## 📜 Dataset & Model Attributions

- **Medical Intelligence**: National Institutes of Health MedQuAD (Medical Question Answering Dataset).
- **Scientific Intelligence**: arXiv.org e-Print Archive (Computer Science AI/ML subset, 100 publications).
- **Sentiment Intelligence**: CardiffNLP Twitter RoBERTa Sentiment Model (`cardiffnlp/twitter-roberta-base-sentiment-latest`).
- **Embedding Foundation**: HuggingFace `sentence-transformers/all-MiniLM-L6-v2` & `hkunlp/instructor-large`.
- **Generative Intelligence**:
  - **Scientific Domain Expert (Task 4)**: Open-Source Hugging Face `google/flan-t5-base` via PyTorch CPU pipeline.
  - **Specialist & Multimodal Domains**: Google Gemini API (`gemini-2.5-flash` / `gemini-1.5-flash`) for Customer Support, Medical Q&A, and Multimodal reasoning.
