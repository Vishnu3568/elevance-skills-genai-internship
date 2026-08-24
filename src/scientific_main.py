"""Streamlit User Interface for Scientific Domain Expert RAG Assistant.

Provides an interactive scientific research assistant powered by an authentic arXiv
AI/ML/NLP corpus, dense retrieval, grounded open-source LLM generation, citation
validation, multi-turn follow-ups, and specialized scientific exploration workflows.
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# Disable NLTK import security hook that blocks regex in CWD
os.environ["NLTK_DISABLE_IMPORT_SECURITY"] = "1"
sys.path = [p for p in sys.path if p and os.path.abspath(p) != os.path.abspath(".")]
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # pyrefly: ignore [missing-import]
    from src.langchain_helper import get_instructor_embeddings, get_llm  # type: ignore
    # pyrefly: ignore [missing-import]
    from src.scientific_kb import (  # type: ignore
        DEFAULT_SCIENTIFIC_VECTOR_STORE_PATH,
        AuthorStatistic,
        CategoryStatistic,
        Citation,
        ConceptCooccurrence,
        ConceptExtractionResult,
        CorpusSummary,
        GroundingValidationResult,
        RelatedPaperMatch,
        ScientificAnswer,
        ScientificConversationSession,
        ScientificExpertResponse,
        ScientificExpertService,
        ScientificExplorationEngine,
        ScientificExplorationGraph,
        ScientificGenerator,
        ScientificRetrievalResult,
        ScientificRetriever,
        TimelineEntry,
        load_scientific_vector_store,
    )
except ImportError:
    # pyrefly: ignore [missing-import]
    from langchain_helper import get_instructor_embeddings, get_llm  # type: ignore
    # pyrefly: ignore [missing-import]
    from scientific_kb import (  # type: ignore
        DEFAULT_SCIENTIFIC_VECTOR_STORE_PATH,
        AuthorStatistic,
        CategoryStatistic,
        Citation,
        ConceptCooccurrence,
        ConceptExtractionResult,
        CorpusSummary,
        GroundingValidationResult,
        RelatedPaperMatch,
        ScientificAnswer,
        ScientificConversationSession,
        ScientificExpertResponse,
        ScientificExpertService,
        ScientificExplorationEngine,
        ScientificExplorationGraph,
        ScientificGenerator,
        ScientificRetrievalResult,
        ScientificRetriever,
        TimelineEntry,
        load_scientific_vector_store,
    )

MISSING_SCIENTIFIC_INDEX_MESSAGE = (
    "Scientific vector store not found at 'faiss_index_scientific/'. "
    "Please build the index first using: python scripts/build_scientific_index.py"
)


def format_source_badge(res: ScientificRetrievalResult) -> str:
    """Format a clean markdown badge for a retrieved paper source."""
    cat = f"`{res.primary_category}`" if res.primary_category else ""
    date = f" ({res.published_date[:4]})" if res.published_date else ""
    link = f"[[arXiv:{res.arxiv_id}]]({res.url})" if res.url else f"`arXiv:{res.arxiv_id}`"
    return f"📄 **{res.title}** {cat}{date} — {link}"


def render_grounding_status(grounded: bool, warning_message: Optional[str] = None):
    """Render a visual grounding indicator badge with optional warning banner."""
    if grounded and not warning_message:
        st.success("✅ **Evidence-Grounded**: All cited facts and arXiv IDs verified against retrieved scientific literature.")
    elif warning_message:
        st.warning(f"⚠️ **Citation Notice**: {warning_message}")
    else:
        st.info("ℹ️ **General Response**: Derived from scientific context without specific paper citations.")


def render_sources_panel(sources: List[ScientificRetrievalResult]):
    """Render an expandable structured evidence and provenance panel."""
    if not sources:
        return

    with st.expander(f"📚 **Retrieved Scientific Evidence ({len(sources)} Papers)**", expanded=False):
        for idx, src in enumerate(sources, 1):
            st.markdown(f"#### {idx}. {src.title}")
            cols = st.columns([2, 1, 1])
            with cols[0]:
                st.markdown(f"**Authors**: {src.authors or 'N/A'}")
                if src.url:
                    st.markdown(f"🔗 **arXiv Page**: [{src.arxiv_id}]({src.url})")
                else:
                    st.markdown(f"**arXiv ID**: `{src.arxiv_id}`")
            with cols[1]:
                st.markdown(f"**Category**: `{src.primary_category}`")
                if src.published_date:
                    st.markdown(f"**Published**: {src.published_date[:10]}")
            with cols[2]:
                if src.score is not None:
                    st.markdown(f"**Distance**: `{src.score:.4f}`")
                if src.doi:
                    st.markdown(f"**DOI**: `{src.doi}`")

            # Show abstract snippet if available in document
            if hasattr(src, "document") and src.document and src.document.page_content:
                st.markdown(f"> *{src.document.page_content[:300]}...*")
            st.divider()


def generate_dot_graph(graph: ScientificExplorationGraph, max_edges: int = 50) -> str:
    """Generate a clean Graphviz DOT string for the exploration graph.

    Args:
        graph (ScientificExplorationGraph): Structured nodes and edges.
        max_edges (int): Upper bound on rendered edges for layout clarity.

    Returns:
        str: Valid Graphviz DOT representation.
    """
    dot_lines = [
        "digraph ScientificKnowledgeGraph {",
        '  graph [rankdir=LR, bgcolor="transparent", fontsize=10, pad="0.2", nodesep="0.4", ranksep="0.6"];',
        '  node [shape=box, style="filled,rounded", fontname="Helvetica", fontsize=9];',
        '  edge [fontname="Helvetica", fontsize=8, arrowsize=0.6];',
    ]

    for node in graph.nodes:
        clean_label = node.label.replace('"', '\\"').replace("\n", " ")
        if node.node_type == "category":
            dot_lines.append(
                f'  "{node.id}" [label="{clean_label}", fillcolor="#e1f5fe", color="#0288d1", fontcolor="#01579b", shape=ellipse, style="filled"];'
            )
        elif node.node_type == "concept":
            dot_lines.append(
                f'  "{node.id}" [label="{clean_label}", fillcolor="#ede7f6", color="#7e57c2", fontcolor="#4527a0", shape=box, style="filled,rounded"];'
            )
        elif node.node_type == "paper":
            dot_lines.append(
                f'  "{node.id}" [label="{clean_label}", fillcolor="#f5f5f5", color="#9e9e9e", fontcolor="#333333", shape=note, style="filled"];'
            )

    for edge in graph.edges[:max_edges]:
        if edge.relation_type == "concept_cooccurrence":
            dot_lines.append(
                f'  "{edge.source}" -> "{edge.target}" [dir=both, style=dashed, color="#7e57c2", label="{int(edge.weight)}"];'
            )
        elif edge.relation_type == "belongs_to_category":
            dot_lines.append(f'  "{edge.source}" -> "{edge.target}" [color="#0288d1"];')
        elif edge.relation_type == "exhibits_concept":
            dot_lines.append(f'  "{edge.source}" -> "{edge.target}" [color="#b39ddb"];')

    dot_lines.append("}")
    return "\n".join(dot_lines)


@st.cache_resource(show_spinner="Loading Scientific AI/ML Knowledge Base...")
def initialize_scientific_service(
    store_path: str = DEFAULT_SCIENTIFIC_VECTOR_STORE_PATH,
) -> Tuple[Optional[ScientificExpertService], Optional[str]]:
    """Initialize and cache the ScientificExpertService and its dependencies.

    Args:
        store_path (str): Relative or absolute path to scientific FAISS directory.

    Returns:
        Tuple[Optional[ScientificExpertService], Optional[str]]: Initialized service or error message.
    """
    proj_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    target_path = proj_root / store_path if not Path(store_path).is_absolute() else Path(store_path)

    if not target_path.exists():
        return None, MISSING_SCIENTIFIC_INDEX_MESSAGE

    try:
        embeddings = get_instructor_embeddings()
        vector_store = load_scientific_vector_store(str(target_path), embeddings=embeddings)
        retriever = ScientificRetriever(vector_store=vector_store, default_k=3)

        # Attempt to load LLM
        try:
            llm = get_llm()
            generator = ScientificGenerator(llm=llm)
        except Exception as llm_err:
            # Fallback callable LLM if API key is unconfigured
            def _fallback_generator(prompt: str) -> str:
                return (
                    "**Scientific Assistant (Local Mode)**:\n\n"
                    "Evidence was successfully retrieved from the scientific corpus. "
                    "To generate full open-source LLM explanations, please configure `GOOGLE_API_KEY` in `.env`."
                )

            generator = ScientificGenerator(llm=_fallback_generator)

        session = ScientificConversationSession(max_history_turns=6)
        service = ScientificExpertService(
            retriever=retriever,
            generator=generator,
            session=session,
        )
        return service, None
    except Exception as exc:
        return None, f"Failed to initialize Scientific Knowledge Base: {exc}"


@st.cache_resource(show_spinner="Initializing Scientific Exploration Engine...")
def initialize_exploration_engine(_vector_store: Any) -> Optional[ScientificExplorationEngine]:
    """Initialize and cache the exploration engine for literature and concept analytics.

    Args:
        _vector_store (Any): Active FAISS vector store.

    Returns:
        Optional[ScientificExplorationEngine]: Exploration engine instance or None.
    """
    if _vector_store is None:
        return None
    try:
        return ScientificExplorationEngine(vector_store=_vector_store)
    except Exception:
        return None


def render_sidebar(service: Optional[ScientificExpertService]) -> Dict[str, Any]:
    """Render sidebar filters and session controls."""
    with st.sidebar:
        st.header("🔬 Scientific Settings")
        st.markdown("Configure retrieval filters and knowledge domain parameters.")

        # Category Filter
        category_options = ["All", "cs.CL", "cs.AI", "cs.LG", "cs.CV", "stat.ML"]
        selected_cat = st.selectbox(
            "Target Category",
            category_options,
            index=0,
            help="Filter retrieval to specific arXiv CS/AI/ML subcategories.",
        )
        cat_filter = None if selected_cat == "All" else selected_cat

        # Year Filter
        min_year = st.slider(
            "Minimum Publication Year",
            min_value=2015,
            max_value=2026,
            value=2018,
            help="Filter papers published on or after this year.",
        )

        # Author Filter
        author_query = st.text_input(
            "Author Filter (optional)",
            placeholder="e.g. Vaswani, Bengio",
            help="Filter papers by author name substring.",
        )
        author_filter = author_query.strip() if author_query.strip() else None

        # Top-K
        top_k = st.slider(
            "Top Evidence Papers (k)",
            min_value=1,
            max_value=8,
            value=3,
            help="Number of relevant papers to retrieve for context synthesis.",
        )

        st.divider()

        # Session reset button
        if st.button("🔄 Clear Conversation", use_container_width=True):
            if service:
                service.clear_session()
            st.session_state["scientific_messages"] = []
            st.session_state["active_paper_summary"] = None
            st.session_state["active_concept_explanation"] = None
            st.session_state["active_comparison"] = None
            st.rerun()

        st.divider()
        st.markdown("### 📊 Corpus Overview")
        if service and hasattr(service.retriever.vector_store, "docstore"):
            total_docs = len(service.retriever.vector_store.docstore._dict)
            st.metric(label="Indexed Papers", value=total_docs)
            st.caption("Domains: `cs.CL`, `cs.AI`, `cs.LG`, `cs.CV`, `stat.ML`")
        else:
            st.caption("Store: `faiss_index_scientific/`")

        return {
            "category_filter": cat_filter,
            "min_year": min_year,
            "author_filter": author_filter,
            "k": top_k,
        }


def main():
    """Main Streamlit application entrypoint."""
    st.set_page_config(
        page_title="Scientific Domain Expert Assistant",
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("🔬 Scientific Domain Expert Assistant")
    st.caption("Domain-Expert AI/ML/NLP Research Assistant with Grounded Scientific Retrieval & Citation Verification")

    # Initialize Service and Exploration Engine
    service, init_error = initialize_scientific_service()
    filters = render_sidebar(service)

    if init_error or service is None:
        st.error(init_error or "Unable to load Scientific Knowledge Base.")
        st.info("Run `python scripts/build_scientific_index.py` from your workspace to construct the index.")
        return

    exploration_engine = initialize_exploration_engine(service.retriever.vector_store)

    # Session State for chat history and active explorer items
    if "scientific_messages" not in st.session_state:
        st.session_state["scientific_messages"] = []
    if "selected_explorer_paper_id" not in st.session_state:
        st.session_state["selected_explorer_paper_id"] = None
    if "selected_explorer_concept" not in st.session_state:
        st.session_state["selected_explorer_concept"] = None

    # Main Tabs
    tab_chat, tab_search, tab_summary, tab_concept, tab_compare, tab_explore = st.tabs([
        "💬 Research Assistant",
        "🔍 Paper Search",
        "📄 Paper Summarizer",
        "🧠 Concept Explainer",
        "⚖️ Technical Comparison",
        "🔬 Research Explorer",
    ])

    # -------------------------------------------------------------
    # TAB 1: Chat Assistant
    # -------------------------------------------------------------
    with tab_chat:
        st.markdown("Ask complex scientific questions, explore research paradigms, and discuss papers with citation provenance.")

        # Render conversation history
        for msg in st.session_state["scientific_messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg.get("sources"):
                    render_sources_panel(msg["sources"])
                if "grounded" in msg:
                    render_grounding_status(msg["grounded"], msg.get("warning"))

        # User chat input
        user_input = st.chat_input("Ask a scientific question (e.g. 'Explain transformer architectures and their limitations')...")
        if user_input:
            with st.chat_message("user"):
                st.markdown(user_input)
            st.session_state["scientific_messages"].append({"role": "user", "content": user_input})

            # Execute RAG query through backend service
            with st.chat_message("assistant"):
                with st.spinner("Retrieving scientific evidence and synthesizing grounded explanation..."):
                    try:
                        response: ScientificExpertResponse = service.ask(
                            query=user_input,
                            category_filter=filters["category_filter"],
                            min_year=filters["min_year"],
                            author_filter=filters["author_filter"],
                            k=filters["k"],
                        )
                        st.markdown(response.answer)
                        render_grounding_status(response.grounded, response.warning_message)
                        render_sources_panel(response.sources)

                        # Record in history
                        st.session_state["scientific_messages"].append({
                            "role": "assistant",
                            "content": response.answer,
                            "sources": response.sources,
                            "grounded": response.grounded,
                            "warning": response.warning_message,
                        })
                    except Exception as err:
                        st.error(f"Error generating scientific response: {err}")

    # -------------------------------------------------------------
    # TAB 2: Paper Search / Lookup
    # -------------------------------------------------------------
    with tab_search:
        st.subheader("🔍 Scientific Paper Search & Metadata Explorer")
        st.markdown("Look up papers by exact arXiv ID (e.g. `2012.10055v2`), title substring, or natural language research query.")

        col_search, col_type = st.columns([3, 1])
        with col_search:
            search_query = st.text_input("Search query / arXiv ID", placeholder="e.g. 2012.10055v2 or speaker diarization")
        with col_type:
            search_mode = st.selectbox("Search By", ["Semantic / Query", "arXiv ID", "Title Match"])

        if st.button("Search Literature", type="primary"):
            if not search_query.strip():
                st.warning("Please enter a search query or arXiv ID.")
            else:
                with st.spinner("Searching scientific vector store..."):
                    if search_mode == "arXiv ID":
                        paper_res = service.find_paper(arxiv_id=search_query.strip())
                        search_results = [paper_res] if paper_res else []
                    elif search_mode == "Title Match":
                        paper_res = service.find_paper(title_query=search_query.strip())
                        search_results = [paper_res] if paper_res else []
                    else:
                        search_results = service.retriever.retrieve(
                            query=search_query.strip(),
                            k=filters["k"],
                            category_filter=filters["category_filter"],
                            min_year=filters["min_year"],
                            author_filter=filters["author_filter"],
                        )

                    if not search_results:
                        st.info(f"No papers found matching '{search_query}'.")
                    else:
                        st.success(f"Found {len(search_results)} relevant paper(s):")
                        for idx, p in enumerate(search_results, 1):
                            with st.container():
                                st.markdown(f"### {idx}. {p.title}")
                                st.markdown(f"**arXiv ID**: `{p.arxiv_id}` | **Category**: `{p.primary_category}` | **Date**: {p.published_date[:10] if p.published_date else 'N/A'}")
                                st.markdown(f"**Authors**: {p.authors}")
                                if p.url:
                                    st.markdown(f"🔗 **URL**: [{p.url}]({p.url})")
                                if p.document and p.document.page_content:
                                    with st.expander("📄 View Abstract / Indexed Content"):
                                        st.markdown(p.document.page_content)
                                st.divider()

    # -------------------------------------------------------------
    # TAB 3: Paper Summarizer
    # -------------------------------------------------------------
    with tab_summary:
        st.subheader("📄 Structured Scientific Paper Summarizer")
        st.markdown("Generate a structured 6-section summary (Contributions, Methodology, Findings, Limitations) grounded in evidence.")

        summary_input = st.text_input("Enter Paper arXiv ID or Title", placeholder="e.g. 2012.10055v2 or Attention Is All You Need")
        if st.button("Generate Paper Summary", type="primary"):
            if not summary_input.strip():
                st.warning("Please enter a paper identifier or title.")
            else:
                with st.spinner("Generating evidence-grounded structured summary..."):
                    summary_resp = service.summarize_paper(summary_input.strip())
                    st.markdown(summary_resp.answer)
                    render_grounding_status(summary_resp.grounded, summary_resp.warning_message)
                    render_sources_panel(summary_resp.sources)

    # -------------------------------------------------------------
    # TAB 4: Concept Explainer
    # -------------------------------------------------------------
    with tab_concept:
        st.subheader("🧠 Two-Tiered Scientific Concept Explainer")
        st.markdown("Generate both technical and intuitive mental models for core AI/ML algorithms grounded in literature.")

        concept_input = st.text_input("Enter AI/ML Concept", placeholder="e.g. Multi-Head Attention, LoRA, Contrastive Learning")
        if st.button("Explain Concept", type="primary"):
            if not concept_input.strip():
                st.warning("Please enter a concept name.")
            else:
                with st.spinner("Retrieving literature and generating two-tiered explanation..."):
                    concept_resp = service.explain_concept(concept_input.strip())
                    st.markdown(concept_resp.answer)
                    render_grounding_status(concept_resp.grounded, concept_resp.warning_message)
                    render_sources_panel(concept_resp.sources)

    # -------------------------------------------------------------
    # TAB 5: Technical Comparison
    # -------------------------------------------------------------
    with tab_compare:
        st.subheader("⚖️ Scientific Paradigm & Method Comparison")
        st.markdown("Generate a side-by-side comparative analysis across mechanism, computation, memory, and trade-offs.")

        col_a, col_b = st.columns(2)
        with col_a:
            topic_a = st.text_input("Topic / Method A", placeholder="e.g. LoRA or Dense Retrieval")
        with col_b:
            topic_b = st.text_input("Topic / Method B", placeholder="e.g. Full Fine-Tuning or BM25")

        if st.button("Compare Paradigms", type="primary"):
            if not topic_a.strip() or not topic_b.strip():
                st.warning("Please enter both Topic A and Topic B.")
            else:
                with st.spinner(f"Retrieving literature for '{topic_a}' and '{topic_b}'..."):
                    comp_resp = service.compare(topic_a.strip(), topic_b.strip())
                    st.markdown(comp_resp.answer)
                    render_grounding_status(comp_resp.grounded, comp_resp.warning_message)
                    render_sources_panel(comp_resp.sources)

    # -------------------------------------------------------------
    # TAB 6: Research Explorer
    # -------------------------------------------------------------
    with tab_explore:
        st.subheader("🔬 arXiv Scientific Research & Concept Explorer")
        st.markdown("Interactive literature exploration, deterministic concept extraction, topic graphs, and semantic paper discovery.")

        if exploration_engine is None or exploration_engine.total_papers == 0:
            st.info("No exploration data available in the current scientific vector store.")
        else:
            explore_subtabs = st.tabs([
                "📊 Corpus Overview",
                "🧠 Concept Explorer",
                "📅 Publication Timeline",
                "📄 Paper Network & Relations",
                "🕸️ Knowledge Graph",
            ])

            # 1. Corpus Overview Subtab
            with explore_subtabs[0]:
                st.markdown("### 📈 Corpus Metadata & Distributions")
                summary: CorpusSummary = exploration_engine.get_corpus_summary()

                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                with col_m1:
                    st.metric("Total Indexed Papers", summary.total_papers)
                with col_m2:
                    st.metric("Unique Concepts", len(exploration_engine.get_concept_statistics()))
                with col_m3:
                    st.metric("Earliest Paper", summary.earliest_date[:10] if summary.earliest_date != "N/A" else "N/A")
                with col_m4:
                    st.metric("Latest Paper", summary.latest_date[:10] if summary.latest_date != "N/A" else "N/A")

                st.divider()

                col_chart1, col_chart2 = st.columns(2)
                with col_chart1:
                    st.markdown("#### Category Distribution")
                    cat_data = [{"Category": cs.category, "Papers": cs.all_count, "Primary": cs.primary_count} for cs in summary.category_distribution]
                    if cat_data:
                        cat_df = pd.DataFrame(cat_data).set_index("Category")
                        st.bar_chart(cat_df[["Papers", "Primary"]])

                with col_chart2:
                    st.markdown("#### Top Extracted Concepts")
                    concept_data = [{"Concept": cs.concept, "Frequency": cs.frequency} for cs in summary.top_concepts[:8]]
                    if concept_data:
                        concept_df = pd.DataFrame(concept_data).set_index("Concept")
                        st.bar_chart(concept_df)

                col_auth, col_yr = st.columns(2)
                with col_auth:
                    st.markdown("#### Top Contributing Authors")
                    auth_data = [{"Author": a.author, "Papers": a.paper_count} for a in summary.top_authors[:10]]
                    st.dataframe(pd.DataFrame(auth_data), use_container_width=True)

                with col_yr:
                    st.markdown("#### Publication Year Distribution")
                    yr_data = [{"Year": str(y), "Count": cnt} for y, cnt in summary.year_distribution.items()]
                    if yr_data:
                        yr_df = pd.DataFrame(yr_data).set_index("Year")
                        st.bar_chart(yr_df)

            # 2. Concept Explorer Subtab
            with explore_subtabs[1]:
                st.markdown("### 🧠 Deterministic Concept & Co-occurrence Explorer")
                all_concepts = exploration_engine.get_concept_statistics()

                if not all_concepts:
                    st.info("No concepts extracted.")
                else:
                    concept_names = [cs.concept for cs in all_concepts]
                    selected_c = st.selectbox(
                        "Select AI/ML/NLP Concept",
                        concept_names,
                        index=0,
                    )

                    c_stat = next((cs for cs in all_concepts if cs.concept == selected_c), None)
                    if c_stat:
                        col_c1, col_c2, col_c3 = st.columns([1, 1, 2])
                        with col_c1:
                            st.metric("Concept Frequency", f"{c_stat.frequency} papers")
                        with col_c2:
                            st.metric("Categories", len(c_stat.categories))
                        with col_c3:
                            st.markdown(f"**Associated Categories**: {', '.join([f'`{cat}`' for cat in c_stat.categories])}")

                        # Co-occurrences with this concept
                        cooccurs = exploration_engine.get_concept_cooccurrence()
                        related_co = [
                            {"Co-occurring Concept": co.concept_b if co.concept_a == selected_c else co.concept_a, "Shared Papers": co.cooccurrence_count}
                            for co in cooccurs
                            if co.concept_a == selected_c or co.concept_b == selected_c
                        ]
                        if related_co:
                            with st.expander("🔗 Co-occurring Concepts", expanded=True):
                                st.dataframe(pd.DataFrame(related_co), use_container_width=True)

                        # Papers containing this concept
                        st.markdown(f"#### 📄 Papers Exhibiting '{selected_c}' ({len(c_stat.paper_ids)})")
                        matching_papers = exploration_engine.get_papers_for_concept(selected_c)
                        for idx, p in enumerate(matching_papers, 1):
                            pid = p.get("arxiv_id") or p.get("id") or ""
                            p_title = p.get("title", pid)
                            p_cat = p.get("primary_category", "")
                            p_date = (p.get("published_date") or "")[:10]
                            p_authors = p.get("authors", "")

                            with st.container():
                                st.markdown(f"**{idx}. {p_title}** (`{pid}`)")
                                st.caption(f"Category: `{p_cat}` | Published: {p_date} | Authors: {p_authors}")
                                st.divider()

            # 3. Publication Timeline Subtab
            with explore_subtabs[2]:
                st.markdown("### 📅 Publication Timeline & Period Breakdown")
                timeline_yearly = exploration_engine.get_timeline(granularity="yearly")
                timeline_monthly = exploration_engine.get_timeline(granularity="monthly")

                years = list(exploration_engine.get_year_statistics().keys())
                if not years:
                    st.info("No timeline data available.")
                else:
                    selected_year = st.selectbox("Select Publication Year", years, index=len(years) - 1)

                    # Monthly distribution chart for selected year
                    months_for_year = [t for t in timeline_monthly if t.year == selected_year]
                    if months_for_year:
                        m_df = pd.DataFrame([{"Month": t.period, "Papers": t.paper_count} for t in months_for_year]).set_index("Month")
                        st.bar_chart(m_df)

                    # Papers in selected year
                    year_papers = [
                        p for p in exploration_engine.papers
                        if p.get("published_date") and p.get("published_date", "").startswith(str(selected_year))
                    ]
                    st.markdown(f"#### Papers Published in {selected_year} ({len(year_papers)} Papers)")
                    for idx, p in enumerate(year_papers, 1):
                        pid = p.get("arxiv_id") or p.get("id") or ""
                        st.markdown(f"**{idx}. {p.get('title', pid)}**")
                        st.caption(f"arXiv: `{pid}` | Category: `{p.get('primary_category')}` | Date: {p.get('published_date', '')[:10]} | Authors: {p.get('authors')}")
                        st.divider()

            # 4. Paper Network & Relations Subtab
            with explore_subtabs[3]:
                st.markdown("### 📄 Paper Details & Semantic Related Recommendations")
                all_pids = [p.get("arxiv_id") or p.get("id") or "" for p in exploration_engine.papers if p.get("arxiv_id") or p.get("id")]
                paper_titles_map = {
                    (p.get("arxiv_id") or p.get("id") or ""): f"[{p.get('arxiv_id') or p.get('id')}] {p.get('title', '')[:70]}..."
                    for p in exploration_engine.papers
                    if p.get("arxiv_id") or p.get("id")
                }

                selected_pid = st.selectbox(
                    "Select Target Paper",
                    all_pids,
                    format_func=lambda pid: paper_titles_map.get(pid, pid),
                    index=0,
                )

                target_p = next(
                    (p for p in exploration_engine.papers if (p.get("arxiv_id") == selected_pid or p.get("id") == selected_pid)),
                    None,
                )

                if target_p:
                    col_p1, col_p2 = st.columns([2, 1])
                    with col_p1:
                        st.markdown(f"### {target_p.get('title')}")
                        st.markdown(f"**Authors**: {target_p.get('authors')}")
                        p_url = f"https://arxiv.org/abs/{selected_pid}"
                        st.markdown(f"🔗 **arXiv Link**: [{selected_pid}]({p_url})")
                        if target_p.get("doi"):
                            st.markdown(f"**DOI**: `{target_p.get('doi')}`")
                    with col_p2:
                        st.markdown(f"**Primary Category**: `{target_p.get('primary_category')}`")
                        st.markdown(f"**Categories**: {target_p.get('categories')}")
                        st.markdown(f"**Published**: {target_p.get('published_date', '')[:10]}")

                    # Extracted Concepts for this paper
                    p_concepts = exploration_engine.get_concepts_for_paper(selected_pid)
                    if p_concepts:
                        st.markdown(f"**Extracted Concepts**: {', '.join([f'`{c}`' for c in p_concepts])}")

                    # Abstract
                    p_abstract = target_p.get("abstract") or target_p.get("page_content") or ""
                    with st.expander("📄 View Abstract", expanded=False):
                        st.markdown(p_abstract)

                    st.markdown("#### 🔍 Semantically Related Papers (KNN Nearest Neighbors)")
                    with st.spinner("Finding semantically related papers in scientific vector store..."):
                        related_matches = exploration_engine.find_related_papers(
                            target=target_p,
                            retriever=service.retriever,
                            top_k=4,
                        )

                        if not related_matches:
                            st.info("No related papers found.")
                        else:
                            for idx, match in enumerate(related_matches, 1):
                                r_paper = match.paper
                                st.markdown(f"##### {idx}. {r_paper.title}")
                                col_r1, col_r2, col_r3 = st.columns([2, 1, 1])
                                with col_r1:
                                    st.caption(f"Authors: {r_paper.authors}")
                                    if r_paper.url:
                                        st.markdown(f"🔗 [{r_paper.arxiv_id}]({r_paper.url})")
                                with col_r2:
                                    st.caption(f"Category: `{r_paper.primary_category}`")
                                    st.caption(f"Distance Score: `{match.similarity_score:.4f}`")
                                with col_r3:
                                    if match.shared_concepts:
                                        st.caption(f"Shared Concepts: {', '.join(match.shared_concepts)}")
                                    if match.shared_categories:
                                        st.caption(f"Shared Categories: {', '.join(match.shared_categories)}")
                                st.divider()

            # 5. Knowledge Graph Subtab
            with explore_subtabs[4]:
                st.markdown("### 🕸️ Tripartite Knowledge Graph (Categories, Concepts, Papers)")
                st.markdown("Visualizes relational pathways connecting arXiv subcategories, canonical concepts, and literature records.")

                col_g1, col_g2 = st.columns(2)
                with col_g1:
                    max_c = st.slider("Max Concepts in Graph", 5, 25, 12)
                with col_g2:
                    max_p = st.slider("Max Papers in Graph", 10, 50, 20)

                graph = exploration_engine.build_exploration_graph(max_concepts=max_c, max_papers=max_p)
                st.caption(f"Graph Topology: **{len(graph.nodes)} Nodes**, **{len(graph.edges)} Edges**")

                try:
                    dot_str = generate_dot_graph(graph, max_edges=45)
                    st.graphviz_chart(dot_str, use_container_width=True)
                except Exception as g_err:
                    st.warning(f"Graphviz visualization unavailable: {g_err}. Showing tabular network representation:")
                    node_df = pd.DataFrame([{"ID": n.id, "Label": n.label, "Type": n.node_type} for n in graph.nodes])
                    edge_df = pd.DataFrame([{"Source": e.source, "Target": e.target, "Relation": e.relation_type, "Weight": e.weight} for e in graph.edges])
                    st.dataframe(node_df, use_container_width=True)
                    st.dataframe(edge_df, use_container_width=True)


if __name__ == "__main__":
    main()
