"""Scientific Exploration Engine for Knowledge Graph, Topic Analytics, and Literature Discovery.

Provides deterministic concept extraction, concept co-occurrence analysis,
hierarchical exploration graph synthesis, corpus metadata aggregation (categories,
publication timeline, authors), and semantic related-paper discovery over arXiv AI/ML/NLP papers.
"""

import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple, Union

try:
    from langchain_core.documents import Document
    from langchain_community.vectorstores import FAISS
except ImportError:
    from langchain.docstore.document import Document
    from langchain.vectorstores import FAISS

try:
    # pyrefly: ignore [missing-import]
    from src.scientific_kb.models import ScientificPaper  # type: ignore
    # pyrefly: ignore [missing-import]
    from src.scientific_kb.retriever import ScientificRetrievalResult, ScientificRetriever  # type: ignore
    # pyrefly: ignore [missing-import]
    from src.scientific_kb.vector_store import (  # type: ignore
        DEFAULT_SCIENTIFIC_VECTOR_STORE_PATH,
        load_scientific_vector_store,
    )
except ImportError:
    # pyrefly: ignore [missing-import]
    from scientific_kb.models import ScientificPaper  # type: ignore
    # pyrefly: ignore [missing-import]
    from scientific_kb.retriever import ScientificRetrievalResult, ScientificRetriever  # type: ignore
    # pyrefly: ignore [missing-import]
    from scientific_kb.vector_store import (  # type: ignore
        DEFAULT_SCIENTIFIC_VECTOR_STORE_PATH,
        load_scientific_vector_store,
    )

# Curated deterministic taxonomy: (Canonical Name, Regex Pattern)
CONCEPT_TAXONOMY: List[Tuple[str, str]] = [
    ("Transformer", r"\b(?:transformers?|transformer-based|transformer architecture)\b"),
    ("Attention", r"\b(?:self[- ]?attention|multi[- ]?head[- ]?attention|attention[- ]?mechanism|attention[- ]?layer|attentive)\b"),
    ("BERT", r"\b(?:roberta|distilbert|albert|bert-base|bert-large|bert)\b"),
    ("GPT", r"\b(?:gpt-2|gpt-3|gpt-4|gpt-neo|gpt)\b"),
    ("Retrieval-Augmented Generation", r"\b(?:retrieval[- ]augmented(?: generation)?|rag)\b"),
    ("Natural Language Processing", r"\b(?:natural language processing|nlp)\b"),
    ("Machine Learning", r"\bmachine learning\b"),
    ("Deep Learning", r"\bdeep learning\b"),
    ("Reinforcement Learning", r"\b(?:reinforcement learning|deep rl|q-learning|policy gradient)\b"),
    ("Transfer Learning", r"\btransfer learning\b"),
    ("Representation Learning", r"\brepresentation learning\b"),
    ("Contrastive Learning", r"\bcontrastive (?:learning|loss|learning-based)\b"),
    ("Graph Neural Networks", r"\b(?:graph neural networks?|gnns?|graph convolutional networks?|gcns?)\b"),
    ("Convolutional Neural Networks", r"\b(?:convolutional neural networks?|cnns?|convolutional network)\b"),
    ("Computer Vision", r"\bcomputer vision\b"),
    ("Multimodal Learning", r"\b(?:multimodal|multi-modal|cross-modal)\b"),
    ("Speech Recognition", r"\b(?:speech recognition|automatic speech recognition|asr|acoustic model(?:ing)?)\b"),
    ("Speaker Diarization", r"\b(?:speaker diarization|diarization)\b"),
    ("Relation Extraction", r"\brelation extraction\b"),
    ("Named Entity Recognition", r"\b(?:named entity recognition|ner)\b"),
    ("Model Compression", r"\b(?:model compression|quantization|pruning|weight quantization|2-bit|compressed model)\b"),
    ("Knowledge Distillation", r"\bknowledge distillation\b"),
    ("Generative Models", r"\bgenerative models?\b"),
    ("GAN", r"\b(?:generative adversarial networks?|gans?)\b"),
    ("VAE", r"\b(?:variational autoencoders?|vaes?)\b"),
    ("Diffusion Models", r"\bdiffusion models?\b"),
    ("Sequence-to-Sequence", r"\b(?:sequence-to-sequence|seq2seq)\b"),
    ("Language Modeling", r"\b(?:language models?|language modeling)\b"),
    ("Pretraining", r"\b(?:pre-training|pretraining|pre-trained|pretrained)\b"),
    ("Fine-Tuning", r"\b(?:fine-tuning|finetuning|fine-tuned|finetuned)\b"),
    ("Embeddings", r"\b(?:word embeddings?|dense embeddings?|sentence embeddings?|node embeddings?)\b"),
    ("Active Learning", r"\bactive learning\b"),
    ("Text Classification", r"\btext classification\b"),
    ("Active Learning", r"\bactive learning\b"),
]

COMPILED_TAXONOMY = [(name, re.compile(pat, re.IGNORECASE)) for name, pat in CONCEPT_TAXONOMY]


@dataclass
class ConceptExtractionResult:
    """Statistics for an extracted scientific concept."""

    concept: str
    frequency: int
    paper_ids: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)


@dataclass
class ConceptCooccurrence:
    """Co-occurrence statistics between two scientific concepts."""

    concept_a: str
    concept_b: str
    cooccurrence_count: int
    paper_ids: List[str] = field(default_factory=list)


@dataclass
class CategoryStatistic:
    """Distribution metrics for an arXiv category."""

    category: str
    primary_count: int
    all_count: int
    percentage: float
    sample_titles: List[str] = field(default_factory=list)


@dataclass
class AuthorStatistic:
    """Publication statistics for a scientific author."""

    author: str
    paper_count: int
    paper_ids: List[str] = field(default_factory=list)


@dataclass
class TimelineEntry:
    """Aggregated timeline metrics for a specific time window."""

    period: str
    year: int
    month: Optional[int]
    paper_count: int
    paper_ids: List[str] = field(default_factory=list)
    top_categories: List[str] = field(default_factory=list)


@dataclass
class RelatedPaperMatch:
    """Semantic relatedness result between scientific papers."""

    paper: ScientificRetrievalResult
    similarity_score: float
    shared_concepts: List[str] = field(default_factory=list)
    shared_categories: List[str] = field(default_factory=list)


@dataclass
class ScientificGraphNode:
    """Graph node representing a paper, concept, or category."""

    id: str
    label: str
    node_type: str  # 'paper', 'concept', 'category'
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScientificGraphEdge:
    """Graph edge representing relationships between scientific entities."""

    source: str
    target: str
    relation_type: str  # 'belongs_to_category', 'exhibits_concept', 'concept_cooccurrence'
    weight: float = 1.0


@dataclass
class ScientificExplorationGraph:
    """Exploration graph containing nodes and relational edges."""

    nodes: List[ScientificGraphNode] = field(default_factory=list)
    edges: List[ScientificGraphEdge] = field(default_factory=list)


@dataclass
class CorpusSummary:
    """High-level summary of the indexed scientific corpus."""

    total_papers: int
    earliest_date: str
    latest_date: str
    category_distribution: List[CategoryStatistic] = field(default_factory=list)
    year_distribution: Dict[int, int] = field(default_factory=dict)
    top_concepts: List[ConceptExtractionResult] = field(default_factory=list)
    top_authors: List[AuthorStatistic] = field(default_factory=list)


def extract_concepts_from_text(title: str, abstract: str) -> List[str]:
    """Extract and normalize canonical scientific concepts from title and abstract.

    Uses word-boundary bounded regular expressions matching against a curated
    AI/ML/NLP taxonomy with deduplication and deterministic sorting.

    Args:
        title (str): Paper title.
        abstract (str): Paper abstract.

    Returns:
        List[str]: Sorted list of unique canonical concept names.
    """
    combined_text = f"{title or ''} {abstract or ''}"
    if not combined_text.strip():
        return []

    found_concepts: Set[str] = set()
    for name, pattern in COMPILED_TAXONOMY:
        if pattern.search(combined_text):
            found_concepts.add(name)

    return sorted(list(found_concepts))


def normalize_author_name(author_str: str) -> str:
    """Normalize author name string by removing excess whitespace and punctuation."""
    if not author_str:
        return ""
    cleaned = re.sub(r"\s+", " ", author_str).strip()
    return cleaned


class ScientificExplorationEngine:
    """Exploration engine analyzing literature topics, timelines, and relationships."""

    def __init__(self, vector_store: Optional[FAISS] = None, papers: Optional[List[Dict[str, Any]]] = None):
        """Initialize the exploration engine from a FAISS vector store or paper records.

        Args:
            vector_store (Optional[FAISS]): FAISS vector store holding indexed documents.
            papers (Optional[List[Dict[str, Any]]]): Explicit list of paper metadata dicts.
        """
        self.papers: List[Dict[str, Any]] = []
        self._concept_cache: Dict[str, List[str]] = {}

        if papers is not None:
            self.papers = list(papers)
        elif vector_store is not None:
            self._load_from_vector_store(vector_store)

        # Build concept cache
        self._index_concepts()

    def _load_from_vector_store(self, vector_store: FAISS) -> None:
        """Extract paper records from FAISS docstore."""
        if not hasattr(vector_store, "docstore") or not hasattr(vector_store.docstore, "_dict"):
            return

        for doc_id, doc in vector_store.docstore._dict.items():
            meta = dict(getattr(doc, "metadata", {}) or {})
            meta["page_content"] = getattr(doc, "page_content", "")
            self.papers.append(meta)

    def _index_concepts(self) -> None:
        """Extract and cache concepts for all loaded papers."""
        self._concept_cache.clear()
        for p in self.papers:
            arxiv_id = p.get("arxiv_id") or p.get("id") or ""
            title = p.get("title", "")
            abstract = p.get("abstract", "")
            if not abstract and "page_content" in p:
                abstract = p["page_content"]
            concepts = extract_concepts_from_text(title, abstract)
            if arxiv_id:
                self._concept_cache[arxiv_id] = concepts
            p["extracted_concepts"] = concepts

    @property
    def total_papers(self) -> int:
        """Return the total number of indexed papers in the exploration corpus."""
        return len(self.papers)

    def get_concepts_for_paper(self, arxiv_id: str) -> List[str]:
        """Return the extracted concepts for a specific paper by arXiv ID."""
        if not arxiv_id:
            return []
        norm_id = arxiv_id.strip()
        if norm_id in self._concept_cache:
            return list(self._concept_cache[norm_id])

        # Version fallback search
        base_id = re.sub(r"v\d+$", "", norm_id)
        for pid, concepts in self._concept_cache.items():
            if re.sub(r"v\d+$", "", pid) == base_id:
                return list(concepts)

        return []

    def get_papers_for_concept(self, concept: str) -> List[Dict[str, Any]]:
        """Return all papers exhibiting a given concept."""
        if not concept or not concept.strip():
            return []
        target = concept.strip().lower()
        matching_papers = []
        for p in self.papers:
            p_concepts = p.get("extracted_concepts", [])
            if any(c.lower() == target for c in p_concepts):
                matching_papers.append(p)
        return matching_papers

    def get_concept_statistics(self, min_frequency: int = 1) -> List[ConceptExtractionResult]:
        """Compute frequency, associated paper IDs, and categories for all extracted concepts."""
        concept_to_pids: Dict[str, List[str]] = defaultdict(list)
        concept_to_cats: Dict[str, Set[str]] = defaultdict(set)

        for p in self.papers:
            pid = p.get("arxiv_id") or p.get("id") or ""
            cats = p.get("categories") or []
            if isinstance(cats, str):
                cats = cats.split()
            prim_cat = p.get("primary_category")
            if prim_cat:
                cats = list(cats) + [prim_cat]

            for c in p.get("extracted_concepts", []):
                concept_to_pids[c].append(pid)
                for cat in cats:
                    if cat:
                        concept_to_cats[c].add(cat)

        results: List[ConceptExtractionResult] = []
        for c, pids in concept_to_pids.items():
            if len(pids) >= min_frequency:
                results.append(
                    ConceptExtractionResult(
                        concept=c,
                        frequency=len(pids),
                        paper_ids=sorted(list(set(pids))),
                        categories=sorted(list(concept_to_cats[c])),
                    )
                )

        # Sort descending by frequency, then alphabetical by concept
        results.sort(key=lambda x: (-x.frequency, x.concept))
        return results

    def get_concept_cooccurrence(self, min_count: int = 1) -> List[ConceptCooccurrence]:
        """Compute co-occurrence frequencies between pairs of concepts."""
        pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        pair_papers: Dict[Tuple[str, str], List[str]] = defaultdict(list)

        for p in self.papers:
            pid = p.get("arxiv_id") or p.get("id") or ""
            concepts = sorted(list(set(p.get("extracted_concepts", []))))
            for i in range(len(concepts)):
                for j in range(i + 1, len(concepts)):
                    pair = (concepts[i], concepts[j])
                    pair_counts[pair] += 1
                    pair_papers[pair].append(pid)

        results: List[ConceptCooccurrence] = []
        for (cA, cB), count in pair_counts.items():
            if count >= min_count:
                results.append(
                    ConceptCooccurrence(
                        concept_a=cA,
                        concept_b=cB,
                        cooccurrence_count=count,
                        paper_ids=sorted(list(set(pair_papers[(cA, cB)]))),
                    )
                )

        # Sort descending by count, then alphabetical
        results.sort(key=lambda x: (-x.cooccurrence_count, x.concept_a, x.concept_b))
        return results

    def get_category_statistics(self) -> List[CategoryStatistic]:
        """Compute primary and overall category distribution across the corpus."""
        if not self.papers:
            return []

        primary_counts: Counter = Counter()
        all_counts: Counter = Counter()
        category_titles: Dict[str, List[str]] = defaultdict(list)

        for p in self.papers:
            prim = p.get("primary_category")
            if prim:
                primary_counts[prim] += 1

            cats = p.get("categories") or []
            if isinstance(cats, str):
                cats = cats.split()
            for cat in set(cats):
                if cat:
                    all_counts[cat] += 1
                    if len(category_titles[cat]) < 3:
                        title = p.get("title", "")
                        if title:
                            category_titles[cat].append(title)

        total = len(self.papers)
        results: List[CategoryStatistic] = []
        for cat, all_cnt in all_counts.items():
            prim_cnt = primary_counts.get(cat, 0)
            pct = round((all_cnt / total) * 100.0, 2)
            results.append(
                CategoryStatistic(
                    category=cat,
                    primary_count=prim_cnt,
                    all_count=all_cnt,
                    percentage=pct,
                    sample_titles=category_titles[cat],
                )
            )

        results.sort(key=lambda x: (-x.all_count, -x.primary_count, x.category))
        return results

    def get_year_statistics(self) -> Dict[int, int]:
        """Compute publication counts by year."""
        year_counts: Counter = Counter()
        for p in self.papers:
            date_str = p.get("published_date") or ""
            if date_str:
                match = re.match(r"^(\d{4})", date_str)
                if match:
                    year_counts[int(match.group(1))] += 1

        return dict(sorted(year_counts.items()))

    def get_timeline(self, granularity: str = "yearly") -> List[TimelineEntry]:
        """Compute timeline entries aggregated yearly or monthly.

        Args:
            granularity (str): 'yearly' or 'monthly'.

        Returns:
            List[TimelineEntry]: Ordered chronological list of timeline entries.
        """
        timeline_buckets: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for p in self.papers:
            date_str = p.get("published_date") or ""
            if not date_str:
                continue

            match = re.match(r"^(\d{4})(?:-(\d{2}))?", date_str)
            if not match:
                continue

            year = int(match.group(1))
            month = int(match.group(2)) if match.group(2) else None

            if granularity == "monthly" and month:
                period_key = f"{year:04d}-{month:02d}"
            else:
                period_key = f"{year:04d}"

            timeline_buckets[period_key].append(p)

        results: List[TimelineEntry] = []
        for period_key in sorted(timeline_buckets.keys()):
            bucket = timeline_buckets[period_key]
            cats_counter: Counter = Counter()
            pids = []
            for p in bucket:
                pid = p.get("arxiv_id") or p.get("id")
                if pid:
                    pids.append(pid)
                prim = p.get("primary_category")
                if prim:
                    cats_counter[prim] += 1

            year_val = int(period_key[:4])
            month_val = int(period_key[5:7]) if len(period_key) >= 7 else None
            top_cats = [c for c, _ in cats_counter.most_common(3)]

            results.append(
                TimelineEntry(
                    period=period_key,
                    year=year_val,
                    month=month_val,
                    paper_count=len(bucket),
                    paper_ids=pids,
                    top_categories=top_cats,
                )
            )

        return results

    def get_author_statistics(self, top_n: int = 20) -> List[AuthorStatistic]:
        """Extract and rank top authors across the corpus."""
        author_papers: Dict[str, List[str]] = defaultdict(list)

        for p in self.papers:
            pid = p.get("arxiv_id") or p.get("id") or ""
            authors_data = p.get("authors") or []
            if isinstance(authors_data, str):
                author_names = [a.strip() for a in authors_data.split(",") if a.strip()]
            elif isinstance(authors_data, list):
                author_names = [str(a).strip() for a in authors_data if str(a).strip()]
            else:
                author_names = []

            for author_raw in author_names:
                norm_author = normalize_author_name(author_raw)
                if norm_author:
                    author_papers[norm_author].append(pid)

        results: List[AuthorStatistic] = []
        for author, pids in author_papers.items():
            results.append(
                AuthorStatistic(
                    author=author,
                    paper_count=len(pids),
                    paper_ids=sorted(list(set(pids))),
                )
            )

        results.sort(key=lambda x: (-x.paper_count, x.author))
        return results[:top_n]

    def get_corpus_summary(self) -> CorpusSummary:
        """Construct a high-level summary of the entire exploration corpus."""
        dates = []
        for p in self.papers:
            d = p.get("published_date")
            if d:
                dates.append(d)

        sorted_dates = sorted(dates)
        earliest = sorted_dates[0] if sorted_dates else "N/A"
        latest = sorted_dates[-1] if sorted_dates else "N/A"

        return CorpusSummary(
            total_papers=len(self.papers),
            earliest_date=earliest,
            latest_date=latest,
            category_distribution=self.get_category_statistics(),
            year_distribution=self.get_year_statistics(),
            top_concepts=self.get_concept_statistics(min_frequency=1)[:10],
            top_authors=self.get_author_statistics(top_n=10),
        )

    def build_exploration_graph(
        self,
        max_concepts: int = 25,
        max_papers: int = 50,
        min_cooccurrence: int = 1,
    ) -> ScientificExplorationGraph:
        """Construct a tripartite exploration graph (Categories, Concepts, Papers).

        Edges:
            - paper -> category (type: belongs_to_category, weight: 1.0)
            - paper -> concept (type: exhibits_concept, weight: 1.0)
            - concept <-> concept (type: concept_cooccurrence, weight: count)

        Args:
            max_concepts (int): Maximum top concepts to include in graph.
            max_papers (int): Maximum papers to include in graph.
            min_cooccurrence (int): Minimum co-occurrence threshold for concept edges.

        Returns:
            ScientificExplorationGraph: Structured nodes and edges.
        """
        nodes_dict: Dict[str, ScientificGraphNode] = {}
        edges: List[ScientificGraphEdge] = []

        # 1. Add Category Nodes
        category_stats = self.get_category_statistics()
        for cat_stat in category_stats:
            cat_id = f"cat_{cat_stat.category}"
            nodes_dict[cat_id] = ScientificGraphNode(
                id=cat_id,
                label=cat_stat.category,
                node_type="category",
                metadata={"all_count": cat_stat.all_count, "primary_count": cat_stat.primary_count},
            )

        # 2. Add Top Concept Nodes
        concept_stats = self.get_concept_statistics(min_frequency=1)[:max_concepts]
        top_concept_names = {cs.concept for cs in concept_stats}
        for cs in concept_stats:
            c_id = f"concept_{cs.concept}"
            nodes_dict[c_id] = ScientificGraphNode(
                id=c_id,
                label=cs.concept,
                node_type="concept",
                metadata={"frequency": cs.frequency, "categories": cs.categories},
            )

        # 3. Add Paper Nodes and Edges
        selected_papers = self.papers[:max_papers]
        for p in selected_papers:
            pid = p.get("arxiv_id") or p.get("id") or ""
            if not pid:
                continue
            p_node_id = f"paper_{pid}"
            title = p.get("title", pid)
            nodes_dict[p_node_id] = ScientificGraphNode(
                id=p_node_id,
                label=title[:40] + ("..." if len(title) > 40 else ""),
                node_type="paper",
                metadata={
                    "arxiv_id": pid,
                    "title": title,
                    "primary_category": p.get("primary_category", ""),
                    "published_date": p.get("published_date", ""),
                },
            )

            # Paper -> Category Edge
            prim_cat = p.get("primary_category")
            if prim_cat:
                cat_node_id = f"cat_{prim_cat}"
                if cat_node_id in nodes_dict:
                    edges.append(
                        ScientificGraphEdge(
                            source=p_node_id,
                            target=cat_node_id,
                            relation_type="belongs_to_category",
                            weight=1.0,
                        )
                    )

            # Paper -> Concept Edges
            for c in p.get("extracted_concepts", []):
                if c in top_concept_names:
                    c_node_id = f"concept_{c}"
                    if c_node_id in nodes_dict:
                        edges.append(
                            ScientificGraphEdge(
                                source=p_node_id,
                                target=c_node_id,
                                relation_type="exhibits_concept",
                                weight=1.0,
                            )
                        )

        # 4. Add Concept <-> Concept Co-occurrence Edges
        cooccurrences = self.get_concept_cooccurrence(min_count=min_cooccurrence)
        for co in cooccurrences:
            if co.concept_a in top_concept_names and co.concept_b in top_concept_names:
                cA_id = f"concept_{co.concept_a}"
                cB_id = f"concept_{co.concept_b}"
                if cA_id in nodes_dict and cB_id in nodes_dict:
                    edges.append(
                        ScientificGraphEdge(
                            source=cA_id,
                            target=cB_id,
                            relation_type="concept_cooccurrence",
                            weight=float(co.cooccurrence_count),
                        )
                    )

        # Deterministic sorting
        sorted_nodes = sorted(nodes_dict.values(), key=lambda n: (n.node_type, n.id))
        sorted_edges = sorted(edges, key=lambda e: (e.relation_type, e.source, e.target))

        return ScientificExplorationGraph(nodes=sorted_nodes, edges=sorted_edges)

    def find_related_papers(
        self,
        target: Union[str, Dict[str, Any]],
        retriever: ScientificRetriever,
        top_k: int = 5,
    ) -> List[RelatedPaperMatch]:
        """Discover semantically related papers using the existing ScientificRetriever.

        Reuses the existing embedding index and vector store without creating
        a second retrieval pipeline. Removes the source paper from recommendations.

        Args:
            target (Union[str, Dict[str, Any]]): arXiv ID or paper metadata dict.
            retriever (ScientificRetriever): Active scientific retriever instance.
            top_k (int): Number of related papers to return.

        Returns:
            List[RelatedPaperMatch]: Ranked list of related papers with shared concepts & categories.
        """
        if retriever is None:
            raise ValueError("retriever cannot be None.")

        # Resolve target paper
        target_paper: Optional[Dict[str, Any]] = None
        target_id = ""

        if isinstance(target, str):
            target_id = target.strip()
            # Lookup in corpus
            norm_id = re.sub(r"v\d+$", "", target_id)
            for p in self.papers:
                pid = p.get("arxiv_id") or p.get("id") or ""
                if pid == target_id or re.sub(r"v\d+$", "", pid) == norm_id:
                    target_paper = p
                    target_id = pid
                    break
        elif isinstance(target, dict):
            target_paper = target
            target_id = target_paper.get("arxiv_id") or target_paper.get("id") or ""

        if not target_paper:
            # Fallback search by string query if target is a general string
            query_str = target if isinstance(target, str) else ""
            if not query_str:
                return []
            raw_results = retriever.retrieve(query=query_str, k=top_k)
            return [
                RelatedPaperMatch(
                    paper=res,
                    similarity_score=res.score,
                    shared_concepts=res.concepts,
                    shared_categories=res.categories,
                )
                for res in raw_results
            ]

        # Construct contextual search query from title + abstract
        title = target_paper.get("title", "")
        abstract = target_paper.get("abstract", "")
        if not abstract and "page_content" in target_paper:
            abstract = target_paper["page_content"]
        search_query = f"{title} {abstract[:250]}".strip()

        # Retrieve top_k + 2 to account for self-match
        retrieval_results = retriever.retrieve(query=search_query, k=top_k + 2)

        target_norm_id = re.sub(r"v\d+$", "", target_id)
        target_concepts = set(self.get_concepts_for_paper(target_id))
        target_categories = set(target_paper.get("categories") or [])
        if isinstance(target_categories, str):
            target_categories = set(target_categories.split())
        if target_paper.get("primary_category"):
            target_categories.add(target_paper["primary_category"])

        matches: List[RelatedPaperMatch] = []
        for res in retrieval_results:
            res_norm_id = re.sub(r"v\d+$", "", res.arxiv_id)
            # Exclude source paper itself
            if res.arxiv_id == target_id or res_norm_id == target_norm_id:
                continue

            res_concepts = set(self.get_concepts_for_paper(res.arxiv_id))
            res_categories = set(res.categories)
            if res.primary_category:
                res_categories.add(res.primary_category)

            shared_concepts = sorted(list(target_concepts.intersection(res_concepts)))
            shared_categories = sorted(list(target_categories.intersection(res_categories)))

            matches.append(
                RelatedPaperMatch(
                    paper=res,
                    similarity_score=res.score,
                    shared_concepts=shared_concepts,
                    shared_categories=shared_categories,
                )
            )

            if len(matches) >= top_k:
                break

        return matches
