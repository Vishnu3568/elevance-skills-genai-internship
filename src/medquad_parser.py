"""MedQuAD XML Parser Module for Medical Q&A Pipeline.

Parses raw MedQuAD XML files into normalized MedicalQARecord dataclass objects,
filtering out records with missing answers and supporting schema variations across all NIH subsets.
"""

import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class MedicalQARecord:
    """Normalized representation of a MedQuAD Question-Answer record."""
    document_id: str
    question_id: str
    source: str
    source_url: str
    focus: str
    synonyms: List[str] = field(default_factory=list)
    cuis: List[str] = field(default_factory=list)
    semantic_types: List[str] = field(default_factory=list)
    semantic_group: str = ""
    question: str = ""
    question_type: str = ""
    answer: str = ""

    def to_embedding_text(self) -> str:
        """Format a rich textual string for vector embedding creation."""
        synonyms_str = ", ".join(self.synonyms) if self.synonyms else "None"
        cuis_str = ", ".join(self.cuis) if self.cuis else "None"
        sem_types_str = ", ".join(self.semantic_types) if self.semantic_types else "None"
        return (
            f"Medical Topic: {self.focus}\n"
            f"Synonyms: {synonyms_str}\n"
            f"UMLS CUIs: {cuis_str}\n"
            f"Semantic Types: {sem_types_str}\n"
            f"Question Type: {self.question_type}\n\n"
            f"Question:\n{self.question}\n\n"
            f"Answer:\n{self.answer}"
        )

    def to_metadata(self) -> Dict[str, Any]:
        """Format metadata dictionary for vector store Document metadata."""
        return {
            "document_id": self.document_id,
            "question_id": self.question_id,
            "source": self.source,
            "source_url": self.source_url,
            "focus": self.focus,
            "synonyms": ", ".join(self.synonyms),
            "cuis": ", ".join(self.cuis),
            "semantic_types": ", ".join(self.semantic_types),
            "semantic_group": self.semantic_group,
            "question": self.question,
            "question_type": self.question_type,
            "domain": "medical"
        }


def _find_tag(element: Optional[ET.Element], tag_names: List[str]) -> Optional[ET.Element]:
    """Find child element matching any tag name in a case-insensitive list."""
    if element is None:
        return None
    targets = {t.strip().lower() for t in tag_names}
    for child in element:
        if child.tag.strip().lower() in targets:
            return child
    return None


def _findall_tags(element: Optional[ET.Element], tag_names: List[str]) -> List[ET.Element]:
    """Find all child elements matching any tag name in a case-insensitive list."""
    if element is None:
        return []
    targets = {t.strip().lower() for t in tag_names}
    matches = []
    for child in element:
        if child.tag.strip().lower() in targets:
            matches.append(child)
    return matches


def parse_medquad_xml_file(file_path: str) -> List[MedicalQARecord]:
    """Parse a single MedQuAD XML file into valid, answerable MedicalQARecord objects.

    Supports dataset variations: multi-CUI, multi-SemanticType, CDC direct UMLS,
    NINDS lowercase tags, missing FocusAnnotations, and missing Focus.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"MedQuAD XML file not found at {file_path}")

    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
    except Exception as err:
        raise ValueError(f"Failed to parse XML file {file_path}: {err}")

    # Root metadata
    doc_id = root.attrib.get("id", "")
    source = root.attrib.get("source", "")
    source_url = root.attrib.get("url", "")

    # Focus resolution (handles Focus, doctitle-focus, title)
    focus_elem = _find_tag(root, ["Focus", "doctitle-focus", "title"])
    focus = focus_elem.text.strip() if focus_elem is not None and focus_elem.text else ""

    synonyms: List[str] = []
    cuis: List[str] = []
    semantic_types: List[str] = []
    semantic_group = ""

    # FocusAnnotations & UMLS Resolution
    focus_annotations = _find_tag(root, ["FocusAnnotations"])
    umls_elem = None

    if focus_annotations is not None:
        umls_elem = _find_tag(focus_annotations, ["UMLS", "umls"])
        # Synonyms
        syns_elem = _find_tag(focus_annotations, ["Synonyms", "synonyms"])
        if syns_elem is not None:
            for syn_item in _findall_tags(syns_elem, ["Synonym", "synonym"]):
                if syn_item.text and syn_item.text.strip():
                    synonyms.append(syn_item.text.strip())

    # Fallback to direct UMLS under root (CDC & NINDS variants)
    if umls_elem is None:
        umls_elem = _find_tag(root, ["UMLS", "umls"])

    if umls_elem is not None:
        # Multi-CUI extraction
        cuis_elem = _find_tag(umls_elem, ["CUIs", "cuis"])
        if cuis_elem is not None:
            for cui_item in _findall_tags(cuis_elem, ["CUI", "cui"]):
                if cui_item.text and cui_item.text.strip():
                    cuis.append(cui_item.text.strip())
        else:
            # Fallback for direct <CUI> under <UMLS>
            for cui_item in _findall_tags(umls_elem, ["CUI", "cui"]):
                if cui_item.text and cui_item.text.strip():
                    cuis.append(cui_item.text.strip())

        # Multi-SemanticType extraction
        sem_types_elem = _find_tag(umls_elem, ["SemanticTypes", "semanticTypes", "semantictypes"])
        if sem_types_elem is not None:
            for st_elem in _findall_tags(sem_types_elem, ["SemanticType", "semanticType", "semantictype"]):
                if st_elem.text and st_elem.text.strip():
                    semantic_types.append(st_elem.text.strip())
        else:
            # Fallback for direct <SemanticType> under <UMLS>
            for st_elem in _findall_tags(umls_elem, ["SemanticType", "semanticType", "semantictype"]):
                if st_elem.text and st_elem.text.strip():
                    semantic_types.append(st_elem.text.strip())

        # SemanticGroup extraction
        sem_grp_elem = _find_tag(umls_elem, ["SemanticGroup", "semanticGroup", "semanticgroup"])
        if sem_grp_elem is not None and sem_grp_elem.text and sem_grp_elem.text.strip():
            semantic_group = sem_grp_elem.text.strip()

    # QA Pairs Extraction
    records: List[MedicalQARecord] = []
    qa_pairs_elem = _find_tag(root, ["QAPairs", "qaPairs", "qapairs"])
    if qa_pairs_elem is not None:
        for qa_pair in _findall_tags(qa_pairs_elem, ["QAPair", "qaPair", "qapair"]):
            q_elem = _find_tag(qa_pair, ["Question", "question"])
            a_elem = _find_tag(qa_pair, ["Answer", "answer"])

            question_text = q_elem.text.strip() if q_elem is not None and q_elem.text else ""
            answer_text = a_elem.text.strip() if a_elem is not None and a_elem.text else ""

            # Filter out records where answer is missing or empty
            if not answer_text or answer_text.lower() in ("answer unavailable", "n/a", "none"):
                continue

            q_id = q_elem.attrib.get("qid", "") if q_elem is not None else ""
            q_type = q_elem.attrib.get("qtype", "") if q_elem is not None else ""

            record = MedicalQARecord(
                document_id=doc_id,
                question_id=q_id,
                source=source,
                source_url=source_url,
                focus=focus,
                synonyms=synonyms,
                cuis=cuis,
                semantic_types=semantic_types,
                semantic_group=semantic_group,
                question=question_text,
                question_type=q_type,
                answer=answer_text
            )
            records.append(record)

    return records


def parse_medquad_directory(dir_path: str) -> List[MedicalQARecord]:
    """Recursively parse all XML files in a MedQuAD dataset directory."""
    if not os.path.exists(dir_path):
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    all_records: List[MedicalQARecord] = []

    for root_dir, _, files in os.walk(dir_path):
        for file_name in sorted(files):
            if file_name.endswith(".xml"):
                full_path = os.path.join(root_dir, file_name)
                try:
                    records = parse_medquad_xml_file(full_path)
                    all_records.extend(records)
                except Exception as err:
                    print(f"Warning: Skipping {full_path} due to error: {err}")

    return all_records


if __name__ == "__main__":
    print("Robust MedQuAD XML Parser module initialized cleanly.")
