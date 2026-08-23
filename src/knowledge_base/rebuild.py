"""Rebuild orchestration module for Customer Service Chatbot Knowledge Base.

Provides safe, transactional full rebuild of the Knowledge Base CSV and FAISS
vector store from clean ground-truth data or incoming update batches containing
UPDATED FAQ records.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    # pyrefly: ignore [missing-import]
    from src.knowledge_base.store import (  # type: ignore
        load_knowledge_base,
        apply_updates,
        save_knowledge_base,
    )
    # pyrefly: ignore [missing-import]
    from src.knowledge_base.ingestion import (  # type: ignore
        load_knowledge_csv,
        classify_updates,
        NEW,
        UPDATED,
        DUPLICATE,
        INVALID,
    )
    # pyrefly: ignore [missing-import]
    from src.knowledge_base.vector_store import create_knowledge_documents  # type: ignore
    # pyrefly: ignore [missing-import]
    from src.knowledge_base.audit import DEFAULT_HISTORY_PATH, record_update  # type: ignore
    # pyrefly: ignore [missing-import]
    from src.langchain_helper import get_instructor_embeddings  # type: ignore
except ImportError:
    # pyrefly: ignore [missing-import]
    from knowledge_base.store import (  # type: ignore
        load_knowledge_base,
        apply_updates,
        save_knowledge_base,
    )
    # pyrefly: ignore [missing-import]
    from knowledge_base.ingestion import (  # type: ignore
        load_knowledge_csv,
        classify_updates,
        NEW,
        UPDATED,
        DUPLICATE,
        INVALID,
    )
    # pyrefly: ignore [missing-import]
    from knowledge_base.vector_store import create_knowledge_documents  # type: ignore
    # pyrefly: ignore [missing-import]
    from knowledge_base.audit import DEFAULT_HISTORY_PATH, record_update  # type: ignore
    # pyrefly: ignore [missing-import]
    from langchain_helper import get_instructor_embeddings  # type: ignore

try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    from langchain.vectorstores import FAISS

DEFAULT_KNOWLEDGE_BASE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "dataset",
    "knowledge_base.csv",
)

DEFAULT_VECTOR_STORE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "faiss_index",
)


def rebuild_knowledge_base(
    knowledge_base_path: str = DEFAULT_KNOWLEDGE_BASE_PATH,
    update_source_path: Optional[str] = None,
    vector_store_path: str = DEFAULT_VECTOR_STORE_PATH,
    history_path: str = DEFAULT_HISTORY_PATH,
    backup_existing: bool = True,
) -> Dict[str, Any]:
    """Perform a safe, transactional full rebuild of the Knowledge Base and FAISS store.

    Supports pure rebuilds (rebuilding from existing knowledge_base.csv) or
    merge-rebuilds (merging updates, including UPDATED records, and rebuilding).

    Args:
        knowledge_base_path (str): Path to active knowledge_base.csv.
        update_source_path (Optional[str]): Path to incoming update CSV, or None for pure rebuild.
        vector_store_path (str): Path to FAISS directory.
        history_path (str): Path to update history .jsonl.
        backup_existing (bool): Whether to create rollback backup during directory swap.

    Returns:
        Dict[str, Any]: Summary dictionary containing execution metrics.

    Raises:
        FileNotFoundError: If knowledge_base_path or update_source_path is missing.
        ValueError: If validation fails or record counts mismatch.
    """
    source_identifier = update_source_path or knowledge_base_path
    temp_stage_dir = None

    try:
        # 1. Load active baseline knowledge base
        existing_df = load_knowledge_base(knowledge_base_path)
        existing_count = len(existing_df)

        # 2. Ingest and classify incoming records if provided
        if update_source_path is not None:
            if not os.path.exists(update_source_path):
                raise FileNotFoundError(f"Update source not found: {update_source_path}")

            incoming_df = load_knowledge_csv(update_source_path)
            incoming_count = len(incoming_df)

            classified = classify_updates(existing_df, incoming_df)
            merged_df = apply_updates(existing_df, classified)

            new_count = sum(1 for c in classified if c["status"] == NEW)
            updated_count = sum(1 for c in classified if c["status"] == UPDATED)
            duplicate_count = sum(1 for c in classified if c["status"] == DUPLICATE)
            invalid_count = sum(1 for c in classified if c["status"] == INVALID)
        else:
            incoming_count = 0
            merged_df = existing_df.copy()
            new_count = 0
            updated_count = 0
            duplicate_count = 0
            invalid_count = 0

        final_count = len(merged_df)
        if final_count == 0:
            raise ValueError("Cannot rebuild knowledge base with 0 records.")

        # 3. Create LangChain Documents for all merged records
        records: List[Dict[str, Any]] = []
        for idx, row in merged_df.iterrows():
            records.append({
                "prompt": row["prompt"],
                "response": row["response"],
                "row": idx,
            })
        documents = create_knowledge_documents(records)

        # 4. Build fresh FAISS store in temporary staging directory
        temp_stage_dir = tempfile.mkdtemp(prefix="faiss_rebuild_stage_")
        staged_faiss_path = os.path.join(temp_stage_dir, "faiss_staged")

        embeddings = get_instructor_embeddings()
        fresh_db = FAISS.from_documents(documents, embeddings)
        fresh_db.save_local(staged_faiss_path)

        # 5. Validate staged FAISS index before touching active state
        loaded_staged_db = FAISS.load_local(
            staged_faiss_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        staged_ntotal = loaded_staged_db.index.ntotal
        staged_doc_count = len(loaded_staged_db.docstore._dict)

        if staged_ntotal != final_count or staged_doc_count != final_count:
            raise ValueError(
                f"Staged FAISS validation failed: expected {final_count} records, "
                f"got ntotal={staged_ntotal}, docstore={staged_doc_count}."
            )

        # 6. Atomic swap / commit
        target_faiss = Path(vector_store_path)
        backup_faiss = target_faiss.with_name(target_faiss.name + "_rebuild_backup")
        had_existing_faiss = target_faiss.exists()

        if had_existing_faiss and backup_existing:
            if backup_faiss.exists():
                shutil.rmtree(backup_faiss, ignore_errors=True)
            shutil.move(str(target_faiss), str(backup_faiss))

        try:
            # Overwrite active CSV
            save_knowledge_base(merged_df, knowledge_base_path)

            # Move staged FAISS to target path
            shutil.copytree(str(staged_faiss_path), str(target_faiss))

            # Cleanup backup if successful
            if had_existing_faiss and backup_existing and backup_faiss.exists():
                shutil.rmtree(backup_faiss, ignore_errors=True)
        except Exception as swap_error:
            # Rollback FAISS directory on swap failure
            if had_existing_faiss and backup_existing and backup_faiss.exists():
                if target_faiss.exists():
                    shutil.rmtree(target_faiss, ignore_errors=True)
                shutil.move(str(backup_faiss), str(target_faiss))
            raise swap_error

        # 7. Record persistent REBUILD_SUCCESS audit entry
        summary = {
            "existing_records": existing_count,
            "incoming_records": incoming_count,
            "final_records": final_count,
            "new": new_count,
            "updated": updated_count,
            "duplicate": duplicate_count,
            "invalid": invalid_count,
        }

        record_update(
            history_path=history_path,
            update_summary=summary,
            source=source_identifier,
            status="REBUILD_SUCCESS",
        )

        return {
            "existing_records": existing_count,
            "incoming_records": incoming_count,
            "merged_records": final_count,
            "updated_records": updated_count,
            "new_records": new_count,
            "duplicate_records": duplicate_count,
            "invalid_records": invalid_count,
            "final_records": final_count,
            "status": "REBUILD_SUCCESS",
        }

    except Exception as exc:
        record_update(
            history_path=history_path,
            source=source_identifier,
            status="REBUILD_FAILED",
            error=str(exc),
        )
        raise exc

    finally:
        if temp_stage_dir and os.path.exists(temp_stage_dir):
            shutil.rmtree(temp_stage_dir, ignore_errors=True)
