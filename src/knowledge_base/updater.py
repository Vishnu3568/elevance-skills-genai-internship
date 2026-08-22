from typing import Dict, Optional

import pandas as pd

try:
    from src.knowledge_base.audit import (
        DEFAULT_HISTORY_PATH,
        record_update,
    )
    from src.knowledge_base.ingestion import (
        DUPLICATE,
        INVALID,
        NEW,
        UPDATED,
        classify_updates,
    )
    from src.knowledge_base.store import (
        apply_updates,
        load_knowledge_base,
        save_knowledge_base,
    )
    from src.knowledge_base.vector_store import (
        add_documents_to_vector_store,
        create_knowledge_documents,
    )
    from src.langchain_helper import get_instructor_embeddings
except ImportError:
    from knowledge_base.audit import (
        DEFAULT_HISTORY_PATH,
        record_update,
    )
    from knowledge_base.ingestion import (
        DUPLICATE,
        INVALID,
        NEW,
        UPDATED,
        classify_updates,
    )
    from knowledge_base.store import (
        apply_updates,
        load_knowledge_base,
        save_knowledge_base,
    )
    from knowledge_base.vector_store import (
        add_documents_to_vector_store,
        create_knowledge_documents,
    )
    from langchain_helper import get_instructor_embeddings
try:
    from langchain_community.vectorstores import FAISS
except ImportError:
    from langchain.vectorstores import FAISS


def update_knowledge_base(
    knowledge_base_path: str,
    update_source_path: str,
    vector_store_path: str,
    history_path: Optional[str] = DEFAULT_HISTORY_PATH,
) -> Dict[str, int]:
    """
    Process incoming knowledge and update the managed
    knowledge base and FAISS index.

    Incremental FAISS updates are currently supported
    for NEW records.
    """

    knowledge_base = load_knowledge_base(
        knowledge_base_path
    )

    incoming = pd.read_csv(
        update_source_path,
        encoding="latin1",
    )

    classified = classify_updates(
        knowledge_base,
        incoming,
    )

    summary = {
        NEW: 0,
        UPDATED: 0,
        DUPLICATE: 0,
        INVALID: 0,
    }

    for item in classified:
        summary[item["status"]] += 1

    # Prevent stale vectors from being silently introduced.
    if summary[UPDATED] > 0:
        raise ValueError(
            "UPDATED records require a vector-store rebuild. "
            "Incremental update aborted."
        )

    updated_knowledge_base = apply_updates(
        knowledge_base,
        classified,
    )

    new_records = [
        {
            "prompt": item["prompt"],
            "response": item["response"],
            "row": len(knowledge_base) + index,
        }
        for index, item in enumerate(classified)
        if item["status"] == NEW
    ]

    if new_records:
        embeddings = get_instructor_embeddings()

        vector_store = FAISS.load_local(
            vector_store_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )

        documents = create_knowledge_documents(
            new_records
        )

        add_documents_to_vector_store(
            vector_store,
            documents,
        )

        vector_store.save_local(
            vector_store_path
        )

    # Persist the managed CSV only after vector store update succeeds
    save_knowledge_base(
        updated_knowledge_base,
        knowledge_base_path,
    )

    result_summary = {
        "existing_records": len(knowledge_base),
        "incoming_records": len(incoming),
        "final_records": len(updated_knowledge_base),
        "new": summary[NEW],
        "updated": summary[UPDATED],
        "duplicate": summary[DUPLICATE],
        "invalid": summary[INVALID],
    }

    # Record persistent audit entry only after FAISS and CSV persistence succeed
    if history_path:
        record_update(
            history_path=history_path,
            update_summary=result_summary,
            source=update_source_path,
            status="SUCCESS",
        )

    return result_summary
