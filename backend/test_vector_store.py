import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langchain_core.documents import Document as LCDoc

from vector_store import build_cv_vector_store, get_relevant_cv_text


def test_uploads_do_not_leak_across_collections():
    """Two separate uploads must not retrieve each other's CV text (privacy)."""
    s1 = build_cv_vector_store(
        [LCDoc(page_content="Alice Zorbax is a Python backend developer with FastAPI experience.")]
    )
    s2 = build_cv_vector_store(
        [LCDoc(page_content="Bob Quibble is a digital marketing manager focused on SEO.")]
    )
    leaked = get_relevant_cv_text(s2, "Python FastAPI backend developer", k=3)
    assert "Alice" not in leaked and "FastAPI" not in leaked, f"CV leaked across uploads: {leaked!r}"
