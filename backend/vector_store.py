"""RAG retrieval: embed CV chunks locally and store/query them in ChromaDB."""

from langchain_chroma import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings

import config

_embeddings = None


def get_embeddings() -> FastEmbedEmbeddings:
    """Build the local embedding model once and reuse it."""
    global _embeddings
    if _embeddings is None:
        _embeddings = FastEmbedEmbeddings(model_name=config.EMBEDDING_MODEL)
    return _embeddings


def build_cv_vector_store(chunks: list) -> Chroma:
    """Store the CV chunks in a fresh in-memory collection (one per upload)."""
    return Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name="cv_collection",
    )


def get_relevant_cv_text(vector_store: Chroma, query: str, k: int = 8) -> str:
    """Return the k most relevant CV chunks joined into one context string."""
    results = vector_store.similarity_search(query, k=k)
    return "\n\n---\n\n".join(doc.page_content for doc in results)
