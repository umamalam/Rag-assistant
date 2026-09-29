"""
store.py
Shared access to the embedding model and the ChromaDB collection.
Everything is lazy so importing this module is cheap and testable.
"""

import os

import chromadb
from chromadb.config import Settings

DB_DIR = os.environ.get("DB_DIR", "../data/chroma")
# v2: created with cosine distance. The old "knowledge_base" collection used
# Chroma's default L2 distance, which made the relevance scores misleading.
COLLECTION_NAME = "knowledge_base_v2"
EMBED_MODEL = os.environ.get("EMBED_MODEL", "all-MiniLM-L6-v2")

_embedder = None
_client = None


def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer(EMBED_MODEL)  # runs locally
    return _embedder


def embed(texts):
    return get_embedder().encode(
        texts, show_progress_bar=False, normalize_embeddings=True
    ).tolist()


def get_client():
    global _client
    if _client is None:
        # telemetry off: nothing about your data or usage leaves the machine
        _client = chromadb.PersistentClient(
            path=DB_DIR, settings=Settings(anonymized_telemetry=False)
        )
    return _client


def get_collection():
    return get_client().get_or_create_collection(
        COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )
