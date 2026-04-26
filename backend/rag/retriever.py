"""
rag/retriever.py
────────────────
ChromaDB-backed vector store for PubMed paper retrieval.

Architecture:
  Query symptoms → BioMedBERT embedding → ChromaDB cosine search → Top-K papers

Why ChromaDB?
  • Runs embedded (no separate server needed for dev)
  • Supports persistent storage
  • Easy swap to Pinecone/Weaviate for production
"""

import os
from typing import Optional

import chromadb
from chromadb.config import Settings
from loguru import logger

from .embedder import get_embedder


class VectorStore:
    """Manages ChromaDB collection for PubMed paper storage and retrieval."""

    def __init__(self):
        self.persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
        self.collection_name = os.getenv("CHROMA_COLLECTION_NAME", "pubmed_papers")
        self.embedder = get_embedder()
        self._client = None
        self._collection = None
        self._init_db()

    def _init_db(self):
        """Initialize ChromaDB client and collection."""
        try:
            self._client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=Settings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},  # Cosine similarity
            )
            logger.info(
                f"ChromaDB initialized — collection '{self.collection_name}' "
                f"has {self._collection.count()} documents"
            )
        except Exception as e:
            logger.error(f"ChromaDB init failed: {e}")
            raise

    # ── Write Operations ──────────────────────────────────────────────────────

    def add_documents(
        self,
        documents: list[str],
        metadatas: list[dict],
        ids: list[str],
    ) -> None:
        """
        Embed and add documents to the collection.
        Skips documents with duplicate IDs automatically.
        """
        if not documents:
            return

        # Filter out already-existing IDs to avoid duplicates
        existing_ids = set(self._collection.get(ids=ids)["ids"])
        new_indices = [i for i, id_ in enumerate(ids) if id_ not in existing_ids]

        if not new_indices:
            logger.debug("All documents already exist — skipping.")
            return

        new_docs = [documents[i] for i in new_indices]
        new_metas = [metadatas[i] for i in new_indices]
        new_ids = [ids[i] for i in new_indices]

        # Generate embeddings in batch
        embeddings = [self.embedder.embed_document(doc) for doc in new_docs]

        self._collection.add(
            documents=new_docs,
            embeddings=embeddings,
            metadatas=new_metas,
            ids=new_ids,
        )
        logger.debug(f"Added {len(new_docs)} new documents to ChromaDB")

    # ── Read Operations ───────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int = 8,
        similarity_threshold: float = 0.60,
        where_filter: Optional[dict] = None,
    ) -> list[dict]:
        """
        Retrieve top-K most relevant papers for a clinical query.

        Args:
            query: Clinical query string (symptoms, chief complaint, etc.)
            top_k: Number of papers to return
            similarity_threshold: Minimum cosine similarity (0-1)
            where_filter: Optional ChromaDB metadata filter

        Returns:
            List of paper dicts with metadata + relevance scores
        """
        if self._collection.count() == 0:
            logger.warning("Vector store is empty — run ingestion first.")
            return []

        # Embed the query with medical-aware prefix
        query_embedding = self.embedder.embed_query(query)

        # Query ChromaDB
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k * 2, self._collection.count()),  # Over-fetch then filter
            "include": ["documents", "metadatas", "distances"],
        }
        if where_filter:
            kwargs["where"] = where_filter

        results = self._collection.query(**kwargs)

        # Convert distances to similarity scores (ChromaDB cosine: 0=identical, 2=opposite)
        papers = []
        for i, (doc, meta, dist) in enumerate(zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )):
            similarity = 1 - (dist / 2)  # Convert cosine distance to similarity

            if similarity < similarity_threshold:
                continue

            papers.append({
                "rank": i + 1,
                "title": meta.get("title", ""),
                "abstract_snippet": meta.get("abstract_snippet", doc[:300]),
                "journal": meta.get("journal", ""),
                "year": meta.get("year", ""),
                "authors": meta.get("authors", ""),
                "pmid": meta.get("pmid", ""),
                "pubmed_url": meta.get("pubmed_url", ""),
                "relevance_score": round(similarity, 4),
                "full_text": doc,
            })

        # Return top_k after filtering
        return papers[:top_k]

    def retrieve_for_diagnosis(self, patient_context: str, condition: str) -> list[dict]:
        """
        Specialized retrieval combining patient context + specific condition.
        Used during reasoning trace generation.
        """
        # Combine query for better specificity
        combined_query = f"{patient_context} {condition} diagnosis treatment guidelines"
        return self.retrieve(combined_query, top_k=5)

    def get_collection_size(self) -> int:
        """Return current number of documents in the collection."""
        try:
            return self._collection.count()
        except Exception:
            return 0

    def clear_collection(self) -> None:
        """Delete and recreate the collection (for testing/reset)."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("Collection cleared.")
