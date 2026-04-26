"""
rag/embedder.py
───────────────
Medical-domain embedding using BioMedBERT.
Falls back to OpenAI embeddings if model can't load locally.

BioMedBERT is pre-trained on PubMed abstracts + full-text articles,
making it far more accurate than generic embeddings for clinical text.
"""

import os
from functools import lru_cache
from typing import Union

import numpy as np
from loguru import logger


class MedicalEmbedder:
    """
    Wraps sentence-transformers BioMedBERT for medical text embeddings.

    Why BioMedBERT over text-embedding-ada-002?
    ─────────────────────────────────────────────
    • Trained on 21M PubMed abstracts — understands clinical jargon
    • "myocardial infarction" ≈ "heart attack" in embedding space
    • Free, runs locally, no API cost per query
    • Proven on BioASQ, MedQA benchmarks
    """

    BIOMEDBERT_MODEL = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    FALLBACK_MODEL = "all-MiniLM-L6-v2"  # Fast fallback

    def __init__(self):
        self.model_name = os.getenv("EMBEDDING_MODEL", self.BIOMEDBERT_MODEL)
        self._model = None
        self._load_model()

    def _load_model(self):
        """Lazy-load sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.model_name}")
            self._model = SentenceTransformer(self.model_name)
            logger.success(f"Embedding model loaded — dim={self._model.get_sentence_embedding_dimension()}")
        except Exception as e:
            logger.warning(f"Failed to load {self.model_name}: {e}. Trying fallback...")
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.FALLBACK_MODEL)
                self.model_name = self.FALLBACK_MODEL
                logger.success("Fallback embedding model loaded.")
            except Exception as e2:
                logger.error(f"Could not load any embedding model: {e2}")
                self._model = None

    def embed(self, texts: Union[str, list[str]]) -> list[list[float]]:
        """
        Embed one or more texts.

        Args:
            texts: Single string or list of strings.

        Returns:
            List of embedding vectors (list of floats).
        """
        if isinstance(texts, str):
            texts = [texts]

        if self._model is None:
            # Last resort: use OpenAI embeddings
            return self._openai_embed(texts)

        vectors = self._model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True,  # Cosine similarity ready
        )
        return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string — optimized for retrieval."""
        # Prefix helps BioMedBERT understand query intent
        prefixed = f"clinical query: {query}"
        return self.embed(prefixed)[0]

    def embed_document(self, doc: str) -> list[float]:
        """Embed a document chunk for indexing."""
        prefixed = f"medical document: {doc}"
        return self.embed(prefixed)[0]

    def _openai_embed(self, texts: list[str]) -> list[list[float]]:
        """Fallback: OpenAI text-embedding-3-small."""
        from openai import OpenAI
        client = OpenAI()
        response = client.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
        )
        return [item.embedding for item in response.data]

    @property
    def dimension(self) -> int:
        if self._model:
            return self._model.get_sentence_embedding_dimension()
        return 1536  # OpenAI fallback dimension


@lru_cache(maxsize=1)
def get_embedder() -> MedicalEmbedder:
    """Singleton embedder — expensive to load, share across requests."""
    return MedicalEmbedder()
