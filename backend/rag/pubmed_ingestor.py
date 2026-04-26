"""
rag/pubmed_ingestor.py
──────────────────────
Fetches papers from PubMed API, parses them, and ingests into ChromaDB.

Usage:
    ingestor = PubMedIngestor()
    await ingestor.ingest_topics(MEDICAL_TOPICS)
"""

import asyncio
import hashlib
import os
import time
from typing import Optional

import aiohttp
from Bio import Entrez
from loguru import logger

from .embedder import MedicalEmbedder
from .retriever import VectorStore


# ─── Medical topics to pre-index ─────────────────────────────────────────────
MEDICAL_TOPICS = [
    "pneumonia diagnosis treatment",
    "pulmonary embolism diagnosis",
    "myocardial infarction clinical presentation",
    "sepsis diagnostic criteria",
    "appendicitis diagnosis imaging",
    "meningitis symptoms diagnosis",
    "deep vein thrombosis diagnosis",
    "stroke acute management",
    "diabetic ketoacidosis treatment",
    "heart failure diagnosis criteria",
    "urinary tract infection diagnosis",
    "acute kidney injury causes",
    "liver failure diagnosis",
    "anemia differential diagnosis",
    "thyroid disorders clinical features",
    "hypertensive emergency management",
    "asthma exacerbation treatment",
    "COPD exacerbation management",
    "gastrointestinal bleeding causes",
    "pancreatitis diagnosis severity",
]


class PubMedIngestor:
    """Fetches PubMed abstracts and ingests them into the vector store."""

    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(self):
        self.api_key = os.getenv("PUBMED_API_KEY", "")
        self.email = os.getenv("PUBMED_EMAIL", "research@medmind.ai")
        self.embedder = MedicalEmbedder()
        self.vector_store = VectorStore()

        # Configure Biopython Entrez
        Entrez.email = self.email
        if self.api_key:
            Entrez.api_key = self.api_key

    # ── Public API ────────────────────────────────────────────────────────────

    async def ingest_topics(
        self,
        topics: list[str] = MEDICAL_TOPICS,
        papers_per_topic: int = 50,
        force_reingest: bool = False,
    ) -> dict:
        """Main entry point: ingest papers for a list of medical topics."""

        if not force_reingest:
            existing = self.vector_store.get_collection_size()
            if existing > 500:
                logger.info(f"Vector store already has {existing} docs — skipping ingest.")
                return {"skipped": True, "existing_docs": existing}

        logger.info(f"Starting PubMed ingest for {len(topics)} topics...")
        total_ingested = 0

        async with aiohttp.ClientSession() as session:
            for topic in topics:
                try:
                    pmids = await self._search_pmids(session, topic, max_results=papers_per_topic)
                    papers = await self._fetch_abstracts(session, pmids)
                    count = self._ingest_papers(papers, topic)
                    total_ingested += count
                    logger.info(f"  ✓ '{topic}': {count} papers ingested")
                    await asyncio.sleep(0.34)  # Respect NCBI rate limits (3 req/s)
                except Exception as e:
                    logger.error(f"  ✗ Failed topic '{topic}': {e}")

        logger.success(f"Ingest complete. Total: {total_ingested} papers.")
        return {
            "papers_ingested": total_ingested,
            "collection_size": self.vector_store.get_collection_size(),
            "topics_covered": topics,
        }

    async def ingest_single_paper(self, pmid: str) -> bool:
        """Fetch and ingest a single paper by PMID."""
        async with aiohttp.ClientSession() as session:
            papers = await self._fetch_abstracts(session, [pmid])
            if papers:
                self._ingest_papers(papers, topic="manual")
                return True
        return False

    # ── Internal Methods ──────────────────────────────────────────────────────

    async def _search_pmids(
        self,
        session: aiohttp.ClientSession,
        query: str,
        max_results: int = 50,
    ) -> list[str]:
        """Search PubMed and return list of PMIDs."""
        params = {
            "db": "pubmed",
            "term": f"{query}[Title/Abstract] AND (\"clinical trial\"[PT] OR \"review\"[PT] OR \"guidelines\"[PT])",
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance",
            "datetype": "pdat",
            "mindate": "2015",   # Focus on recent literature
            "maxdate": "2024",
        }
        if self.api_key:
            params["api_key"] = self.api_key

        url = f"{self.BASE_URL}/esearch.fcgi"
        async with session.get(url, params=params) as resp:
            data = await resp.json()
            return data.get("esearchresult", {}).get("idlist", [])

    async def _fetch_abstracts(
        self,
        session: aiohttp.ClientSession,
        pmids: list[str],
        batch_size: int = 20,
    ) -> list[dict]:
        """Fetch abstract details for a list of PMIDs in batches."""
        all_papers = []

        for i in range(0, len(pmids), batch_size):
            batch = pmids[i : i + batch_size]
            params = {
                "db": "pubmed",
                "id": ",".join(batch),
                "retmode": "xml",
                "rettype": "abstract",
            }
            if self.api_key:
                params["api_key"] = self.api_key

            url = f"{self.BASE_URL}/efetch.fcgi"
            async with session.get(url, params=params) as resp:
                xml_text = await resp.text()
                papers = self._parse_xml(xml_text, batch)
                all_papers.extend(papers)
                await asyncio.sleep(0.1)

        return all_papers

    def _parse_xml(self, xml_text: str, pmids: list[str]) -> list[dict]:
        """Parse PubMed XML response into structured dicts."""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(xml_text, "lxml-xml")
        papers = []

        for article in soup.find_all("PubmedArticle"):
            try:
                pmid_tag = article.find("PMID")
                pmid = pmid_tag.text if pmid_tag else ""

                title_tag = article.find("ArticleTitle")
                title = title_tag.get_text(strip=True) if title_tag else "No title"

                abstract_tag = article.find("AbstractText")
                if not abstract_tag:
                    continue  # Skip papers without abstracts
                abstract = abstract_tag.get_text(separator=" ", strip=True)

                journal_tag = article.find("Title")  # Journal title
                journal = journal_tag.text if journal_tag else "Unknown Journal"

                year_tag = article.find("PubDate")
                year = ""
                if year_tag:
                    year_inner = year_tag.find("Year")
                    year = year_inner.text if year_inner else ""

                # Author list
                authors = []
                for author in article.find_all("Author")[:3]:
                    last = author.find("LastName")
                    if last:
                        authors.append(last.text)
                author_str = ", ".join(authors) + (" et al." if len(authors) >= 3 else "")

                papers.append({
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract,
                    "journal": journal,
                    "year": year,
                    "authors": author_str,
                    "pubmed_url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                })
            except Exception as e:
                logger.warning(f"Could not parse article: {e}")
                continue

        return papers

    def _ingest_papers(self, papers: list[dict], topic: str) -> int:
        """Embed papers and add to ChromaDB vector store."""
        if not papers:
            return 0

        documents = []
        metadatas = []
        ids = []

        for paper in papers:
            # Combine title + abstract for richer embedding
            full_text = f"TITLE: {paper['title']}\n\nABSTRACT: {paper['abstract']}"

            # Deduplicate by PMID
            doc_id = hashlib.md5(paper["pmid"].encode()).hexdigest()

            documents.append(full_text)
            metadatas.append({
                "pmid": paper["pmid"],
                "title": paper["title"],
                "journal": paper["journal"],
                "year": paper.get("year", ""),
                "authors": paper.get("authors", ""),
                "pubmed_url": paper["pubmed_url"],
                "topic": topic,
                "abstract_snippet": paper["abstract"][:400],
            })
            ids.append(doc_id)

        # Batch add to vector store
        self.vector_store.add_documents(documents, metadatas, ids)
        return len(documents)
