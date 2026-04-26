"""
main.py
────────
FastAPI application entry point.

Endpoints:
  POST /diagnose          → Run full diagnosis pipeline
  GET  /health            → System health check
  POST /ingest            → Trigger PubMed paper ingestion
  GET  /papers/search     → Search vector store directly
  POST /feedback          → Submit clinician feedback on a diagnosis
"""

import os
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel

load_dotenv()

from models.schemas import PatientInput, DiagnosisResponse, HealthResponse, IngestResponse
from agents.diagnosis_agent import DiagnosisAgent
from rag.retriever import VectorStore
from rag.pubmed_ingestor import PubMedIngestor, MEDICAL_TOPICS


# ─── Lifespan (startup / shutdown) ────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup."""
    logger.info("🚀 MedMind AI starting up...")

    # Initialize vector store connection
    app.state.vector_store = VectorStore()
    app.state.agent = DiagnosisAgent()

    # Auto-ingest if collection is empty
    collection_size = app.state.vector_store.get_collection_size()
    if collection_size < 100:
        logger.info(f"Vector store has only {collection_size} docs — triggering background ingest...")
        ingestor = PubMedIngestor()
        # Run a minimal ingest synchronously for first 3 topics
        import asyncio
        asyncio.create_task(ingestor.ingest_topics(MEDICAL_TOPICS[:3], papers_per_topic=20))

    logger.success(f"✅ MedMind AI ready — {collection_size} papers in knowledge base")
    yield

    logger.info("MedMind AI shutting down...")


# ─── App Instance ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="MedMind AI — Clinical Decision Support API",
    description=(
        "AI-powered differential diagnosis engine with RAG over PubMed, "
        "LangGraph multi-agent reasoning, chain-of-thought traces, and safety guardrails."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:5173").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Request/Response Logging Middleware ──────────────────────────────────────

@app.middleware("http")
async def log_requests(request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    logger.info(f"{request.method} {request.url.path} → {response.status_code} ({duration:.2f}s)")
    return response


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.post(
    "/diagnose",
    response_model=DiagnosisResponse,
    summary="Run Differential Diagnosis",
    description=(
        "Accepts patient data and runs the full LangGraph diagnosis pipeline: "
        "patient parsing → RAG retrieval → diagnosis generation → self-critique → "
        "reasoning traces → guardrail checks."
    ),
    tags=["Diagnosis"],
)
async def diagnose(patient_input: PatientInput) -> DiagnosisResponse:
    """
    Main diagnosis endpoint.

    Example request body:
    ```json
    {
      "age": 54,
      "sex": "Male",
      "chief_complaint": "3 days of productive cough and fever",
      "symptoms": "Fever 38.7°C, productive cough with yellow sputum, right-sided chest pain worse on inspiration, shortness of breath. SpO2 94% on room air.",
      "vitals_raw": "BP 128/82, HR 108, RR 22, Temp 38.7°C, SpO2 94%",
      "medical_history": "Hypertension, Type 2 DM. Non-smoker.",
      "medications": "Metformin 500mg BD, Lisinopril 10mg OD",
      "max_diagnoses": 3
    }
    ```
    """
    try:
        agent: DiagnosisAgent = app.state.agent
        result = await agent.run(patient_input)
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(e))
    except Exception as e:
        logger.error(f"Diagnosis failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Diagnosis pipeline failed. Please try again.",
        )


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="System Health Check",
    tags=["System"],
)
async def health_check() -> HealthResponse:
    """Check system status including vector DB document count."""
    vector_store: VectorStore = app.state.vector_store
    return HealthResponse(
        status="healthy",
        vector_db_documents=vector_store.get_collection_size(),
        llm_provider=os.getenv("LLM_PROVIDER", "openai"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "BioMedBERT"),
    )


@app.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Trigger PubMed Ingestion",
    tags=["Data"],
)
async def trigger_ingest(
    background_tasks: BackgroundTasks,
    topics: list[str] = None,
    papers_per_topic: int = 50,
    force: bool = False,
) -> IngestResponse:
    """
    Trigger PubMed paper ingestion into the vector store.
    Runs in background — returns immediately with status.
    """
    ingestor = PubMedIngestor()
    topics_to_use = topics or MEDICAL_TOPICS

    background_tasks.add_task(
        ingestor.ingest_topics,
        topics=topics_to_use,
        papers_per_topic=papers_per_topic,
        force_reingest=force,
    )

    return IngestResponse(
        status="ingestion_started",
        papers_ingested=0,  # Will be updated in background
        collection_size=app.state.vector_store.get_collection_size(),
        topics_covered=topics_to_use,
    )


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


@app.post(
    "/papers/search",
    summary="Search Knowledge Base",
    tags=["Data"],
)
async def search_papers(request: SearchRequest) -> dict:
    """
    Search the PubMed vector store directly.
    Useful for testing retrieval quality.
    """
    vector_store: VectorStore = app.state.vector_store
    papers = vector_store.retrieve(request.query, top_k=request.top_k)
    return {
        "query": request.query,
        "results": papers,
        "total_found": len(papers),
    }


class FeedbackRequest(BaseModel):
    session_id: str
    diagnosis_rank: int
    clinician_agreed: bool
    actual_diagnosis: str = ""
    notes: str = ""


@app.post(
    "/feedback",
    summary="Submit Clinician Feedback",
    tags=["Feedback"],
)
async def submit_feedback(feedback: FeedbackRequest) -> dict:
    """
    Collect clinician feedback for model improvement.
    In production: store in PostgreSQL, use for fine-tuning.
    """
    # TODO: Persist to database
    logger.info(
        f"Feedback received — session: {feedback.session_id}, "
        f"agreed: {feedback.clinician_agreed}, "
        f"actual: {feedback.actual_diagnosis}"
    )
    return {"status": "feedback_received", "message": "Thank you for improving MedMind AI."}


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
