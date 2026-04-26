#  MedMind AI — Clinical Decision Support System

> **Portfolio Project** | LLM + RAG + LangGraph + FastAPI + React  
> An AI system that reads patient symptoms, retrieves evidence from PubMed, and generates differential diagnoses with explainable reasoning traces.

---

##  Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MEDMIND AI PIPELINE                         │
│                                                                     │
│  Patient Input                                                      │
│      │                                                              │
│      ▼                                                              │
│  ┌─────────────┐    ┌──────────────────┐    ┌──────────────────┐   │
│  │   FastAPI   │───▶│  LangGraph Agent │───▶│  RAG Retriever   │   │
│  │  (REST API) │    │   (Orchestrator) │    │ ChromaDB+BioMED  │   │
│  └─────────────┘    └──────────────────┘    └──────────────────┘   │
│                              │                       │             │
│                              ▼                       ▼             │
│                    ┌──────────────────┐    ┌──────────────────┐   │
│                    │   GPT-4o /       │    │  PubMed API      │   │
│                    │   Claude 3.5     │    │  23K+ Papers     │   │
│                    └──────────────────┘    └──────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│                    ┌──────────────────┐                             │
│                    │  Self-Critique   │  ← Reflection Pattern      │
│                    │  + Guardrails    │                             │
│                    └──────────────────┘                             │
│                              │                                      │
│                              ▼                                      │
│                    Diagnosis + Reasoning Trace + Citations          │
└─────────────────────────────────────────────────────────────────────┘
```

##  Key Technical Features

| Feature | Implementation | Why It Matters |
|---|---|---|
| **Medical RAG** | BioMedBERT + ChromaDB | Domain-specific embeddings outperform generic ones on clinical text |
| **Multi-node Agent** | LangGraph StateGraph | Structured, debuggable agent flow vs. single prompt |
| **Self-Critique** | Reflection node | LLM reviews its own output for hallucinations before returning |
| **Reasoning Traces** | Chain-of-thought extraction | Clinicians can see *why* each diagnosis was suggested |
| **Safety Guardrails** | Must-not-miss checker | Ensures life-threatening conditions aren't overlooked |
| **Confidence Calibration** | Score validation | Prevents overconfident outputs on uncertain presentations |

---

##  Project Structure

```
medmind-ai/
├── backend/
│   ├── main.py                      # FastAPI app, endpoints, lifespan
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── .env.example                 # Copy to .env and fill in keys
│   │
│   ├── models/
│   │   └── schemas.py               # All Pydantic v2 models
│   │
│   ├── rag/
│   │   ├── embedder.py              # BioMedBERT embedding wrapper
│   │   ├── retriever.py             # ChromaDB vector store
│   │   └── pubmed_ingestor.py       # PubMed API fetcher + ingestion
│   │
│   ├── agents/
│   │   ├── diagnosis_agent.py       # LangGraph graph + all nodes
│   │   ├── reasoning_tracer.py      # Chain-of-thought builder
│   │   └── guardrails.py            # Safety checks
│   │
│   └── tests/
│       └── test_pipeline.py
│
├── frontend/
│   └── src/
│       └── App.jsx                  # React UI (MedMind dashboard)
│
└── docker-compose.yml
```

---

##  Quick Start

### 1. Clone & Configure

```bash
git clone https://github.com/yourusername/medmind-ai
cd medmind-ai/backend

cp .env.example .env
# Edit .env — add your OPENAI_API_KEY or ANTHROPIC_API_KEY
```

### 2. Install & Run Backend

```bash
pip install -r requirements.txt

# Start the API
uvicorn main:app --reload --port 8000
```

### 3. Ingest PubMed Papers

```bash
# Trigger ingestion (runs in background)
curl -X POST http://localhost:8000/ingest

# Check progress
curl http://localhost:8000/health
```

### 4. Run a Diagnosis

```bash
curl -X POST http://localhost:8000/diagnose \
  -H "Content-Type: application/json" \
  -d '{
    "age": 54,
    "sex": "Male",
    "chief_complaint": "3 days of productive cough and fever",
    "symptoms": "Fever 38.7C, productive cough with yellow sputum, right-sided chest pain, SpO2 94%",
    "vitals_raw": "BP 128/82, HR 108, RR 22, Temp 38.7C, SpO2 94%",
    "medical_history": "Hypertension, Type 2 DM",
    "medications": "Metformin 500mg, Lisinopril 10mg",
    "max_diagnoses": 3
  }'
```

### 5. Docker (Full Stack)

```bash
cd medmind-ai
docker-compose up --build
# Frontend: http://localhost:3000
# API docs: http://localhost:8000/docs
```

---

##  Running Tests

```bash
cd backend
pip install pytest pytest-asyncio
python -m pytest tests/ -v
```

---

##  API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/diagnose` | Run full diagnosis pipeline |
| `GET` | `/health` | System health + vector DB size |
| `POST` | `/ingest` | Trigger PubMed ingestion |
| `POST` | `/papers/search` | Search vector store directly |
| `POST` | `/feedback` | Submit clinician feedback |

Full interactive docs at: `http://localhost:8000/docs`

---

##  LangGraph Agent Flow

```
parse_patient → retrieve_evidence → generate_diagnoses
                                           ↓
                             self_critique (reflection)
                                           ↓
                             build_reasoning_traces
                                           ↓
                               check_guardrails
                                           ↓
                                format_output → END
```

Each node is independently testable and the state is fully typed with TypedDict.

---


##  Disclaimer

This is a **portfolio/research project** demonstrating AI/ML engineering skills. It is **NOT** intended for clinical use, medical advice, or patient care. Always consult licensed healthcare professionals for medical decisions.

---

##  Tech Stack

`Python 3.11` · `FastAPI` · `LangGraph` · `LangChain` · `GPT-4o / Claude 3.5` · `BioMedBERT` · `ChromaDB` · `PubMed API` · `React` · `Docker`
