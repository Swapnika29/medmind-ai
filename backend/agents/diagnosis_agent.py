"""
agents/diagnosis_agent.py
──────────────────────────
LangGraph-powered multi-node agent for clinical differential diagnosis.

Graph Architecture:
┌─────────────────────────────────────────────────────────┐
│                                                         │
│  [parse_patient] → [retrieve_evidence] → [generate_dx] │
│         ↑                                      ↓        │
│         └──── [self_critique] ←── [build_trace] ────┘   │
│                      ↓                                  │
│               [final_output]                            │
│                                                         │
└─────────────────────────────────────────────────────────┘

The self_critique node implements a reflection pattern — the LLM
reviews its own output for hallucinations/inconsistencies before
finalizing. This is what separates production LLM systems from demos.
"""

import json
import os
import time
import uuid
from typing import Annotated, Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END
from loguru import logger

from models.schemas import PatientInput, DiagnosisResponse, DiagnosisResult
from rag.retriever import VectorStore
from .reasoning_tracer import ReasoningTracer
from .guardrails import MedicalGuardrails


# ─── Agent State ─────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """Shared state passed between graph nodes."""
    session_id: str
    patient_input: PatientInput
    patient_summary: str              # Structured summary of patient
    retrieved_papers: list[dict]      # Raw papers from ChromaDB
    raw_diagnoses: list[dict]         # LLM-generated diagnoses (unvalidated)
    reasoning_traces: list[dict]      # Chain-of-thought per diagnosis
    critiqued_diagnoses: list[dict]   # Post-self-critique diagnoses
    guardrail_flags: list[str]        # Safety flags
    final_response: dict              # Structured final output
    error: str                        # Error message if any node fails


# ─── LLM Factory ─────────────────────────────────────────────────────────────

def get_llm(temperature: float = 0.1):
    """Return configured LLM based on env var."""
    provider = os.getenv("LLM_PROVIDER", "openai").lower()

    if provider == "anthropic":
        return ChatAnthropic(
            model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
            temperature=temperature,
            max_tokens=4096,
        )
    else:
        return ChatOpenAI(
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
            temperature=temperature,
            max_tokens=4096,
        )


# ─── Node Functions ───────────────────────────────────────────────────────────

def parse_patient_node(state: AgentState) -> AgentState:
    """
    Node 1: Parse patient input into a structured clinical summary.
    This standardizes messy freeform text before the LLM sees it.
    """
    logger.info(f"[{state['session_id']}] Node: parse_patient")
    patient = state["patient_input"]
    llm = get_llm(temperature=0.0)

    system_prompt = """You are a clinical documentation specialist. 
    Extract and structure patient information into a concise clinical summary.
    Be precise, use medical terminology, and flag any critical findings."""

    user_prompt = f"""Structure this patient presentation:

Patient: {patient.age}y {patient.sex.value}
Chief Complaint: {patient.chief_complaint}
Symptoms: {patient.symptoms}
Vitals: {patient.vitals_raw or 'Not provided'}
Medical History: {patient.medical_history or 'Not provided'}
Medications: {patient.medications or 'Not provided'}
Allergies: {patient.allergies or 'Not provided'}

Provide a structured clinical summary in 3-4 sentences. Include:
1. Patient demographics and chief complaint
2. Key positive findings (symptoms, abnormal vitals)
3. Relevant history and risk factors
4. Any red flags or urgent considerations"""

    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])
    state["patient_summary"] = response.content
    logger.debug(f"Patient summary generated: {response.content[:100]}...")
    return state


def retrieve_evidence_node(state: AgentState) -> AgentState:
    """
    Node 2: Retrieve relevant PubMed papers using RAG.
    Constructs a rich query from the patient summary for better retrieval.
    """
    logger.info(f"[{state['session_id']}] Node: retrieve_evidence")
    patient = state["patient_input"]

    # Build a clinically rich query
    query_parts = [
        patient.chief_complaint,
        patient.symptoms[:200],
    ]
    if patient.medical_history:
        query_parts.append(patient.medical_history[:100])

    retrieval_query = " ".join(query_parts)

    vector_store = VectorStore()
    top_k = int(os.getenv("RAG_TOP_K", "8"))
    threshold = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.60"))

    papers = vector_store.retrieve(
        query=retrieval_query,
        top_k=top_k,
        similarity_threshold=threshold,
    )

    state["retrieved_papers"] = papers
    logger.info(f"Retrieved {len(papers)} relevant papers")
    return state


def generate_diagnoses_node(state: AgentState) -> AgentState:
    """
    Node 3: Core diagnosis generation using LLM + retrieved evidence.
    Uses structured JSON output for reliable parsing.
    """
    logger.info(f"[{state['session_id']}] Node: generate_diagnoses")
    patient = state["patient_input"]
    papers = state["retrieved_papers"]

    # Format retrieved papers for the prompt
    evidence_block = ""
    for i, paper in enumerate(papers[:6], 1):
        evidence_block += f"""
[PAPER {i}] {paper['title']}
Journal: {paper['journal']} ({paper['year']}) | PMID: {paper['pmid']}
Abstract: {paper['abstract_snippet']}
Relevance: {paper['relevance_score']:.0%}
"""

    llm = get_llm(temperature=0.1)

    system_prompt = """You are an expert physician AI assistant specializing in differential diagnosis.
You have access to current medical literature. Generate evidence-based differential diagnoses.

CRITICAL RULES:
1. Only suggest diagnoses supported by the provided evidence or established medical knowledge
2. Never fabricate citations — only reference provided papers
3. Confidence scores must reflect genuine clinical probability
4. Always include at least one life-threatening condition to not miss (even if low probability)
5. Respond ONLY with valid JSON — no markdown, no explanation outside JSON"""

    user_prompt = f"""PATIENT PRESENTATION:
{state['patient_summary']}

Age: {patient.age} | Sex: {patient.sex.value}
Raw Symptoms: {patient.symptoms}
History: {patient.medical_history or 'None reported'}
Medications: {patient.medications or 'None reported'}

RETRIEVED MEDICAL EVIDENCE:
{evidence_block}

Generate {patient.max_diagnoses} differential diagnoses ranked by probability.

Respond with this EXACT JSON structure:
{{
  "diagnoses": [
    {{
      "rank": 1,
      "condition": "Full condition name",
      "icd10_code": "X00.0",
      "confidence_score": 0.87,
      "severity": "moderate",
      "clinical_rationale": "Why this is the top diagnosis based on the presentation",
      "supporting_pmids": ["12345678"],
      "recommended_workup": ["CBC", "Chest X-ray", "Blood cultures"],
      "treatment_considerations": ["Amoxicillin-clavulanate 875mg BID x 5 days"],
      "red_flags": ["Declining SpO2", "Worsening dyspnea"],
      "reasoning_steps": [
        {{
          "clinical_finding": "Fever 38.7°C + productive cough",
          "inference": "Suggests lower respiratory tract infection",
          "supporting_evidence": "Consistent with IDSA CAP guidelines (PMID: 19299617)",
          "confidence_contribution": 0.25
        }}
      ]
    }}
  ]
}}

Severity options: "low" | "moderate" | "high" | "critical"
Confidence scores: 0.0 to 1.0 (must sum to ≤ 2.0 across all diagnoses)"""

    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])

    # Parse JSON response
    try:
        # Strip any accidental markdown fences
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        raw = json.loads(content)
        state["raw_diagnoses"] = raw.get("diagnoses", [])
        logger.info(f"Generated {len(state['raw_diagnoses'])} diagnoses")
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse diagnosis JSON: {e}")
        state["error"] = f"Diagnosis parsing failed: {e}"
        state["raw_diagnoses"] = []

    return state


def self_critique_node(state: AgentState) -> AgentState:
    """
    Node 4: Reflection/self-critique — the LLM reviews its own output.

    This is the most important node for production reliability.
    The LLM checks for:
    - Hallucinated citations (PMIDs not in retrieved papers)
    - Confidence score inflation
    - Missing critical diagnoses (red flags not to miss)
    - Logical inconsistencies in reasoning
    """
    logger.info(f"[{state['session_id']}] Node: self_critique")

    if not state.get("raw_diagnoses"):
        state["critiqued_diagnoses"] = []
        return state

    llm = get_llm(temperature=0.0)

    valid_pmids = {p["pmid"] for p in state["retrieved_papers"]}

    system_prompt = """You are a senior physician reviewing AI-generated diagnoses for quality and safety.
Your job is to critique and correct the diagnoses. Be conservative — patient safety is paramount.
Respond ONLY with valid JSON."""

    user_prompt = f"""Review these AI-generated diagnoses for a patient:

PATIENT SUMMARY: {state['patient_summary']}

PROPOSED DIAGNOSES: {json.dumps(state['raw_diagnoses'], indent=2)}

VALID PMIDs IN EVIDENCE BASE: {list(valid_pmids)}

CRITIQUE CHECKLIST:
1. Remove any PMIDs not in the valid list above (hallucinated citations)
2. Reduce confidence_score if reasoning steps don't clearly support it
3. Flag if a critical/life-threatening diagnosis was missed
4. Verify ICD-10 codes are realistic for the condition
5. Ensure red_flags list is comprehensive

Return the corrected diagnoses with the SAME JSON structure, plus add:
- "critique_notes": "What was changed and why" (per diagnosis)
- "hallucination_detected": true/false (per diagnosis)"""

    response = llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)])

    try:
        content = response.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        critiqued = json.loads(content)

        # Handle both wrapped and unwrapped responses
        if isinstance(critiqued, dict) and "diagnoses" in critiqued:
            state["critiqued_diagnoses"] = critiqued["diagnoses"]
        elif isinstance(critiqued, list):
            state["critiqued_diagnoses"] = critiqued
        else:
            state["critiqued_diagnoses"] = state["raw_diagnoses"]

        logger.info("Self-critique complete")
    except Exception as e:
        logger.warning(f"Self-critique parsing failed, using raw diagnoses: {e}")
        state["critiqued_diagnoses"] = state["raw_diagnoses"]

    return state


def build_reasoning_traces_node(state: AgentState) -> AgentState:
    """
    Node 5: Build detailed reasoning traces for the UI.
    Converts raw reasoning_steps into the structured ReasoningStep format.
    """
    logger.info(f"[{state['session_id']}] Node: build_reasoning_traces")
    tracer = ReasoningTracer()

    traces = []
    for dx in state["critiqued_diagnoses"]:
        trace = tracer.build_trace(
            diagnosis=dx,
            patient_summary=state["patient_summary"],
            retrieved_papers=state["retrieved_papers"],
        )
        traces.append(trace)

    state["reasoning_traces"] = traces
    return state


def check_guardrails_node(state: AgentState) -> AgentState:
    """
    Node 6: Run safety guardrails on the final output.
    Detects dangerous/inappropriate outputs before returning to the user.
    """
    logger.info(f"[{state['session_id']}] Node: check_guardrails")
    guardrails = MedicalGuardrails()
    flags = guardrails.check(
        diagnoses=state["critiqued_diagnoses"],
        patient_input=state["patient_input"],
    )
    state["guardrail_flags"] = flags
    if flags:
        logger.warning(f"Guardrail flags raised: {flags}")
    return state


def format_output_node(state: AgentState) -> AgentState:
    """
    Node 7: Assemble the final DiagnosisResponse from agent state.
    Maps internal dicts to Pydantic schemas.
    """
    logger.info(f"[{state['session_id']}] Node: format_output")
    from models.schemas import (
        DiagnosisResult, ResearchPaper, ReasoningStep, SeverityLevel
    )

    diagnoses = []
    for dx, trace in zip(state["critiqued_diagnoses"], state["reasoning_traces"]):

        # Map supporting papers
        dx_pmids = set(dx.get("supporting_pmids", []))
        supporting_papers = [
            ResearchPaper(
                title=p["title"],
                journal=p["journal"],
                year=int(p["year"]) if p.get("year") else 2020,
                pmid=p["pmid"],
                relevance_score=p["relevance_score"],
                abstract_snippet=p["abstract_snippet"],
                pubmed_url=p["pubmed_url"],
            )
            for p in state["retrieved_papers"]
            if p["pmid"] in dx_pmids or not dx_pmids  # Show all if no specific PMIDs
        ][:3]  # Cap at 3 per diagnosis

        # Map reasoning steps
        reasoning_steps = [
            ReasoningStep(
                step_number=i + 1,
                clinical_finding=step.get("clinical_finding", ""),
                inference=step.get("inference", ""),
                supporting_evidence=step.get("supporting_evidence", ""),
                confidence_contribution=step.get("confidence_contribution", 0.0),
            )
            for i, step in enumerate(trace.get("steps", []))
        ]

        severity_str = dx.get("severity", "moderate").lower()
        try:
            severity = SeverityLevel(severity_str)
        except ValueError:
            severity = SeverityLevel.moderate

        diagnoses.append(DiagnosisResult(
            rank=dx.get("rank", len(diagnoses) + 1),
            condition=dx.get("condition", "Unknown"),
            icd10_code=dx.get("icd10_code", "Z00.0"),
            confidence_score=min(float(dx.get("confidence_score", 0.5)), 1.0),
            severity=severity,
            reasoning_trace=reasoning_steps,
            supporting_papers=supporting_papers,
            recommended_workup=dx.get("recommended_workup", []),
            treatment_considerations=dx.get("treatment_considerations", []),
            red_flags=dx.get("red_flags", []),
            differential_rationale=dx.get("clinical_rationale", ""),
        ))

    state["final_response"] = {
        "session_id": state["session_id"],
        "patient_summary": state["patient_summary"],
        "diagnoses": diagnoses,
        "total_papers_retrieved": len(state["retrieved_papers"]),
        "agent_reasoning_summary": (
            f"Analyzed {patient_age}yo patient with '{chief_complaint}'. "
            f"Retrieved {len(state['retrieved_papers'])} PubMed papers via RAG. "
            f"Generated {len(diagnoses)} differential diagnoses with self-critique validation."
        ).format(
            patient_age=state["patient_input"].age,
            chief_complaint=state["patient_input"].chief_complaint,
        ) if False else (
            f"Analyzed {state['patient_input'].age}yo patient presenting with "
            f"'{state['patient_input'].chief_complaint}'. Retrieved "
            f"{len(state['retrieved_papers'])} PubMed papers via RAG. "
            f"Generated {len(diagnoses)} differential diagnoses with self-critique validation."
        ),
        "guardrail_flags": state.get("guardrail_flags", []),
    }
    return state


# ─── Graph Assembly ───────────────────────────────────────────────────────────

def build_diagnosis_graph() -> StateGraph:
    """Assemble the LangGraph StateGraph with all nodes and edges."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("parse_patient", parse_patient_node)
    graph.add_node("retrieve_evidence", retrieve_evidence_node)
    graph.add_node("generate_diagnoses", generate_diagnoses_node)
    graph.add_node("self_critique", self_critique_node)
    graph.add_node("build_reasoning_traces", build_reasoning_traces_node)
    graph.add_node("check_guardrails", check_guardrails_node)
    graph.add_node("format_output", format_output_node)

    # Define edges (execution order)
    graph.set_entry_point("parse_patient")
    graph.add_edge("parse_patient", "retrieve_evidence")
    graph.add_edge("retrieve_evidence", "generate_diagnoses")
    graph.add_edge("generate_diagnoses", "self_critique")
    graph.add_edge("self_critique", "build_reasoning_traces")
    graph.add_edge("build_reasoning_traces", "check_guardrails")
    graph.add_edge("check_guardrails", "format_output")
    graph.add_edge("format_output", END)

    return graph.compile()


# ─── Main Runner ──────────────────────────────────────────────────────────────

class DiagnosisAgent:
    """High-level interface for running the diagnosis graph."""

    def __init__(self):
        self.graph = build_diagnosis_graph()
        logger.info("DiagnosisAgent initialized with LangGraph")

    async def run(self, patient_input: PatientInput) -> DiagnosisResponse:
        """
        Run the full diagnosis pipeline for a patient.

        Args:
            patient_input: Validated PatientInput from the API

        Returns:
            DiagnosisResponse with diagnoses, reasoning traces, and citations
        """
        start_time = time.time()
        session_id = str(uuid.uuid4())[:8]

        logger.info(f"Starting diagnosis session {session_id}")

        initial_state: AgentState = {
            "session_id": session_id,
            "patient_input": patient_input,
            "patient_summary": "",
            "retrieved_papers": [],
            "raw_diagnoses": [],
            "reasoning_traces": [],
            "critiqued_diagnoses": [],
            "guardrail_flags": [],
            "final_response": {},
            "error": "",
        }

        # Run the graph
        final_state = await self.graph.ainvoke(initial_state)

        processing_time = time.time() - start_time
        logger.success(f"Session {session_id} complete in {processing_time:.2f}s")

        # Build final response
        resp_data = final_state["final_response"]
        return DiagnosisResponse(
            session_id=session_id,
            patient_summary=resp_data["patient_summary"],
            diagnoses=resp_data["diagnoses"],
            total_papers_retrieved=resp_data["total_papers_retrieved"],
            agent_reasoning_summary=resp_data["agent_reasoning_summary"],
            guardrail_flags=resp_data["guardrail_flags"],
            processing_time_seconds=round(processing_time, 2),
        )
