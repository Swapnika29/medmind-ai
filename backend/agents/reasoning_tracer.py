"""
agents/reasoning_tracer.py
───────────────────────────
Builds structured reasoning traces for each diagnosis.

This is the "explainability layer" — it maps each diagnosis back to
specific clinical findings, inferences, and supporting evidence.
This is what makes the system trustworthy to clinicians.
"""

import os
import json
from loguru import logger
from langchain_core.messages import HumanMessage, SystemMessage


class ReasoningTracer:
    """Generates step-by-step reasoning traces for each differential diagnosis."""

    def build_trace(
        self,
        diagnosis: dict,
        patient_summary: str,
        retrieved_papers: list[dict],
    ) -> dict:
        """
        Build a detailed reasoning trace for a single diagnosis.

        Strategy:
        1. Use the LLM-generated reasoning_steps if available (already good quality)
        2. Enrich with specific paper references from retrieved documents
        3. Add a summary trace explaining the overall inference chain
        """

        # Use existing reasoning steps from diagnosis generation
        raw_steps = diagnosis.get("reasoning_steps", [])

        # If no steps were generated, create them from clinical rationale
        if not raw_steps:
            raw_steps = self._generate_steps_from_rationale(
                diagnosis=diagnosis,
                patient_summary=patient_summary,
                papers=retrieved_papers,
            )

        # Enrich steps with paper references
        enriched_steps = self._enrich_with_citations(raw_steps, retrieved_papers)

        return {
            "condition": diagnosis.get("condition"),
            "steps": enriched_steps,
            "summary": self._build_summary(diagnosis, enriched_steps),
            "evidence_quality": self._assess_evidence_quality(retrieved_papers),
        }

    def _generate_steps_from_rationale(
        self,
        diagnosis: dict,
        patient_summary: str,
        papers: list[dict],
    ) -> list[dict]:
        """Generate reasoning steps from clinical rationale using LLM."""
        from langchain_openai import ChatOpenAI
        from langchain_anthropic import ChatAnthropic

        provider = os.getenv("LLM_PROVIDER", "openai").lower()
        if provider == "anthropic":
            llm = ChatAnthropic(model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"), temperature=0.0, max_tokens=1500)
        else:
            llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o"), temperature=0.0, max_tokens=1500)

        paper_refs = "\n".join([f"- {p['title']} (PMID: {p['pmid']})" for p in papers[:4]])

        prompt = f"""Break down the clinical reasoning for diagnosing '{diagnosis.get("condition")}' in this patient:

Patient: {patient_summary}
Clinical Rationale: {diagnosis.get("clinical_rationale", "")}

Available evidence:
{paper_refs}

Generate 3-4 reasoning steps as JSON array:
[
  {{
    "clinical_finding": "specific symptom/sign from the patient",
    "inference": "what this finding suggests about the diagnosis",
    "supporting_evidence": "which guideline/paper supports this reasoning",
    "confidence_contribution": 0.20
  }}
]

JSON only, no explanation."""

        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            content = response.content.strip().lstrip("```json").rstrip("```").strip()
            return json.loads(content)
        except Exception as e:
            logger.warning(f"Step generation failed for {diagnosis.get('condition')}: {e}")
            return [
                {
                    "clinical_finding": patient_summary[:100],
                    "inference": diagnosis.get("clinical_rationale", ""),
                    "supporting_evidence": "Clinical judgment based on presented symptoms",
                    "confidence_contribution": diagnosis.get("confidence_score", 0.5),
                }
            ]

    def _enrich_with_citations(
        self,
        steps: list[dict],
        papers: list[dict],
    ) -> list[dict]:
        """
        Enrich reasoning steps with specific paper references.
        Matches each step to the most relevant retrieved paper.
        """
        enriched = []
        for i, step in enumerate(steps):
            enriched_step = dict(step)

            # Try to find a matching paper for this step
            finding_lower = step.get("clinical_finding", "").lower()
            best_paper = None
            best_score = 0

            for paper in papers:
                # Simple keyword matching for citation enrichment
                paper_text = (paper.get("title", "") + " " + paper.get("abstract_snippet", "")).lower()
                # Count matching words
                words = set(finding_lower.split())
                matches = sum(1 for w in words if w in paper_text and len(w) > 4)
                if matches > best_score:
                    best_score = matches
                    best_paper = paper

            if best_paper and best_score >= 2:
                enriched_step["paper_reference"] = {
                    "title": best_paper["title"],
                    "pmid": best_paper["pmid"],
                    "journal": best_paper["journal"],
                    "year": best_paper.get("year", ""),
                }

            enriched.append(enriched_step)

        return enriched

    def _build_summary(self, diagnosis: dict, steps: list[dict]) -> str:
        """Build a one-paragraph natural language summary of the reasoning."""
        condition = diagnosis.get("condition", "Unknown")
        confidence = diagnosis.get("confidence_score", 0)
        findings = [s.get("clinical_finding", "") for s in steps[:2]]

        return (
            f"The diagnosis of {condition} (confidence: {confidence:.0%}) was reached through "
            f"a {len(steps)}-step clinical reasoning process. Key findings driving this assessment "
            f"include: {'; '.join(findings[:2])}. "
            f"This diagnosis was supported by {len([s for s in steps if s.get('paper_reference')])} "
            f"peer-reviewed publications retrieved from the PubMed evidence base."
        )

    def _assess_evidence_quality(self, papers: list[dict]) -> str:
        """Assess overall quality of retrieved evidence."""
        if not papers:
            return "No direct evidence retrieved — based on clinical knowledge only"

        high_relevance = sum(1 for p in papers if p.get("relevance_score", 0) > 0.80)
        if high_relevance >= 3:
            return "Strong — multiple highly relevant publications (>80% relevance)"
        elif high_relevance >= 1:
            return "Moderate — some relevant publications with supporting evidence"
        else:
            return "Limited — evidence is somewhat tangential; clinical judgment essential"
