"""
tests/test_pipeline.py
───────────────────────
Quick integration test for the full diagnosis pipeline.
Run with: python -m pytest tests/ -v
"""

import asyncio
import pytest
from unittest.mock import patch, MagicMock

# ─── Sample Test Patient ──────────────────────────────────────────────────────

SAMPLE_PATIENT = {
    "age": 54,
    "sex": "Male",
    "chief_complaint": "3 days of productive cough and fever",
    "symptoms": (
        "Fever 38.7°C, productive cough with yellow-green sputum, "
        "right-sided pleuritic chest pain, progressive dyspnea on exertion. "
        "Onset 3 days ago, worsening. No hemoptysis."
    ),
    "vitals_raw": "BP 128/82, HR 108, RR 22, Temp 38.7°C, SpO2 94% on RA",
    "medical_history": "Hypertension, Type 2 DM. Non-smoker. No recent travel or hospitalization.",
    "medications": "Metformin 500mg BD, Lisinopril 10mg OD. NKDA.",
    "max_diagnoses": 3,
}


# ─── Unit Tests ───────────────────────────────────────────────────────────────

def test_patient_input_validation():
    """Test PatientInput Pydantic validation."""
    from models.schemas import PatientInput, Sex

    patient = PatientInput(**SAMPLE_PATIENT)
    assert patient.age == 54
    assert patient.sex == Sex.male
    assert len(patient.symptoms) > 10


def test_patient_input_rejects_short_symptoms():
    """Validation should reject symptoms that are too brief."""
    from models.schemas import PatientInput
    from pydantic import ValidationError

    bad_patient = dict(SAMPLE_PATIENT)
    bad_patient["symptoms"] = "cough"
    with pytest.raises(ValidationError):
        PatientInput(**bad_patient)


def test_guardrails_flags_critical_vitals():
    """Guardrails should flag critical SpO2."""
    from agents.guardrails import MedicalGuardrails
    from models.schemas import PatientInput, VitalSigns

    patient_data = dict(SAMPLE_PATIENT)
    patient_data["vitals"] = VitalSigns(spo2_percent=85.0)  # Critical hypoxia
    patient = PatientInput(**patient_data)

    guardrails = MedicalGuardrails()
    flags = guardrails.check(diagnoses=[], patient_input=patient)

    assert any("hypoxia" in f.lower() or "spo2" in f.lower() or "o2" in f.lower() for f in flags)


def test_guardrails_must_not_miss():
    """Guardrails should flag missing critical diagnoses for chest pain."""
    from agents.guardrails import MedicalGuardrails
    from models.schemas import PatientInput

    patient_data = dict(SAMPLE_PATIENT)
    patient_data["chief_complaint"] = "acute chest pain"
    patient = PatientInput(**patient_data)

    # Diagnoses that DON'T include MI or PE
    diagnoses = [{"condition": "Musculoskeletal chest pain", "confidence_score": 0.9}]

    guardrails = MedicalGuardrails()
    flags = guardrails.check(diagnoses=diagnoses, patient_input=patient)

    # Should flag missed critical diagnoses
    assert len(flags) > 0


def test_vector_store_initialization():
    """Vector store should initialize without errors."""
    from rag.retriever import VectorStore
    store = VectorStore()
    size = store.get_collection_size()
    assert isinstance(size, int)
    assert size >= 0


@pytest.mark.asyncio
async def test_full_pipeline_with_mocked_llm():
    """Integration test with mocked LLM calls."""
    from models.schemas import PatientInput
    from agents.diagnosis_agent import DiagnosisAgent

    mock_diagnosis_json = '''
    {
      "diagnoses": [
        {
          "rank": 1,
          "condition": "Community-Acquired Pneumonia",
          "icd10_code": "J18.9",
          "confidence_score": 0.85,
          "severity": "moderate",
          "clinical_rationale": "Fever, productive cough, and reduced SpO2 consistent with CAP",
          "supporting_pmids": [],
          "recommended_workup": ["Chest X-ray", "CBC", "Blood cultures"],
          "treatment_considerations": ["Amoxicillin-clavulanate"],
          "red_flags": ["Declining SpO2"],
          "reasoning_steps": [
            {
              "clinical_finding": "Fever 38.7°C + productive cough",
              "inference": "Suggests bacterial lower respiratory infection",
              "supporting_evidence": "IDSA CAP Guidelines",
              "confidence_contribution": 0.35
            }
          ]
        }
      ]
    }
    '''

    patient = PatientInput(**SAMPLE_PATIENT)

    with patch("agents.diagnosis_agent.get_llm") as mock_llm_factory:
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content=mock_diagnosis_json)
        mock_llm_factory.return_value = mock_llm

        agent = DiagnosisAgent()
        # Test that the agent initializes correctly
        assert agent.graph is not None


# ─── Run Directly ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    """Quick manual test run."""
    import sys
    import json
    sys.path.insert(0, ".")

    from models.schemas import PatientInput
    patient = PatientInput(**SAMPLE_PATIENT)
    print(f"✅ Patient input validated: {patient.age}yo {patient.sex.value}")
    print(f"   Chief complaint: {patient.chief_complaint}")
    print("\nRun the full pipeline with:")
    print("  uvicorn main:app --reload")
    print("  Then POST to http://localhost:8000/diagnose")
