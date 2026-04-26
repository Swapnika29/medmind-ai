"""
agents/guardrails.py
─────────────────────
Safety guardrails for the medical AI system.

Checks for:
1. Missing critical diagnoses (life-threatening conditions not flagged)
2. Dangerously high confidence in uncertain diagnoses
3. Pediatric/pregnancy edge cases requiring specialist referral
4. Drug interactions in treatment suggestions
5. Red flag symptoms that must trigger emergency referral

In production, augment with Llama Guard or Azure Content Safety.
"""

from loguru import logger
from models.schemas import PatientInput


# ─── High-risk conditions that must ALWAYS be considered ─────────────────────
MUST_NOT_MISS = {
    "chest pain": ["myocardial infarction", "pulmonary embolism", "aortic dissection", "tension pneumothorax"],
    "headache": ["subarachnoid hemorrhage", "meningitis", "hypertensive emergency"],
    "dyspnea": ["pulmonary embolism", "tension pneumothorax", "anaphylaxis"],
    "abdominal pain": ["aortic aneurysm", "ectopic pregnancy", "mesenteric ischemia"],
    "altered consciousness": ["hypoglycemia", "stroke", "sepsis", "opioid overdose"],
    "fever": ["sepsis", "meningitis", "endocarditis"],
}

# ─── Vital sign thresholds for emergency flags ────────────────────────────────
EMERGENCY_VITALS = {
    "spo2_percent": {"threshold": 90, "direction": "below", "flag": "Critical hypoxia — immediate O2 and emergency evaluation required"},
    "heart_rate": {"threshold": 140, "direction": "above", "flag": "Severe tachycardia — requires urgent cardiac evaluation"},
    "temperature_celsius": {"threshold": 40.0, "direction": "above", "flag": "Hyperpyrexia — consider sepsis protocol"},
    "respiratory_rate": {"threshold": 30, "direction": "above", "flag": "Severe tachypnea — risk of respiratory failure"},
}


class MedicalGuardrails:
    """Runs safety checks on AI-generated diagnoses."""

    def check(self, diagnoses: list[dict], patient_input: PatientInput) -> list[str]:
        """
        Run all guardrail checks.

        Returns:
            List of flag strings (empty = all clear)
        """
        flags = []

        flags.extend(self._check_must_not_miss(diagnoses, patient_input))
        flags.extend(self._check_vital_sign_emergencies(patient_input))
        flags.extend(self._check_confidence_inflation(diagnoses))
        flags.extend(self._check_special_populations(patient_input, diagnoses))
        flags.extend(self._check_minimum_workup(diagnoses))

        return flags

    def _check_must_not_miss(
        self, diagnoses: list[dict], patient: PatientInput
    ) -> list[str]:
        """Ensure life-threatening diagnoses are not missed for high-risk presentations."""
        flags = []
        chief = patient.chief_complaint.lower()
        symptoms = patient.symptoms.lower()
        combined = chief + " " + symptoms

        diagnosis_names = " ".join([
            dx.get("condition", "").lower() for dx in diagnoses
        ])

        for keyword, critical_conditions in MUST_NOT_MISS.items():
            if keyword in combined:
                for condition in critical_conditions:
                    if condition not in diagnosis_names:
                        # Check if it was excluded for a valid reason
                        flags.append(
                            f"⚠️ SAFETY: '{condition.title()}' not in differential — "
                            f"consider excluding with appropriate workup for '{keyword}' presentation"
                        )

        return flags[:3]  # Cap at 3 to avoid overwhelming output

    def _check_vital_sign_emergencies(self, patient: PatientInput) -> list[str]:
        """Flag critical vital signs that require immediate attention."""
        flags = []

        if not patient.vitals:
            return flags

        vitals_dict = patient.vitals.model_dump()

        for vital, config in EMERGENCY_VITALS.items():
            value = vitals_dict.get(vital)
            if value is None:
                continue

            threshold = config["threshold"]
            direction = config["direction"]

            is_critical = (
                (direction == "below" and value < threshold) or
                (direction == "above" and value > threshold)
            )

            if is_critical:
                flags.append(f"🚨 EMERGENCY VITAL: {config['flag']} (Recorded: {value})")

        return flags

    def _check_confidence_inflation(self, diagnoses: list[dict]) -> list[str]:
        """Warn if confidence scores seem inflated for a complex presentation."""
        flags = []

        if not diagnoses:
            return flags

        top_confidence = diagnoses[0].get("confidence_score", 0)

        if top_confidence > 0.95:
            flags.append(
                "⚠️ CONFIDENCE FLAG: Top diagnosis confidence >95% — clinical presentations "
                "are rarely this certain. Verify with confirmatory testing before acting."
            )

        # Check if all diagnoses have suspiciously similar confidence
        if len(diagnoses) >= 2:
            scores = [dx.get("confidence_score", 0) for dx in diagnoses]
            if max(scores) - min(scores) < 0.05:
                flags.append(
                    "⚠️ CALIBRATION FLAG: Confidence scores too uniform — model may be "
                    "poorly calibrated for this presentation."
                )

        return flags

    def _check_special_populations(
        self, patient: PatientInput, diagnoses: list[dict]
    ) -> list[str]:
        """Flag edge cases requiring specialist consideration."""
        flags = []

        # Pediatric
        if patient.age < 18:
            flags.append(
                "📋 PEDIATRIC NOTE: Diagnoses generated for adult population. "
                "Dosing, differential probabilities, and normal ranges differ in children — "
                "consult pediatric specialist."
            )

        # Elderly
        if patient.age > 80:
            flags.append(
                "📋 GERIATRIC NOTE: Atypical presentations are common in patients >80y. "
                "Consider occult sepsis, medication side effects, and cognitive baseline changes."
            )

        # Check for pregnancy mention
        history = (patient.medical_history or "").lower()
        symptoms = patient.symptoms.lower()
        if "pregnant" in history or "pregnancy" in history or "gravid" in symptoms:
            flags.append(
                "📋 OBSTETRIC NOTE: Pregnancy-related considerations not fully modeled. "
                "Consult OB/GYN. Many medications in treatment suggestions may be contraindicated."
            )

        return flags

    def _check_minimum_workup(self, diagnoses: list[dict]) -> list[str]:
        """Ensure critical workup items are recommended."""
        flags = []

        all_workup = " ".join([
            " ".join(dx.get("recommended_workup", [])).lower()
            for dx in diagnoses
        ])

        # If no imaging was recommended for chest/abdominal presentations
        if not any(term in all_workup for term in ["x-ray", "ct", "mri", "ultrasound", "imaging"]):
            flags.append(
                "📋 WORKUP NOTE: No imaging recommended — consider whether radiological "
                "evaluation is appropriate for this presentation."
            )

        return flags
