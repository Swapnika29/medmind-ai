import { useState, useEffect, useRef } from "react";

const PHASES = ["symptoms", "analyzing", "results"];

const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

function parseVitalsRaw(vitalsRaw = "") {
  const text = vitalsRaw.toLowerCase();

  const systolicDiastolic = text.match(/bp[:\s]*([0-9]{2,3})\s*\/\s*([0-9]{2,3})/i);
  const hr = text.match(/(?:hr|heart rate)[:\s]*([0-9]{2,3})/i);
  const rr = text.match(/(?:rr|respiratory rate)[:\s]*([0-9]{1,2})/i);
  const temp = text.match(/(?:temp|temperature)[:\s]*([0-9]+(?:\.[0-9]+)?)/i);
  const spo2 = text.match(/(?:spo2|spo₂|o2 sat|oxygen saturation)[:\s]*([0-9]{2,3})/i);
  const weight = text.match(/(?:weight)[:\s]*([0-9]+(?:\.[0-9]+)?)/i);

  return {
    blood_pressure_systolic: systolicDiastolic ? Number(systolicDiastolic[1]) : undefined,
    blood_pressure_diastolic: systolicDiastolic ? Number(systolicDiastolic[2]) : undefined,
    heart_rate: hr ? Number(hr[1]) : undefined,
    respiratory_rate: rr ? Number(rr[1]) : undefined,
    temperature_celsius: temp ? Number(temp[1]) : undefined,
    spo2_percent: spo2 ? Number(spo2[1]) : undefined,
    weight_kg: weight ? Number(weight[1]) : undefined,
  };
}

function inferSeverity(confidence) {
  if (confidence >= 75) return "moderate";
  if (confidence >= 55) return "high";
  return "low";
}

function normalizeConfidence(score) {
  // backend may return 0-1 or 0-100
  if (score == null || Number.isNaN(Number(score))) return 0;
  const n = Number(score);
  return n <= 1 ? Math.round(n * 100) : Math.round(n);
}

function mapApiDiagnosisToUi(d, idx) {
  const confidence = normalizeConfidence(d.confidence_score);

  const reasoning =
    Array.isArray(d.reasoning_trace) && d.reasoning_trace.length
      ? d.reasoning_trace.map((step) => {
          const parts = [
            step?.clinical_finding,
            step?.inference,
            step?.supporting_evidence,
          ].filter(Boolean);
          return parts.join(" — ") || `Reasoning step ${step?.step_number ?? idx + 1}`;
        })
      : ["No reasoning trace returned."];

  const papers =
    Array.isArray(d.supporting_papers)
      ? d.supporting_papers.map((p) => ({
          title: p.title || "Untitled paper",
          journal: p.journal || "Unknown journal",
          year: p.year || "",
          pmid: p.pmid || "N/A",
        }))
      : [];

  const treatment = [
    ...(Array.isArray(d.recommended_workup) ? d.recommended_workup : []),
    ...(Array.isArray(d.treatment_considerations) ? d.treatment_considerations : []),
    ...(Array.isArray(d.red_flags) && d.red_flags.length
      ? d.red_flags.map((rf) => `Red flag: ${rf}`)
      : []),
  ];

  return {
    id: idx + 1,
    condition: d.condition || `Diagnosis ${idx + 1}`,
    icd: d.icd10_code || "N/A",
    confidence,
    severity: d.severity || inferSeverity(confidence),
    reasoning,
    papers,
    treatment: treatment.length ? treatment : ["No suggested next steps returned."],
    differentialRationale: d.differential_rationale || "",
  };
}

const mockDiagnoses = [
  {
    id: 1,
    condition: "Community-Acquired Pneumonia",
    icd: "J18.9",
    confidence: 87,
    severity: "moderate",
    reasoning: [
      "Productive cough with purulent sputum strongly correlates with bacterial pneumonia",
      "Fever >38.5°C combined with tachycardia suggests systemic inflammatory response",
      "Right lower lobe crackles on auscultation indicate consolidation pattern",
      "SpO2 94% reflects impaired gas exchange consistent with parenchymal involvement",
    ],
    papers: [
      { title: "IDSA/ATS Consensus Guidelines on CAP in Adults", journal: "AJRCCM", year: 2019, pmid: "19299617" },
      { title: "Procalcitonin-guided antibiotic therapy in CAP", journal: "Lancet", year: 2021, pmid: "34111411" },
    ],
    treatment: ["Amoxicillin-Clavulanate 875mg BID × 5 days", "Azithromycin 500mg QD × 3 days (atypical coverage)", "Chest X-ray confirmatory imaging"],
  },
  {
    id: 2,
    condition: "Pulmonary Embolism",
    icd: "I26.99",
    confidence: 61,
    severity: "high",
    reasoning: [
      "Acute dyspnea onset with pleuritic chest pain raises PE suspicion",
      "Tachycardia (HR 108) without clear infectious source warrants vascular consideration",
      "Wells Score estimated 4.5 — moderate probability category",
      "Recent immobilization history increases pre-test probability",
    ],
    papers: [
      { title: "ESC Guidelines for acute PE diagnosis and management", journal: "Eur Heart J", year: 2020, pmid: "31504429" },
      { title: "CT Pulmonary Angiography vs V/Q Scan in suspected PE", journal: "NEJM", year: 2022, pmid: "35443107" },
    ],
    treatment: ["D-dimer assay stat (if <500 μg/L, PE unlikely)", "CTPA if D-dimer elevated", "Anticoagulation initiation pending imaging"],
  },
  {
    id: 3,
    condition: "Acute Bronchitis",
    icd: "J20.9",
    confidence: 44,
    severity: "low",
    reasoning: [
      "Cough duration <3 weeks consistent with acute bronchitis timeline",
      "Absence of lobar consolidation pattern would support airways-only inflammation",
      "Viral etiology most probable given seasonal presentation",
      "Lower confidence due to fever severity atypical for simple bronchitis",
    ],
    papers: [
      { title: "Antibiotic prescribing for acute bronchitis: Cochrane Review", journal: "Cochrane DB", year: 2017, pmid: "28881005" },
    ],
    treatment: ["Supportive care: hydration, rest", "Antipyretics PRN", "Avoid antibiotics unless bacterial superinfection confirmed"],
  },
];

const severityConfig = {
  high: { color: "#ff4757", bg: "rgba(255,71,87,0.12)", label: "HIGH RISK" },
  moderate: { color: "#ffa502", bg: "rgba(255,165,2,0.12)", label: "MODERATE" },
  low: { color: "#2ed573", bg: "rgba(46,213,115,0.12)", label: "LOW RISK" },
};

function ConfidenceRing({ value, color }) {
  const r = 36;
  const circ = 2 * Math.PI * r;
  const [animated, setAnimated] = useState(0);
  useEffect(() => {
    const t = setTimeout(() => setAnimated(value), 300);
    return () => clearTimeout(t);
  }, [value]);
  const dash = (animated / 100) * circ;
  return (
    <svg width="90" height="90" style={{ transform: "rotate(-90deg)" }}>
      <circle cx="45" cy="45" r={r} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="6" />
      <circle
        cx="45" cy="45" r={r} fill="none"
        stroke={color} strokeWidth="6"
        strokeDasharray={`${dash} ${circ}`}
        strokeLinecap="round"
        style={{ transition: "stroke-dasharray 1.2s cubic-bezier(0.34,1.56,0.64,1)" }}
      />
      <text
        x="45" y="45" textAnchor="middle" dominantBaseline="central"
        fill="white" fontSize="13" fontWeight="700" fontFamily="'DM Mono', monospace"
        style={{ transform: "rotate(90deg)", transformOrigin: "45px 45px" }}
      >
        {animated}%
      </text>
    </svg>
  );
}

function TypewriterText({ text, speed = 18 }) {
  const [displayed, setDisplayed] = useState("");
  useEffect(() => {
    setDisplayed("");
    let i = 0;
    const iv = setInterval(() => {
      if (i < text.length) { setDisplayed(text.slice(0, i + 1)); i++; }
      else clearInterval(iv);
    }, speed);
    return () => clearInterval(iv);
  }, [text]);
  return <span>{displayed}<span style={{ opacity: displayed.length < text.length ? 1 : 0, animation: "blink 1s infinite" }}>▋</span></span>;
}

function AnalyzingScreen() {
  const steps = [
    "Parsing patient history & vitals...",
    "Vectorizing symptom embeddings...",
    "Querying PubMed knowledge base (23,847 papers)...",
    "Running differential diagnosis model...",
    "Generating reasoning traces...",
    "Calculating confidence intervals...",
    "Compiling citations & evidence...",
  ];
  const [current, setCurrent] = useState(0);
  const [done, setDone] = useState([]);
  useEffect(() => {
    const iv = setInterval(() => {
      setCurrent(c => {
        setDone(d => [...d, c]);
        return c + 1;
      });
    }, 600);
    return () => clearInterval(iv);
  }, []);
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", minHeight: "60vh", gap: "32px" }}>
      <div style={{ position: "relative", width: "120px", height: "120px" }}>
        {[0, 1, 2].map(i => (
          <div key={i} style={{
            position: "absolute", inset: `${i * 15}px`, borderRadius: "50%",
            border: `2px solid rgba(0,212,255,${0.6 - i * 0.18})`,
            animation: `spin ${1.5 + i * 0.4}s linear infinite${i % 2 ? " reverse" : ""}`,
          }} />
        ))}
        <div style={{
          position: "absolute", inset: "35px", background: "rgba(0,212,255,0.15)",
          borderRadius: "50%", display: "flex", alignItems: "center", justifyContent: "center",
          boxShadow: "0 0 30px rgba(0,212,255,0.4)",
        }}>
          <span style={{ fontSize: "22px" }}>🧠</span>
        </div>
      </div>
      <div style={{ width: "380px" }}>
        {steps.map((step, i) => (
          <div key={i} style={{
            display: "flex", alignItems: "center", gap: "12px", padding: "8px 0",
            opacity: i <= current ? 1 : 0.2,
            transition: "opacity 0.4s ease",
            fontFamily: "'DM Mono', monospace", fontSize: "12px",
            color: done.includes(i) ? "#2ed573" : i === current ? "#00d4ff" : "rgba(255,255,255,0.3)",
          }}>
            <span style={{ fontSize: "14px" }}>{done.includes(i) ? "✓" : i === current ? "◈" : "○"}</span>
            {i === current ? <TypewriterText text={step} /> : step}
          </div>
        ))}
      </div>
    </div>
  );
}

export default function ClinicalAI() {
  const [phase, setPhase] = useState("symptoms");
  const [selected, setSelected] = useState(0);
  const [expandedReason, setExpandedReason] = useState(null);
  const [form, setForm] = useState({
    age: "", sex: "Male", chiefComplaint: "", symptoms: "", vitals: "", history: "", medications: "",
  });
  const [diagnoses, setDiagnoses] = useState(mockDiagnoses);
  const [apiMeta, setApiMeta] = useState({
  totalPapers: 14,
  patientSummary: "",
  disclaimer: "",
  processingTimeMs: null,
  guardrailFlags: [],
  agentReasoningSummary: "",
  });
  const [errorMsg, setErrorMsg] = useState("");

  const handleAnalyze = async () => {
  setErrorMsg("");

  // Basic validation (beginner-friendly)
  if (!form.age || !form.chiefComplaint || !form.symptoms) {
    setErrorMsg("Please fill at least Age, Chief Complaint, and Symptoms before analyzing.");
    return;
  }

  const ageNum = Number(form.age);
  if (Number.isNaN(ageNum) || ageNum <= 0) {
    setErrorMsg("Please enter a valid age.");
    return;
  }

  setPhase("analyzing");

  try {
    const payload = {
      age: ageNum,
      sex: form.sex,
      chief_complaint: form.chiefComplaint,
      symptoms: form.symptoms,
      vitals: parseVitalsRaw(form.vitals),
      vitals_raw: form.vitals,
      medical_history: form.history || "",
      medications: form.medications || "",
      allergies: "",
      family_history: "",
      max_diagnoses: 3,
      include_rare: false,
    };

    const res = await fetch(`${API_BASE}/diagnose`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const errText = await res.text();
      throw new Error(`API error ${res.status}: ${errText}`);
    }

    const data = await res.json();

    const mappedDiagnoses = Array.isArray(data.diagnoses)
      ? data.diagnoses.map(mapApiDiagnosisToUi)
      : [];

    if (!mappedDiagnoses.length) {
      throw new Error("No diagnoses returned from backend.");
    }

    setDiagnoses(mappedDiagnoses);
    setSelected(0);
    setExpandedReason(null);

    setApiMeta({
      totalPapers: data.total_papers_retrieved ?? 0,
      patientSummary: data.patient_summary || "",
      disclaimer: data.disclaimer || "",
      processingTimeMs: data.processing_time_seconds
        ? Math.round(Number(data.processing_time_seconds) * 1000)
        : null,
      guardrailFlags: Array.isArray(data.guardrail_flags) ? data.guardrail_flags : [],
      agentReasoningSummary: data.agent_reasoning_summary || "",
    });

    setPhase("results");
  } catch (err) {
    console.error("Diagnose request failed:", err);
    setErrorMsg(err.message || "Failed to analyze patient data.");
    setPhase("symptoms");
  }
};

  const diag = diagnoses[selected];
  const sev = severityConfig[diag?.severity];

  return (
    <div style={{
      minHeight: "100vh", background: "#080c14",
      fontFamily: "'Inter', sans-serif", color: "white",
      backgroundImage: `radial-gradient(ellipse at 20% 0%, rgba(0,100,255,0.08) 0%, transparent 60%), radial-gradient(ellipse at 80% 100%, rgba(0,212,150,0.06) 0%, transparent 60%)`,
    }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@700;800&family=Inter:wght@300;400;500;600&display=swap');
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes blink { 0%,100%{opacity:1}50%{opacity:0} }
        @keyframes fadeUp { from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:translateY(0)} }
        @keyframes pulse { 0%,100%{box-shadow:0 0 0 0 rgba(0,212,255,0.3)}50%{box-shadow:0 0 0 8px rgba(0,212,255,0)} }
        * { box-sizing: border-box; }
        ::-webkit-scrollbar { width: 4px; }
        ::-webkit-scrollbar-track { background: transparent; }
        ::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 2px; }
      `}</style>

      {/* Header */}
      <div style={{
        borderBottom: "1px solid rgba(255,255,255,0.06)", padding: "18px 40px",
        display: "flex", alignItems: "center", justifyContent: "space-between",
        background: "rgba(8,12,20,0.8)", backdropFilter: "blur(20px)",
        position: "sticky", top: 0, zIndex: 100,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: "14px" }}>
          <div style={{
            width: "38px", height: "38px", borderRadius: "10px",
            background: "linear-gradient(135deg, #00d4ff, #0066ff)",
            display: "flex", alignItems: "center", justifyContent: "center", fontSize: "18px",
          }}>⚕</div>
          <div>
            <div style={{ fontFamily: "'Syne', sans-serif", fontWeight: 800, fontSize: "17px", letterSpacing: "-0.3px" }}>
              MedMind <span style={{ color: "#00d4ff" }}>AI</span>
            </div>
            <div style={{ fontSize: "10px", color: "rgba(255,255,255,0.35)", fontFamily: "'DM Mono', monospace", letterSpacing: "1px" }}>
              CLINICAL DECISION SUPPORT v2.4
            </div>
          </div>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: "24px" }}>
          <div style={{ display: "flex", gap: "6px" }}>
            {["symptoms", "analyzing", "results"].map((p, i) => (
              <div key={p} style={{
                width: "28px", height: "3px", borderRadius: "2px",
                background: phase === p ? "#00d4ff" : i < PHASES.indexOf(phase) ? "rgba(0,212,255,0.4)" : "rgba(255,255,255,0.1)",
                transition: "all 0.4s",
              }} />
            ))}
          </div>
          <div style={{
            padding: "6px 14px", borderRadius: "6px", fontSize: "11px",
            fontFamily: "'DM Mono', monospace", letterSpacing: "1px",
            background: "rgba(46,213,115,0.12)", color: "#2ed573",
            border: "1px solid rgba(46,213,115,0.2)",
          }}>● LIVE</div>
        </div>
      </div>

      <div style={{ maxWidth: "1100px", margin: "0 auto", padding: "40px 24px" }}>

        {/* SYMPTOM INPUT PHASE */}
        {phase === "symptoms" && (
          <div style={{ animation: "fadeUp 0.5s ease" }}>
            <div style={{ marginBottom: "36px" }}>
              <h1 style={{ fontFamily: "'Syne', sans-serif", fontWeight: 800, fontSize: "32px", letterSpacing: "-1px", margin: 0 }}>
                Patient Intake <span style={{ color: "#00d4ff" }}>Assessment</span>
              </h1>
              <p style={{ color: "rgba(255,255,255,0.4)", marginTop: "8px", fontSize: "14px" }}>
                Enter patient data for AI-powered differential diagnosis with evidence-based reasoning
              </p>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px" }}>
              {/* Left col */}
              <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "12px" }}>
                  <Field label="Age" placeholder="e.g. 54" value={form.age} onChange={v => setForm(f => ({...f, age: v}))} />
                  <div>
                    <label style={labelStyle}>Biological Sex</label>
                    <select value={form.sex} onChange={e => setForm(f => ({...f, sex: e.target.value}))} style={inputStyle}>
                      <option>Male</option><option>Female</option><option>Other</option>
                    </select>
                  </div>
                </div>
                <Field label="Chief Complaint" placeholder="Primary reason for visit..." value={form.chiefComplaint} onChange={v => setForm(f => ({...f, chiefComplaint: v}))} />
                <Textarea label="Symptoms & Duration" placeholder="Describe symptoms, onset, duration, aggravating/relieving factors..." value={form.symptoms} onChange={v => setForm(f => ({...f, symptoms: v}))} rows={4} />
                <Textarea label="Vital Signs" placeholder="BP: 128/82, HR: 108, RR: 22, Temp: 38.7°C, SpO2: 94%..." value={form.vitals} onChange={v => setForm(f => ({...f, vitals: v}))} rows={3} />
              </div>
              {/* Right col */}
              <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
                <Textarea label="Medical History & Comorbidities" placeholder="Hypertension, Type 2 DM, prior PE 2019, recent hospitalization..." value={form.history} onChange={v => setForm(f => ({...f, history: v}))} rows={5} />
                <Textarea label="Current Medications & Allergies" placeholder="Metformin 500mg BD, Lisinopril 10mg OD. NKDA..." value={form.medications} onChange={v => setForm(f => ({...f, medications: v}))} rows={4} />
                <div style={{
                  padding: "16px", borderRadius: "12px",
                  background: "rgba(0,212,255,0.05)", border: "1px solid rgba(0,212,255,0.15)",
                  fontSize: "12px", color: "rgba(255,255,255,0.45)", lineHeight: "1.6",
                }}>
                  <span style={{ color: "#00d4ff", fontWeight: 600 }}>⚡ RAG Pipeline Active</span><br />
                  Querying 23,847 indexed PubMed papers · BioMedBERT embeddings · ChromaDB vector store
                </div>
              </div>
            </div>

            {errorMsg && (
              <div style={{
                marginTop: "12px",
                marginBottom: "8px",
                padding: "12px 14px",
                borderRadius: "10px",
                background: "rgba(255,71,87,0.08)",
                border: "1px solid rgba(255,71,87,0.2)",
                color: "#ff8a95",
                fontSize: "12px",
                lineHeight: "1.5",
              }}>
                {errorMsg}
              </div>
            )}

            <div style={{ display: "flex", justifyContent: "center", marginTop: "32px" }}>
              <button onClick={handleAnalyze} style={{
                padding: "16px 48px", borderRadius: "12px", border: "none", cursor: "pointer",
                background: "linear-gradient(135deg, #00d4ff, #0066ff)",
                color: "white", fontSize: "15px", fontWeight: 600, letterSpacing: "0.3px",
                boxShadow: "0 8px 32px rgba(0,102,255,0.35)",
                transition: "transform 0.2s, box-shadow 0.2s",
                fontFamily: "'Inter', sans-serif",
              }}
                onMouseEnter={e => { e.target.style.transform = "translateY(-2px)"; e.target.style.boxShadow = "0 12px 40px rgba(0,102,255,0.5)"; }}
                onMouseLeave={e => { e.target.style.transform = "translateY(0)"; e.target.style.boxShadow = "0 8px 32px rgba(0,102,255,0.35)"; }}
              >
                Run Differential Diagnosis →
              </button>
            </div>
          </div>
        )}

        {/* ANALYZING PHASE */}
        {phase === "analyzing" && <AnalyzingScreen />}

        {/* RESULTS PHASE */}
        {phase === "results" && (
          <div style={{ animation: "fadeUp 0.5s ease" }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "28px" }}>
              <div>
                <h1 style={{ fontFamily: "'Syne', sans-serif", fontWeight: 800, fontSize: "28px", letterSpacing: "-0.8px", margin: 0 }}>
                  Differential <span style={{ color: "#00d4ff" }}>Diagnoses</span>
                </h1>
                <p style={{ color: "rgba(255,255,255,0.35)", marginTop: "6px", fontSize: "13px", fontFamily: "'DM Mono', monospace" }}>
                  {diagnoses.length} conditions identified · {apiMeta.totalPapers ?? 0} papers retrieved · Reasoning traces available
                </p>
              </div>
              <button onClick={() => setPhase("symptoms")} style={{
                padding: "10px 20px", borderRadius: "8px", border: "1px solid rgba(255,255,255,0.12)",
                background: "transparent", color: "rgba(255,255,255,0.5)", cursor: "pointer", fontSize: "13px",
              }}>← New Patient</button>
            </div>

            {/* Disclaimer */}
            <div style={{
              padding: "12px 18px", borderRadius: "10px", marginBottom: "24px",
              background: "rgba(255,165,2,0.08)", border: "1px solid rgba(255,165,2,0.2)",
              fontSize: "12px", color: "rgba(255,200,100,0.8)", display: "flex", gap: "10px", alignItems: "center",
            }}>
              <span>⚠️</span>
              <span>
                {apiMeta.disclaimer || "AI-generated suggestions only. Not a substitute for clinical judgment. Always verify with physical examination and confirmatory diagnostics."}
              </span>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "320px 1fr", gap: "20px" }}>
              {/* Diagnosis list */}
              <div style={{ display: "flex", flexDirection: "column", gap: "12px" }}>
                {diagnoses.map((d, i) => {
                  const rawSeverity = String(d.severity || "").toLowerCase();

                  const normalizedSeverity =
                    rawSeverity.includes("high") ? "high" :
                    rawSeverity.includes("moderate") || rawSeverity.includes("medium") ? "moderate" :
                    rawSeverity.includes("low") ? "low" :
                    "moderate";

                  const s = severityStyles?.[normalizedSeverity] || severityStyles?.moderate || { color: "#fbbf24" };
                  return (
                    <div key={d.id} onClick={() => setSelected(i)} style={{
                      padding: "18px", borderRadius: "14px", cursor: "pointer",
                      border: `1px solid ${selected === i ? "rgba(0,212,255,0.4)" : "rgba(255,255,255,0.07)"}`,
                      background: selected === i ? "rgba(0,212,255,0.06)" : "rgba(255,255,255,0.02)",
                      transition: "all 0.25s", animation: `fadeUp 0.4s ease ${i * 0.12}s both`,
                    }}>
                      <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: "12px" }}>
                        <div style={{ flex: 1 }}>
                          <div style={{ fontWeight: 600, fontSize: "14px", marginBottom: "4px" }}>{d.condition}</div>
                          <div style={{ fontFamily: "'DM Mono', monospace", fontSize: "11px", color: "rgba(255,255,255,0.3)" }}>{d.icd}</div>
                        </div>
                        <ConfidenceRing value={d.confidence} color={s.color} />
                      </div>
                      <div style={{
                        display: "inline-block", marginTop: "10px",
                        padding: "3px 10px", borderRadius: "4px", fontSize: "10px",
                        fontFamily: "'DM Mono', monospace", letterSpacing: "1px",
                        background: s.bg, color: s.color,
                      }}>{s.label}</div>
                    </div>
                  );
                })}
              </div>

              {/* Detail panel */}
              {diag && (
                <div style={{ display: "flex", flexDirection: "column", gap: "16px", animation: "fadeUp 0.35s ease" }} key={selected}>
                  {/* Header */}
                  <div style={{
                    padding: "24px", borderRadius: "16px",
                    background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.07)",
                  }}>
                    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start" }}>
                      <div>
                        <h2 style={{ fontFamily: "'Syne', sans-serif", fontWeight: 800, fontSize: "22px", margin: "0 0 6px" }}>{diag.condition}</h2>
                        <div style={{ fontFamily: "'DM Mono', monospace", fontSize: "12px", color: "rgba(255,255,255,0.3)" }}>ICD-10: {diag.icd}</div>
                      </div>
                      <div style={{ textAlign: "right" }}>
                        <div style={{ fontSize: "36px", fontWeight: 800, fontFamily: "'Syne', sans-serif", color: sev.color }}>{diag.confidence}%</div>
                        <div style={{ fontSize: "11px", color: "rgba(255,255,255,0.3)", fontFamily: "'DM Mono', monospace" }}>CONFIDENCE</div>
                      </div>
                    </div>
                    <div style={{ marginTop: "16px", background: "rgba(255,255,255,0.04)", borderRadius: "8px", height: "6px", overflow: "hidden" }}>
                      <div style={{
                        height: "100%", borderRadius: "8px", width: `${diag.confidence}%`,
                        background: `linear-gradient(90deg, ${sev.color}88, ${sev.color})`,
                        transition: "width 1s cubic-bezier(0.34,1.56,0.64,1)",
                      }} />
                    </div>
                  </div>

                  {/* Reasoning Trace */}
                  <div style={{
                    padding: "20px", borderRadius: "16px",
                    background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.07)",
                  }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "16px" }}>
                      <div style={{
                        width: "28px", height: "28px", borderRadius: "8px",
                        background: "rgba(0,212,255,0.15)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: "14px",
                      }}>🔍</div>
                      <div style={{ fontWeight: 600, fontSize: "14px" }}>AI Reasoning Trace</div>
                      <div style={{
                        marginLeft: "auto", padding: "3px 10px", borderRadius: "4px", fontSize: "10px",
                        fontFamily: "'DM Mono', monospace", letterSpacing: "1px",
                        background: "rgba(0,212,255,0.1)", color: "#00d4ff",
                      }}>CHAIN-OF-THOUGHT</div>
                    </div>
                    <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
                      {diag.reasoning.map((r, i) => (
                        <div key={i} style={{
                          display: "flex", gap: "14px", padding: "12px 14px", borderRadius: "10px",
                          background: expandedReason === i ? "rgba(0,212,255,0.06)" : "rgba(255,255,255,0.02)",
                          border: `1px solid ${expandedReason === i ? "rgba(0,212,255,0.2)" : "rgba(255,255,255,0.05)"}`,
                          cursor: "pointer", transition: "all 0.2s",
                          animation: `fadeUp 0.3s ease ${i * 0.08}s both`,
                        }} onClick={() => setExpandedReason(expandedReason === i ? null : i)}>
                          <div style={{
                            minWidth: "22px", height: "22px", borderRadius: "50%",
                            background: "rgba(0,212,255,0.15)", display: "flex", alignItems: "center", justifyContent: "center",
                            fontFamily: "'DM Mono', monospace", fontSize: "11px", color: "#00d4ff", flexShrink: 0,
                          }}>{i + 1}</div>
                          <div style={{ fontSize: "13px", color: "rgba(255,255,255,0.75)", lineHeight: "1.55" }}>{r}</div>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Evidence */}
                  <div style={{
                    padding: "20px", borderRadius: "16px",
                    background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.07)",
                  }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "16px" }}>
                      <div style={{
                        width: "28px", height: "28px", borderRadius: "8px",
                        background: "rgba(46,213,115,0.15)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: "14px",
                      }}>📄</div>
                      <div style={{ fontWeight: 600, fontSize: "14px" }}>Evidence Base</div>
                      <div style={{
                        marginLeft: "auto", padding: "3px 10px", borderRadius: "4px", fontSize: "10px",
                        fontFamily: "'DM Mono', monospace", background: "rgba(46,213,115,0.1)", color: "#2ed573",
                      }}>{diag.papers.length} PAPERS</div>
                    </div>
                    {diag.papers.map((p, i) => (
                      <div key={i} style={{
                        padding: "12px 14px", borderRadius: "10px", marginBottom: "8px",
                        background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.05)",
                        animation: `fadeUp 0.3s ease ${i * 0.1}s both`,
                      }}>
                        <div style={{ fontSize: "13px", fontWeight: 500, marginBottom: "6px" }}>{p.title}</div>
                        <div style={{ display: "flex", gap: "12px" }}>
                          <span style={{ fontFamily: "'DM Mono', monospace", fontSize: "11px", color: "#2ed573" }}>{p.journal}</span>
                          <span style={{ fontFamily: "'DM Mono', monospace", fontSize: "11px", color: "rgba(255,255,255,0.3)" }}>{p.year}</span>
                          <span style={{ fontFamily: "'DM Mono', monospace", fontSize: "11px", color: "rgba(0,212,255,0.6)" }}>PMID: {p.pmid}</span>
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* Treatment */}
                  <div style={{
                    padding: "20px", borderRadius: "16px",
                    background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.07)",
                  }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "10px", marginBottom: "16px" }}>
                      <div style={{
                        width: "28px", height: "28px", borderRadius: "8px",
                        background: "rgba(255,165,2,0.15)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: "14px",
                      }}>💊</div>
                      <div style={{ fontWeight: 600, fontSize: "14px" }}>Suggested Next Steps</div>
                    </div>
                    <div style={{ display: "flex", flexDirection: "column", gap: "8px" }}>
                      {diag.treatment.map((t, i) => (
                        <div key={i} style={{
                          display: "flex", gap: "12px", alignItems: "flex-start",
                          animation: `fadeUp 0.3s ease ${i * 0.08}s both`,
                        }}>
                          <div style={{ color: "#ffa502", marginTop: "1px", fontSize: "14px" }}>→</div>
                          <div style={{ fontSize: "13px", color: "rgba(255,255,255,0.7)", lineHeight: "1.5" }}>{t}</div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

// Helper components
const labelStyle = { display: "block", fontSize: "11px", fontFamily: "'DM Mono', monospace", letterSpacing: "0.8px", color: "rgba(255,255,255,0.4)", marginBottom: "8px", textTransform: "uppercase" };
const inputStyle = { width: "100%", padding: "11px 14px", borderRadius: "10px", border: "1px solid rgba(255,255,255,0.08)", background: "rgba(255,255,255,0.04)", color: "white", fontSize: "13px", outline: "none", fontFamily: "'Inter', sans-serif", transition: "border-color 0.2s", appearance: "none" };

function Field({ label, placeholder, value, onChange }) {
  return (
    <div>
      <label style={labelStyle}>{label}</label>
      <input value={value} onChange={e => onChange(e.target.value)} placeholder={placeholder}
        style={{ ...inputStyle, "::placeholder": { color: "rgba(255,255,255,0.2)" } }}
        onFocus={e => e.target.style.borderColor = "rgba(0,212,255,0.4)"}
        onBlur={e => e.target.style.borderColor = "rgba(255,255,255,0.08)"}
      />
    </div>
  );
}
function Textarea({ label, placeholder, value, onChange, rows = 3 }) {
  return (
    <div>
      <label style={labelStyle}>{label}</label>
      <textarea value={value} onChange={e => onChange(e.target.value)} placeholder={placeholder} rows={rows}
        style={{ ...inputStyle, resize: "vertical", lineHeight: "1.6" }}
        onFocus={e => e.target.style.borderColor = "rgba(0,212,255,0.4)"}
        onBlur={e => e.target.style.borderColor = "rgba(255,255,255,0.08)"}
      />
    </div>
  );
}
