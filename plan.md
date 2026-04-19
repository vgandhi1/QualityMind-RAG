# Manufacturing Quality Engineering Assistant
## Project Plan — Multi-Source RAG + Text-to-SQL

**Author:** Vinay Gandhi  
**Base Architecture:** Multi-Source RAG + Text-to-SQL (FastAPI, Pinecone, AWS Lambda)  
**Domain Specialization:** Manufacturing Quality Engineering — CAPA, 8D, PFMEA, DFMEA, QMS, Root Cause Analysis  
**Status:** Planning

---

## 1. Problem Statement

Quality engineers in high-volume manufacturing spend significant time manually searching through PFMEA records, CAPA logs, supplier corrective action databases, and QMS documentation to answer questions they face daily:

- *"What failure modes have we seen on this component before?"*
- *"Which supplier CAPAs are still open past 90 days?"*
- *"What does the control plan say about this process step?"*
- *"We have a recurring defect — what have we tried before?"*

This assistant replaces manual document search and fragmented SQL queries with a single natural-language interface that reasons across both structured defect databases and unstructured quality documents simultaneously.

---

## 2. Goals

| Goal | Description |
|------|-------------|
| **G1 — Document Intelligence** | Enable natural-language search across PFMEA, DFMEA, CAPA, 8D, and QMS documentation |
| **G2 — Structured Data Query** | Convert plain-English questions into SQL against defect, NCR, and corrective action databases |
| **G3 — Root Cause Assistance** | Guide engineers through structured 5-Why and fishbone (Ishikawa) analysis using historical data |
| **G4 — Problem-Solving Workflows** | Generate CAPA and 8D draft structures pre-populated from retrieved context |
| **G5 — Evaluation & Trust** | Validate AI output quality with RAGAS metrics to ensure responses are faithful and relevant |
| **G6 — Production Deployment** | Deploy on AWS Lambda (ARM64) with CI/CD, scalable to factory-wide use |

---

## 3. System Architecture

### 3.1 High-Level Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        Engineer / User                          │
│         (Quality Engineer, Process Engineer, Supplier QE)       │
└───────────────────────────┬─────────────────────────────────────┘
                            │  Natural Language Query
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   FastAPI Application                           │
│              (AWS Lambda ARM64 + API Gateway)                   │
│         OPIK Monitoring  |  CloudWatch Logging                  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Intelligent Query Router                       │
│                                                                 │
│   SQL Keywords          Doc Keywords         Hybrid             │
│   (counts, defects,     (explain, PFMEA,     (data + context)   │
│    open CAPAs, NCRs)     policy, 8D, why)                       │
└────────┬────────────────────┬────────────────────┬──────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────┐   ┌───────────────────┐   ┌──────────────────┐
│  Text-to-   │   │  Document RAG     │   │  Hybrid Engine   │
│  SQL Engine │   │  Pipeline         │   │  (Both + Merge)  │
│  (Vanna.ai) │   │  (LangGraph)      │   │                  │
└──────┬──────┘   └────────┬──────────┘   └────────┬─────────┘
       │                   │                        │
       ▼                   ▼                        │
┌─────────────┐   ┌───────────────────┐            │
│ PostgreSQL  │   │ Pinecone Vector   │◄───────────┘
│             │   │ Store             │
│ • defects   │   │                   │
│ • capa_log  │   │ • PFMEA chunks    │
│ • ncr       │   │ • DFMEA chunks    │
│ • suppliers │   │ • CAPA records    │
│ • insp_res  │   │ • 8D reports      │
│ • actions   │   │ • QMS docs        │
└─────────────┘   │ • Control Plans   │
                  │ • Work Instruc.   │
                  └───────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              Specialized Quality Agents (LangGraph)             │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────┐   │
│  │  5-Why       │  │  Fishbone    │  │  CAPA / 8D Draft    │   │
│  │  Agent       │  │  Agent       │  │  Generator          │   │
│  └──────────────┘  └──────────────┘  └─────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     RAGAS Evaluation Layer                      │
│         Faithfulness | Answer Relevancy | Context Precision     │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Key Architectural Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Base framework | FastAPI + AWS Lambda ARM64 | Proven in existing project; 20% cost savings vs x86 |
| Vector DB | Pinecone (cosine, 1536-dim) | Already configured; scales to millions of quality doc chunks |
| Embeddings | OpenAI text-embedding-3-small | Best quality/cost balance for technical manufacturing text |
| LLM | GPT-4o | Required for structured output (JSON CAPA/8D drafts) and multi-step reasoning |
| SQL engine | Vanna.ai trained on quality schema | Natural language → SQL with schema-aware training |
| Orchestration | LangGraph | Multi-step agent workflows for 5-Why chains and fishbone construction |
| Evaluation | RAGAS | Faithfulness + relevancy scores; same rigor as Gage R&R for AI outputs |
| Monitoring | OPIK + CloudWatch | End-to-end request tracing |

---

## 4. Data Layer Design

### 4.1 Document Corpus (RAG — Pinecone)

Documents ingested and chunked via Docling HybridChunker for structure preservation:

| Document Type | Format | Chunking Strategy | Key Metadata Tags |
|---------------|--------|-------------------|-------------------|
| PFMEA | XLSX / DOCX | Row-level (one chunk per failure mode) | `part_number`, `process_step`, `severity`, `occurrence`, `detection`, `RPN` |
| DFMEA | XLSX / DOCX | Row-level per failure mode | `component`, `function`, `failure_effect`, `severity` |
| CAPA Records | DOCX / PDF | Section-level (Problem / Root Cause / Action / Verification) | `capa_id`, `supplier`, `status`, `open_date` |
| 8D Reports | DOCX / PDF | Discipline-level (D1–D8 as separate chunks) | `8d_id`, `part_number`, `team`, `containment_date` |
| QMS Procedures | PDF / DOCX | Paragraph-level with heading context | `procedure_id`, `revision`, `process_area` |
| Control Plans | XLSX | Row-level per control point | `part_number`, `operation`, `characteristic`, `method` |
| Work Instructions | PDF / DOCX | Step-level | `wi_number`, `revision`, `station` |
| Engineering Change Notices | PDF | Section-level | `ecn_number`, `affected_parts`, `effective_date` |

### 4.2 Structured Database Schema (Text-to-SQL — PostgreSQL)

```sql
-- Core defect tracking
CREATE TABLE defects (
    id              SERIAL PRIMARY KEY,
    part_number     VARCHAR(50),
    description     TEXT,
    failure_mode    VARCHAR(200),
    detection_station VARCHAR(100),
    disposition     VARCHAR(50),   -- rework, scrap, use-as-is
    severity        INTEGER,       -- 1-10
    date_found      DATE,
    shift           VARCHAR(10),
    operator_id     VARCHAR(20),
    supplier_id     INTEGER REFERENCES suppliers(id),
    ncr_id          INTEGER REFERENCES ncr(id)
);

-- Non-conformance records
CREATE TABLE ncr (
    id              SERIAL PRIMARY KEY,
    ncr_number      VARCHAR(30) UNIQUE,
    part_number     VARCHAR(50),
    description     TEXT,
    quantity        INTEGER,
    disposition     VARCHAR(50),
    status          VARCHAR(20),   -- open, closed, on-hold
    opened_date     DATE,
    closed_date     DATE,
    root_cause      TEXT,
    cost_impact     DECIMAL(10,2)
);

-- CAPA log
CREATE TABLE capa_log (
    id              SERIAL PRIMARY KEY,
    capa_number     VARCHAR(30) UNIQUE,
    title           TEXT,
    problem_statement TEXT,
    root_cause      TEXT,
    corrective_action TEXT,
    preventive_action TEXT,
    owner           VARCHAR(100),
    supplier_id     INTEGER REFERENCES suppliers(id),
    status          VARCHAR(20),   -- open, closed, verified
    due_date        DATE,
    opened_date     DATE,
    closed_date     DATE,
    verified_by     VARCHAR(100),
    recurrence_flag BOOLEAN DEFAULT FALSE
);

-- 8D tracker
CREATE TABLE eight_d (
    id              SERIAL PRIMARY KEY,
    report_number   VARCHAR(30) UNIQUE,
    part_number     VARCHAR(50),
    problem_statement TEXT,
    d1_team         TEXT,
    d2_problem_desc TEXT,
    d3_containment  TEXT,
    d4_root_cause   TEXT,
    d5_perm_action  TEXT,
    d6_implemented  TEXT,
    d7_prevention   TEXT,
    d8_closure      TEXT,
    status          VARCHAR(20),
    opened_date     DATE,
    closed_date     DATE,
    supplier_id     INTEGER REFERENCES suppliers(id)
);

-- Supplier master
CREATE TABLE suppliers (
    id              SERIAL PRIMARY KEY,
    supplier_code   VARCHAR(20) UNIQUE,
    name            VARCHAR(200),
    tier            INTEGER,       -- 1, 2, 3
    commodity       VARCHAR(100),
    quality_rating  DECIMAL(4,2),  -- 0-100
    active          BOOLEAN DEFAULT TRUE
);

-- Inspection results (SPC data)
CREATE TABLE inspection_results (
    id              SERIAL PRIMARY KEY,
    part_number     VARCHAR(50),
    characteristic  VARCHAR(200),
    measured_value  DECIMAL(12,6),
    nominal         DECIMAL(12,6),
    usl             DECIMAL(12,6),
    lsl             DECIMAL(12,6),
    cp              DECIMAL(6,3),
    cpk             DECIMAL(6,3),
    measurement_date TIMESTAMP,
    station         VARCHAR(100),
    gauge_id        VARCHAR(50),
    operator_id     VARCHAR(20)
);

-- Corrective actions (linked to CAPA or 8D)
CREATE TABLE corrective_actions (
    id              SERIAL PRIMARY KEY,
    action_text     TEXT,
    owner           VARCHAR(100),
    due_date        DATE,
    completed_date  DATE,
    status          VARCHAR(20),
    capa_id         INTEGER REFERENCES capa_log(id),
    eight_d_id      INTEGER REFERENCES eight_d(id),
    verified        BOOLEAN DEFAULT FALSE
);
```

---

## 5. Specialized Quality Agents

This is the core differentiation beyond the base RAG + SQL system. Three LangGraph agents handle structured quality problem-solving workflows.

### 5.1 5-Why Agent

**Trigger keywords:** "why", "root cause", "5 why", "five why", "reason for"

**Workflow:**
```
User: "Why are we seeing recurring delamination on EDV fascia?"
  │
  ├── Step 1: RAG lookup → retrieve PFMEA entries for EDV fascia, 
  │           prior CAPA records, any 8D reports on delamination
  │
  ├── Step 2: SQL query → SELECT defects WHERE part LIKE '%EDV fascia%' 
  │           ORDER BY date DESC LIMIT 20
  │
  ├── Step 3: LangGraph 5-Why chain
  │     Why 1: Surface finish adhesion failure (from PFMEA)
  │     Why 2: Ultrasonic welding parameter drift (from process data)
  │     Why 3: No SPC control on weld amplitude (from control plan gap)
  │     Why 4: Control plan not updated after design change (from ECN)
  │     Why 5: ECN review process does not trigger control plan update
  │
  └── Output: Structured JSON → rendered as 5-Why table
              + linked source documents for each Why
              + suggested corrective action per level
```

**Output format:**
```json
{
  "problem_statement": "Recurring delamination on EDV fascia",
  "whys": [
    {"level": 1, "why": "...", "evidence_source": "PFMEA-EDV-042, D4 section"},
    {"level": 2, "why": "...", "evidence_source": "defects table, 2024-Q3"},
    ...
  ],
  "root_cause": "...",
  "recommended_action": "...",
  "confidence_score": 0.84
}
```

---

### 5.2 Fishbone (Ishikawa) Diagram Agent

**Trigger keywords:** "fishbone", "Ishikawa", "cause and effect", "categories of cause"

**Workflow:**
```
User: "Build a fishbone diagram for high RPN fastener torque failures at station 12"
  │
  ├── Step 1: RAG → retrieve PFMEA entries, control plan for station 12,
  │           related work instructions
  │
  ├── Step 2: SQL → SELECT failure_mode, COUNT(*) FROM defects 
  │           WHERE detection_station = 'Station 12' GROUP BY failure_mode
  │
  ├── Step 3: LangGraph categorization across 6M bones:
  │     Man    → operator training gaps, certification status
  │     Machine → torque tool calibration, tool wear
  │     Method  → work instruction ambiguity, sequence errors
  │     Material → fastener dimensional variation, supplier lot
  │     Measurement → gauge R&R status on torque tools
  │     Environment → temperature effects on torque readings
  │
  └── Output: Structured JSON for fishbone rendering
              + evidence citations per cause
              + RPN weighting per branch
```

**Output format:**
```json
{
  "effect": "Fastener torque failures at Station 12",
  "bones": {
    "Man":         [{"cause": "...", "evidence": "...", "weight": 0.7}],
    "Machine":     [{"cause": "...", "evidence": "...", "weight": 0.9}],
    "Method":      [{"cause": "...", "evidence": "...", "weight": 0.6}],
    "Material":    [{"cause": "...", "evidence": "...", "weight": 0.4}],
    "Measurement": [{"cause": "...", "evidence": "...", "weight": 0.8}],
    "Environment": [{"cause": "...", "evidence": "...", "weight": 0.3}]
  }
}
```

> **Frontend note:** The JSON output can be rendered as an interactive SVG fishbone diagram in a Streamlit or React UI in a future phase.

---

### 5.3 CAPA / 8D Draft Generator

**Trigger keywords:** "draft CAPA", "generate 8D", "write corrective action", "create problem report"

**Workflow:**
```
User: "Draft an 8D for the recurring torque failures at Station 12"
  │
  ├── Step 1: Pull all context — defect records, prior CAPAs, 
  │           fishbone output (if run), PFMEA entries
  │
  ├── Step 2: LangGraph populates each discipline:
  │     D1: Team → suggests cross-functional members from org context
  │     D2: Problem Description → quantified from SQL defect data
  │     D3: Containment → pulls from similar prior 8D D3 sections
  │     D4: Root Cause → from 5-Why agent output if available
  │     D5: Permanent Corrective Action → generated + sourced
  │     D6: Implementation → dates, owners (templated)
  │     D7: Prevention → PFMEA update, control plan revision flags
  │     D8: Closure → verification criteria from control plan
  │
  └── Output: Structured 8D JSON + exportable DOCX draft
```

---

## 6. API Endpoints

### New Endpoints (Quality-Specific)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/quality/five-why` | POST | Run 5-Why analysis on a problem statement |
| `/quality/fishbone` | POST | Generate fishbone diagram data for an issue |
| `/quality/draft-capa` | POST | Generate pre-populated CAPA draft |
| `/quality/draft-8d` | POST | Generate pre-populated 8D report draft |
| `/quality/capa-status` | GET | Query open/overdue CAPAs by supplier or part |
| `/quality/pfmea-search` | POST | Search PFMEA for failure modes by part or process |
| `/quality/spc-summary` | POST | Summarize Cp/Cpk trends for a characteristic |
| `/quality/ncr-history` | POST | Retrieve NCR history for a part or supplier |

### Inherited Endpoints (from base system)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/upload` | POST | Ingest quality documents (PFMEA, CAPA, 8D, QMS) |
| `/query` | POST | Unified hybrid query (SQL + Documents) |
| `/query/documents` | POST | RAG-only query |
| `/query/sql/generate` | POST | Text-to-SQL generation |
| `/query/sql/execute` | POST | Execute approved SQL |
| `/health` | GET | Health check |
| `/documents` | GET | List ingested documents |

---

## 7. Query Routing — Quality-Specialized Keywords

### SQL Route
Structured data queries against defect/CAPA/NCR database:

> "How many open CAPAs", "defect count", "top failure modes", "overdue corrective actions", "Cpk for", "NCR history", "supplier quality rating", "scrap rate", "first pass yield", "how many", "list all", "show me", "which suppliers"

### Document Route
Unstructured document search across PFMEA, 8D, QMS, Control Plans:

> "what does the PFMEA say", "explain the control plan", "work instruction for", "what is the procedure", "8D for", "CAPA history", "root cause for", "failure mode", "detection method", "severity rating", "according to QMS"

### Hybrid Route
Both sources combined — data + document context:

> "show defect data and explain root cause", "failure counts and PFMEA context", "open CAPAs and corrective actions taken"

### Agent Route (New)
Structured problem-solving workflows:

> "5-why for", "fishbone", "Ishikawa", "draft 8D", "draft CAPA", "root cause analysis for", "cause and effect"

---

## 8. RAGAS Evaluation Dataset

20 curated question/answer/context triplets for continuous quality validation:

| # | Question Type | Example Question | Expected Source |
|---|---------------|-----------------|-----------------|
| 1 | SQL — count | "How many CAPAs are open past their due date?" | `capa_log` table |
| 2 | SQL — supplier | "Which suppliers have more than 5 NCRs this quarter?" | `ncr` + `suppliers` |
| 3 | SQL — SPC | "What parts have Cpk below 1.33 at Station 12?" | `inspection_results` |
| 4 | Doc — PFMEA | "What is the RPN for improper torque on assembly step 7?" | PFMEA document |
| 5 | Doc — 8D | "What containment action was taken for the fastener issue?" | 8D report D3 |
| 6 | Doc — QMS | "What is the escalation procedure for a Severity 9 defect?" | QMS procedure doc |
| 7 | Doc — Control Plan | "What gauge is specified for measuring flange thickness?" | Control plan |
| 8 | Hybrid | "Show NCR counts for EDV fascia and explain the control plan" | Both |
| 9 | 5-Why | "Why do torque failures recur at Station 12?" | Agent + both sources |
| 10 | Fishbone | "What are the Man and Machine causes for seal leakage?" | Agent + PFMEA + defects |
| 11–20 | Mixed | Additional manufacturing-specific Q&A pairs | Various |

**Target RAGAS scores:**
- Faithfulness: > 0.75 (answers grounded in retrieved context)
- Answer Relevancy: > 0.80 (answers directly address the question)
- Context Precision: > 0.70 (retrieved chunks are relevant)

---

## 9. Build Phases

### Phase 1 — Data Foundation (Week 1–2)
- [ ] Define and create PostgreSQL schema (defects, ncr, capa_log, eight_d, suppliers, inspection_results, corrective_actions)
- [ ] Generate realistic sample data (Python script, 500–1000 rows per table)
- [ ] Prepare sample quality documents: 2 PFMEA, 2 CAPA records, 1 8D report, 1 QMS procedure, 1 control plan
- [ ] Validate Docling chunking on PFMEA/8D formats — confirm structure preservation
- [ ] Ingest documents into Pinecone with quality-specific metadata tags
- [ ] Train Vanna.ai on quality schema + 20 seed SQL question/query pairs

### Phase 2 — Core Query Layer (Week 3–4)
- [ ] Fork base Multi-Source RAG repo
- [ ] Extend query router with quality-specific keyword sets and agent route
- [ ] Implement `/quality/pfmea-search`, `/quality/capa-status`, `/quality/ncr-history` endpoints
- [ ] Implement `/quality/spc-summary` endpoint with Cp/Cpk narrative generation
- [ ] End-to-end test: 10 SQL queries, 10 RAG queries, 5 hybrid queries
- [ ] Tune chunking parameters for PFMEA row-level fidelity

### Phase 3 — Quality Agents (Week 5–6)
- [ ] Build 5-Why LangGraph agent with RAG + SQL context injection
- [ ] Build Fishbone LangGraph agent with 6M categorization and evidence weighting
- [ ] Build CAPA/8D Draft Generator with structured JSON output
- [ ] Add `/quality/five-why`, `/quality/fishbone`, `/quality/draft-capa`, `/quality/draft-8d` endpoints
- [ ] Test agent outputs against 5 real-world quality problem scenarios
- [ ] Validate structured JSON output schema for downstream rendering

### Phase 4 — Evaluation & Hardening (Week 7)
- [ ] Build RAGAS evaluation dataset (20 Q&A/context triplets)
- [ ] Run baseline RAGAS scores; iterate on chunking/prompts to hit targets
- [ ] Add OPIK tracing to all quality agent endpoints
- [ ] Input validation: file type enforcement, query length limits, SQL safety checks
- [ ] Error handling: structured responses for empty retrievals, agent timeouts
- [ ] Load test: 50 concurrent requests against Lambda endpoint

### Phase 5 — Deployment & Documentation (Week 8)
- [ ] Deploy to AWS Lambda ARM64 via CI/CD (GitHub Actions)
- [ ] CloudWatch dashboard: invocations, error rate, agent latency per endpoint
- [ ] Update README with quality engineering use case, sample queries, schema docs
- [ ] Record a 3-minute demo: ingest PFMEA → hybrid query → 5-Why → 8D draft
- [ ] Update portfolio site with project entry and live API link
- [ ] Update resume bullet to reflect specialized system

---

## 10. Resume & Portfolio Framing

### Resume Bullet (updated after Phase 5)

> **Manufacturing Quality Engineering Assistant** | *LangGraph, FastAPI, Pinecone, Vanna.ai, AWS Lambda ARM64, RAGAS, OPIK*
> Built a production-grade GenAI assistant specializing in manufacturing quality workflows: natural-language querying of PFMEA records, CAPA logs, and NCR databases via hybrid RAG + Text-to-SQL; LangGraph agents automate 5-Why chains, fishbone diagram generation, and 8D/CAPA draft pre-population from retrieved evidence — deployed on AWS Lambda ARM64 with RAGAS evaluation scoring AI output quality.

### Apple Interview Talking Points

1. **Lab → Production translation:** *"I didn't just prototype this in a notebook — it's containerized, deployed serverless on Lambda, and validated with RAGAS metrics before any output reaches a user."*
2. **Connecting supply chain data to understand variation early:** *"The Text-to-SQL layer lets engineers ask 'which suppliers have recurring NCRs this quarter' in plain English — no BI tool, no analyst needed."*
3. **GenAI for day-to-day manufacturing challenges:** *"The 5-Why agent pulls from both historical defect data and PFMEA documentation simultaneously — the same reasoning a senior quality engineer does manually, but in seconds."*
4. **Gage R&R parallel for LLM outputs:** *"RAGAS evaluation plays the same role as measurement system analysis — before I trust the system's answers in production, I validate its faithfulness and relevancy scores the same way I'd validate a gauge."*

---

## 11. Technology Stack Summary

| Layer | Technology | Purpose |
|-------|-----------|---------|
| API Framework | FastAPI | REST endpoints, input validation |
| Orchestration | LangGraph + LangChain | Multi-step quality agents |
| Document Parsing | Docling HybridChunker | Structure-aware PFMEA/8D chunking |
| Vector Store | Pinecone (cosine, 1536-dim) | PFMEA, CAPA, QMS retrieval |
| Embeddings | OpenAI text-embedding-3-small | Document + query embeddings |
| LLM | GPT-4o | Reasoning, structured JSON output |
| Text-to-SQL | Vanna.ai | Natural language → quality database SQL |
| Database | PostgreSQL (Supabase) | Defects, CAPA, NCR, inspection results |
| Evaluation | RAGAS | Faithfulness, relevancy, context precision |
| Monitoring | OPIK + CloudWatch | Request tracing, latency, error rates |
| Deployment | AWS Lambda ARM64 | Serverless, auto-scaling, 20% cost savings |
| Containers | Docker (linux/arm64) | Reproducible builds |
| CI/CD | GitHub Actions | Push-to-deploy pipeline |
| Caching | SHA-256 content hash | 40–60% reduction in embedding API calls |

---

## 12. Open Questions / Decisions Needed

| # | Question | Options | Notes |
|---|----------|---------|-------|
| Q1 | Real vs synthetic data for initial build? | Synthetic (generated) vs anonymized real PFMEA/CAPA | Synthetic is faster to start; real data increases demo credibility |
| Q2 | Fishbone diagram rendering? | Text-only JSON / Streamlit SVG / React component | Phase 1 = JSON output; visual rendering in Phase 5 if time permits |
| Q3 | 8D/CAPA export format? | JSON only / DOCX export via python-docx | DOCX export makes demo more compelling for interview |
| Q4 | Scope of QMS documents? | Generic ISO 9001 procedures vs IATF 16949 automotive | IATF 16949 more relevant to Apple's supply chain context |
| Q5 | User authentication? | None (demo) / API key / OAuth | API key sufficient for portfolio; skip OAuth for now |

---

*This plan is a living document. Update status checkboxes in Phase 9 as work progresses.*
