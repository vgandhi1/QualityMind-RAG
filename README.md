<div align="center">

# QualityMind RAG

### AI-Powered Manufacturing Quality Engineering Assistant

*One natural-language interface across your PFMEA documents, CAPA logs, defect databases, and structured quality records*

### ▶ [**Live presentation**](https://vgandhi1.github.io/QualityMind-RAG/presentation.html) · [GitHub](https://github.com/vgandhi1/QualityMind-RAG)

[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![OpenAI GPT-4o](https://img.shields.io/badge/OpenAI-GPT--4o-412991?style=flat-square&logo=openai&logoColor=white)](https://openai.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-Agents-FF6B35?style=flat-square)](https://langchain-ai.github.io/langgraph/)
[![Pinecone](https://img.shields.io/badge/Pinecone-Vector%20DB-00A67E?style=flat-square)](https://pinecone.io)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Quality%20DB-336791?style=flat-square&logo=postgresql&logoColor=white)](https://postgresql.org)
[![AWS Lambda](https://img.shields.io/badge/AWS-Lambda%20ARM64-FF9900?style=flat-square&logo=amazonaws&logoColor=white)](https://aws.amazon.com/lambda/)
[![RAGAS](https://img.shields.io/badge/RAGAS-Evaluation-8B5CF6?style=flat-square)](https://docs.ragas.io)

</div>

---

## What Is This?

Quality engineers in high-volume manufacturing spend hours manually searching PFMEA spreadsheets, CAPA logs, 8D reports, and supplier databases to answer questions they face daily. **QualityMind RAG** replaces that fragmented workflow with a single conversational interface that reasons simultaneously across:

- **Unstructured documents** — PFMEA, DFMEA, 8D reports, CAPA records, control plans, work instructions, QMS procedures
- **Structured operational data** — defects, NCRs, supplier quality ratings, SPC inspection results, corrective actions

Beyond retrieval, three **LangGraph agent workflows** automate the problem-solving steps that senior engineers perform manually: tracing root causes with **5-Why chains**, building **Ishikawa fishbone diagrams**, and drafting **CAPA / 8D reports** pre-populated from retrieved evidence.

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          Engineer / User                                  │
│            Natural Language Query  ──  "Why are torque failures           │
│                                         recurring at Station 12?"         │
└────────────────────────────┬─────────────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                      FastAPI Application                                  │
│              POST /query  ·  POST /quality/*  ·  POST /upload             │
│          OPIK Tracing  │  Redis Query Cache  │  CloudWatch Logging        │
└────────────────────────┬─────────────────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                     Intelligent Query Router                               │
│                                                                            │
│   ┌─────────────┐   ┌──────────────┐   ┌────────────┐   ┌────────────┐  │
│   │  SQL Route  │   │  Doc Route   │   │  Hybrid    │   │   AGENT    │  │
│   │  (counts,   │   │  (PFMEA,     │   │  (data +   │   │  (5-Why,   │  │
│   │  CAPAs,     │   │  procedures, │   │  context   │   │  fishbone, │  │
│   │  NCRs, Cpk) │   │  8D, QMS)    │   │  combined) │   │  CAPA/8D)  │  │
│   └──────┬──────┘   └──────┬───────┘   └─────┬──────┘   └─────┬──────┘  │
└──────────┼─────────────────┼─────────────────┼────────────────┼──────────┘
           │                 │                  │                │
           ▼                 ▼                  │                ▼
┌──────────────┐   ┌──────────────────┐         │   ┌────────────────────────┐
│  Text-to-SQL │   │  Document RAG    │         │   │  Quality Agent         │
│  Engine      │   │  Pipeline        │◄────────┘   │  Workflows             │
│              │   │                  │             │                        │
│  Vanna 2.0   │   │  Docling         │             │  ┌──────────────────┐  │
│  GPT-4o      │   │  HybridChunker   │             │  │  5-Why LangGraph  │  │
│  PostgreSQL  │   │  Pinecone        │             │  │  (gather → chain  │  │
│              │   │  GPT-4o          │             │  │  → synthesize)    │  │
└──────┬───────┘   └─────────┬────────┘             │  └──────────────────┘  │
       │                     │                      │  ┌──────────────────┐  │
       ▼                     ▼                      │  │  Fishbone (6M)   │  │
┌──────────────┐   ┌──────────────────┐             │  │  Ishikawa graph  │  │
│  PostgreSQL  │   │  Pinecone        │             │  └──────────────────┘  │
│  (Supabase)  │   │  Vector Store    │             │  ┌──────────────────┐  │
│              │   │  (1536-dim)      │             │  │  CAPA / 8D Draft │  │
│  defects     │   │                  │             │  │  Generator       │  │
│  ncr         │   │  PFMEA chunks    │             │  └──────────────────┘  │
│  capa_log    │   │  CAPA records    │             └────────────────────────┘
│  eight_d     │   │  8D sections     │
│  suppliers   │   │  QMS procedures  │
│  insp_results│   │  Control plans   │
│  corr_actions│   │  Work instruc.   │
└──────────────┘   └──────────────────┘
                            │
                            ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                      Evaluation & Observability                            │
│                                                                            │
│  RAGAS (faithfulness > 0.75 · relevancy > 0.80 · agent structural valid) │
│  OPIK traces · Redis cache stats · CloudWatch dashboards                  │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## Feature Highlights

### Document Intelligence (RAG)

| Capability | Detail |
|---|---|
| **Document ingestion** | PDF, DOCX, XLSX, CSV, TXT, JSON — up to 50 MB |
| **Context-aware chunking** | Docling HybridChunker preserves PFMEA row structure, 8D discipline boundaries, heading hierarchy |
| **Quality metadata tagging** | Every chunk is tagged with `doc_type` (pfmea / 8d / capa / qms / control_plan / spc), extracted `part_numbers`, and `quality_keywords` for filtered retrieval |
| **Semantic search** | Pinecone cosine similarity (1536-dim, OpenAI text-embedding-3-small) |
| **Heading-aware citations** | Sources returned with full heading breadcrumb (e.g. `Control Plan > Station 12 > Torque Verification`) |
| **Document cache** | SHA-256 content-hash deduplication — skip re-embedding identical files (S3 or local) |

### Text-to-SQL (Structured Data)

| Capability | Detail |
|---|---|
| **Natural language → SQL** | Vanna 2.0 Agent Framework + GPT-4o with deterministic settings (temperature=0, seed=42) |
| **Human-in-the-loop approval** | SQL returned for review before execution; stale queries auto-evicted after 1 hour |
| **Schema-aware generation** | Full quality schema context (7 tables, join hints, 10 example Q&A pairs) injected per query |
| **Result caching** | SELECT queries cached in Redis for 15 minutes — instant repeat answers |
| **Safety guard** | Dangerous patterns (DROP, DELETE, TRUNCATE, ALTER) blocked before any execution |
| **Pinecone agent memory** | Optional persistent SQL training memory across sessions |

### Quality Agent Workflows (LangGraph)

| Workflow | Trigger Keywords | Output |
|---|---|---|
| **5-Why Analysis** | "5-why", "five why", "root cause analysis for" | JSON: `whys[]` chain + `root_cause` + `confidence_score` [0–1] |
| **Fishbone (Ishikawa)** | "fishbone", "ishikawa", "cause and effect" | JSON: 6M bones (Man / Machine / Method / Material / Measurement / Environment) with evidence + weight |
| **CAPA Draft** | "draft capa", "write corrective action" | JSON: pre-populated CAPA with root cause, actions, owner, due date |
| **8D Report Draft** | "draft 8d", "generate 8d" | JSON: D1–D8 disciplines populated from retrieved PFMEA + defect context |

Each workflow: **parallel RAG retrieval + SQL context injection → structured LLM synthesis → validated JSON output**. Graphs are pre-compiled at startup for zero per-request compilation cost.

### Quality Data API

Direct parameterized queries against the manufacturing schema — no SQL knowledge required:

- **CAPA Status** — filter by supplier, overdue-only flag, limit
- **NCR History** — by part number with date-sorted results
- **SPC Summary** — Cp/Cpk aggregates by part + characteristic + station with GPT-4o-mini narrative
- **PFMEA Search** — semantic search scoped to PFMEA document chunks

---

## Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| **API Framework** | FastAPI 0.115 | REST endpoints, Pydantic validation, async handlers |
| **LLM** | GPT-4o | RAG Q&A, 5-Why / fishbone / CAPA / 8D synthesis |
| **Narrative LLM** | GPT-4o-mini | SPC metric summaries (cost-efficient) |
| **Embeddings** | OpenAI text-embedding-3-small | 1536-dim document and query embeddings |
| **Vector Store** | Pinecone (cosine, serverless) | PFMEA, CAPA, 8D, QMS semantic retrieval |
| **Agent Orchestration** | LangGraph | Multi-step quality workflows with parallel gather nodes |
| **Text-to-SQL** | Vanna 2.0 Agent Framework | Natural language → audited PostgreSQL |
| **Database** | PostgreSQL (Supabase) | Defects, NCR, CAPA, 8D, suppliers, SPC results |
| **DB Connection Pool** | psycopg2 ThreadedConnectionPool | Reuses TCP connections across concurrent requests |
| **Document Parsing** | Docling HybridChunker | Structure-preserving PFMEA / 8D chunking |
| **Document Parsing (fallback)** | Unstructured.io | PDF, DOCX without Docling |
| **Query Cache** | Upstash Redis | RAG responses (1h TTL), embeddings (7d), SQL gen (24h), SQL results (15m) |
| **Document Cache** | S3 / local filesystem | SHA-256 chunk + embedding deduplication |
| **Evaluation** | RAGAS | Faithfulness, answer relevancy, agent structural validation |
| **Observability** | OPIK + CloudWatch | End-to-end request tracing, cost tracking |
| **Deployment** | AWS Lambda ARM64 | Serverless, auto-scaling (20% cost saving vs x86) |
| **Containers** | Docker (linux/arm64) | Multi-stage builds |
| **CI/CD** | GitHub Actions | Push-to-deploy pipeline |

---

## Database Schema

Seven tables covering the full manufacturing quality workflow:

```
suppliers ──────────────────────────────────────────────────────────┐
  id, supplier_code, name, tier(1-3), commodity,                    │
  quality_rating(0-100), active                                      │
                                                                     │
ncr ───────────────────────────────────────────────────────────┐    │
  id, ncr_number, part_number, description, quantity,          │    │
  disposition, status[open|closed|on-hold],                    │    │
  opened_date, closed_date, root_cause, cost_impact            │    │
                                                               │    │
defects ──────────────────────────────────┐                    │    │
  id, part_number, description,           │                    │    │
  failure_mode, detection_station,        ├── FK ──► ncr       │    │
  disposition, severity(1-10),            │                    │    │
  date_found, shift, operator_id          ├── FK ──────────────────► suppliers
                                          │
capa_log ──────────────────────────────────────────────────────┐    │
  id, capa_number, title, problem_statement, root_cause,       │    │
  corrective_action, preventive_action, owner,                 ├── FK ──► suppliers
  status[open|closed|verified|cancelled],                      │
  due_date, opened_date, closed_date, recurrence_flag          │
                                                               │
eight_d ──────────────────────────────────────────────────────────── FK ──► suppliers
  id, report_number, part_number, problem_statement,
  d1_team … d8_closure, status[open|closed|on-hold],
  opened_date, closed_date

inspection_results
  id, part_number, characteristic, measured_value,
  nominal, usl, lsl, cp, cpk, measurement_date, station,
  gauge_id, operator_id

corrective_actions ──────────────────────────────────────────────── FK ──► capa_log / eight_d
  id, action_text, owner, due_date, completed_date,
  status[open|closed|overdue|verified], verified
```

All `status`, `disposition`, `severity`, `tier`, and `quality_rating` columns are enforced with **PostgreSQL CHECK constraints**. Thirteen performance indexes cover the most common analytics query patterns.

---

## Quick Start

### Option A — Docker Compose (recommended)

PostgreSQL is provisioned automatically; schema and seed data applied on first run.

```bash
git clone https://github.com/your-org/qualitymind-rag.git
cd qualitymind-rag

cp .env.example .env
# Fill in your keys (see Configuration section below)

docker compose up --build -d
docker compose run --rm app python data/generate_sample_data.py
```

Open **[http://localhost:8000/docs](http://localhost:8000/docs)** — full Swagger UI with all endpoints.

### Option B — Local Python

```bash
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# Set DATABASE_URL=postgresql://user:pass@localhost:5432/quality_db

# Apply schema and generate sample data
psql "$DATABASE_URL" -f data/sql/schema.sql
python data/generate_sample_data.py

uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Configuration

Copy `.env.example` → `.env` and configure:

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | **Yes** | GPT-4o for RAG, agents, SQL generation |
| `PINECONE_API_KEY` | **Yes** | Document vector store |
| `PINECONE_INDEX_NAME` | **Yes** | Index for quality documents (1536-dim, cosine) |
| `PINECONE_ENVIRONMENT` | **Yes** | e.g. `us-east-1-aws` |
| `DATABASE_URL` | **Yes** | PostgreSQL connection string |
| `VANNA_PINECONE_INDEX` | Recommended | Separate index for SQL agent memory |
| `STORAGE_BACKEND` | No | `local` (default) or `s3` |
| `S3_CACHE_BUCKET` | If S3 | S3 bucket for document chunk cache |
| `UPSTASH_REDIS_URL` | No | Redis query cache (app degrades gracefully without it) |
| `UPSTASH_REDIS_TOKEN` | No | Redis auth token |
| `OPIK_API_KEY` | No | OPIK observability tracing |
| `RAG_MODEL` | No | Override LLM for RAG (default: `gpt-4o`) |
| `AGENT_MODEL` | No | Override LLM for agents (default: `gpt-4o`) |
| `NARRATIVE_MODEL` | No | Override LLM for SPC narratives (default: `gpt-4o-mini`) |
| `DB_POOL_MIN_CONNECTIONS` | No | Min DB pool size (default: 1) |
| `DB_POOL_MAX_CONNECTIONS` | No | Max DB pool size (default: 5) |

---

## API Reference

### Core Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service status — all dependencies, cache state |
| `GET` | `/info` | Application metadata and feature flags |
| `GET` | `/stats` | Document count, SQL queue depth, cache hit rates |
| `POST` | `/upload` | Ingest documents into Pinecone |
| `GET` | `/documents` | List uploaded documents |
| `POST` | `/query` | **Unified query** — auto-routes to SQL / RAG / Hybrid / Agent |
| `POST` | `/query/documents` | RAG-only document Q&A |
| `POST` | `/query/sql/generate` | Generate SQL from natural language (returns for approval) |
| `POST` | `/query/sql/execute` | Execute an approved SQL query |
| `GET` | `/query/sql/pending` | List queries awaiting approval |

### Quality Engineering Endpoints (`/quality`)

| Method | Endpoint | Body | Description |
|---|---|---|---|
| `POST` | `/quality/five-why` | `problem_statement` | LangGraph 5-Why root cause analysis |
| `POST` | `/quality/fishbone` | `effect`, `station?` | Ishikawa 6M fishbone diagram data |
| `POST` | `/quality/draft-capa` | `problem_statement`, `part_number?` | Pre-populated CAPA draft |
| `POST` | `/quality/draft-8d` | `problem_statement`, `part_number?` | Pre-populated 8D report draft |
| `GET` | `/quality/capa-status` | `supplier_id?`, `overdue_only?`, `limit?` | CAPA query with filters |
| `POST` | `/quality/pfmea-search` | `query`, `top_k?` | Semantic PFMEA chunk retrieval |
| `POST` | `/quality/ncr-history` | `part_number?`, `limit?` | NCR history by part |
| `POST` | `/quality/spc-summary` | `part_number`, `characteristic?`, `station?` | Cp/Cpk aggregates + narrative |

### Cache Management

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/cache/stats` | Document cache statistics |
| `DELETE` | `/cache/clear` | Clear document cache (all or by doc ID) |
| `GET` | `/cache/query/stats` | Redis query cache hit rates and cost savings |
| `DELETE` | `/cache/query` | Clear query cache (all or by type) |

---

## Sample Queries

### Unified Query — Automatic Routing

```bash
# Routes to SQL: structured count query
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "How many CAPAs are open past their due date?", "auto_approve_sql": true}'

# Routes to DOCUMENTS: policy question
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the escalation procedure in the QMS manual?"}'

# Routes to AGENT → 5-Why workflow
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "Do a 5-why analysis for recurring torque failures at Station 12"}'
```

### 5-Why Analysis

```bash
curl -X POST "http://localhost:8000/quality/five-why" \
  -H "Content-Type: application/json" \
  -d '{"problem_statement": "Recurring weld delamination on EDV fascia assembly"}'
```

*Illustrative response (actual content is LLM-generated; the structure below is what the schema + `app/agent_validation.py` enforce and what `tests/test_agent_validation.py` verifies):*

```json
{
  "status": "success",
  "analysis": {
    "problem_statement": "Recurring weld delamination on EDV fascia assembly",
    "whys": [
      {"level": 1, "why": "Ultrasonic weld joint failing at adhesion interface", "evidence_source": "PFMEA-EDV-042 §3.2"},
      {"level": 2, "why": "Weld amplitude drifting outside ±5% control window", "evidence_source": "inspection_results, Q3 2024"},
      {"level": 3, "why": "No SPC control chart active for weld amplitude", "evidence_source": "Control Plan Rev C gap"},
      {"level": 4, "why": "Control plan not updated after ECN-2024-0187", "evidence_source": "ECN review log"},
      {"level": 5, "why": "ECN process does not trigger mandatory control plan review", "evidence_source": "QMS-PROC-014 §6.1"}
    ],
    "root_cause": "ECN change management process missing mandatory control plan update gate",
    "recommended_action": "Update QMS-PROC-014 to require control plan sign-off for all ECNs affecting process parameters",
    "confidence_score": 0.82
  }
}
```

### Fishbone Diagram

```bash
curl -X POST "http://localhost:8000/quality/fishbone" \
  -H "Content-Type: application/json" \
  -d '{"effect": "Fastener torque failures", "station": "Station 12"}'
```

*Illustrative response (LLM-generated content; structure enforced by the 6M validator):*

```json
{
  "status": "success",
  "fishbone": {
    "effect": "Fastener torque failures at Station 12",
    "bones": {
      "Man":         [{"cause": "Operator re-certification gap", "evidence": "Training records", "weight": 0.6}],
      "Machine":     [{"cause": "Torque tool calibration overdue", "evidence": "Gauge R&R report", "weight": 0.9}],
      "Method":      [{"cause": "Work instruction step ambiguity", "evidence": "WI-ST12-007 §4", "weight": 0.7}],
      "Material":    [{"cause": "Fastener lot dimensional variation", "evidence": "NCR-2024-0891", "weight": 0.5}],
      "Measurement": [{"cause": "Torque wrench resolution inadequate", "evidence": "MSA study 2023", "weight": 0.8}],
      "Environment": [{"cause": "Temperature effect on thread engagement", "evidence": "PFMEA-ST12-003", "weight": 0.3}]
    }
  }
}
```

### Text-to-SQL

```bash
# Generate SQL (returned for approval)
curl -X POST "http://localhost:8000/query/sql/generate" \
  -d "question=Which suppliers have more than 5 NCRs opened this year?"

# Execute after review
curl -X POST "http://localhost:8000/query/sql/execute?query_id=<id>&approved=true"
```

### SPC Summary

```bash
curl -X POST "http://localhost:8000/quality/spc-summary" \
  -H "Content-Type: application/json" \
  -d '{"part_number": "EDV-FASCIA-01", "station": "Station 12"}'
```

---

## Document Upload

```bash
# Upload a PFMEA spreadsheet
curl -X POST "http://localhost:8000/upload" \
  -F "file=@PFMEA_EDV_Fascia_Rev3.pdf"

# Upload a QMS procedure
curl -X POST "http://localhost:8000/upload" \
  -F "file=@QMS-PROC-014-Escalation.pdf"

# Upload an 8D report
curl -X POST "http://localhost:8000/upload" \
  -F "file=@8D-2024-Station12-Torque.docx"
```

Documents are automatically:
1. Parsed with Docling (structure-aware) or Unstructured.io (fallback)
2. Tagged with quality domain metadata (`doc_type`, `part_numbers`, `quality_keywords`)
3. Chunked with heading hierarchy preserved
4. Embedded and stored in Pinecone
5. Cached by content hash (skip re-processing identical files)

---

## Evaluation

```bash
python evaluate.py
```

Evaluates two dimensions:

**RAGAS** (retrieval quality — Documents / SQL / Hybrid queries). Thresholds
below are the **acceptance targets** the harness asserts against; running it
requires `OPENAI_API_KEY` + `PINECONE_*` and a seeded database:

| Metric | Target | Measures |
|---|---|---|
| Faithfulness | > 0.75 | Answers grounded in retrieved context |
| Answer Relevancy | > 0.80 | Answers directly address the question |

**Structural Validation** (agent workflow outputs):

| Workflow | Validates |
|---|---|
| 5-Why | All 5 JSON keys present · 3–5 why levels · confidence_score ∈ [0,1] |
| Fishbone | All 6M bone categories present and non-empty |
| CAPA Draft | problem_statement · root_cause · corrective_action · preventive_action |
| 8D Draft | All disciplines D1–D8 populated |

Test cases are defined in `data/eval/ragas_dataset.json` (17 Q&A pairs across SQL, Documents, Hybrid, and Agent types).

The **structural validators** (`app/agent_validation.py`) and the **query router**
and **input/SQL-safety guards** (`app/utils.py`) run fully offline — no API keys —
and are covered by the unit suite:

```bash
pytest tests/test_router.py tests/test_validators.py tests/test_agent_validation.py -q
```

These pin the agent output contracts (5-Why keys + 3–5 whys + confidence ∈ [0,1],
6M fishbone bones, CAPA fields, 8D D1–D8) and the dangerous-SQL block
(DROP/DELETE/TRUNCATE/ALTER) independently of any live LLM call.

---

## Deployment

### AWS Lambda (Production)

```bash
# Build ARM64 Lambda image
docker build -f Dockerfile.lambda -t qualitymind-rag:lambda --platform linux/arm64 .

# Push to ECR and update Lambda function
aws ecr get-login-password | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
docker tag qualitymind-rag:lambda <account>.dkr.ecr.<region>.amazonaws.com/qualitymind-rag:latest
docker push <account>.dkr.ecr.<region>.amazonaws.com/qualitymind-rag:latest
aws lambda update-function-code --function-name qualitymind-rag \
  --image-uri <account>.dkr.ecr.<region>.amazonaws.com/qualitymind-rag:latest
```

**Lambda configuration notes:**
- Set `ROOT_PATH=/prod` and `ENVIRONMENT=production` in Lambda env vars
- `STORAGE_BACKEND=s3` — document cache uses S3 (Lambda `/tmp` is ephemeral)
- Memory: 1024 MB recommended; timeout: 300s for agent workflows
- Attach IAM role with `AmazonS3FullAccess` on the cache bucket

### Local / VM

```bash
docker build -t qualitymind-rag .
docker run -p 8000:8000 --env-file .env qualitymind-rag
```

### CI/CD

GitHub Actions workflow in `.github/workflows/` — configure `ECR_REPOSITORY` and `LAMBDA_FUNCTION_NAME` as repository secrets for push-to-deploy.

---

## Development

```bash
# Run tests
pytest tests/ -q

# Lint and format
ruff check app/
ruff format app/

# Run evaluation suite
python evaluate.py

# Generate fresh sample data
python data/generate_sample_data.py

# Apply schema changes
psql "$DATABASE_URL" -f data/sql/schema.sql
```

<details>
<summary>Project structure</summary>

```
qualitymind-rag/
├── app/
│   ├── api/
│   │   └── quality_routes.py        # /quality/* endpoints
│   ├── services/
│   │   ├── cache_service.py         # Document chunk/embed cache (S3/local)
│   │   ├── docling_service.py       # Context-aware PDF/DOCX chunking
│   │   ├── document_service.py      # Document parsing + quality metadata extraction
│   │   ├── embedding_service.py     # OpenAI embeddings with Redis cache
│   │   ├── local_storage.py         # Local file storage backend
│   │   ├── quality_data_service.py  # Parameterized SQL queries + DB pool
│   │   ├── quality_langgraph.py     # 5-Why / fishbone / CAPA / 8D LangGraph agents
│   │   ├── query_cache_service.py   # Redis query-level cache
│   │   ├── rag_service.py           # Full RAG pipeline (retrieve → generate)
│   │   ├── router_service.py        # Keyword router + agent type detection
│   │   ├── s3_storage.py            # S3 storage backend
│   │   ├── service_registry.py      # Singleton registry (breaks circular imports)
│   │   ├── sql_service.py           # Vanna 2.0 Text-to-SQL + approval workflow
│   │   ├── storage_backend.py       # Storage abstraction interface
│   │   └── vector_service.py        # Pinecone upsert and search
│   ├── config.py                    # Pydantic settings (all env vars)
│   ├── logging_config.py
│   ├── main.py                      # FastAPI app + startup + all core endpoints
│   └── utils.py                     # File/query validation, error responses
├── data/
│   ├── eval/
│   │   └── ragas_dataset.json       # 17 Q&A evaluation test cases
│   ├── sql/
│   │   └── schema.sql               # PostgreSQL schema with CHECK constraints + indexes
│   └── generate_sample_data.py
├── tests/
│   └── test_storage_backends.py
├── evaluate.py                      # RAGAS + agent structural evaluation
├── lambda_handler.py                # AWS Lambda entry point (Mangum)
├── Dockerfile                       # Standard image
├── Dockerfile.lambda                # Lambda ARM64 image
├── docker-compose.yml               # App + PostgreSQL
├── pyproject.toml
└── requirements.txt
```

</details>

---

## Troubleshooting

| Symptom | Check |
|---|---|
| `503` on SQL / quality DB endpoints | `DATABASE_URL` set · PostgreSQL running · schema applied (`data/sql/schema.sql`) · seed data generated |
| `503` on RAG / upload | `OPENAI_API_KEY` and `PINECONE_*` configured · Pinecone index exists at 1536-dim cosine |
| `503` on quality agent endpoints | `OPENAI_API_KEY` set · RAG service initialized (check `/health`) |
| SQL query never executes | Approval flow is human-in-the-loop by default — call `POST /query/sql/execute?query_id=<id>&approved=true`, or set `auto_approve_sql=true` in `/query` for testing only |
| Docling timeout on Lambda | Expected — simple `.txt`/`.csv`/`.json` files bypass Docling automatically; large PDF parsing requires Tesseract Lambda layer |
| Agent outputs missing keys | Check OPIK traces or logs for the raw LLM response — usually indicates insufficient context from RAG retrieval |
| Cache not hitting | Verify `UPSTASH_REDIS_URL` and `UPSTASH_REDIS_TOKEN` are set; `/cache/query/stats` shows enabled state |
| DB schema already exists | Run `docker compose down -v` to reset the Postgres volume (destructive — drops all data) |

---

## Design Decisions

| Decision | Choice | Rationale |
|---|---|---|
| SQL generation safety | Human-in-the-loop approval before execution | Prevents LLM hallucinations from mutating production data; auto-approve only for testing |
| Router implementation | Keyword matching with quality-domain specificity | Deterministic, zero-latency, zero-cost; LLM-based classification reserved for future fine-tuning |
| Agent SQL context | Blocked by `check_dangerous_sql()` before auto-execution | Workflow SQL is generated without user review — safety check compensates |
| DB connection strategy | `ThreadedConnectionPool` (psycopg2) | Reuses connections across concurrent requests; avoids 100ms+ TCP handshake per query |
| Graph pre-compilation | Compiled once in `__init__` | LangGraph compilation is expensive; reusing compiled graphs halves agent latency |
| Model separation | GPT-4o for reasoning, GPT-4o-mini for narratives | Cost optimization: SPC summaries don't require full reasoning capability |

---

<div align="center">

Built with FastAPI · LangGraph · Vanna 2.0 · Pinecone · OpenAI · PostgreSQL

*Bringing AI-grade root cause analysis to manufacturing quality teams*

</div>
