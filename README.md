# Manufacturing Quality Engineering Assistant

A production-style **FastAPI** application for **manufacturing quality engineering**: one natural-language interface over **PFMEA / CAPA / 8D / NCR / QMS-style documents** (RAG) and **structured defect, supplier, and SPC data** (text-to-SQL). It adds **LangGraph** workflows for **5-Why**, **fishbone (Ishikawa)**, and **CAPA / 8D draft** generation on top of a proven hybrid RAG + SQL stack.

**Live API docs (local):** [http://localhost:8000/docs](http://localhost:8000/docs)

---

## Features

### Core (hybrid RAG + SQL)

- **Document RAG** — Upload and query PDF, DOCX, TXT, CSV, JSON, XLSX; Docling-aware chunking and Pinecone retrieval  
- **Text-to-SQL** — Vanna 2.0 + PostgreSQL; generate SQL, approve, execute  
- **Intelligent routing** — `POST /query` routes to SQL, documents, **HYBRID**, or an **AGENT** hint for quality workflows  

### Manufacturing & quality

- **Quality REST API** — `/quality/*` for CAPA status, NCR history, SPC-style summaries, PFMEA-oriented search, 5-Why, fishbone, CAPA/8D drafts  
- **Manufacturing schema** — Defects, NCR, CAPA, 8D, suppliers, inspection results, corrective actions (`data/sql/schema.sql`)  
- **Synthetic seed data** — `data/generate_sample_data.py` for demos and tests  

### Performance & operations

- **Caching** — Document chunk/embed cache and optional Redis query cache (same patterns as the reference stack)  
- **AWS Lambda** — `Dockerfile.lambda`, ARM64-friendly builds, `lambda_handler.py` lazy init  
- **CI/CD** — GitHub Actions (tests + deploy workflow templates under `.github/workflows/`)  

### Evaluation & monitoring

- **RAGAS-oriented eval seeds** — `data/eval/ragas_dataset.json`  
- **OPIK** — Optional tracing (when configured)  
- **Validation** — File and query limits; parameterized SQL on quality endpoints  

---

## Table of contents

- [Quick start](#quick-start)  
- [Prerequisites](#prerequisites)  
- [Installation](#installation)  
- [Configuration](#configuration)  
- [API endpoints](#api-endpoints)  
- [Query routing](#query-routing)  
- [Deployment](#deployment)  
- [Development](#development)  
- [Troubleshooting](#troubleshooting)  
- [Documentation & lineage](#documentation--lineage)  

---

## Quick start

### Option A — Docker Compose (recommended)

Includes **PostgreSQL** and applies **`data/sql/schema.sql`** on first DB init.

```bash
cd mfg-quality-assistant
cp .env.example .env
# Set OPENAI_API_KEY, PINECONE_API_KEY, indexes, etc.

docker compose up --build -d
docker compose run --rm app python data/generate_sample_data.py
```

Then open [http://localhost:8000/docs](http://localhost:8000/docs) and [http://localhost:8000/health](http://localhost:8000/health).

### Option B — Local API (Postgres already running)

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# DATABASE_URL=postgresql://mfg:mfg@localhost:5432/quality_db
# Apply schema if needed: psql "$DATABASE_URL" -f data/sql/schema.sql

python data/generate_sample_data.py
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Optional: `hatch run generate-sample-data` if you use Hatch.

---

## Prerequisites

| Item | Notes |
|------|--------|
| **Python 3.12+** | For local runs |
| **Docker + Compose** | Optional; easiest path for DB + app |
| **OpenAI API key** | Embeddings, RAG, Vanna SQL, quality agents |
| **Pinecone** | Document index + separate Vanna memory index (see `.env.example`) |
| **PostgreSQL** | Quality + SQL; Compose provides it locally |
| **OPIK / Redis** | Optional monitoring and query cache |

---

## Installation

**System packages** (for Docling / PDF pipelines): same as a typical Docling setup — e.g. on Ubuntu: `libmagic1`, `poppler-utils`, `tesseract-ocr` (see reference project README for detail).

**Python:**

```bash
pip install -r requirements.txt
# or: uv pip install -r requirements.txt
```

---

## Configuration

Copy **`.env.example`** → **`.env`** and set at least:

- `OPENAI_API_KEY`  
- `PINECONE_API_KEY`, `PINECONE_INDEX_NAME`, `PINECONE_ENVIRONMENT`  
- `VANNA_PINECONE_INDEX`, `VANNA_NAMESPACE` (SQL agent memory)  
- `DATABASE_URL` (Compose sets this for the `app` service automatically)  

Optional: `OPIK_API_KEY`, `UPSTASH_REDIS_URL` / `UPSTASH_REDIS_TOKEN`, `STORAGE_BACKEND`, S3 settings for document cache.

---

## API endpoints

### Core

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health and dependency flags |
| POST | `/upload` | Ingest documents into Pinecone |
| POST | `/query` | Unified query (routing) |
| POST | `/query/documents` | RAG-only |
| POST | `/query/sql/generate` | Generate SQL (approval flow) |
| POST | `/query/sql/execute` | Execute approved SQL |
| GET | `/documents` | List uploads |
| — | `/cache/*`, `/vectors/clear`, `/stats`, … | Caching and ops (see OpenAPI) |

### Quality (`/quality`)

| Method | Path | Description |
|--------|------|-------------|
| POST | `/quality/five-why` | 5-Why analysis (JSON body: `problem_statement`) |
| POST | `/quality/fishbone` | Ishikawa data (`effect`, optional `station`) |
| POST | `/quality/draft-capa` | CAPA draft JSON |
| POST | `/quality/draft-8d` | 8D draft JSON |
| GET | `/quality/capa-status` | Query params: `supplier_id`, `overdue_only`, `limit` |
| POST | `/quality/pfmea-search` | Semantic chunk search (`query`, `top_k`) |
| POST | `/quality/ncr-history` | NCR list (`part_number` optional) |
| POST | `/quality/spc-summary` | Cp/Cpk aggregates + short narrative |

Full schemas: **Swagger UI** at `/docs`.

---

## Query routing

`POST /query` uses keyword-style routing:

- **SQL** — counts, CAPAs, NCRs, defects, Cpk, suppliers, …  
- **DOCUMENTS** — PFMEA, procedures, control plan language, “what does the doc say”, …  
- **HYBRID** — explicit “data + explain” style prompts  
- **AGENT** — suggests using `/quality/*` for structured workflows (5-Why, fishbone, drafts)  

---

## Deployment

- **Local / VM:** `Dockerfile` + `uvicorn`  
- **AWS Lambda:** build with **`Dockerfile.lambda`** (ARM64 in CI); handler **`lambda_handler.py`**  
- **GitHub Actions:** `.github/workflows/` — adjust `ECR_REPOSITORY` / `LAMBDA_FUNCTION` env in `deploy.yml` for your account  

---

## Development

```bash
ruff check app/
black app/                    # optional
pytest --override-ini='addopts=-q'   # omit override if pytest-cov installed
```

---

## Troubleshooting

| Symptom | Check |
|---------|--------|
| 503 on SQL / quality DB | `DATABASE_URL`, Postgres up, `data/sql/schema.sql` applied, seed run |
| 503 on RAG / upload | `OPENAI_API_KEY`, `PINECONE_*`, index dimensions (1536) |
| 503 on quality agents | `OPENAI_API_KEY`; RAG + SQL initialized for full context |
| DB init did not run | First-time only; for a clean slate: `docker compose down -v` (destroys DB volume) |

---

## Documentation & lineage

**Product specification (phases, architecture, data model, goals)** — see **[`plan.md`](plan.md)**.

**How this repo relates to the open-source base stack** (synced paths, what is manufacturing-only, how to merge upstream) — see **[`docs/REFERENCE_MULTIDATA.md`](docs/REFERENCE_MULTIDATA.md)** and the upstream repository:

**[https://github.com/sourangshupal/multidata-rag-project](https://github.com/sourangshupal/multidata-rag-project)**

This project inherits the **MIT** spirit of that codebase; keep upstream copyright/license notice in distributions. A root `LICENSE` file is recommended if you publish the repo publicly.

For a longer, tutorial-style README (Pinecone setup, determinism knobs, extended deployment narrative), compare **`reference/multidata-rag-project/README.md`** in this workspace.
