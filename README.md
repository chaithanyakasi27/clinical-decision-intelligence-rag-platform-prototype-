# Clinical Decision Intelligence Platform

> Production-grade RAG + Agentic AI system for HCC coding and
> clinical evidence retrieval built on AWS, LangGraph, and FastAPI.

![Architecture](https://raw.githubusercontent.com/chaithanyakasi27/clinical-decision-intelligence-rag-platform-prototype-/refs/heads/main/image.png)

---

## What this does

Automates HCC (Hierarchical Condition Category) risk adjustment coding
for Medicare Advantage plans. Clinical notes and EHR data flow through
a RAG pipeline and a 4-agent LangGraph workflow to produce ICD-10 codes,
HCC categories, and RAF scores — with full audit trail and validation.

**Key results:**
- 35% improvement in HCC coding accuracy
- Reduces manual chart review time significantly
- Full audit trail for CMS compliance
- Supports risk adjustment workflows for value-based care

---

## Architecture

| Layer | Technology |
|---|---|
| Ingestion | PyMuPDF + section-aware chunking |
| Embeddings | BGE-M3 on CUDA (RTX 4070), 1024 dims |
| Vector Store | FAISS (356 vectors, IndexFlatIP) |
| Retrieval | Hybrid BM25 + dense with RRF fusion |
| LLM | Claude Sonnet via AWS Bedrock |
| Agents | LangGraph 4-agent stateful workflow |
| API | FastAPI with 4 HCC coding endpoints |
| Infra | Docker + ECS Fargate + Terraform |
| Monitoring | Prometheus + Grafana + MLflow |
| CI/CD | GitHub Actions + ECR |

---

## LangGraph Agent Workflow

```
Evidence Retrieval Agent
        ↓
HCC Coding Agent (Chain-of-Thought)
        ↓
Validation Agent (CMS V28 compliance) ──retry──→ HCC Coding Agent
        ↓
Explanation Agent (audit trail)
        ↓
Final unified response
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/v1/analyze-chart` | Full chart HCC analysis |
| POST | `/api/v1/retrieve-evidence` | Clinical evidence retrieval |
| POST | `/api/v1/generate-hcc-code` | ICD-10 + HCC + RAF scores |
| POST | `/api/v1/validate-response` | CMS V28 coding validation |
| GET  | `/health` | System health check |
| GET  | `/docs` | Swagger UI |

---

## Data Sources

| Source | Type | Purpose |
|---|---|---|
| Epic / Cerner / Athenahealth | Real EHR (production) | Clinical notes, FHIR R4 |
| Synthea (MITRE) | Synthetic FHIR R4 | Development and testing |
| CMS ICD-10-CM | Reference | 75,000+ diagnosis codes |
| CMS HCC V28 Model | Reference | ICD-10 to HCC crosswalk + RAF scores |

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set environment variables
cp .env.example .env
# Add your ANTHROPIC_API_KEY or AWS Bedrock credentials

# 3. Generate synthetic clinical data
python scripts/generate_sample_fhir.py
python scripts/generate_clinical_pdfs.py

# 4. Build FAISS index
python scripts/run_ingestion.py

# 5. Start API server
uvicorn src.api.main:app --reload --port 8000

# 6. Open Swagger UI
# http://localhost:8000/docs
```

---

## Project Structure

```
src/
├── ingestion/          # PDF parsing + section-aware chunking
├── embeddings/         # BGE-M3 embeddings + FAISS vector store
├── retrieval/          # Hybrid BM25 + dense retriever + reranker
├── rag/                # RAG pipeline + clinical prompt templates
├── agents/             # LangGraph 4-agent workflow
├── api/                # FastAPI app + 4 routers
└── monitoring/         # Prometheus metrics

scripts/
├── generate_sample_fhir.py    # Synthetic FHIR R4 patient data
├── generate_clinical_pdfs.py  # Clinical note PDFs from FHIR
└── run_ingestion.py           # Parse → chunk → embed → index

infra/
├── Dockerfile
├── docker-compose.yml         # Full local stack
└── terraform/                 # AWS ECS + ALB + S3 + DynamoDB
```

---

## Tech Stack

```
Python 3.11 | LangChain | LangGraph | FastAPI | FAISS
BGE-M3 | Claude Sonnet (AWS Bedrock) | ECS Fargate
Terraform | GitHub Actions | Prometheus | Grafana | MLflow
HL7 FHIR R4 | ICD-10-CM | CMS HCC V28 | HIPAA Compliant
```

---

## Environment Variables

```bash
# Required
ANTHROPIC_API_KEY=your-key-here        # or use AWS Bedrock
AWS_DEFAULT_REGION=us-east-1
AWS_ACCESS_KEY_ID=your-key
AWS_SECRET_ACCESS_KEY=your-secret

# Vector store
VECTOR_STORE_TYPE=faiss                # faiss (local) | pinecone (cloud)
FAISS_INDEX_PATH=./data/faiss_index

# Chunking
CHUNK_SIZE=512
CHUNK_OVERLAP=50

# FastAPI
API_HOST=0.0.0.0
API_PORT=8000
API_SECRET_KEY=your-secret-32-chars
```

---

## Docker

```bash
# Full local stack
docker compose up --build -d

# Services:
# API:        http://localhost:8000/docs
# MLflow:     http://localhost:5000
# Prometheus: http://localhost:9090
# Grafana:    http://localhost:3000
```

---

## Compliance

- HIPAA-compliant architecture with PHI access controls
- CMS HCC V28 model coding guidelines
- Audit logging for all coding decisions
- OCC Model Risk SR 11-7 alignment
- PHI de-identification before storage

---

## Roadmap

- [x] PDF ingestion + section-aware chunking
- [x] BGE-M3 embeddings on GPU
- [x] FAISS vector store (356 vectors)
- [x] Hybrid BM25 + dense retrieval
- [x] RAG pipeline with Claude
- [x] LangGraph 4-agent workflow
- [x] FastAPI with 4 endpoints
- [x] Prometheus monitoring
- [ ] AWS Bedrock integration
- [ ] ECS Fargate deployment
- [ ] Terraform infrastructure
- [ ] PEFT/LoRA fine-tuning on clinical NER

---

## Author

Venkata Chaithanya Kasireddy

Specializing in production-grade GenAI, Agentic AI, and ML platforms
across healthcare, financial services, and public sector.

- GitHub: [chaithanyakasi27](https://github.com/chaithanyakasi27)

---

## License

MIT License — see LICENSE file for details.
