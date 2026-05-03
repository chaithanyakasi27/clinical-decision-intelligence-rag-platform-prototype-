# ============================================================
# Clinical Decision Intelligence Platform — Makefile
# Usage: make <target>
# ============================================================
 
.PHONY: help setup install env data ingest index run test lint clean docker-up docker-down
 
PYTHON     := python3.11
VENV       := .venv
PIP        := $(VENV)/bin/pip
PYTHON_BIN := $(VENV)/bin/python
UVICORN    := $(VENV)/bin/uvicorn

# ── Default target ───────────────────────────────────────────
help:
	@echo ""
	@echo "  Clinical Decision Intelligence Platform"
	@echo "  ───────────────────────────────────────"
	@echo "  make setup       → full first-time setup (venv + install + .env)"
	@echo "  make install     → install/update pip dependencies"
	@echo "  make env         → copy .env.example to .env"
	@echo "  make data        → run Synthea + generate clinical PDFs"
	@echo "  make ingest      → parse PDFs → chunk → embed → build FAISS index"
	@echo "  make run         → start FastAPI server (localhost:8000)"
	@echo "  make test        → run pytest with coverage"
	@echo "  make lint        → ruff check + mypy"
	@echo "  make docker-up   → start full local stack (API + MLflow + Prometheus)"
	@echo "  make docker-down → stop docker stack"
	@echo "  make mlflow      → open MLflow UI (localhost:5000)"
	@echo "  make clean       → remove venv, cache, FAISS index"
	@echo ""

	# ── First-time setup ─────────────────────────────────────────
setup: env
	@echo "→ Creating Python 3.11 virtual environment..."
	$(PYTHON) -m venv $(VENV)
	@echo "→ Upgrading pip..."
	$(PIP) install --upgrade pip setuptools wheel
	@echo "→ Installing dependencies..."
	$(PIP) install -r requirements.txt
	@echo "→ Installing pre-commit hooks..."
	$(VENV)/bin/pre-commit install
	@echo ""
	@echo "✓ Setup complete. Edit .env with your API keys, then run: make data"

	# ── Install deps ─────────────────────────────────────────────
install:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
 
# ── Copy env file ────────────────────────────────────────────
env:
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "✓ Created .env from .env.example — add your API keys now"; \
	else \
		echo "· .env already exists, skipping"; \
	fi
 
# ── Data generation ──────────────────────────────────────────
data:
	@echo "→ Generating synthetic clinical dataset..."
	$(PYTHON_BIN) scripts/generate_dataset.py
	@echo "✓ Dataset ready in data/"
 
 # ── Ingest pipeline ──────────────────────────────────────────
ingest:
	@echo "→ Running ingestion pipeline (parse → chunk → embed → index)..."
	$(PYTHON_BIN) scripts/run_ingestion.py
	@echo "✓ FAISS index built at data/faiss_index/"
 
# ── Run API server ───────────────────────────────────────────
run:
	@echo "→ Starting FastAPI on http://localhost:8000"
	@echo "   Swagger docs: http://localhost:8000/docs"
	$(UVICORN) src.api.main:app --host 0.0.0.0 --port 8000 --reload
 
# ── Tests ────────────────────────────────────────────────────
test:
	$(VENV)/bin/pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html
	@echo "✓ Coverage report: htmlcov/index.html"
 
test-fast:
	$(VENV)/bin/pytest tests/ -v -x --ignore=tests/test_agents.py
 
# ── Linting ──────────────────────────────────────────────────
lint:
	$(VENV)/bin/ruff check src/ tests/ scripts/
	$(VENV)/bin/mypy src/ --ignore-missing-imports
 
format:
	$(VENV)/bin/ruff format src/ tests/ scripts/

# ── Docker stack ─────────────────────────────────────────────
docker-up:
	@echo "→ Starting local stack: FastAPI + MLflow + Prometheus + Grafana + LocalStack"
	docker compose up --build -d
	@echo ""
	@echo "  Services running:"
	@echo "  → API:        http://localhost:8000/docs"
	@echo "  → MLflow:     http://localhost:5000"
	@echo "  → Prometheus: http://localhost:9090"
	@echo "  → Grafana:    http://localhost:3000  (admin/admin)"
	@echo "  → LocalStack: http://localhost:4566"
 
docker-down:
	docker compose down
 
docker-logs:
	docker compose logs -f api

# ── MLflow UI ────────────────────────────────────────────────
mlflow:
	@echo "→ Opening MLflow at http://localhost:5000"
	$(VENV)/bin/mlflow ui --host 0.0.0.0 --port 5000
 
# ── Jupyter notebook ─────────────────────────────────────────
notebook:
	$(VENV)/bin/jupyter lab --port 8888
# ── AWS deploy (phase 6) ─────────────────────────────────────
tf-init:
	cd infra/terraform && terraform init
 
tf-plan:
	cd infra/terraform && terraform plan
 
tf-apply:
	cd infra/terraform && terraform apply -auto-approve
 
ecr-push:
	@echo "→ Building and pushing to ECR..."
	$(PYTHON_BIN) scripts/ecr_push.py
 
# ── Clean ────────────────────────────────────────────────────
clean:
	rm -rf $(VENV) .pytest_cache htmlcov .ruff_cache .mypy_cache
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -name "*.pyc" -delete
	@echo "✓ Cleaned"
 
clean-data:
	rm -rf data/faiss_index data/synthea_output data/clinical_notes
	@echo "✓ Data directories cleaned (reference data preserved)"
 