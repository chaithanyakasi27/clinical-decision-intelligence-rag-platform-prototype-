# Prometheus metrics for the Clinical Decision Intelligence API
#
# What this does:
#   Defines counters and histograms that track API usage.
#   Prometheus scrapes these at /metrics every 15 seconds.
#   Grafana displays them as dashboards.
#
# Metrics tracked:
#   - Total API requests per endpoint
#   - Request latency per endpoint
#   - Total tokens used (cost tracking)
#   - FAISS retrieval hit count
#   - Agent pipeline execution count
# ============================================================

from prometheus_client import Counter, Histogram, Gauge

# ── Request counters ──────────────────────────────────────────
# Incremented every time an endpoint is called

API_REQUESTS_TOTAL = Counter(
    "cdip_api_requests_total",
    "Total API requests by endpoint and status",
    ["endpoint", "status"],  # labels
)

# ── Latency histograms ────────────────────────────────────────
# Measures how long each endpoint takes to respond
# Buckets: 0.1s, 0.5s, 1s, 2s, 5s, 10s, 30s

API_LATENCY_SECONDS = Histogram(
    "cdip_api_latency_seconds",
    "API request latency in seconds",
    ["endpoint"],
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
)

# ── Token usage counter ───────────────────────────────────────
# Tracks Claude API token consumption for cost monitoring

LLM_TOKENS_TOTAL = Counter(
    "cdip_llm_tokens_total",
    "Total LLM tokens consumed by model and type",
    ["model", "token_type"],  # input_tokens, output_tokens
)

# ── Retrieval metrics ─────────────────────────────────────────
# Tracks FAISS vector search performance

RETRIEVAL_HITS = Histogram(
    "cdip_retrieval_chunks_returned",
    "Number of chunks returned per retrieval",
    buckets=[1, 2, 3, 5, 8, 10, 15, 20],
)

# ── Agent pipeline metrics ────────────────────────────────────
# Tracks 4-agent LangGraph pipeline execution

AGENT_PIPELINE_TOTAL = Counter(
    "cdip_agent_pipeline_total",
    "Total agent pipeline executions by outcome",
    ["outcome"],  # success, failure, validation_retry
)

# ── FAISS index gauge ─────────────────────────────────────────
# Current number of vectors in the index

FAISS_VECTOR_COUNT = Gauge(
    "cdip_faiss_vector_count",
    "Current number of vectors in FAISS index",
)

# ── Helper functions ──────────────────────────────────────────
# Call these from router files to record metrics

def record_request(endpoint: str, status: str = "success"):
    """Record an API request."""
    API_REQUESTS_TOTAL.labels(
        endpoint=endpoint,
        status=status,
    ).inc()


def record_tokens(model: str, input_tokens: int, output_tokens: int):
    """Record LLM token usage."""
    LLM_TOKENS_TOTAL.labels(
        model=model,
        token_type="input",
    ).inc(input_tokens)
    LLM_TOKENS_TOTAL.labels(
        model=model,
        token_type="output",
    ).inc(output_tokens)


def record_retrieval(chunk_count: int):
    """Record retrieval result count."""
    RETRIEVAL_HITS.observe(chunk_count)


def update_faiss_count(count: int):
    """Update FAISS vector count gauge."""
    FAISS_VECTOR_COUNT.set(count)