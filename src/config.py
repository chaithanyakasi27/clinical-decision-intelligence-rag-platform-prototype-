# src/config.py
# ============================================================
# Centralised settings — loaded from .env via pydantic-settings
# Import this everywhere instead of os.getenv() directly
# ============================================================

from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
 
 
BASE_DIR = Path(__file__).resolve().parent.parent
 
 
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )
 
    # ── Project ──────────────────────────────────────────────
    project_name: str = "clinical-decision-intelligence"
    environment: str = "local"
    log_level: str = "INFO"
    debug: bool = True

    # ── OpenAI ───────────────────────────────────────────────
    openai_api_key: str = ""
    openai_embedding_model: str = "text-embedding-ada-002"
    openai_chat_model: str = "claude-sonnet-4-20250514"
    openai_max_tokens: int = 4096
    openai_temperature: float = 0.0
 
    # ── Anthropic ────────────────────────────────────────────
    anthropic_api_key: str = ""
 
    # ── AWS ──────────────────────────────────────────────────
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_default_region: str = "us-east-1"
    aws_s3_bucket_raw: str = "cdip-raw-clinical-docs-dev"
    aws_s3_bucket_processed: str = "cdip-processed-docs-dev"
    aws_dynamodb_table: str = "cdip-metadata-dev"
    use_localstack: bool = True
    localstack_endpoint: str = "http://localhost:4566"

     # ── Vector store ─────────────────────────────────────────
    vector_store_type: str = "faiss"
    faiss_index_path: Path = BASE_DIR / "data" / "faiss_index"
    pinecone_api_key: str = ""
    pinecone_environment: str = "us-east-1-aws"
    pinecone_index_name: str = "cdip-clinical-embeddings"

    # ── Chunking ─────────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 50
    embedding_batch_size: int = 32

     # ── Retrieval ────────────────────────────────────────────
    retriever_top_k: int = 10
    reranker_top_k: int = 5
    bm25_weight: float = 0.3
    dense_weight: float = 0.7

    # ── FastAPI ───────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = 2
    api_secret_key: str = "change-this-secret-key-in-production"
    api_algorithm: str = "HS256"
    api_access_token_expire_minutes: int = 60
    rate_limit_per_minute: int = 60

     # ── MLflow ────────────────────────────────────────────────
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "cdip-clinical-rag"

    # ── Reference data paths ─────────────────────────────────
    icd10_codes_path: Path = BASE_DIR / "data/reference/icd10cm_codes_2024.txt"
    hcc_mappings_path: Path = BASE_DIR / "data/reference/2024_cms_hcc_mappings.csv"
    hcc_coefficients_path: Path = BASE_DIR / "data/reference/hcc_coefficients.csv"
    synthea_output_dir: Path = BASE_DIR / "data/synthea_output"
    clinical_notes_dir: Path = BASE_DIR / "data/clinical_notes"

    @property
    def aws_endpoint_url(self) -> str | None:
        """Returns LocalStack endpoint when running locally."""
        return self.localstack_endpoint if self.use_localstack else None
 
    @property
    def is_production(self) -> bool:
        return self.environment == "production"
    
@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton — call get_settings() anywhere."""
    return Settings()
 
 
# Convenience alias used throughout the codebase
settings = get_settings()