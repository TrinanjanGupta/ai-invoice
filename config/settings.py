from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    app_name: str = "Invoice Digitizer"
    app_env: str = "development"
    secret_key: str = "change-me"

    database_url: str = "postgresql+asyncpg://postgres:password@localhost:5432/invoice_db"
    redis_url: str = "redis://localhost:6379/0"

    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "invoices"
    minio_secure: bool = False

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "invoice-expert"

    yolo_model_path: str = "data/models/doclayout_yolo_v8s/weights/best.pt"
    layoutlm_model_path: str = "data/models/layoutlmv3-finetuned"
    layoutlm_base_model: str = "microsoft/layoutlmv3-base"

    ocr_languages: str = "en,hi,bn"
    confidence_threshold: float = 0.80
    llm_fallback_threshold: float = 0.60
    max_file_size_mb: int = 50
    supported_formats: str = "pdf,jpg,jpeg,png,tiff,webp"

    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()
