import os
import warnings

# Suppress library warnings and noisy C++/Python logger messages across the application
warnings.filterwarnings("ignore")
os.environ.setdefault("ORT_LOG_LEVEL", "3")
os.environ.setdefault("PADDLE_PDX_LOG_LEVEL", "ERROR")
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("FLAGS_enable_pir_api", "0")
os.environ.setdefault("PADDLE_DISABLE_PIR", "1")

from typing import Optional
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
    ollama_vision_model: str = "minicpm-v:latest"
    enable_llm_fallback: bool = True
    enable_vision_fallback: bool = False
    ollama_timeout: float = 90.0
    ollama_num_ctx: int = 2048
    ollama_keep_alive: str = "15m"
    ollama_num_thread: Optional[int] = None

    yolo_model_path: str = "data/models/doclayout_yolo_v8s/weights/best.pt"
    layoutlm_model_path: str = "data/models/layoutlmv3-finetuned"
    layoutlm_base_model: str = "microsoft/layoutlmv3-base"

    ocr_languages: str = "en,hi,bn,ta,te,gu,mr"
    confidence_threshold: float = 0.80
    llm_fallback_threshold: float = 0.60
    max_file_size_mb: int = 50
    supported_formats: str = "pdf,jpg,jpeg,png,tiff,webp"
    cors_origins: str = "*"

    # Handwriting processing
    handwriting_confidence_penalty: float = 0.85
    handwriting_strict_review_fields: str = "grand_total,invoice_number,vendor_gstin,tax_amount"
    handwriting_permissive_fields: str = "remarks,payment_terms"
    enable_ruled_line_removal: bool = True
    enable_illumination_normalization: bool = True
    enable_ink_detection: bool = True

    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()
