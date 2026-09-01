"""
FastAPI request/response models.
"""

from pydantic import BaseModel
from typing import Optional, Any


class JobResponse(BaseModel):
    job_id: str
    status: str
    filename: str
    stage: Optional[str] = "preprocessing"
    stage_index: Optional[int] = 1
    stage_label: Optional[str] = "Pre-processing: Initializing"
    progress_pct: Optional[int] = 10


class BatchJobResponse(BaseModel):
    total_queued: int
    jobs: list[JobResponse]
    errors: list[dict[str, str]] = []


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    filename: str
    stage: Optional[str] = None
    stage_index: Optional[int] = 0
    stage_label: Optional[str] = None
    progress_pct: Optional[int] = 0
    invoice: Optional[dict] = None
    invoice_builder_data: Optional[dict] = None
    overall_confidence: Optional[float] = None
    needs_review: bool = False
    review_reasons: list[str] = []
    error_message: Optional[str] = None
    created_at: Optional[str] = None


class JobListResponse(BaseModel):
    jobs: list[JobStatusResponse]
    total: int


class InvoiceUpdateRequest(BaseModel):
    corrections: dict[str, Any]
    status: Optional[str] = None  # "reviewed" | "partially_reviewed" | "done"
    is_verified: Optional[bool] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    ollama_available: bool
    yolo_loaded: bool
    layoutlm_loaded: bool
    ollama_model: str
    tie_healthy: bool = True
    tie_templates_loaded: int = 0
