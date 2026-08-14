"""
FastAPI request/response models.
"""

from pydantic import BaseModel
from typing import Optional, Any


class JobResponse(BaseModel):
    job_id: str
    status: str
    filename: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    filename: str
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


class HealthResponse(BaseModel):
    status: str
    version: str
    ollama_available: bool
    yolo_loaded: bool
    layoutlm_loaded: bool
    ollama_model: str
