"""
FastAPI application — main entry point.
All routes for invoice upload, job status, review, and output download.
"""

import uuid
import json
import tempfile
from pathlib import Path
from contextlib import asynccontextmanager
from loguru import logger

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config.settings import get_settings, Settings
from api.models import (
    JobResponse, JobStatusResponse, InvoiceUpdateRequest,
    JobListResponse, HealthResponse
)
from api.dependencies import get_pipeline, get_db, get_minio
from storage.db import DatabaseManager, MinIOManager


# ------------------------------------------------------------------
# App lifespan (startup / shutdown)
# ------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("Starting Invoice Digitizer API...")

    # Init DB
    db = DatabaseManager(settings.database_url)
    await db.init_db()
    app.state.db = db

    # Init MinIO
    try:
        minio = MinIOManager(
            endpoint=settings.minio_endpoint,
            access_key=settings.minio_access_key,
            secret_key=settings.minio_secret_key,
            bucket=settings.minio_bucket,
            secure=settings.minio_secure,
        )
        app.state.minio = minio
        logger.info("MinIO connected")
    except Exception as e:
        logger.warning(f"MinIO not available: {e} — file storage disabled")
        app.state.minio = None

    # Warm up pipeline
    from api.pipeline_runner import InvoicePipeline
    app.state.pipeline = InvoicePipeline(settings)

    yield

    logger.info("Shutting down...")


# ------------------------------------------------------------------
# App instance
# ------------------------------------------------------------------

app = FastAPI(
    title="Invoice Digitizer",
    description="Convert any invoice image or PDF into a structured digital invoice. 100% free & open source.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check API health and pipeline component status."""
    settings = get_settings()
    pipeline = getattr(app.state, "pipeline", None)

    ollama_ok = False
    if pipeline:
        try:
            ollama_ok = pipeline.llm.is_available()
        except Exception:
            pass

    return HealthResponse(
        status="ok",
        version="1.0.0",
        ollama_available=ollama_ok,
        yolo_loaded=pipeline.detector.model is not None if pipeline else False,
        layoutlm_loaded=pipeline.extractor.model is not None if pipeline else False,
        ollama_model=settings.ollama_model,
    )


@app.post("/api/invoices/upload", response_model=JobResponse, tags=["Invoices"])
async def upload_invoice(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
):
    """
    Upload an invoice (PDF, JPG, PNG, TIFF, WEBP).
    Returns a job_id to poll for results.
    """
    # Validate file type
    allowed = settings.supported_formats.split(",")
    suffix = Path(file.filename or "").suffix.lower().lstrip(".")
    if suffix not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: .{suffix}. Allowed: {allowed}"
        )

    # Validate file size
    file_bytes = await file.read()
    max_bytes = settings.max_file_size_mb * 1024 * 1024
    if len(file_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: {settings.max_file_size_mb} MB"
        )

    job_id = str(uuid.uuid4())
    filename = file.filename or f"invoice_{job_id}.{suffix}"

    # Store to DB
    db: DatabaseManager = app.state.db
    await db.create_job(job_id=job_id, filename=filename)

    # Upload raw file to MinIO
    minio: MinIOManager = getattr(app.state, "minio", None)
    storage_key = None
    if minio:
        storage_key = f"raw/{job_id}/{filename}"
        minio.upload_file(file_bytes, storage_key)

    # Run pipeline in background
    background_tasks.add_task(
        _run_pipeline_task,
        job_id=job_id,
        file_bytes=file_bytes,
        filename=filename,
        storage_key=storage_key,
        db=db,
        minio=minio,
        pipeline=app.state.pipeline,
        settings=settings,
    )

    logger.info(f"Job queued: {job_id} ({filename})")
    return JobResponse(job_id=job_id, status="processing", filename=filename)


@app.get("/api/invoices/{job_id}", response_model=JobStatusResponse, tags=["Invoices"])
async def get_invoice_status(job_id: str):
    """Poll job status and get extracted invoice data when complete."""
    db: DatabaseManager = app.state.db
    record = await db.get_job(job_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    builder_data = None
    if record.output_json:
        try:
            from validation.validator import InvoiceSchema
            schema_obj = InvoiceSchema(**record.output_json)
            builder_data = schema_obj.to_invoice_builder_json()
        except Exception:
            pass

    return JobStatusResponse(
        job_id=record.job_id,
        status=record.status,
        filename=record.filename,
        invoice=record.output_json,
        invoice_builder_data=builder_data,
        overall_confidence=record.overall_confidence,
        needs_review=record.needs_review,
        review_reasons=record.review_reasons or [],
        error_message=record.error_message,
        created_at=str(record.created_at),
    )


@app.get("/api/invoices/{job_id}/html", response_class=HTMLResponse, tags=["Invoices"])
async def get_invoice_html(job_id: str):
    """Get the rendered HTML invoice."""
    db: DatabaseManager = app.state.db
    record = await db.get_job(job_id)
    if not record or record.status != "done":
        raise HTTPException(status_code=404, detail="Invoice not ready or not found")

    if not record.output_json:
        raise HTTPException(status_code=404, detail="No invoice data")

    from validation.validator import InvoiceSchema
    from output.renderer import InvoiceRenderer
    invoice = InvoiceSchema(**record.output_json)
    renderer = InvoiceRenderer()
    return HTMLResponse(content=renderer.to_html(invoice))


@app.get("/api/invoices/{job_id}/pdf", tags=["Invoices"])
async def download_invoice_pdf(job_id: str):
    """Download the rendered PDF invoice."""
    db: DatabaseManager = app.state.db
    record = await db.get_job(job_id)
    if not record or record.status != "done":
        raise HTTPException(status_code=404, detail="Invoice not ready")

    minio: MinIOManager = getattr(app.state, "minio", None)
    if minio and record.output_pdf_key:
        # Generate presigned URL and redirect
        url = minio.get_presigned_url(record.output_pdf_key)
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url=url)

    # Fallback: regenerate PDF on the fly
    if record.output_json:
        from validation.validator import InvoiceSchema
        from output.renderer import InvoiceRenderer
        import io
        invoice = InvoiceSchema(**record.output_json)
        renderer = InvoiceRenderer()
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
            tmp_path = Path(f.name)
        renderer.to_pdf(invoice, tmp_path)
        return FileResponse(
            str(tmp_path),
            media_type="application/pdf",
            filename=f"invoice_{job_id}.pdf",
        )

    raise HTTPException(status_code=404, detail="PDF not available")


@app.patch("/api/invoices/{job_id}", tags=["Invoices"])
async def update_invoice(job_id: str, update: InvoiceUpdateRequest):
    """
    Human review correction endpoint.
    Accepts corrected field values and re-validates.
    """
    db: DatabaseManager = app.state.db
    record = await db.get_job(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Job not found")

    # Merge corrections into existing output
    current = record.output_json or {}
    corrections = update.corrections
    merged = {**current, **corrections, "needs_review": False, "review_reasons": []}

    # Re-validate
    from validation.validator import InvoiceSchema, InvoiceValidator
    from understanding.layoutlm import ExtractedInvoice, ExtractedField

    try:
        schema = InvoiceSchema(**merged)
        # Mark corrected fields
        schema.needs_review = False
        schema.review_reasons = []

        await db.update_job(
            job_id,
            output_json=schema.model_dump(),
            needs_review=False,
            review_reasons=[],
            status="reviewed",
        )
        return {"status": "updated", "job_id": job_id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Validation error: {e}")


@app.get("/api/invoices", response_model=JobListResponse, tags=["Invoices"])
async def list_invoices(limit: int = 20, offset: int = 0):
    """List recent invoice jobs."""
    db: DatabaseManager = app.state.db
    records = await db.list_jobs(limit=limit, offset=offset)
    return JobListResponse(
        jobs=[
            JobStatusResponse(
                job_id=r.job_id,
                status=r.status,
                filename=r.filename,
                overall_confidence=r.overall_confidence,
                needs_review=r.needs_review,
                review_reasons=r.review_reasons or [],
                created_at=str(r.created_at),
            )
            for r in records
        ],
        total=len(records),
    )


@app.delete("/api/invoices/{job_id}", tags=["Invoices"])
async def delete_invoice(job_id: str):
    """Delete a job record."""
    db: DatabaseManager = app.state.db
    record = await db.get_job(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Job not found")
    await db.update_job(job_id, status="deleted")
    return {"status": "deleted", "job_id": job_id}


# ------------------------------------------------------------------
# Background task
# ------------------------------------------------------------------

async def _run_pipeline_task(
    job_id: str,
    file_bytes: bytes,
    filename: str,
    storage_key,
    db: DatabaseManager,
    minio,
    pipeline,
    settings: Settings,
):
    """Runs in background after upload returns."""
    try:
        await db.update_job(job_id, status="processing")

        with tempfile.TemporaryDirectory() as tmpdir:
            result = pipeline.process(
                file_bytes=file_bytes,
                filename=filename,
                job_id=job_id,
                output_dir=tmpdir,
            )

        # Upload PDF to MinIO
        pdf_key = None
        if minio and result.pdf_path and result.pdf_path.exists():
            pdf_key = f"output/{job_id}/invoice.pdf"
            minio.upload_pdf(result.pdf_path, pdf_key)

        await db.update_job(
            job_id,
            status="done",
            output_json=result.invoice.model_dump(),
            output_pdf_key=pdf_key,
            storage_key=storage_key,
            overall_confidence=result.invoice.overall_confidence,
            needs_review=result.invoice.needs_review,
            review_reasons=result.invoice.review_reasons,
        )
        logger.info(f"[{job_id}] Pipeline complete ✓")

    except Exception as e:
        logger.exception(f"[{job_id}] Pipeline failed: {e}")
        await db.update_job(job_id, status="failed", error_message=str(e))
