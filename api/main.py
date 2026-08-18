"""
FastAPI application — main entry point.
All routes for invoice upload, job status, review, and output download.
"""

import uuid
import json
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any
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

    # Save local copy to data/raw for instant document viewing
    try:
        raw_dir = Path("data/raw")
        raw_dir.mkdir(parents=True, exist_ok=True)
        local_raw_path = raw_dir / f"{job_id}_{filename}"
        with open(local_raw_path, "wb") as f:
            f.write(file_bytes)
    except Exception as e:
        logger.warning(f"Failed to save local raw file: {e}")

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


def _find_invoice_file(job_id: str, filename: str) -> Optional[Path]:
    clean_stem = Path(filename).stem
    candidates = [
        Path(f"data/raw/{job_id}_{filename}"),
        Path(f"data/raw/{filename}"),
        Path(f"data/uploads/{job_id}_{filename}"),
        Path(f"data/uploads/{filename}"),
    ]
    for c in candidates:
        if c.exists() and c.is_file():
            return c

    # Search in data/images_to_annotate and data/annotations
    search_dirs = [Path("data/images_to_annotate"), Path("data/annotations"), Path("data/raw"), Path("data")]
    for s_dir in search_dirs:
        if s_dir.exists():
            for p in s_dir.rglob(f"*{clean_stem}*"):
                if p.is_file() and p.suffix.lower() in [".pdf", ".png", ".jpg", ".jpeg", ".webp"]:
                    return p
            for p in s_dir.rglob(f"*{job_id}*"):
                if p.is_file() and p.suffix.lower() in [".pdf", ".png", ".jpg", ".jpeg", ".webp"]:
                    return p
    return None


@app.get("/api/invoices/{job_id}/original", tags=["Invoices"])
async def get_original_invoice_file(job_id: str):
    """
    Serve the original uploaded invoice document (PDF, PNG, JPG)
    so the human reviewer can view the real scanned file in the Review UI.
    """
    db: DatabaseManager = app.state.db
    record = await db.get_job(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Job not found")

    doc_path = _find_invoice_file(job_id, record.filename)
    if doc_path:
        ext = doc_path.suffix.lower().lstrip(".")
        mime = "application/pdf" if ext == "pdf" else f"image/{ext}"
        if mime == "image/jpg":
            mime = "image/jpeg"
        return FileResponse(
            str(doc_path),
            media_type=mime,
            headers={"Content-Disposition": "inline"}
        )

    # Check MinIO
    minio: MinIOManager = getattr(app.state, "minio", None)
    if minio and record.storage_key:
        url = minio.get_presigned_url(record.storage_key)
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url=url)

    raise HTTPException(status_code=404, detail="Original document file not found on disk")


@app.get("/api/invoices/{job_id}/preview-image", tags=["Invoices"])
async def get_invoice_preview_image(job_id: str, page: int = 0):
    """
    Render a crisp PNG image of the requested page of the invoice.
    Works for both PDF files and image files. Never triggers browser downloads.
    """
    from fastapi.responses import Response
    import pymupdf

    db: DatabaseManager = app.state.db
    record = await db.get_job(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Job not found")

    doc_path = _find_invoice_file(job_id, record.filename)
    if not doc_path or not doc_path.exists():
        # Fallback to graceful SVG placeholder so no broken image is displayed
        clean_name = record.filename
        svg_placeholder = f"""<svg xmlns="http://www.w3.org/2000/svg" width="600" height="800" viewBox="0 0 600 800">
            <rect width="600" height="800" fill="#f8fafc" rx="12" stroke="#e2e8f0" stroke-width="2"/>
            <rect x="40" y="40" width="520" height="720" fill="#ffffff" rx="8" stroke="#cbd5e1" stroke-dasharray="6,6"/>
            <circle cx="300" cy="340" r="40" fill="#eff6ff"/>
            <text x="300" y="348" font-family="system-ui, sans-serif" font-size="28" text-anchor="middle">📄</text>
            <text x="300" y="420" font-family="system-ui, sans-serif" font-size="16" font-weight="bold" fill="#1e293b" text-anchor="middle">Original Document: {clean_name}</text>
            <text x="300" y="450" font-family="system-ui, sans-serif" font-size="12" fill="#64748b" text-anchor="middle">Processed prior to local disk storage. Please switch to Standard HTML</text>
            <text x="300" y="475" font-family="system-ui, sans-serif" font-size="12" fill="#64748b" text-anchor="middle">or upload new invoices to view live document side-by-side.</text>
        </svg>"""
        return Response(content=svg_placeholder.encode("utf-8"), media_type="image/svg+xml")

    ext = doc_path.suffix.lower().lstrip(".")
    if ext in ["png", "jpg", "jpeg", "webp"]:
        with open(doc_path, "rb") as f:
            content = f.read()
        mime = "image/png" if ext == "png" else f"image/{ext}"
        if mime == "image/jpg":
            mime = "image/jpeg"
        return Response(content=content, media_type=mime, headers={"Cache-Control": "public, max-age=3600"})

    # Render PDF page to PNG with PyMuPDF
    try:
        import pymupdf
        doc = pymupdf.open(str(doc_path))
        if page < 0 or page >= len(doc):
            page = 0
        pix = doc[page].get_pixmap(dpi=150)
        img_bytes = pix.tobytes("png")
        doc.close()
        return Response(
            content=img_bytes,
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=3600"}
        )
    except Exception as e:
        logger.exception(f"Failed to render preview image: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to render image: {e}")


@app.get("/api/invoices/{job_id}/doc-info", tags=["Invoices"])
async def get_invoice_doc_info(job_id: str):
    """Get metadata about the original document (page count, filename, format)."""
    import pymupdf

    db: DatabaseManager = app.state.db
    record = await db.get_job(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Job not found")

    doc_path = None
    raw_candidates = [
        Path(f"data/raw/{job_id}_{record.filename}"),
        Path(f"data/raw/{record.filename}"),
        Path(f"data/uploads/{job_id}_{record.filename}"),
    ]
    for c in raw_candidates:
        if c.exists() and c.is_file():
            doc_path = c
            break

    pages = 1
    is_pdf = False
    if doc_path and doc_path.suffix.lower() == ".pdf":
        try:
            doc = pymupdf.open(str(doc_path))
            pages = len(doc)
            is_pdf = True
            doc.close()
        except Exception:
            pass

    return {
        "job_id": job_id,
        "filename": record.filename,
        "is_pdf": is_pdf,
        "pages": pages,
    }


# ── Training Endpoints ─────────────────────────────────────────────

TRAINING_STATE = {
    "is_training": False,
    "current_model": None,
    "progress": "",
    "last_trained": None,
}


def _run_yolo_training_task():
    global TRAINING_STATE
    import subprocess
    import sys
    from datetime import datetime

    TRAINING_STATE["is_training"] = True
    TRAINING_STATE["current_model"] = "yolo"
    TRAINING_STATE["progress"] = "Fine-tuning YOLOv8 model on annotations..."

    try:
        cmd = [sys.executable, "scripts/train_yolo.py", "--data", "data/annotations/dataset.yaml", "--epochs", "60"]
        logger.info(f"Starting background YOLO training: {' '.join(cmd)}")
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if res.returncode == 0:
            TRAINING_STATE["last_trained"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            TRAINING_STATE["progress"] = "YOLO training completed successfully! Model active."
            logger.info("YOLO background training complete ✓")
            pipeline = getattr(app.state, "pipeline", None)
            if pipeline and hasattr(pipeline, "detector"):
                pipeline.detector._try_load_model()
        else:
            TRAINING_STATE["progress"] = f"YOLO training error: {res.stderr[:200]}"
            logger.error(f"YOLO training failed: {res.stderr}")
    except Exception as e:
        TRAINING_STATE["progress"] = f"Training failed: {e}"
        logger.exception("Training error")
    finally:
        TRAINING_STATE["is_training"] = False


def _run_layoutlm_training_task():
    global TRAINING_STATE
    import subprocess
    import sys
    from datetime import datetime

    TRAINING_STATE["is_training"] = True
    TRAINING_STATE["current_model"] = "layoutlmv3"
    TRAINING_STATE["progress"] = "Exporting reviewed invoices and fine-tuning LayoutLMv3..."

    try:
        # 1. Export dataset
        logger.info("Exporting reviewed invoices to LayoutLMv3 dataset...")
        subprocess.run(
            [sys.executable, "scripts/export_reviewed_to_layoutlm.py"],
            check=True,
            encoding="utf-8",
            errors="replace"
        )

        # 2. Train LayoutLMv3
        logger.info("Training LayoutLMv3...")
        cmd = [sys.executable, "scripts/train_layoutlm.py", "--epochs", "10"]
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if res.returncode == 0:
            TRAINING_STATE["last_trained"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            TRAINING_STATE["progress"] = "LayoutLMv3 fine-tuning complete! Model active."
            logger.info("LayoutLMv3 training complete ✓")
            pipeline = getattr(app.state, "pipeline", None)
            if pipeline and hasattr(pipeline, "extractor"):
                pipeline.extractor._try_load_model()
        else:
            TRAINING_STATE["progress"] = f"LayoutLMv3 training error: {res.stderr[:200]}"
            logger.error(f"LayoutLMv3 training failed: {res.stderr}")
    except Exception as e:
        TRAINING_STATE["progress"] = f"LayoutLMv3 training failed: {e}"
        logger.exception("LayoutLMv3 training error")
    finally:
        TRAINING_STATE["is_training"] = False


@app.post("/api/train/yolo", tags=["Training"])
async def trigger_yolo_training(background_tasks: BackgroundTasks):
    """Trigger YOLOv8 retraining on reviewed invoices in background."""
    if TRAINING_STATE["is_training"]:
        raise HTTPException(status_code=409, detail=f"Training already in progress: {TRAINING_STATE['progress']}")
    background_tasks.add_task(_run_yolo_training_task)
    return {"status": "started", "model": "yolo"}


@app.post("/api/train/layoutlm", tags=["Training"])
async def trigger_layoutlm_training(background_tasks: BackgroundTasks):
    """Trigger LayoutLMv3 fine-tuning on reviewed invoices in background."""
    if TRAINING_STATE["is_training"]:
        raise HTTPException(status_code=409, detail=f"Training already in progress: {TRAINING_STATE['progress']}")
    background_tasks.add_task(_run_layoutlm_training_task)
    return {"status": "started", "model": "layoutlmv3"}


def _run_auto_annotate_task(min_conf: float = 0.35, status: str = "reviewed"):
    """
    Background task: run auto_annotate_from_pipeline.py to generate YOLO
    annotation files from all reviewed invoices, then invalidate YOLO caches.
    """
    global TRAINING_STATE
    import subprocess, sys
    from datetime import datetime

    TRAINING_STATE["is_training"] = True
    TRAINING_STATE["current_model"] = "auto_annotate"
    TRAINING_STATE["progress"] = "Generating YOLO annotations from reviewed invoices..."

    try:
        cmd = [
            sys.executable, "scripts/auto_annotate_from_pipeline.py",
            "--min-conf", str(min_conf),
            "--status", status,
        ]
        logger.info(f"Starting auto-annotation: {' '.join(cmd)}")
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if res.returncode == 0:
            TRAINING_STATE["last_trained"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            TRAINING_STATE["progress"] = "Auto-annotation complete! Verify boxes in Label Studio, then retrain YOLO."
            logger.info("Auto-annotation complete ✓")
            if res.stdout:
                logger.info(res.stdout[-1000:])
        else:
            TRAINING_STATE["progress"] = f"Auto-annotation error: {res.stderr[:200]}"
            logger.error(f"Auto-annotation failed: {res.stderr}")
    except Exception as e:
        TRAINING_STATE["progress"] = f"Auto-annotation failed: {e}"
        logger.exception("Auto-annotation error")
    finally:
        TRAINING_STATE["is_training"] = False


@app.post("/api/train/auto-annotate", tags=["Training"])
async def trigger_auto_annotate(
    background_tasks: BackgroundTasks,
    min_conf: float = 0.35,
    status: str = "reviewed",
):
    """
    Auto-generate YOLO bounding-box annotation files from pipeline detections
    on all reviewed invoices. Writes images + .txt label files to
    data/annotations/ in YOLOv8 format, ready for Label Studio import or
    direct retraining.

    - min_conf: Minimum YOLO confidence threshold (default 0.35)
    - status: 'reviewed' | 'completed' | 'all'
    """
    if TRAINING_STATE["is_training"]:
        raise HTTPException(
            status_code=409,
            detail=f"Already running: {TRAINING_STATE['progress']}"
        )
    background_tasks.add_task(_run_auto_annotate_task, min_conf=min_conf, status=status)
    return {"status": "started", "action": "auto_annotate"}


@app.get("/api/train/status", tags=["Training"])
async def get_training_status():
    """Get status of background model training."""
    pipeline = getattr(app.state, "pipeline", None)
    return {
        **TRAINING_STATE,
        "yolo_loaded": pipeline.detector.model is not None if pipeline else False,
        "layoutlm_loaded": pipeline.extractor.model is not None if pipeline else False,
    }


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

    # Re-validate
    from validation.validator import InvoiceSchema, InvoiceValidator

    try:
        if any(k in corrections for k in ["company", "client", "meta", "items", "totals", "bankDetails"]):
            schema = InvoiceSchema.from_invoice_builder_json(corrections)
        else:
            merged = {**current, **corrections, "needs_review": False, "review_reasons": []}
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
