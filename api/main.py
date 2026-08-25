"""
FastAPI application — main entry point.
All routes for invoice upload, job status, review, and output download.
"""

import uuid
import json
import hashlib
from datetime import datetime
import asyncio
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager
from loguru import logger

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, BackgroundTasks, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from config.settings import get_settings, Settings
from api.models import (
    JobResponse, BatchJobResponse, JobStatusResponse, InvoiceUpdateRequest,
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

    # Warm up pipeline asynchronously in background thread so lifespan finishes immediately and API is ready instantly
    app.state.pipeline = None

    async def _warmup():
        try:
            from api.pipeline_runner import InvoicePipeline
            loop = asyncio.get_running_loop()
            pipeline = await loop.run_in_executor(None, lambda: InvoicePipeline(settings))
            app.state.pipeline = pipeline
            logger.info("Pipeline warmed up in background")
        except Exception as ex:
            logger.warning(f"Background pipeline warmup: {ex}")

    asyncio.create_task(_warmup())

    # Start Autonomous Active Learning Background Worker (polls every 15 mins)
    try:
        from active_learning.auto_trainer import continuous_learning_worker
        app.state.learning_worker = asyncio.create_task(continuous_learning_worker(interval_seconds=900))
    except Exception as ex:
        logger.warning(f"Could not start active learning background worker: {ex}")
        app.state.learning_worker = None

    yield

    logger.info("Shutting down...")
    if getattr(app.state, "learning_worker", None):
        app.state.learning_worker.cancel()


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
# Live Job Stage Progress Tracker
# ------------------------------------------------------------------
_active_job_progress: dict[str, dict] = {}

# Per-job asyncio queues for SSE push — key: job_id, value: list of Queue listeners
_job_sse_queues: dict[str, list] = {}

# Global broadcast queues for the invoice-list page (receives events for ALL jobs)
_global_sse_queues: list = []


def _push_progress(job_id: str, payload: dict):
    """Write progress to in-memory store AND fan-out to all live SSE listeners."""
    _active_job_progress[job_id] = payload
    # Per-job listeners (UploadPage / ReviewPage)
    for q in _job_sse_queues.get(job_id, []):
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            pass  # Listener is slow; skip this frame — it will catch up
    # Global listeners (InvoiceListPage)
    for q in _global_sse_queues:
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            pass


# ------------------------------------------------------------------
# Routes
# ------------------------------------------------------------------


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check(request: Request):
    """Check API health and pipeline component status."""
    settings = get_settings()
    pipeline = get_pipeline(request)

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


@app.get("/api/stream/jobs", tags=["Invoices"])
async def stream_all_jobs(request: Request):
    """
    Global Server-Sent Events endpoint — broadcasts real-time progress events for
    ALL currently processing jobs. The invoice list page opens one connection and
    receives patches for each row as stages complete, instead of polling every 3s.
    """
    q: asyncio.Queue = asyncio.Queue(maxsize=128)
    _global_sse_queues.append(q)

    # Seed the queue with the current state of every in-progress job
    for jid, prog in _active_job_progress.items():
        if prog.get("status") not in ("done", "reviewed", "failed", None):
            await q.put({**prog, "job_id": jid})

    async def event_generator():
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    payload = await asyncio.wait_for(q.get(), timeout=2.0)
                except asyncio.TimeoutError:
                    yield ": keep-alive\n\n"
                    continue
                yield f"data: {json.dumps(payload)}\n\n"
        finally:
            try:
                _global_sse_queues.remove(q)
            except ValueError:
                pass

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
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
    doc_hash = hashlib.sha256(file_bytes).hexdigest()

    # Store to DB
    db: DatabaseManager = app.state.db
    await db.create_job(job_id=job_id, filename=filename, document_hash=doc_hash)

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


@app.post("/api/invoices/upload-batch", response_model=BatchJobResponse, tags=["Invoices"])
async def upload_batch_invoices(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    settings: Settings = Depends(get_settings),
):
    """
    Batch upload multiple invoices simultaneously (PDF, JPG, PNG, TIFF, WEBP).
    Queues all valid files for AI processing and returns their job IDs.
    """
    db: DatabaseManager = app.state.db
    minio: MinIOManager = getattr(app.state, "minio", None)
    allowed = [s.strip().lower() for s in settings.supported_formats.split(",")]
    max_bytes = settings.max_file_size_mb * 1024 * 1024

    queued_jobs: list[JobResponse] = []
    errors: list[dict[str, str]] = []

    for file in files:
        filename = file.filename or "uploaded_invoice"
        suffix = Path(filename).suffix.lower().lstrip(".")
        if suffix not in allowed:
            errors.append({"filename": filename, "error": f"Unsupported format (.{suffix})"})
            continue

        file_bytes = await file.read()
        if len(file_bytes) > max_bytes:
            errors.append({"filename": filename, "error": f"File exceeds {settings.max_file_size_mb}MB limit"})
            continue

        job_id = str(uuid.uuid4())
        await db.create_job(job_id=job_id, filename=filename)

        # Save local copy
        try:
            raw_dir = Path("data/raw")
            raw_dir.mkdir(parents=True, exist_ok=True)
            local_raw_path = raw_dir / f"{job_id}_{filename}"
            with open(local_raw_path, "wb") as f:
                f.write(file_bytes)
        except Exception as e:
            logger.warning(f"Failed to save local raw file: {e}")

        # Upload to MinIO
        storage_key = None
        if minio:
            storage_key = f"raw/{job_id}/{filename}"
            try:
                minio.upload_file(file_bytes, storage_key)
            except Exception as e:
                logger.warning(f"MinIO upload error: {e}")

        # Queue pipeline task
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

        logger.info(f"[Batch] Queued job: {job_id} ({filename})")
        queued_jobs.append(JobResponse(job_id=job_id, status="processing", filename=filename))

    return BatchJobResponse(
        total_queued=len(queued_jobs),
        jobs=queued_jobs,
        errors=errors,
    )


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

    # Real-time pipeline stage calculation
    prog = _active_job_progress.get(job_id, {})
    cur_stage = prog.get("stage")
    cur_idx = prog.get("stage_index", 0)
    cur_label = prog.get("stage_label")
    cur_pct = prog.get("progress_pct", 0)

    if record.status in ["done", "reviewed", "partially_reviewed"]:
        cur_stage = "done"
        cur_idx = 6
        cur_label = "Complete"
        cur_pct = 100
    elif record.status == "failed":
        cur_stage = "failed"
        cur_idx = 0
        cur_label = record.error_message or "Failed"
        cur_pct = 0
    elif record.status == "processing" and not cur_stage:
        cur_stage = "preprocessing"
        cur_idx = 1
        cur_label = "Pre-processing: Analyzing document"
        cur_pct = 15

    return JobStatusResponse(
        job_id=record.job_id,
        status=record.status,
        filename=record.filename,
        stage=cur_stage,
        stage_index=cur_idx,
        stage_label=cur_label,
        progress_pct=cur_pct,
        invoice=record.output_json,
        invoice_builder_data=builder_data,
        overall_confidence=record.overall_confidence,
        needs_review=record.needs_review,
        review_reasons=record.review_reasons or [],
        error_message=record.error_message,
        created_at=str(record.created_at),
    )

@app.get("/api/invoices/{job_id}/stream", tags=["Invoices"])
async def stream_job_progress(job_id: str, request: Request):
    """
    Server-Sent Events endpoint — streams real-time pipeline progress.
    The client opens one long-lived connection; the server pushes events
    as each pipeline stage completes. Connection auto-closes when done/failed.
    """
    db: DatabaseManager = app.state.db

    record = await db.get_job(job_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    # If job is already finished, send a single terminal event immediately
    if record.status in ("done", "reviewed", "partially_reviewed"):
        async def _done_stream():
            payload = json.dumps({
                "job_id": job_id, "status": record.status,
                "stage": "done", "stage_index": 6,
                "stage_label": "Digitization Complete", "progress_pct": 100,
            })
            yield f"data: {payload}\n\n"
        return StreamingResponse(_done_stream(), media_type="text/event-stream",
                                  headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    if record.status == "failed":
        async def _fail_stream():
            payload = json.dumps({
                "job_id": job_id, "status": "failed",
                "stage": "failed", "stage_index": 0,
                "stage_label": record.error_message or "Failed", "progress_pct": 0,
            })
            yield f"data: {payload}\n\n"
        return StreamingResponse(_fail_stream(), media_type="text/event-stream",
                                  headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    # ── Register a per-listener queue ─────────────────────────────────────────
    q: asyncio.Queue = asyncio.Queue(maxsize=32)
    listeners = _job_sse_queues.setdefault(job_id, [])
    listeners.append(q)

    # Immediately send current known progress so the client has something on connect
    initial = _active_job_progress.get(job_id, {
        "stage": "preprocessing", "stage_index": 1,
        "stage_label": "Pre-processing: Initializing document", "progress_pct": 10,
    })
    await q.put({**initial, "job_id": job_id, "status": "processing"})

    async def event_generator():
        try:
            while True:
                # Check for client disconnect
                if await request.is_disconnected():
                    break

                try:
                    payload = await asyncio.wait_for(q.get(), timeout=2.0)
                except asyncio.TimeoutError:
                    # Send a keep-alive comment so the connection doesn't time out
                    yield ": keep-alive\n\n"
                    continue

                status = payload.get("status", "processing")
                data_str = json.dumps(payload)
                yield f"data: {data_str}\n\n"

                if status in ("done", "reviewed", "partially_reviewed", "failed"):
                    break
        finally:
            # Always clean up this listener when connection closes
            try:
                listeners.remove(q)
            except ValueError:
                pass
            if not listeners and job_id in _job_sse_queues:
                del _job_sse_queues[job_id]

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )



@app.get("/api/invoices/{job_id}", response_model=JobStatusResponse, tags=["Invoices"])
async def get_invoice(job_id: str):
    """
    Get detailed invoice data for human review.
    Returns:
    - job_id, status, filename, overall_confidence, needs_review, review_reasons
    - invoice: InvoiceSchema dictionary
    - invoice_builder_data: Nested JSON object ready for the frontend review UI
    """
    db: DatabaseManager = app.state.db
    record = await db.get_job(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Job not found")

    from validation.validator import InvoiceSchema
    invoice_dict = record.output_json
    builder_data = None

    if invoice_dict:
        try:
            if any(k in invoice_dict for k in ["company", "client", "meta", "items", "totals", "bankDetails"]):
                schema_obj = InvoiceSchema.from_invoice_builder_json(invoice_dict)
                builder_data = invoice_dict
                invoice_dict = schema_obj.model_dump()
            else:
                schema_obj = InvoiceSchema(**invoice_dict)
                builder_data = schema_obj.to_invoice_builder_json()
                invoice_dict = schema_obj.model_dump()
        except Exception as e:
            logger.warning(f"Failed to normalize invoice data for {job_id}: {e}")

    return JobStatusResponse(
        job_id=record.job_id,
        status=record.status,
        filename=record.filename,
        overall_confidence=record.overall_confidence,
        needs_review=record.needs_review,
        review_reasons=record.review_reasons or [],
        invoice=invoice_dict,
        invoice_builder_data=builder_data,
        error_message=record.error_message,
        created_at=str(record.created_at),
    )


@app.get("/api/invoices/{job_id}/html", response_class=HTMLResponse, tags=["Invoices"])
async def get_invoice_html(job_id: str):
    """Get the rendered HTML invoice."""
    db: DatabaseManager = app.state.db
    record = await db.get_job(job_id)
    if not record or record.status not in ("done", "reviewed", "partially_reviewed"):
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
    if not record or record.status not in ("done", "reviewed", "partially_reviewed"):
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
        Path(f"data/uploads/{record.filename}"),
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
    Accepts corrected field values and saves them.

    - If is_verified=True or status="reviewed": marks invoice as VERIFIED (Ground Truth for AI training).
    - If is_verified=False or status="partially_reviewed": saves progress as a PARTIAL DRAFT without marking verified.
    """
    db: DatabaseManager = app.state.db
    record = await db.get_job(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Job not found")

    # Merge corrections into existing output
    current = record.output_json or {}
    corrections = update.corrections

    # Determine verification intent
    is_verified = False
    if update.is_verified is True or update.status == "reviewed":
        is_verified = True
    elif update.is_verified is False or update.status in ("partially_reviewed", "partial"):
        is_verified = False
    elif update.status:
        is_verified = (update.status == "reviewed")
    else:
        is_verified = False

    target_status = "reviewed" if is_verified else "partially_reviewed"
    logger.info(f"[{job_id}] Update requested: is_verified={is_verified} -> target_status='{target_status}'")

    from validation.validator import InvoiceSchema, InvoiceValidator

    try:
        if any(k in corrections for k in ["company", "client", "meta", "items", "totals", "bankDetails"]):
            schema = InvoiceSchema.from_invoice_builder_json(corrections)
        else:
            merged = {**current, **corrections}
            schema = InvoiceSchema(**merged)

        # Set verification flags
        schema.needs_review = not is_verified
        if is_verified:
            schema.review_reasons = []
        else:
            schema.review_reasons = schema.review_reasons or ["Partially reviewed draft"]

        # Active learning: track human field corrections against initial AI output
        from active_learning.correction_tracker import compute_field_corrections, classify_review_status
        ai_data = record.ai_output_json or record.output_json or {}
        human_data = schema.model_dump()
        corrections_list = compute_field_corrections(ai_data, human_data)
        review_status = classify_review_status(corrections_list, schema.needs_review, schema.overall_confidence)

        if len(corrections_list) > 0 and is_verified:
            ground_truth_source = "human_corrected"
        elif is_verified:
            ground_truth_source = "human_confirmed"
        else:
            ground_truth_source = "partial"

        schema.ground_truth_source = ground_truth_source

        updated_rec = await db.update_job(
            job_id,
            output_json=schema.model_dump(),
            field_confidences=schema.field_confidences,
            corrections=corrections_list,
            review_status=review_status,
            ground_truth_source=ground_truth_source,
            needs_review=not is_verified,
            review_reasons=schema.review_reasons,
            status=target_status,
        )
        logger.info(
            f"[{job_id}] Saved to DB: status='{updated_rec.status}', review_status='{review_status}', "
            f"corrections={len(corrections_list)}, needs_review={updated_rec.needs_review}"
        )
        return {
            "status": target_status,
            "job_id": job_id,
            "is_verified": is_verified,
            "review_status": review_status,
            "corrections_recorded": len(corrections_list),
            "message": "Invoice marked as verified ground truth" if is_verified else "Invoice progress saved as partial draft"
        }
    except Exception as e:
        logger.exception(f"Error saving invoice corrections: {e}")
        raise HTTPException(status_code=400, detail=f"Validation error: {e}")


@app.get("/api/invoices", response_model=JobListResponse, tags=["Invoices"])
async def list_invoices(
    limit: int = 20,
    offset: int = 0,
    status: Optional[str] = "non_pending",
    search: Optional[str] = None,
    needs_review: Optional[bool] = None,
):
    """
    List invoice jobs with pagination, status filtering, and search.
    - limit: Page size (default 20)
    - offset: Page offset (default 0)
    - status: 'all' | 'non_pending' (default) | 'done' | 'partially_reviewed' | 'reviewed' | 'processing' | 'failed'
    - search: Search keyword by filename or job_id
    - needs_review: Optional boolean flag
    """
    db: DatabaseManager = app.state.db
    records, total_count = await db.list_jobs(
        limit=limit,
        offset=offset,
        status=status,
        search=search,
        needs_review=needs_review,
    )

    jobs_out = []
    for r in records:
        prog = _active_job_progress.get(r.job_id, {})
        cur_stage = prog.get("stage")
        cur_idx = prog.get("stage_index", 0)
        cur_label = prog.get("stage_label")
        cur_pct = prog.get("progress_pct", 0)

        if r.status in ["done", "reviewed", "partially_reviewed"]:
            cur_stage = "done"
            cur_idx = 6
            cur_label = "Complete"
            cur_pct = 100
        elif r.status == "failed":
            cur_stage = "failed"
            cur_idx = 0
            cur_label = r.error_message or "Failed"
            cur_pct = 0
        elif r.status == "processing" and not cur_stage:
            cur_stage = "preprocessing"
            cur_idx = 1
            cur_label = "Pre-processing: Analyzing document"
            cur_pct = 15

        jobs_out.append(
            JobStatusResponse(
                job_id=r.job_id,
                status=r.status,
                filename=r.filename,
                stage=cur_stage,
                stage_index=cur_idx,
                stage_label=cur_label,
                progress_pct=cur_pct,
                overall_confidence=r.overall_confidence,
                needs_review=r.needs_review,
                review_reasons=r.review_reasons or [],
                created_at=str(r.created_at),
            )
        )

    return JobListResponse(
        jobs=jobs_out,
        total=total_count,
    )


@app.delete("/api/invoices/{job_id}", tags=["Invoices"])
async def delete_invoice(job_id: str):
    """Delete a job record."""
    db: DatabaseManager = app.state.db
    record = await db.get_job(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Job not found")
    await db.update_job(job_id, status="deleted")
    if job_id in _active_job_progress:
        del _active_job_progress[job_id]
    return {"status": "deleted", "job_id": job_id}


@app.post("/api/invoices/bulk-delete", tags=["Invoices"])
async def bulk_delete_invoices(
    clear_pending: bool = True,
    clear_unreviewed: bool = True,
    clear_failed: bool = True,
):
    """
    Bulk clear/delete invoices matching specified statuses.
    Sets status to 'deleted' so they are removed from all views and queues.
    Keeps verified 'reviewed' and 'partially_reviewed' invoices intact.
    """
    db: DatabaseManager = app.state.db
    target_statuses = []
    if clear_pending:
        target_statuses.append("pending")
    if clear_unreviewed:
        target_statuses.append("done")
    if clear_failed:
        target_statuses.append("failed")

    if not target_statuses:
        return {"deleted_count": 0, "message": "No statuses selected"}

    count = await db.bulk_delete_jobs(target_statuses)

    # Clear memory cache for deleted items
    for j_id in list(_active_job_progress.keys()):
        prog = _active_job_progress.get(j_id, {})
        if prog.get("status") in target_statuses:
            del _active_job_progress[j_id]

    logger.info(f"Bulk cleared {count} invoices with status in {target_statuses}")
    return {
        "deleted_count": count,
        "statuses": target_statuses,
        "message": f"Successfully cleared {count} invoice(s) from queue",
    }


def _get_raw_file_bytes(job_id: str, filename: str, storage_key: Optional[str] = None, minio: Optional[MinIOManager] = None) -> Optional[bytes]:
    """Retrieve raw file bytes from local disk or MinIO storage."""
    doc_path = _find_invoice_file(job_id, filename)
    if doc_path and doc_path.exists() and doc_path.is_file():
        try:
            return doc_path.read_bytes()
        except Exception as e:
            logger.warning(f"Failed to read local file {doc_path}: {e}")

    if minio and storage_key:
        try:
            return minio.download_file(storage_key)
        except Exception as e:
            logger.warning(f"Failed to download from MinIO ({storage_key}): {e}")

    return None


@app.post("/api/invoices/{job_id}/reprocess", tags=["Invoices"])
async def reprocess_invoice(
    job_id: str,
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(get_settings),
):
    """
    Re-run the updated AI pipeline on an existing invoice.
    Re-extracts fields, re-runs table extraction, and re-validates.
    """
    db: DatabaseManager = app.state.db
    record = await db.get_job(job_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    minio: MinIOManager = getattr(app.state, "minio", None)
    file_bytes = _get_raw_file_bytes(job_id, record.filename, record.storage_key, minio)
    if not file_bytes:
        raise HTTPException(status_code=404, detail=f"Raw document file for {record.filename} not found on disk or MinIO")

    # Set status to processing
    await db.update_job(job_id, status="processing")

    background_tasks.add_task(
        _run_pipeline_task,
        job_id=job_id,
        file_bytes=file_bytes,
        filename=record.filename,
        storage_key=record.storage_key,
        db=db,
        minio=minio,
        pipeline=app.state.pipeline,
        settings=settings,
    )
    return {"status": "processing", "job_id": job_id, "filename": record.filename}


@app.post("/api/invoices/reprocess-all", tags=["Invoices"])
async def reprocess_all_invoices(
    background_tasks: BackgroundTasks,
    only_unreviewed: bool = True,
    limit: int = 100,
    settings: Settings = Depends(get_settings),
):
    """
    Batch re-process invoices using the latest AI pipeline.
    - only_unreviewed: If true (default), reprocesses only invoices not marked as 'reviewed'.
    - limit: Maximum number of invoices to reprocess (default 100).
    """
    db: DatabaseManager = app.state.db
    minio: MinIOManager = getattr(app.state, "minio", None)
    records, _ = await db.list_jobs(limit=limit, offset=0)

    queued = []
    skipped = []

    for r in records:
        if only_unreviewed and r.status == "reviewed":
            skipped.append(r.job_id)
            continue

        file_bytes = _get_raw_file_bytes(r.job_id, r.filename, r.storage_key, minio)
        if not file_bytes:
            skipped.append(r.job_id)
            continue

        await db.update_job(r.job_id, status="processing")
        background_tasks.add_task(
            _run_pipeline_task,
            job_id=r.job_id,
            file_bytes=file_bytes,
            filename=r.filename,
            storage_key=r.storage_key,
            db=db,
            minio=minio,
            pipeline=app.state.pipeline,
            settings=settings,
        )
        queued.append({"job_id": r.job_id, "filename": r.filename})

    return {
        "status": "started",
        "total_queued": len(queued),
        "total_skipped": len(skipped),
        "queued": queued,
    }


@app.post("/api/invoices/cancel-processing", tags=["Invoices"])
async def cancel_all_processing():
    """
    Instantly cancel/stop all currently running and queued invoice re-scans.
    Resets processing jobs back to 'done' and clears memory queues.
    """
    db: DatabaseManager = app.state.db
    from sqlalchemy import update
    async with db.session_factory() as session:
        stmt = (
            update(InvoiceRecord)
            .where(InvoiceRecord.status == "processing")
            .values(status="done")
        )
        res = await session.execute(stmt)
        await session.commit()
        count = res.rowcount or 0

    # Clear active progress trackers and notify listeners
    for j_id in list(_active_job_progress.keys()):
        _push_progress(j_id, {
            "job_id": j_id,
            "status": "done",
            "stage": "done",
            "stage_label": "Cancelled by user",
            "progress_pct": 100,
        })
        if j_id in _active_job_progress:
            del _active_job_progress[j_id]

    logger.info(f"Cancelled all processing: reset {count} jobs back to done")
    return {
        "status": "cancelled",
        "stopped_count": count,
        "message": f"Successfully stopped {count} running/queued invoice scan(s)",
    }


@app.post("/api/invoices/{job_id}/retry", tags=["Invoices"])
async def retry_invoice(
    job_id: str,
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(get_settings),
):
    """
    Retry a failed or queued invoice digitization job.
    Adds it back to the end of the background processing queue.
    """
    db: DatabaseManager = app.state.db
    record = await db.get_job(job_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    minio: MinIOManager = getattr(app.state, "minio", None)
    file_bytes = _get_raw_file_bytes(job_id, record.filename, record.storage_key, minio)
    if not file_bytes:
        raise HTTPException(status_code=404, detail=f"Raw document file for {record.filename} not found on disk or MinIO")

    await db.update_job(job_id, status="processing", error_message=None)
    _push_progress(job_id, {
        "job_id": job_id,
        "status": "processing",
        "stage": "preprocessing",
        "stage_index": 1,
        "progress_pct": 5,
        "stage_label": "Queued for retry: Waiting for worker slot...",
    })

    background_tasks.add_task(
        _run_pipeline_task,
        job_id=job_id,
        file_bytes=file_bytes,
        filename=record.filename,
        storage_key=record.storage_key,
        db=db,
        minio=minio,
        pipeline=app.state.pipeline,
        settings=settings,
    )
    return {"status": "processing", "job_id": job_id, "filename": record.filename}


@app.post("/api/invoices/retry-all-failed", tags=["Invoices"])
async def retry_all_failed_invoices(
    background_tasks: BackgroundTasks,
    settings: Settings = Depends(get_settings),
):
    """
    Retry all failed invoice jobs, adding each to the end of the queue.
    """
    db: DatabaseManager = app.state.db
    minio: MinIOManager = getattr(app.state, "minio", None)
    records, _ = await db.list_jobs(limit=200, status="failed")

    retried = []
    for r in records:
        file_bytes = _get_raw_file_bytes(r.job_id, r.filename, r.storage_key, minio)
        if not file_bytes:
            continue

        await db.update_job(r.job_id, status="processing", error_message=None)
        _push_progress(r.job_id, {
            "job_id": r.job_id,
            "status": "processing",
            "stage": "preprocessing",
            "stage_index": 1,
            "progress_pct": 5,
            "stage_label": "Queued for retry: Waiting for worker slot...",
        })
        background_tasks.add_task(
            _run_pipeline_task,
            job_id=r.job_id,
            file_bytes=file_bytes,
            filename=r.filename,
            storage_key=r.storage_key,
            db=db,
            minio=minio,
            pipeline=app.state.pipeline,
            settings=settings,
        )
        retried.append({"job_id": r.job_id, "filename": r.filename})

    return {"status": "started", "total_retried": len(retried), "jobs": retried}


@app.delete("/api/invoices/{job_id}", tags=["Invoices"])
async def delete_invoice(job_id: str):
    """
    Delete an invoice job (whether queued, processing, failed, or completed).
    Removes it from the database and broadcasts a deletion event via SSE.
    """
    db: DatabaseManager = app.state.db
    record = await db.get_job(job_id)
    if not record:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    deleted = await db.delete_job(job_id)
    if not deleted:
        raise HTTPException(status_code=500, detail="Failed to delete invoice record")

    # Clean in-memory tracking
    _active_job_progress.pop(job_id, None)

    # Clean local raw files if any
    try:
        raw_candidates = [
            Path(f"data/raw/{job_id}_{record.filename}"),
            Path(f"data/uploads/{job_id}_{record.filename}"),
        ]
        for rc in raw_candidates:
            if rc.exists():
                rc.unlink(missing_ok=True)
    except Exception as e:
        logger.warning(f"Could not delete local file: {e}")

    # Broadcast deletion event so UI removes the row in real time
    _push_progress(job_id, {
        "job_id": job_id,
        "status": "deleted",
    })

    return {"deleted": True, "job_id": job_id, "filename": record.filename}


# ------------------------------------------------------------------
# Background task & Concurrency Limiter
# ------------------------------------------------------------------

_PIPELINE_SEMAPHORE = asyncio.Semaphore(2)

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
    loop = asyncio.get_event_loop()

    def _stage_cb(stage: str, stage_idx: int, progress: int, label: str):
        """Called from worker thread — safe to use call_soon_threadsafe."""
        payload = {
            "job_id": job_id,
            "status": "processing",
            "stage": stage,
            "stage_index": stage_idx,
            "progress_pct": progress,
            "stage_label": label,
        }
        loop.call_soon_threadsafe(_push_progress, job_id, payload)

    try:
        _push_progress(job_id, {
            "job_id": job_id,
            "status": "processing",
            "stage": "preprocessing",
            "stage_index": 1,
            "progress_pct": 5,
            "stage_label": "Queued: Waiting for worker slot...",
        })
        await db.update_job(job_id, status="processing")

        async with _PIPELINE_SEMAPHORE:
            _push_progress(job_id, {
                "job_id": job_id,
                "status": "processing",
                "stage": "preprocessing",
                "stage_index": 1,
                "progress_pct": 10,
                "stage_label": "Pre-processing: Initializing document...",
            })

            with tempfile.TemporaryDirectory() as tmpdir:
                result = await asyncio.to_thread(
                    pipeline.process,
                    file_bytes=file_bytes,
                    filename=filename,
                    job_id=job_id,
                    output_dir=tmpdir,
                    stage_callback=_stage_cb,
                )

        # Upload PDF to MinIO
        pdf_key = None
        if minio and result.pdf_path and result.pdf_path.exists():
            pdf_key = f"output/{job_id}/invoice.pdf"
            minio.upload_pdf(result.pdf_path, pdf_key)

        manifest = {
            "pipeline_version": "2.0.0",
            "ocr_engine": "paddleocr-v3",
            "yolo_model": str(settings.yolo_model_path),
            "layoutlm_model": str(settings.layoutlm_model_path),
            "llm_model": str(settings.ollama_model),
            "timestamp": datetime.now().isoformat(),
            "model_used": getattr(result, "model_used", "hybrid"),
        }

        await db.update_job(
            job_id,
            status="done",
            output_json=result.invoice.model_dump(),
            ai_output_json=result.invoice.model_dump(),
            model_manifest=manifest,
            document_type=getattr(result, "doc_type", "unknown"),
            quality_score=getattr(result, "quality_score", 1.0),
            field_confidences=result.invoice.field_confidences,
            review_status="auto_accepted" if not result.invoice.needs_review else "pending",
            ground_truth_source="auto_accepted" if not result.invoice.needs_review else "partial",
            output_pdf_key=pdf_key,
            storage_key=storage_key,
            overall_confidence=result.invoice.overall_confidence,
            needs_review=result.invoice.needs_review,
            review_reasons=result.invoice.review_reasons,
            page_count=getattr(result, "page_count", 1),
        )
        # Push terminal "done" event so SSE clients close immediately
        _push_progress(job_id, {
            "job_id": job_id,
            "status": "done",
            "stage": "done",
            "stage_index": 6,
            "progress_pct": 100,
            "stage_label": "Digitization Complete",
        })
        logger.info(f"[{job_id}] Pipeline complete ✓")
    except Exception as e:
        logger.exception(f"[{job_id}] Pipeline failed: {e}")
        _push_progress(job_id, {
            "job_id": job_id,
            "status": "failed",
            "stage": "failed",
            "stage_index": 0,
            "progress_pct": 0,
            "stage_label": f"Failed: {str(e)}",
        })
    
# ------------------------------------------------------------------
# Active Learning & Intelligent Review Queue Endpoints
# ------------------------------------------------------------------

@app.get("/api/active-learning/queue", tags=["Active Learning"])
async def get_active_learning_queue(limit: int = 50):
    """
    Returns pending invoices ranked by Active Learning Informativeness Score.
    Reviewers are presented with the most informative/uncertain invoices first.
    """
    db: DatabaseManager = app.state.db
    from sqlalchemy import select
    from active_learning.sample_selector import prioritize_review_queue

    async with db.session() as session:
        stmt = (
            select(InvoiceRecord)
            .where(
                InvoiceRecord.output_json.isnot(None),
                InvoiceRecord.status.in_(["done", "partially_reviewed", "pending"]),
                InvoiceRecord.needs_review == True,
            )
            .limit(limit * 2)
        )
        result = await session.execute(stmt)
        records = result.scalars().all()

    invoices_data = []
    for r in records:
        inv = r.output_json or {}
        invoices_data.append({
            "job_id": r.job_id,
            "filename": r.filename,
            "status": r.status,
            "review_status": r.review_status,
            "overall_confidence": r.overall_confidence or 0.0,
            "field_confidences": r.field_confidences or inv.get("field_confidences", {}),
            "fields_needing_review": inv.get("fields_needing_review", []),
            "auto_accepted_fields": inv.get("auto_accepted_fields", []),
            "review_reasons": r.review_reasons or inv.get("review_reasons", []),
            "vendor_name": inv.get("vendor_name") or inv.get("company", {}).get("name"),
            "grand_total": inv.get("grand_total") or inv.get("totals", {}).get("grandTotal"),
            "created_at": r.created_at.isoformat() if r.created_at else None,
        })

    ranked = prioritize_review_queue(invoices_data)[:limit]
    return {
        "total_in_queue": len(ranked),
        "queue": ranked,
    }


@app.post("/api/active-learning/auto-accept", tags=["Active Learning"])
async def auto_accept_high_confidence():
    """
    Batch auto-accepts all invoices with confidence >= 0.85 and 0 error flags.
    Converts them directly into verified Ground Truth without manual button clicking.
    """
    db: DatabaseManager = app.state.db
    from sqlalchemy import select, update

    async with db.session() as session:
        stmt = select(InvoiceRecord.job_id, InvoiceRecord.review_reasons, InvoiceRecord.output_json).where(
            InvoiceRecord.output_json.isnot(None),
            InvoiceRecord.status.in_(["done", "partially_reviewed", "pending"]),
            InvoiceRecord.overall_confidence >= 0.85,
        )
        result = await session.execute(stmt)
        rows = result.all()

    eligible_job_ids = []
    for job_id, review_reasons, output_json in rows:
        inv = output_json or {}
        reasons = review_reasons or inv.get("review_reasons", [])
        if not reasons or all("error" not in str(err).lower() for err in reasons):
            eligible_job_ids.append(job_id)

    if eligible_job_ids:
        async with db.session() as session:
            stmt = (
                update(InvoiceRecord)
                .where(InvoiceRecord.job_id.in_(eligible_job_ids))
                .values(
                    status="reviewed",
                    review_status="auto_accepted",
                    needs_review=False,
                    review_reasons=[],
                )
            )
            await session.execute(stmt)
            await session.commit()

    return {
        "auto_accepted_count": len(eligible_job_ids),
        "message": f"Successfully auto-accepted {len(eligible_job_ids)} high-confidence invoices as verified ground truth.",
    }


@app.get("/api/active-learning/stats", tags=["Active Learning"])
async def get_active_learning_stats():
    """
    Returns active learning dataset, ground truth tiers, and human feedback loop metrics.
    """
    db: DatabaseManager = app.state.db
    from sqlalchemy import select, func
    from active_learning.auto_trainer import get_champion_metadata, check_retraining_trigger

    async with db.session() as session:
        total = (await session.execute(select(func.count(InvoiceRecord.id)))).scalar() or 0
        verified = (await session.execute(select(func.count(InvoiceRecord.id)).where(InvoiceRecord.status == "reviewed"))).scalar() or 0
        pending_review = (await session.execute(select(func.count(InvoiceRecord.id)).where(InvoiceRecord.needs_review == True))).scalar() or 0
        gold_corrected = (await session.execute(select(func.count(InvoiceRecord.id)).where(InvoiceRecord.ground_truth_source == "human_corrected"))).scalar() or 0
        silver_confirmed = (await session.execute(select(func.count(InvoiceRecord.id)).where(InvoiceRecord.ground_truth_source == "human_confirmed"))).scalar() or 0
        bronze_auto = (await session.execute(select(func.count(InvoiceRecord.id)).where(InvoiceRecord.ground_truth_source == "auto_accepted"))).scalar() or 0

    champion = get_champion_metadata()
    trigger_status = await check_retraining_trigger()

    return {
        "total_invoices": total,
        "verified_ground_truth": verified,
        "pending_review": pending_review,
        "tiers": {
            "gold_human_corrected": gold_corrected,
            "silver_human_confirmed": silver_confirmed,
            "bronze_auto_accepted": bronze_auto,
        },
        "champion_model": champion,
        "retraining_trigger": trigger_status,
    }


@app.get("/api/active-learning/champion-status", tags=["Active Learning"])
async def get_champion_status():
    """
    Returns current production champion model performance and holdout benchmark accuracy.
    """
    from active_learning.auto_trainer import get_champion_metadata
    return get_champion_metadata()


@app.post("/api/active-learning/auto-train", tags=["Active Learning"])
async def trigger_champion_retraining(epochs: int = 10):
    """
    Triggers Champion/Challenger candidate training run and auto-promotion gate.
    """
    from active_learning.auto_trainer import run_champion_challenger_retraining
    try:
        res = await run_champion_challenger_retraining(epochs=epochs)
        return res
    except Exception as e:
        logger.exception(f"Auto-training failed: {e}")
        raise HTTPException(status_code=500, detail=f"Training error: {e}")


@app.post("/api/active-learning/rollback", tags=["Active Learning"])
async def trigger_champion_rollback():
    """
    Rolls back the active LayoutLM champion to the most recent archived version.
    """
    from active_learning.auto_trainer import rollback_champion
    res = rollback_champion()
    if res.get("status") == "ERROR":
        raise HTTPException(status_code=400, detail=res.get("message"))
    return res


# ------------------------------------------------------------------
# Serve built React UI (production / LAN hosting)

# Build first: cd review_ui && npm run build
# ------------------------------------------------------------------

import os as _os

_UI_DIST = _os.path.join(_os.path.dirname(__file__), "..", "review_ui", "dist")
_UI_ASSETS = _os.path.join(_UI_DIST, "assets")

if _os.path.isdir(_UI_DIST):
    # Static assets (hashed JS/CSS bundles)
    if _os.path.isdir(_UI_ASSETS):
        app.mount("/assets", StaticFiles(directory=_UI_ASSETS), name="ui-assets")

    # Catch-all: serve static files in dist/ if they exist, otherwise index.html for React Router
    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        target = _os.path.join(_UI_DIST, full_path)
        if full_path and _os.path.isfile(target):
            return FileResponse(target)
        index_html = _os.path.join(_UI_DIST, "index.html")
        return FileResponse(index_html)

    logger.info(f"Serving React UI from {_UI_DIST}")
else:
    logger.warning(
        "React UI dist not found — run  cd review_ui && npm run build  "
        "to enable LAN hosting via FastAPI."
    )

