"""
worker/tasks.py

Celery worker for high-throughput (100k+/month) asynchronous invoice processing.
Features:
- Lazy warm singleton InvoicePipeline (models loaded ONCE per worker process).
- Zero-payload Redis queues: passes lightweight MinIO storage keys instead of multi-megabyte base64 blobs.
- Concurrency-safe database updates and immutable extraction provenance recording.

Start worker with:
    celery -A worker.tasks.celery_app worker --loglevel=info --concurrency=4
"""

import base64
import asyncio
from pathlib import Path
from typing import Optional
from celery import Celery
from loguru import logger
from config.settings import get_settings
from storage.db import DatabaseManager, MinIOManager, HAS_MINIO

settings = get_settings()

celery_app = Celery(
    "invoice_digitizer",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Kolkata",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
)

# ── Module-Level Warm Singleton ───────────────────────────────────────────────
_worker_pipeline = None
_worker_db = None
_worker_minio = None


def get_worker_components():
    global _worker_pipeline, _worker_db, _worker_minio
    if _worker_pipeline is None:
        logger.info("[Celery Worker] Initializing warm singleton models (PaddleOCR, YOLO, LayoutLM, TIE)...")
        app_settings = get_settings()
        _worker_db = DatabaseManager(app_settings.database_url)
        _worker_minio = None
        if HAS_MINIO and app_settings.minio_endpoint:
            try:
                _worker_minio = MinIOManager(
                    endpoint=app_settings.minio_endpoint,
                    access_key=app_settings.minio_access_key,
                    secret_key=app_settings.minio_secret_key,
                    bucket=app_settings.minio_bucket,
                    secure=app_settings.minio_secure,
                )
            except Exception as me:
                logger.warning(f"[Celery Worker] MinIO init notice: {me}")

        from api.pipeline_runner import InvoicePipeline
        _worker_pipeline = InvoicePipeline(settings=app_settings, db_manager=_worker_db, minio_manager=_worker_minio)
        try:
            asyncio.run(_worker_pipeline.template_retriever.load_templates_from_db())
            logger.info("[Celery Worker] Pre-warmed TIE template cache from database.")
        except Exception as tpl_err:
            logger.debug(f"[Celery Worker] Template cache pre-warm notice: {tpl_err}")
        logger.info("[Celery Worker] Warm singleton initialized successfully.")

    return _worker_pipeline, _worker_db, _worker_minio


@celery_app.task(bind=True, max_retries=3, default_retry_delay=10)
def process_invoice_task(
    self,
    job_id: str,
    storage_key: Optional[str] = None,
    filename: Optional[str] = None,
    file_bytes_b64: Optional[str] = None,
):
    """
    Celery task for processing an invoice.
    Passes lightweight MinIO storage_key to avoid sending heavy binaries through Redis.
    """
    try:
        pipeline, db, minio = get_worker_components()
        fn = filename or "uploaded_invoice"

        # 1. Retrieve raw file bytes from MinIO or local cache
        file_bytes = None
        if storage_key and minio:
            try:
                file_bytes = minio.download_file(storage_key)
            except Exception as dl_err:
                logger.warning(f"[Celery Worker] MinIO download failed for {storage_key}: {dl_err}")

        if not file_bytes:
            # Fallback to local raw file cache
            local_raw_path = Path("data/raw") / f"{job_id}_{fn}"
            if local_raw_path.exists():
                with open(local_raw_path, "rb") as f:
                    file_bytes = f.read()

        if not file_bytes and file_bytes_b64:
            file_bytes = base64.b64decode(file_bytes_b64)

        if not file_bytes:
            raise FileNotFoundError(f"Could not locate raw invoice bytes for job {job_id} (key={storage_key})")

        # 2. Run pipeline inference using warm singleton
        result = pipeline.process(
            file_bytes=file_bytes,
            filename=fn,
            job_id=job_id,
        )

        # 3. Synchronize database state
        async def sync_db():
            inv = result.invoice
            rev_status = "auto_accepted" if not inv.needs_review else "pending"
            await db.update_job(
                job_id,
                status="done",
                output_json=inv.model_dump(),
                ai_output_json=inv.model_dump(),
                field_confidences=inv.field_confidences,
                review_status=rev_status,
                ground_truth_source="auto_accepted" if not inv.needs_review else "partial",
                overall_confidence=inv.overall_confidence,
                needs_review=inv.needs_review,
                review_reasons=inv.review_reasons,
                template_version_id=inv.template_version_id,
                disagreement_score=inv.disagreement_score,
                quality_score=getattr(result, "quality_score", 1.0),
            )

        asyncio.run(sync_db())
        logger.info(f"[Celery Worker] Successfully processed job {job_id} ({fn})")
        return {"job_id": job_id, "status": "done", "overall_confidence": result.invoice.overall_confidence}

    except Exception as exc:
        logger.exception(f"[Celery Worker] Task failed for job {job_id}: {exc}")
        self.retry(exc=exc)
