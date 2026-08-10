"""
Celery worker — alternative to FastAPI background tasks.
Use this for high-volume production deployments.

Start with:
    celery -A worker.celery_app worker --loglevel=info --concurrency=2
"""

from celery import Celery
from loguru import logger
from config.settings import get_settings

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


@celery_app.task(bind=True, max_retries=3, default_retry_delay=10)
def process_invoice_task(self, job_id: str, file_bytes_b64: str, filename: str):
    """
    Celery task for processing an invoice.
    file_bytes_b64: base64-encoded file bytes (JSON-safe).
    """
    import base64
    import asyncio
    from api.pipeline_runner import InvoicePipeline
    from storage.db import DatabaseManager, MinIOManager

    try:
        file_bytes = base64.b64decode(file_bytes_b64)
        settings = get_settings()
        pipeline = InvoicePipeline(settings)

        result = pipeline.process(
            file_bytes=file_bytes,
            filename=filename,
            job_id=job_id,
        )

        # Sync DB update
        import asyncio
        db = DatabaseManager(settings.database_url)

        async def update():
            await db.update_job(
                job_id,
                status="done",
                output_json=result.invoice.model_dump(),
                overall_confidence=result.invoice.overall_confidence,
                needs_review=result.invoice.needs_review,
                review_reasons=result.invoice.review_reasons,
            )

        asyncio.get_event_loop().run_until_complete(update())
        logger.info(f"Celery task complete: {job_id}")
        return {"job_id": job_id, "status": "done"}

    except Exception as exc:
        logger.error(f"Celery task failed: {job_id} — {exc}")
        self.retry(exc=exc)
