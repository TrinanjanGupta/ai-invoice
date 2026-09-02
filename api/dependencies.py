"""FastAPI dependency injection helpers."""

from fastapi import Request
from config.settings import get_settings


def get_pipeline(request: Request):
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        from api.pipeline_runner import InvoicePipeline
        db = getattr(request.app.state, "db", None)
        minio = getattr(request.app.state, "minio", None)
        pipeline = InvoicePipeline(get_settings(), db_manager=db, minio_manager=minio)
        request.app.state.pipeline = pipeline
    return pipeline


def get_db(request: Request):
    return request.app.state.db


def get_minio(request: Request):
    return getattr(request.app.state, "minio", None)
