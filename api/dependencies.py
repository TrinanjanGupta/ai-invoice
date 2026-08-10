"""FastAPI dependency injection helpers."""

from fastapi import Request
from config.settings import get_settings


def get_pipeline(request: Request):
    return request.app.state.pipeline


def get_db(request: Request):
    return request.app.state.db


def get_minio(request: Request):
    return getattr(request.app.state, "minio", None)
