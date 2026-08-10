"""
Storage layer: PostgreSQL (via SQLAlchemy async) + MinIO (S3-compatible).
"""

import uuid
import io
from datetime import datetime
from pathlib import Path
from loguru import logger
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, mapped_column, Mapped
from sqlalchemy import String, Float, Boolean, DateTime, Text, JSON, func
from typing import Optional, Any
from minio import Minio
from minio.error import S3Error


# ------------------------------------------------------------------
# SQLAlchemy models
# ------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


class InvoiceRecord(Base):
    __tablename__ = "invoices"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    filename: Mapped[str] = mapped_column(String(255))
    status: Mapped[str] = mapped_column(String(50), default="pending")
    storage_key: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    output_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    output_pdf_key: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    overall_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    needs_review: Mapped[bool] = mapped_column(Boolean, default=False)
    review_reasons: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


# ------------------------------------------------------------------
# Database manager
# ------------------------------------------------------------------

class DatabaseManager:
    def __init__(self, database_url: str):
        self.engine = create_async_engine(database_url, echo=False)
        self.session_factory = async_sessionmaker(self.engine, expire_on_commit=False)

    async def init_db(self):
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables initialised")

    async def create_job(self, job_id: str, filename: str) -> InvoiceRecord:
        async with self.session_factory() as session:
            record = InvoiceRecord(
                id=str(uuid.uuid4()),
                job_id=job_id,
                filename=filename,
                status="pending",
            )
            session.add(record)
            await session.commit()
            await session.refresh(record)
            return record

    async def update_job(self, job_id: str, **kwargs) -> Optional[InvoiceRecord]:
        from sqlalchemy import select, update
        async with self.session_factory() as session:
            stmt = (
                update(InvoiceRecord)
                .where(InvoiceRecord.job_id == job_id)
                .values(**kwargs)
                .returning(InvoiceRecord)
            )
            result = await session.execute(stmt)
            await session.commit()
            return result.scalar_one_or_none()

    async def get_job(self, job_id: str) -> Optional[InvoiceRecord]:
        from sqlalchemy import select
        async with self.session_factory() as session:
            stmt = select(InvoiceRecord).where(InvoiceRecord.job_id == job_id)
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

    async def list_jobs(self, limit: int = 50, offset: int = 0) -> list[InvoiceRecord]:
        from sqlalchemy import select
        async with self.session_factory() as session:
            stmt = (
                select(InvoiceRecord)
                .order_by(InvoiceRecord.created_at.desc())
                .limit(limit)
                .offset(offset)
            )
            result = await session.execute(stmt)
            return list(result.scalars().all())


# ------------------------------------------------------------------
# MinIO manager
# ------------------------------------------------------------------

class MinIOManager:
    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        secure: bool = False,
    ):
        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )
        self.bucket = bucket
        self._ensure_bucket()

    def _ensure_bucket(self):
        try:
            if not self.client.bucket_exists(self.bucket):
                self.client.make_bucket(self.bucket)
                logger.info(f"MinIO bucket created: {self.bucket}")
        except S3Error as e:
            logger.error(f"MinIO bucket error: {e}")

    def upload_file(self, file_bytes: bytes, object_name: str, content_type: str = "application/octet-stream") -> str:
        self.client.put_object(
            self.bucket,
            object_name,
            io.BytesIO(file_bytes),
            length=len(file_bytes),
            content_type=content_type,
        )
        logger.debug(f"Uploaded to MinIO: {object_name}")
        return object_name

    def upload_pdf(self, pdf_path: str | Path, object_name: str) -> str:
        self.client.fput_object(
            self.bucket,
            object_name,
            str(pdf_path),
            content_type="application/pdf",
        )
        return object_name

    def get_presigned_url(self, object_name: str, expires_hours: int = 24) -> str:
        from datetime import timedelta
        return self.client.presigned_get_object(
            self.bucket,
            object_name,
            expires=timedelta(hours=expires_hours),
        )

    def download_file(self, object_name: str) -> bytes:
        response = self.client.get_object(self.bucket, object_name)
        return response.read()
