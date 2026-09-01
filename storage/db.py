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
# SQLAlchemy models
# ------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


class TemplateFamilyRecord(Base):
    __tablename__ = "template_families"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    family_code: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    family_fingerprint: Mapped[str] = mapped_column(String(64), index=True)
    vendor_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    vendor_gstin: Mapped[Optional[str]] = mapped_column(String(32), nullable=True, index=True)
    vendor_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    status: Mapped[str] = mapped_column(String(32), default="active")
    sample_count: Mapped[int] = mapped_column(default=1)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    last_seen: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class TemplateVersionRecord(Base):
    __tablename__ = "template_versions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    family_id: Mapped[str] = mapped_column(String(36), index=True)
    version_num: Mapped[int] = mapped_column(default=1)
    version_fingerprint: Mapped[str] = mapped_column(String(64), index=True)
    aspect_ratio: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    page_count: Mapped[int] = mapped_column(default=1)
    anchor_signature: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    layout_signature: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    topology_spec: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    sample_count: Mapped[int] = mapped_column(default=1)
    success_count: Mapped[int] = mapped_column(default=0)
    correction_count: Mapped[int] = mapped_column(default=0)
    success_rate: Mapped[float] = mapped_column(Float, default=1.0)
    status: Mapped[str] = mapped_column(String(32), default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    last_promoted_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class TemplateFieldRuleRecord(Base):
    __tablename__ = "template_field_rules"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    version_id: Mapped[str] = mapped_column(String(36), index=True)
    field_name: Mapped[str] = mapped_column(String(64), index=True)
    strategy: Mapped[str] = mapped_column(String(64))  # anchor_relative, regex_pattern, semantic_numeric, spatial_table, constant
    anchors: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    search_region: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)  # [x1, y1, x2, y2] in 0-1000
    relative_box: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)   # [dx1, dy1, dx2, dy2] relative to anchor
    parser_spec: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    validator_spec: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    confidence_score: Mapped[float] = mapped_column(Float, default=0.95)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class TemplateStatisticsRecord(Base):
    __tablename__ = "template_statistics"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    version_id: Mapped[str] = mapped_column(String(36), index=True)
    field_name: Mapped[str] = mapped_column(String(64), index=True)
    sample_count: Mapped[int] = mapped_column(default=1)
    correct_count: Mapped[int] = mapped_column(default=1)
    correction_count: Mapped[int] = mapped_column(default=0)
    last_updated: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now())


class InvoiceRecord(Base):
    __tablename__ = "invoices"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    filename: Mapped[str] = mapped_column(String(255))
    document_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    document_type: Mapped[Optional[str]] = mapped_column(String(32), default="unknown")
    quality_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(50), default="pending")
    storage_key: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    output_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    ai_output_json: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    field_confidences: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    model_manifest: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    corrections: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    review_status: Mapped[str] = mapped_column(String(50), default="pending")
    ground_truth_source: Mapped[str] = mapped_column(String(50), default="auto_accepted")
    template_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    template_family_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    template_version_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    disagreement_score: Mapped[Optional[float]] = mapped_column(Float, default=0.0)
    output_pdf_key: Mapped[Optional[str]] = mapped_column(String(512), nullable=True)
    overall_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    needs_review: Mapped[bool] = mapped_column(Boolean, default=False)
    review_reasons: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    page_count: Mapped[Optional[int]] = mapped_column(default=1)
    ocr_word_count: Mapped[Optional[int]] = mapped_column(default=0)
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

    def session(self):
        """Context manager helper for database sessions."""
        return self.session_factory()

    async def init_db(self):
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            from sqlalchemy import text
            columns_to_add = [
                ("ai_output_json", "JSON"),
                ("field_confidences", "JSON"),
                ("corrections", "JSON"),
                ("review_status", "VARCHAR(50) DEFAULT 'pending'"),
                ("ground_truth_source", "VARCHAR(50) DEFAULT 'auto_accepted'"),
                ("template_id", "VARCHAR(64)"),
                ("template_family_id", "VARCHAR(36)"),
                ("template_version_id", "VARCHAR(36)"),
                ("disagreement_score", "FLOAT DEFAULT 0.0"),
                ("document_hash", "VARCHAR(64)"),
                ("document_type", "VARCHAR(32) DEFAULT 'unknown'"),
                ("quality_score", "FLOAT"),
                ("model_manifest", "JSON"),
                ("page_count", "INTEGER DEFAULT 1"),
                ("ocr_word_count", "INTEGER DEFAULT 0"),
            ]
            for col_name, col_type in columns_to_add:
                try:
                    await conn.execute(text(f"ALTER TABLE invoices ADD COLUMN IF NOT EXISTS {col_name} {col_type};"))
                except Exception as ex:
                    logger.debug(f"Migration notice for {col_name}: {ex}")
        logger.info("Database tables initialised and migrated")

    async def create_job(self, job_id: str, filename: str, document_hash: Optional[str] = None) -> InvoiceRecord:
        async with self.session_factory() as session:
            record = InvoiceRecord(
                id=str(uuid.uuid4()),
                job_id=job_id,
                filename=filename,
                document_hash=document_hash,
                status="pending",
            )
            session.add(record)
            await session.commit()
            await session.refresh(record)
            return record

    async def get_job_by_hash(self, document_hash: str) -> Optional[InvoiceRecord]:
        from sqlalchemy import select
        async with self.session_factory() as session:
            stmt = (
                select(InvoiceRecord)
                .where(InvoiceRecord.document_hash == document_hash)
                .order_by(InvoiceRecord.created_at.desc())
                .limit(1)
            )
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

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

    async def delete_job(self, job_id: str) -> bool:
        from sqlalchemy import delete
        async with self.session_factory() as session:
            stmt = delete(InvoiceRecord).where(InvoiceRecord.job_id == job_id)
            result = await session.execute(stmt)
            await session.commit()
            return result.rowcount > 0

    async def bulk_delete_jobs(self, statuses: list[str]) -> int:
        from sqlalchemy import update
        async with self.session_factory() as session:
            stmt = (
                update(InvoiceRecord)
                .where(InvoiceRecord.status.in_(statuses))
                .values(status="deleted")
            )
            result = await session.execute(stmt)
            await session.commit()
            return result.rowcount or 0

    async def list_jobs(
        self,
        limit: int = 50,
        offset: int = 0,
        status: Optional[str] = None,
        search: Optional[str] = None,
        needs_review: Optional[bool] = None,
        exclude_pending: bool = False,
    ) -> tuple[list[InvoiceRecord], int]:
        from sqlalchemy import select, func, or_
        async with self.session_factory() as session:
            query = select(InvoiceRecord)
            count_query = select(func.count(InvoiceRecord.id))

            conditions = []
            if exclude_pending:
                conditions.append(InvoiceRecord.status != "pending")
            elif status and status != "all":
                if status == "non_pending":
                    conditions.append(InvoiceRecord.status != "pending")
                elif status in ("reviewed", "partially_reviewed", "done", "processing", "failed", "pending"):
                    conditions.append(InvoiceRecord.status == status)

            if search and search.strip():
                s = f"%{search.strip()}%"
                conditions.append(
                    or_(
                        InvoiceRecord.filename.ilike(s),
                        InvoiceRecord.job_id.ilike(s),
                    )
                )

            if needs_review is not None:
                conditions.append(InvoiceRecord.needs_review == needs_review)

            for cond in conditions:
                query = query.where(cond)
                count_query = count_query.where(cond)

            # Get total matching count
            total_count_res = await session.execute(count_query)
            total_count = total_count_res.scalar() or 0

            # Get paginated slice
            query = query.order_by(InvoiceRecord.created_at.desc()).limit(limit).offset(offset)
            result = await session.execute(query)
            records = list(result.scalars().all())

            return records, total_count

    # ── Template Registry Methods ──────────────────────────────────────────

    async def get_or_create_template_family(
        self,
        family_fingerprint: str,
        vendor_name: Optional[str] = None,
        vendor_gstin: Optional[str] = None,
    ) -> TemplateFamilyRecord:
        from sqlalchemy import select
        async with self.session_factory() as session:
            stmt = select(TemplateFamilyRecord).where(
                TemplateFamilyRecord.family_fingerprint == family_fingerprint
            )
            result = await session.execute(stmt)
            family = result.scalar_one_or_none()

            if not family:
                family_code = f"fam_{family_fingerprint[:10]}"
                family = TemplateFamilyRecord(
                    id=str(uuid.uuid4()),
                    family_code=family_code,
                    family_fingerprint=family_fingerprint,
                    vendor_name=vendor_name,
                    vendor_gstin=vendor_gstin,
                    status="active",
                    sample_count=1,
                )
                session.add(family)
            else:
                family.sample_count += 1
                if vendor_name and not family.vendor_name:
                    family.vendor_name = vendor_name
                if vendor_gstin and not family.vendor_gstin:
                    family.vendor_gstin = vendor_gstin

            await session.commit()
            await session.refresh(family)
            return family

    async def get_template_family(self, family_id: str) -> Optional[TemplateFamilyRecord]:
        from sqlalchemy import select
        async with self.session_factory() as session:
            stmt = select(TemplateFamilyRecord).where(TemplateFamilyRecord.id == family_id)
            result = await session.execute(stmt)
            return result.scalar_one_or_none()

    async def get_or_create_template_version(
        self,
        family_id: str,
        version_fingerprint: str,
        aspect_ratio: Optional[float] = None,
        page_count: int = 1,
        anchor_signature: Optional[str] = None,
        layout_signature: Optional[str] = None,
        topology_spec: Optional[dict] = None,
    ) -> TemplateVersionRecord:
        from sqlalchemy import select, func
        async with self.session_factory() as session:
            stmt = select(TemplateVersionRecord).where(
                TemplateVersionRecord.version_fingerprint == version_fingerprint
            )
            result = await session.execute(stmt)
            version = result.scalar_one_or_none()

            if not version:
                # Get next version number in family
                ver_count_stmt = select(func.count(TemplateVersionRecord.id)).where(
                    TemplateVersionRecord.family_id == family_id
                )
                cnt_res = await session.execute(ver_count_stmt)
                next_vnum = (cnt_res.scalar() or 0) + 1

                version = TemplateVersionRecord(
                    id=str(uuid.uuid4()),
                    family_id=family_id,
                    version_num=next_vnum,
                    version_fingerprint=version_fingerprint,
                    aspect_ratio=aspect_ratio,
                    page_count=page_count,
                    anchor_signature=anchor_signature,
                    layout_signature=layout_signature,
                    topology_spec=topology_spec,
                    sample_count=1,
                    status="active",
                )
                session.add(version)
            else:
                version.sample_count += 1

            await session.commit()
            await session.refresh(version)
            return version

    async def get_all_active_template_versions(self) -> list[TemplateVersionRecord]:
        from sqlalchemy import select
        async with self.session_factory() as session:
            stmt = (
                select(TemplateVersionRecord)
                .where(TemplateVersionRecord.status == "active")
                .order_by(TemplateVersionRecord.sample_count.desc())
            )
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def save_field_rules(self, version_id: str, rules: list[dict[str, Any]]):
        from sqlalchemy import delete
        async with self.session_factory() as session:
            # Clear old rules for this version
            await session.execute(
                delete(TemplateFieldRuleRecord).where(TemplateFieldRuleRecord.version_id == version_id)
            )
            for r in rules:
                rule_rec = TemplateFieldRuleRecord(
                    id=str(uuid.uuid4()),
                    version_id=version_id,
                    field_name=r["field_name"],
                    strategy=r.get("strategy", "anchor_relative"),
                    anchors=r.get("anchors", []),
                    search_region=r.get("search_region"),
                    relative_box=r.get("relative_box"),
                    parser_spec=r.get("parser_spec"),
                    validator_spec=r.get("validator_spec"),
                    confidence_score=r.get("confidence_score", 0.95),
                )
                session.add(rule_rec)
            await session.commit()

    async def get_field_rules_for_version(self, version_id: str) -> list[TemplateFieldRuleRecord]:
        from sqlalchemy import select
        async with self.session_factory() as session:
            stmt = select(TemplateFieldRuleRecord).where(
                TemplateFieldRuleRecord.version_id == version_id
            )
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def record_template_feedback(
        self,
        version_id: str,
        was_correct: bool,
        field_corrections: Optional[list[dict]] = None,
    ):
        from sqlalchemy import update, select
        async with self.session_factory() as session:
            stmt = select(TemplateVersionRecord).where(TemplateVersionRecord.id == version_id)
            res = await session.execute(stmt)
            ver = res.scalar_one_or_none()
            if ver:
                if was_correct:
                    ver.success_count += 1
                else:
                    ver.correction_count += 1
                tot = ver.success_count + ver.correction_count
                ver.success_rate = round(ver.success_count / tot, 3) if tot > 0 else 1.0

            if field_corrections:
                for fc in field_corrections:
                    f_name = fc.get("field")
                    if not f_name:
                        continue
                    st_stmt = select(TemplateStatisticsRecord).where(
                        TemplateStatisticsRecord.version_id == version_id,
                        TemplateStatisticsRecord.field_name == f_name,
                    )
                    st_res = await session.execute(st_stmt)
                    stat = st_res.scalar_one_or_none()
                    if not stat:
                        stat = TemplateStatisticsRecord(
                            id=str(uuid.uuid4()),
                            version_id=version_id,
                            field_name=f_name,
                            sample_count=1,
                            correct_count=0,
                            correction_count=1,
                        )
                        session.add(stat)
                    else:
                        stat.sample_count += 1
                        stat.correction_count += 1

            await session.commit()



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
