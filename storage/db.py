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

try:
    from minio import Minio
    from minio.error import S3Error
    HAS_MINIO = True
except ImportError:
    Minio = None
    S3Error = Exception
    HAS_MINIO = False


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


class ExtractionRunRecord(Base):
    __tablename__ = "extraction_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id: Mapped[str] = mapped_column(String(36), index=True)
    pipeline_version: Mapped[str] = mapped_column(String(32), default="2.0.0")
    document_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True, index=True)
    template_version_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    routing_decision: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    overall_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    decision: Mapped[str] = mapped_column(String(32), default="REVIEW")  # AUTO_ACCEPT | REVIEW
    is_auto_accepted: Mapped[bool] = mapped_column(Boolean, default=False)
    model_manifest: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class FieldDecisionRecord(Base):
    __tablename__ = "field_decisions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    run_id: Mapped[str] = mapped_column(String(36), index=True)
    job_id: Mapped[str] = mapped_column(String(36), index=True)
    field_name: Mapped[str] = mapped_column(String(64), index=True)
    selected_value: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    source: Mapped[str] = mapped_column(String(32))  # tie, layoutlm, heuristic, llm, vision_llm, ocr
    page: Mapped[int] = mapped_column(default=1)
    bbox: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    ocr_confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    validation_status: Mapped[str] = mapped_column(String(32), default="passed")
    evidence_summary: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class ReviewEventRecord(Base):
    __tablename__ = "review_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    job_id: Mapped[str] = mapped_column(String(36), index=True)
    field_name: Mapped[str] = mapped_column(String(64), index=True)
    old_value: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    new_value: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    reason: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    reviewer_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class DocumentSegmentRecord(Base):
    __tablename__ = "document_segments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    parent_job_id: Mapped[str] = mapped_column(String(36), index=True)
    child_job_id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    segment_index: Mapped[int] = mapped_column(default=0)
    page_start: Mapped[int] = mapped_column(default=1)
    page_end: Mapped[int] = mapped_column(default=1)
    detected_invoice_number: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    detected_vendor: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


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

    async def get_all_active_templates_joined(
        self,
    ) -> list[tuple[TemplateVersionRecord, Optional[TemplateFamilyRecord], list[TemplateFieldRuleRecord]]]:
        """
        Loads all active template versions with their families and field rules in exactly 3 batched queries.
        Completely eliminates N+1 query overhead.
        """
        from sqlalchemy import select
        async with self.session_factory() as session:
            v_stmt = select(TemplateVersionRecord).where(TemplateVersionRecord.status == "active")
            v_res = await session.execute(v_stmt)
            versions = list(v_res.scalars().all())
            if not versions:
                return []

            v_ids = [v.id for v in versions]
            fam_ids = list(set(v.family_id for v in versions if v.family_id))

            # Batch fetch families
            fam_map: dict[str, TemplateFamilyRecord] = {}
            if fam_ids:
                fam_stmt = select(TemplateFamilyRecord).where(TemplateFamilyRecord.id.in_(fam_ids))
                fam_res = await session.execute(fam_stmt)
                for f in fam_res.scalars().all():
                    fam_map[f.id] = f

            # Batch fetch field rules
            rules_map: dict[str, list[TemplateFieldRuleRecord]] = {vid: [] for vid in v_ids}
            if v_ids:
                r_stmt = select(TemplateFieldRuleRecord).where(TemplateFieldRuleRecord.version_id.in_(v_ids))
                r_res = await session.execute(r_stmt)
                for r in r_res.scalars().all():
                    rules_map[r.version_id].append(r)

            return [(v, fam_map.get(v.family_id), rules_map.get(v.id, [])) for v in versions]

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
        field_results: Optional[list[dict]] = None,
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
                ver.sample_count = tot
                ver.success_rate = round(ver.success_count / tot, 3) if tot > 0 else 1.0

            results_to_process = field_results if field_results else (
                [{"field": fc.get("field"), "status": "CORRECTED"} for fc in (field_corrections or [])]
            )

            for fr in results_to_process:
                f_name = fr.get("field")
                status = fr.get("status", "CORRECTED")
                if not f_name or status in ("NOT_PRESENT", "NOT_APPLICABLE"):
                    continue

                st_stmt = select(TemplateStatisticsRecord).where(
                    TemplateStatisticsRecord.version_id == version_id,
                    TemplateStatisticsRecord.field_name == f_name,
                )
                st_res = await session.execute(st_stmt)
                stat = st_res.scalar_one_or_none()
                is_correct = (status == "CORRECT")

                if not stat:
                    stat = TemplateStatisticsRecord(
                        id=str(uuid.uuid4()),
                        version_id=version_id,
                        field_name=f_name,
                        sample_count=1,
                        correct_count=1 if is_correct else 0,
                        correction_count=0 if is_correct else 1,
                    )
                    session.add(stat)
                else:
                    stat.sample_count += 1
                    if is_correct:
                        stat.correct_count += 1
                    else:
                        stat.correction_count += 1

            await session.commit()

    async def record_extraction_run(
        self,
        job_id: str,
        pipeline_version: str = "2.0.0",
        document_hash: Optional[str] = None,
        template_version_id: Optional[str] = None,
        routing_decision: Optional[str] = None,
        overall_confidence: float = 0.0,
        decision: str = "REVIEW",
        is_auto_accepted: bool = False,
        model_manifest: Optional[dict] = None,
        field_decisions: Optional[list[dict]] = None,
    ) -> str:
        """Records an immutable extraction run and its field-level decisions."""
        run_id = str(uuid.uuid4())
        async with self.session_factory() as session:
            run_rec = ExtractionRunRecord(
                id=run_id,
                job_id=job_id,
                pipeline_version=pipeline_version,
                document_hash=document_hash,
                template_version_id=template_version_id,
                routing_decision=routing_decision,
                overall_confidence=overall_confidence,
                decision=decision,
                is_auto_accepted=is_auto_accepted,
                model_manifest=model_manifest or {},
            )
            session.add(run_rec)

            if field_decisions:
                for fd in field_decisions:
                    fd_rec = FieldDecisionRecord(
                        id=str(uuid.uuid4()),
                        run_id=run_id,
                        job_id=job_id,
                        field_name=fd["field_name"],
                        selected_value=str(fd.get("value", "")),
                        confidence=float(fd.get("confidence", 0.0)),
                        source=fd.get("source", "ocr"),
                        page=int(fd.get("page", 1)),
                        bbox=fd.get("bbox"),
                        ocr_confidence=float(fd.get("ocr_confidence", 0.0)) if fd.get("ocr_confidence") is not None else None,
                        validation_status=fd.get("validation_status", "passed"),
                        evidence_summary=fd.get("evidence_summary"),
                    )
                    session.add(fd_rec)

            await session.commit()
            return run_id

    async def record_review_event(
        self,
        job_id: str,
        field_name: str,
        old_value: Optional[str],
        new_value: Optional[str],
        reason: Optional[str] = None,
        reviewer_id: Optional[str] = None,
    ):
        """Records an immutable review edit event for full human auditability."""
        async with self.session_factory() as session:
            ev = ReviewEventRecord(
                id=str(uuid.uuid4()),
                job_id=job_id,
                field_name=field_name,
                old_value=old_value,
                new_value=new_value,
                reason=reason,
                reviewer_id=reviewer_id,
            )
            session.add(ev)
            await session.commit()

    async def record_document_segment(
        self,
        parent_job_id: str,
        child_job_id: str,
        segment_index: int,
        page_start: int,
        page_end: int,
        detected_invoice_number: Optional[str] = None,
        detected_vendor: Optional[str] = None,
    ) -> str:
        """Records a detected document segment linking parent multi-invoice job to child extraction."""
        rec_id = str(uuid.uuid4())
        async with self.session_factory() as session:
            seg_rec = DocumentSegmentRecord(
                id=rec_id,
                parent_job_id=parent_job_id,
                child_job_id=child_job_id,
                segment_index=segment_index,
                page_start=page_start,
                page_end=page_end,
                detected_invoice_number=detected_invoice_number,
                detected_vendor=detected_vendor,
            )
            session.add(seg_rec)
            await session.commit()
            return rec_id

    async def get_extraction_audit_trail(self, job_id: str) -> dict[str, Any]:
        """Fetches the complete immutable provenance trail for a processed invoice."""
        from sqlalchemy import select
        async with self.session_factory() as session:
            # Fetch latest run
            r_stmt = select(ExtractionRunRecord).where(ExtractionRunRecord.job_id == job_id).order_by(ExtractionRunRecord.created_at.desc())
            r_res = await session.execute(r_stmt)
            latest_run = r_res.scalars().first()

            decisions = []
            if latest_run:
                d_stmt = select(FieldDecisionRecord).where(FieldDecisionRecord.run_id == latest_run.id)
                d_res = await session.execute(d_stmt)
                for d in d_res.scalars().all():
                    decisions.append({
                        "field_name": d.field_name,
                        "value": d.selected_value,
                        "confidence": d.confidence,
                        "source": d.source,
                        "page": d.page,
                        "bbox": d.bbox,
                        "ocr_confidence": d.ocr_confidence,
                        "validation_status": d.validation_status,
                    })

            # Fetch review history
            rev_stmt = select(ReviewEventRecord).where(ReviewEventRecord.job_id == job_id).order_by(ReviewEventRecord.created_at.asc())
            rev_res = await session.execute(rev_stmt)
            review_events = [
                {
                    "field_name": re.field_name,
                    "old_value": re.old_value,
                    "new_value": re.new_value,
                    "reason": re.reason,
                    "reviewer_id": re.reviewer_id,
                    "timestamp": re.created_at.isoformat() if re.created_at else None,
                }
                for re in rev_res.scalars().all()
            ]

            return {
                "job_id": job_id,
                "run": {
                    "run_id": latest_run.id if latest_run else None,
                    "pipeline_version": latest_run.pipeline_version if latest_run else None,
                    "document_hash": latest_run.document_hash if latest_run else None,
                    "template_version_id": latest_run.template_version_id if latest_run else None,
                    "routing_decision": latest_run.routing_decision if latest_run else None,
                    "overall_confidence": latest_run.overall_confidence if latest_run else None,
                    "decision": latest_run.decision if latest_run else None,
                } if latest_run else None,
                "field_decisions": decisions,
                "review_history": review_events,
            }

    async def record_document_segment(
        self,
        parent_job_id: str,
        child_job_id: str,
        segment_index: int,
        page_start: int,
        page_end: int,
        detected_invoice_number: Optional[str] = None,
        detected_vendor: Optional[str] = None,
    ):
        """Maps a sub-invoice child job to its parent multi-invoice upload."""
        async with self.session_factory() as session:
            seg = DocumentSegmentRecord(
                id=str(uuid.uuid4()),
                parent_job_id=parent_job_id,
                child_job_id=child_job_id,
                segment_index=segment_index,
                page_start=page_start,
                page_end=page_end,
                detected_invoice_number=detected_invoice_number,
                detected_vendor=detected_vendor,
            )
            session.add(seg)
            await session.commit()

    async def get_child_segments(self, parent_job_id: str) -> list[DocumentSegmentRecord]:
        """Retrieves all sub-invoice children belonging to a segmented parent job."""
        from sqlalchemy import select
        async with self.session_factory() as session:
            stmt = select(DocumentSegmentRecord).where(
                DocumentSegmentRecord.parent_job_id == parent_job_id
            ).order_by(DocumentSegmentRecord.segment_index.asc())
            res = await session.execute(stmt)
            return list(res.scalars().all())



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
        self.bucket = bucket
        if not HAS_MINIO:
            logger.warning("MinIO package not available. MinIOManager running in disabled mode.")
            self.client = None
            return

        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )
        self._ensure_bucket()

    def _ensure_bucket(self):
        if not self.client:
            return
        try:
            if not self.client.bucket_exists(self.bucket):
                self.client.make_bucket(self.bucket)
                logger.info(f"MinIO bucket created: {self.bucket}")
        except Exception as e:
            logger.error(f"MinIO bucket error: {e}")

    def upload_file(self, file_bytes: bytes, object_name: str, content_type: str = "application/octet-stream") -> str:
        if not self.client:
            return object_name
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
        if not self.client:
            return object_name
        self.client.fput_object(
            self.bucket,
            object_name,
            str(pdf_path),
            content_type="application/pdf",
        )
        return object_name

    def get_presigned_url(self, object_name: str, expires_hours: int = 24) -> str:
        if not self.client:
            return ""
        from datetime import timedelta
        return self.client.presigned_get_object(
            self.bucket,
            object_name,
            expires=timedelta(hours=expires_hours),
        )

    def download_file(self, object_name: str) -> bytes:
        if not self.client:
            return b""
        response = self.client.get_object(self.bucket, object_name)
        return response.read()
