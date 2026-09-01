"""
tests/test_audit_trail.py

Unit tests for immutable extraction runs, field decisions, and review event audit logging.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from storage.db import (
    DatabaseManager,
    ExtractionRunRecord,
    FieldDecisionRecord,
    ReviewEventRecord,
    DocumentSegmentRecord,
)


@pytest.mark.asyncio
async def test_record_extraction_run_and_decisions():
    mock_session = AsyncMock()
    mock_session.add = MagicMock()
    mock_session_factory = MagicMock()
    mock_session_factory.return_value.__aenter__.return_value = mock_session
    mock_session_factory.return_value.__aexit__.return_value = None

    db = MagicMock(spec=DatabaseManager)
    db.session_factory = mock_session_factory
    db.record_extraction_run = DatabaseManager.record_extraction_run.__get__(db, DatabaseManager)

    run_id = await db.record_extraction_run(
        job_id="job_test_101",
        pipeline_version="2.0.0",
        document_hash="abc123hash",
        template_version_id="ver_001",
        routing_decision="DIGITAL_PDF",
        overall_confidence=0.98,
        decision="AUTO_ACCEPT",
        is_auto_accepted=True,
        model_manifest={"ocr": "native", "tie": "fast_path"},
        field_decisions=[
            {
                "field_name": "grand_total",
                "value": "15600.00",
                "confidence": 0.99,
                "source": "tie",
                "page": 1,
                "bbox": [100, 200, 300, 250],
                "ocr_confidence": 0.99,
                "validation_status": "passed",
            }
        ],
    )

    assert run_id is not None
    assert mock_session.add.call_count == 2  # 1 run record + 1 field decision record
    assert mock_session.commit.call_count == 1


@pytest.mark.asyncio
async def test_record_review_event():
    mock_session = AsyncMock()
    mock_session.add = MagicMock()
    mock_session_factory = MagicMock()
    mock_session_factory.return_value.__aenter__.return_value = mock_session
    mock_session_factory.return_value.__aexit__.return_value = None

    db = MagicMock(spec=DatabaseManager)
    db.session_factory = mock_session_factory
    db.record_review_event = DatabaseManager.record_review_event.__get__(db, DatabaseManager)

    await db.record_review_event(
        job_id="job_test_101",
        field_name="vendor_name",
        old_value="Acme",
        new_value="Acme Corp",
        reason="Missing suffix",
        reviewer_id="user_123",
    )

    assert mock_session.add.call_count == 1
    assert mock_session.commit.call_count == 1


@pytest.mark.asyncio
async def test_record_document_segment():
    mock_session = AsyncMock()
    mock_session.add = MagicMock()
    mock_session_factory = MagicMock()
    mock_session_factory.return_value.__aenter__.return_value = mock_session
    mock_session_factory.return_value.__aexit__.return_value = None

    db = MagicMock(spec=DatabaseManager)
    db.session_factory = mock_session_factory
    db.record_document_segment = DatabaseManager.record_document_segment.__get__(db, DatabaseManager)

    await db.record_document_segment(
        parent_job_id="parent_1",
        child_job_id="child_1",
        segment_index=0,
        page_start=1,
        page_end=2,
        detected_invoice_number="INV-001",
        detected_vendor="Acme",
    )

    assert mock_session.add.call_count == 1
    assert mock_session.commit.call_count == 1
