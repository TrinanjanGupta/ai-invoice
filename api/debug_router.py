"""
api/debug_router.py

Provides high-resolution debug inspection data for the Invoice Review UI:
- Visual bounding boxes for YOLO regions, OCR tokens/lines, TIE anchors, and handwriting zones.
- Normalized to standard 0-1000 coordinate space for direct SVG overlay over the preview image.
- Comprehensive Field Evidence Layer: selected values, competing candidates, sources,
  selection reasoning, validation status, and disagreement scores.
"""

import json
from pathlib import Path
from typing import Optional, Any
from loguru import logger
from fastapi import APIRouter, HTTPException, Request

from preprocessing.document_profile import DocumentProfile, normalize_box


debug_router = APIRouter(prefix="/api/invoices", tags=["Debug"])

# In-memory fast cache for debug overlays: job_id -> dict of page_idx -> overlay_data
_job_debug_cache: dict[str, dict[int, dict]] = {}


def normalize_to_1000(
    box: Any,
    width: int = 1000,
    height: int = 1000,
    is_raw: Optional[bool] = None,
) -> Optional[list[int]]:
    """Guarantees bounding box coordinates are in 0-1000 integer range."""
    if not box or len(box) < 4:
        return None
    try:
        if isinstance(box[0], (list, tuple)):
            xs = [p[0] for p in box]
            ys = [p[1] for p in box]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        else:
            x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])

        # Normalized 0..1 float
        if max(x1, y1, x2, y2) <= 1.0:
            return [
                int(round(x1 * 1000)),
                int(round(y1 * 1000)),
                int(round(x2 * 1000)),
                int(round(y2 * 1000)),
            ]

        # Explicitly marked raw pixels or coordinates exceeding 1000
        w = max(1.0, float(width))
        h = max(1.0, float(height))
        if is_raw is True or (is_raw is None and (x2 > 1000.0 or y2 > 1000.0)):
            nx1 = max(0, min(1000, int(round(1000.0 * x1 / w))))
            ny1 = max(0, min(1000, int(round(1000.0 * y1 / h))))
            nx2 = max(0, min(1000, int(round(1000.0 * x2 / w))))
            ny2 = max(0, min(1000, int(round(1000.0 * y2 / h))))
            if nx2 <= nx1:
                nx2 = min(1000, nx1 + 2)
            if ny2 <= ny1:
                ny2 = min(1000, ny1 + 2)
            return [nx1, ny1, nx2, ny2]

        # Already normalized 0..1000
        nx1 = max(0, min(1000, int(round(x1))))
        ny1 = max(0, min(1000, int(round(y1))))
        nx2 = max(0, min(1000, int(round(x2))))
        ny2 = max(0, min(1000, int(round(y2))))
        if nx2 <= nx1:
            nx2 = min(1000, nx1 + 2)
        if ny2 <= ny1:
            ny2 = min(1000, ny1 + 2)
        return [nx1, ny1, nx2, ny2]
    except Exception as ex:
        logger.debug(f"normalize_to_1000 error on {box}: {ex}")
        return None


def build_debug_overlay_payload(
    job_id: str,
    page_idx: int,
    doc_profile: Optional[DocumentProfile],
    invoice_dict: Optional[dict] = None,
    validation_report: Optional[Any] = None,
    page_dimensions: Optional[dict] = None,
) -> dict:
    """
    Assembles a complete, self-contained debug overlay payload for a single page.
    """
    page_dimensions = page_dimensions or {}
    p_num = page_idx + 1

    pw, ph = 1200, 1600
    if page_dimensions and p_num in page_dimensions:
        pw, ph = page_dimensions[p_num]
    elif doc_profile and doc_profile.width and doc_profile.height:
        pw, ph = doc_profile.width, doc_profile.height

    total_pages = doc_profile.page_count if doc_profile else 1

    # 1. YOLO visual regions
    yolo_regions = []
    if doc_profile and doc_profile.regions:
        for r in doc_profile.regions:
            if getattr(r, "page", 1) == p_num or total_pages == 1:
                yolo_regions.append({
                    "label": r.label,
                    "bbox_norm": list(r.bbox_norm),
                    "bbox_raw": list(r.bbox_raw) if getattr(r, "bbox_raw", None) else None,
                    "confidence": round(float(r.confidence), 3),
                    "page": getattr(r, "page", 1),
                    "is_handwritten": bool(getattr(r, "is_handwritten", False) or r.label == "handwriting"),
                })

    # 2. OCR text word / line bounding boxes
    ocr_boxes = []
    if doc_profile and doc_profile.words:
        for w in doc_profile.words:
            if getattr(w, "page", 1) == p_num or total_pages == 1:
                ocr_boxes.append({
                    "text": w.text,
                    "bbox_norm": list(w.bbox_norm),
                    "bbox_raw": list(w.bbox_raw) if getattr(w, "bbox_raw", None) else None,
                    "confidence": round(float(w.confidence), 3),
                    "source": getattr(w, "source", "ocr"),
                    "page": getattr(w, "page", 1),
                    "block_no": getattr(w, "block_no", 0),
                    "line_no": getattr(w, "line_no", 0),
                })

    # 3. TIE anchor occurrences
    tie_anchors = []
    if doc_profile and doc_profile.anchor_occurrences:
        for anc_name, occ_list in doc_profile.anchor_occurrences.items():
            for occ in occ_list:
                if occ.get("page", 1) == p_num or total_pages == 1:
                    tie_anchors.append({
                        "anchor": anc_name,
                        "bbox_norm": list(occ.get("bbox_norm", [0, 0, 0, 0])),
                        "center_norm": list(occ.get("center_norm", [0, 0])),
                        "page": occ.get("page", 1),
                    })

    # 4. Handwriting patches
    handwriting_regions = [
        r for r in yolo_regions
        if r.get("is_handwritten") or r.get("label") in ("handwriting", "signature", "stamp")
    ]

    # 5. Field Evidence Layer
    field_evidence = {}
    inv = invoice_dict or {}
    prov_map = inv.get("field_provenance") or {}
    conf_map = inv.get("field_confidences") or {}
    review_reasons = inv.get("review_reasons") or []

    for fname in [
        "invoice_number", "invoice_date", "due_date", "po_number", "place_of_supply",
        "vendor_name", "vendor_gstin", "vendor_pan", "vendor_email", "vendor_phone",
        "buyer_name", "buyer_gstin", "buyer_phone",
        "subtotal", "tax_amount", "grand_total", "cgst", "sgst", "igst",
        "bank_name", "branch_name", "account_name", "account_number", "ifsc_code"
    ]:
        f_val = inv.get(fname)
        if f_val is not None and str(f_val).strip() and str(f_val).strip() != "None":
            prov = prov_map.get(fname, {})
            f_page = prov.get("page", 1)

            # Check if this field should be shown on this page
            if f_page == p_num or total_pages == 1 or total_pages == 0:
                raw_bbox = prov.get("bbox")
                bbox_norm = normalize_to_1000(raw_bbox, pw, ph) if raw_bbox else None

                # Fallback: if no bbox recorded yet, locate first matching OCR word
                if not bbox_norm and doc_profile and doc_profile.words:
                    f_val_clean = str(f_val).strip().lower().replace(",", "")
                    for w in doc_profile.words:
                        if (getattr(w, "page", 1) == p_num or total_pages == 1) and f_val_clean in w.text.lower().replace(",", ""):
                            bbox_norm = list(w.bbox_norm)
                            break

                is_flagged = any(fname in str(r).lower() for r in review_reasons)
                val_status = "flagged" if is_flagged else "passed"

                cand_list = prov.get("candidates") or []
                if not cand_list:
                    # Provide default candidate if none stored
                    cand_list = [{
                        "value": str(f_val),
                        "source": prov.get("source", "ocr"),
                        "confidence": conf_map.get(fname, prov.get("confidence", 0.90)),
                        "bbox_norm": bbox_norm,
                        "page": f_page,
                    }]
                else:
                    # Normalize bboxes in candidates
                    for c in cand_list:
                        if "bbox_norm" not in c and "bbox" in c:
                            c["bbox_norm"] = normalize_to_1000(c["bbox"], pw, ph)

                field_evidence[fname] = {
                    "field_name": fname,
                    "selected_value": str(f_val),
                    "selected_source": prov.get("source", "heuristic"),
                    "confidence": round(float(conf_map.get(fname, prov.get("confidence", 0.90))), 3),
                    "selection_reason": prov.get("selection_reason") or f"Selected via {prov.get('source', 'model')}",
                    "bbox_norm": bbox_norm,
                    "page": f_page,
                    "validation_status": val_status,
                    "candidates": cand_list,
                    "disagreement_score": round(float(prov.get("disagreement_score", 0.0)), 3),
                }

    disagreements = []
    for fn, ev in field_evidence.items():
        if ev.get("disagreement_score", 0.0) > 0.05 or len(ev.get("candidates", [])) > 1:
            vals = {c.get("value") for c in ev.get("candidates", []) if c.get("value")}
            if len(vals) > 1:
                disagreements.append({
                    "field": fn,
                    "score": ev.get("disagreement_score", 0.0),
                    "reason": f"Disagreement among candidates: {', '.join(str(v) for v in vals)}",
                    "candidates": ev.get("candidates", []),
                })

    payload = {
        "job_id": job_id,
        "page": page_idx,
        "page_number": p_num,
        "total_pages": total_pages,
        "page_width": pw,
        "page_height": ph,
        "is_digital_native": getattr(doc_profile, "is_digital_native", False) if doc_profile else False,
        "quality_score": getattr(doc_profile, "quality_score", 1.0) if doc_profile else 1.0,
        "template_id": inv.get("template_id"),
        "template_family_id": inv.get("template_family_id"),
        "yolo_regions": yolo_regions,
        "ocr_boxes": ocr_boxes,
        "tie_anchors": tie_anchors,
        "handwriting_regions": handwriting_regions,
        "field_evidence": field_evidence,
        "disagreements": disagreements,
        "summary": {
            "yolo_regions_count": len(yolo_regions),
            "ocr_words_count": len(ocr_boxes),
            "tie_anchors_count": len(tie_anchors),
            "handwriting_count": len(handwriting_regions),
            "fields_located_count": len([f for f in field_evidence.values() if f.get("bbox_norm")]),
            "disagreements_count": len(disagreements),
        }
    }

    return payload


def cache_debug_overlay(job_id: str, page_idx: int, payload: dict):
    """Store debug overlay payload in memory cache and persist to disk."""
    if job_id not in _job_debug_cache:
        _job_debug_cache[job_id] = {}
    _job_debug_cache[job_id][page_idx] = payload

    try:
        debug_dir = Path("data/debug")
        debug_dir.mkdir(parents=True, exist_ok=True)
        job_file = debug_dir / f"{job_id}.json"
        
        all_pages = {}
        if job_file.exists():
            try:
                with open(job_file, "r", encoding="utf-8") as f:
                    all_pages = json.load(f)
            except Exception:
                all_pages = {}
        
        all_pages[str(page_idx)] = payload
        with open(job_file, "w", encoding="utf-8") as f:
            json.dump(all_pages, f, indent=2)
    except Exception as ex:
        logger.debug(f"Failed to persist debug overlay to disk: {ex}")


@debug_router.get("/{job_id}/debug-overlay", tags=["Debug"])
async def get_invoice_debug_overlay(job_id: str, page: int = 0, request: Request = None):
    """
    Returns visual bounding boxes and candidate evidentiary trail for Invoice Debug Mode.
    - page: 0-indexed page number (0 = Page 1)
    """
    # 1. Check in-memory cache
    if job_id in _job_debug_cache and page in _job_debug_cache[job_id]:
        return _job_debug_cache[job_id][page]

    # 2. Check disk cache
    debug_file = Path(f"data/debug/{job_id}.json")
    if debug_file.exists():
        try:
            with open(debug_file, "r", encoding="utf-8") as f:
                disk_data = json.load(f)
                if str(page) in disk_data:
                    if job_id not in _job_debug_cache:
                        _job_debug_cache[job_id] = {}
                    _job_debug_cache[job_id][page] = disk_data[str(page)]
                    return disk_data[str(page)]
        except Exception as e:
            logger.debug(f"Error reading debug disk cache: {e}")

    # 3. Dynamic On-Demand Reconstruction
    db = getattr(request.app.state, "db", None) if request else None
    pipeline = getattr(request.app.state, "pipeline", None) if request else None
    
    if not db:
        raise HTTPException(status_code=503, detail="Database not initialized")

    record = await db.get_job(job_id)
    if not record:
        raise HTTPException(status_code=404, detail="Invoice job not found")

    invoice_dict = record.output_json or record.ai_output_json or {}

    # Check if DocumentProfile is cached in memory
    from api.main import _job_profiles, _find_invoice_file, _get_raw_file_bytes
    doc_profile = _job_profiles.get(job_id)
    page_dimensions = {}

    if not doc_profile and pipeline:
        try:
            minio = getattr(request.app.state, "minio", None) if request else None
            file_bytes = _get_raw_file_bytes(job_id, record.filename, record.storage_key, minio)
            if file_bytes:
                suffix = Path(record.filename).suffix.lower()
                if suffix == ".pdf":
                    pages = pipeline.pdf_converter.convert_bytes(file_bytes)
                else:
                    pages = [pipeline.preprocessor.process(file_bytes)]

                all_regions_flat = []
                combined_ocr = {}
                for p_idx, p_obj in enumerate(pages):
                    p_num = p_idx + 1
                    p_w, p_h = 1200, 1600
                    if hasattr(p_obj.image, "shape"):
                        p_h, p_w = p_obj.image.shape[:2]
                    elif hasattr(p_obj.image, "size"):
                        p_w, p_h = p_obj.image.size
                    page_dimensions[p_num] = (p_w, p_h)

                    det_res = pipeline.detector.detect(p_obj.image)
                    for r in det_res.regions:
                        r.page = p_num
                    all_regions_flat.extend(det_res.regions)

                    from preprocessing.pdf_converter import NativePDFPage
                    if isinstance(p_obj, NativePDFPage):
                        from api.pipeline_runner import _native_page_to_ocr_results
                        p_ocr = _native_page_to_ocr_results(p_obj, det_res.regions)
                    else:
                        full_res = pipeline.ocr.extract_full_page(p_obj.image)
                        p_ocr = {f"full_page_p{p_num}": full_res}
                        if det_res.regions:
                            p_ocr.update(pipeline._assign_tokens_to_regions(full_res, det_res.regions))

                    for k, v in p_ocr.items():
                        combined_ocr[f"{k}_p{p_num}"] = v

                pw, ph = page_dimensions.get(1, (1200, 1600))
                doc_profile = DocumentProfile.from_ocr_and_regions(
                    ocr_results=combined_ocr,
                    regions=all_regions_flat,
                    width=pw,
                    height=ph,
                    page_count=len(pages),
                    page_dimensions=page_dimensions,
                )
                _job_profiles[job_id] = doc_profile
        except Exception as ex:
            logger.warning(f"On-demand debug profile reconstruction: {ex}")

    payload = build_debug_overlay_payload(
        job_id=job_id,
        page_idx=page,
        doc_profile=doc_profile,
        invoice_dict=invoice_dict,
        page_dimensions=page_dimensions,
    )

    cache_debug_overlay(job_id, page, payload)
    return payload
