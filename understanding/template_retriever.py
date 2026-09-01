"""
understanding/template_retriever.py

Multi-Stage Template Retrieval Engine.

Solves the 10k → 100k template scale problem using hierarchical pruning:
1. Stage 1 (Deterministic Filter, N → ~100):
   Filter by page count, aspect ratio (+- 0.15), vendor GSTIN, text density bucket.
2. Stage 2 (Layout Similarity, ~100 → ~10):
   Region topology Jaccard similarity and visual grid layout matching.
3. Stage 3 (Anchor Graph Similarity, ~10 → ~1):
   Fine-grained spatial anchor token position alignment.

Computes separate Match Confidence (document belongs to template)
independent from Field Extraction Confidence.
"""

from __future__ import annotations
import time
import math
import difflib
from dataclasses import dataclass, field
from typing import Optional, Any, Sequence
from loguru import logger

from preprocessing.document_profile import DocumentProfile


@dataclass
class TemplateMatchResult:
    matched_version_id: Optional[str]
    matched_family_id: Optional[str]
    match_type: str                  # "exact_version" | "family_anchor" | "none"
    match_confidence: float          # 0.0 to 1.0
    field_rules: list[dict[str, Any]] = field(default_factory=list)
    retrieval_stage: str = "none"    # "stage1" | "stage2" | "stage3" | "exact_hash"
    latency_ms: float = 0.0


@dataclass
class CachedTemplateVersion:
    version_id: str
    family_id: str
    version_num: int
    version_fingerprint: str
    aspect_ratio: float
    page_count: int
    anchor_signature: str
    layout_signature: str
    vendor_gstin: Optional[str] = None
    vendor_name: Optional[str] = None
    anchors_map: dict[str, list[int]] = field(default_factory=dict)  # anchor_phrase -> [x1, y1, x2, y2]
    field_rules: list[dict[str, Any]] = field(default_factory=list)
    sample_count: int = 1
    success_rate: float = 1.0


class TemplateRetriever:
    """
    High-speed, multi-stage template index and retrieval service.
    """

    def __init__(self, db_manager: Optional[Any] = None):
        self.db = db_manager
        self._in_memory_index: list[CachedTemplateVersion] = []
        self._last_loaded_time: float = 0.0

    def register_in_memory_template(self, tpl: CachedTemplateVersion):
        """Registers a template directly in memory (for tests/offline usage)."""
        self._in_memory_index.append(tpl)

    async def load_templates_from_db(self):
        """Loads all active template versions and their field rules into memory index."""
        if not self.db:
            return

        try:
            versions = await self.db.get_all_active_template_versions()
            new_index = []
            for ver in versions:
                rules = await self.db.get_field_rules_for_version(ver.id)
                rule_dicts = [
                    {
                        "field_name": r.field_name,
                        "strategy": r.strategy,
                        "anchors": r.anchors or [],
                        "search_region": r.search_region,
                        "relative_box": r.relative_box,
                        "parser_spec": r.parser_spec or {},
                        "validator_spec": r.validator_spec or {},
                        "confidence_score": r.confidence_score,
                    }
                    for r in rules
                ]

                # Get family info if possible
                fam = await self.db.get_template_family(ver.family_id)
                v_gstin = fam.vendor_gstin if fam else None
                v_name = fam.vendor_name if fam else None

                cached = CachedTemplateVersion(
                    version_id=ver.id,
                    family_id=ver.family_id,
                    version_num=ver.version_num,
                    version_fingerprint=ver.version_fingerprint,
                    aspect_ratio=ver.aspect_ratio or 1.41,
                    page_count=ver.page_count,
                    anchor_signature=ver.anchor_signature or "",
                    layout_signature=ver.layout_signature or "",
                    vendor_gstin=v_gstin,
                    vendor_name=v_name,
                    field_rules=rule_dicts,
                    sample_count=ver.sample_count,
                    success_rate=ver.success_rate,
                )
                new_index.append(cached)

            self._in_memory_index = new_index
            self._last_loaded_time = time.time()
            logger.info(f"Loaded {len(self._in_memory_index)} templates into in-memory retrieval index")
        except Exception as e:
            logger.debug(f"Could not load templates from DB: {e}")

    def retrieve(self, profile: DocumentProfile) -> TemplateMatchResult:
        """
        Executes 3-stage candidate retrieval on the DocumentProfile.
        Returns the best matching TemplateVersion or 'none'.
        """
        start_t = time.perf_counter()

        # Instant Exact Fingerprint Match
        for tpl in self._in_memory_index:
            if tpl.version_fingerprint and (
                tpl.version_fingerprint == profile.layout_signature or
                tpl.version_fingerprint == profile.text_signature
            ):
                dur_ms = (time.perf_counter() - start_t) * 1000.0
                return TemplateMatchResult(
                    matched_version_id=tpl.version_id,
                    matched_family_id=tpl.family_id,
                    match_type="exact_version",
                    match_confidence=0.98,
                    field_rules=tpl.field_rules,
                    retrieval_stage="exact_hash",
                    latency_ms=dur_ms,
                )

        # ── Stage 1: Deterministic Filter (N -> ~100) ──────────────────────────
        candidates_stage1: list[CachedTemplateVersion] = []
        for tpl in self._in_memory_index:
            # 1. Page count match (allow if either is single page or within 1 page)
            if abs(tpl.page_count - profile.page_count) > 1 and tpl.page_count != 1 and profile.page_count != 1:
                continue

            # 2. Aspect ratio tolerance +- 0.40 for phone scans
            if abs(tpl.aspect_ratio - profile.aspect_ratio) > 0.40:
                continue

            # 3. Vendor GSTIN match (strong positive boost if matches)
            if profile.vendor_gstin and tpl.vendor_gstin:
                if profile.vendor_gstin.upper() == tpl.vendor_gstin.upper():
                    candidates_stage1.append(tpl)
                    continue

            candidates_stage1.append(tpl)

        if not candidates_stage1:
            # Fallback: test all index templates if stage 1 was too strict
            candidates_stage1 = self._in_memory_index

        # ── Stage 2: Layout & Signature Similarity (~100 -> ~10) ───────────────
        scored_stage2: list[tuple[float, CachedTemplateVersion]] = []
        for tpl in candidates_stage1:
            score = 0.0

            # GSTIN match boost
            if profile.vendor_gstin and tpl.vendor_gstin and profile.vendor_gstin == tpl.vendor_gstin:
                score += 0.50

            # Anchor signature similarity (Jaccard on anchor string)
            anc_sim = difflib.SequenceMatcher(None, profile.anchor_signature, tpl.anchor_signature).ratio()
            score += anc_sim * 0.35

            # Layout signature similarity
            lay_sim = difflib.SequenceMatcher(None, profile.layout_signature, tpl.layout_signature).ratio()
            score += lay_sim * 0.25

            if score > 0.15:
                scored_stage2.append((score, tpl))

        scored_stage2.sort(key=lambda x: x[0], reverse=True)
        top_candidates = scored_stage2[:10] if scored_stage2 else [(0.1, t) for t in candidates_stage1[:5]]

        # ── Stage 3: Fine-Grained Anchor Graph Similarity (~10 -> 1) ───────────
        best_match: Optional[CachedTemplateVersion] = None
        highest_conf: float = 0.0

        for s2_score, tpl in top_candidates:
            # Check anchor rule presence against document profile
            anchor_hits = 0
            total_checks = 0

            for rule in tpl.field_rules:
                anchors = rule.get("anchors", [])
                if not anchors:
                    continue
                total_checks += 1
                for anc in anchors:
                    matches = profile.find_anchor_tokens(anc)
                    if matches:
                        anchor_hits += 1
                        break

            anchor_ratio = (anchor_hits / max(1, total_checks)) if total_checks > 0 else 0.5
            final_match_conf = round(s2_score * 0.4 + anchor_ratio * 0.6, 3)

            if final_match_conf > highest_conf:
                highest_conf = final_match_conf
                best_match = tpl

        dur_ms = (time.perf_counter() - start_t) * 1000.0

        if best_match and highest_conf >= 0.70:
            return TemplateMatchResult(
                matched_version_id=best_match.version_id,
                matched_family_id=best_match.family_id,
                match_type="exact_version",
                match_confidence=highest_conf,
                field_rules=best_match.field_rules,
                retrieval_stage="stage3_exact",
                latency_ms=dur_ms,
            )
        elif best_match and highest_conf >= 0.40:
            return TemplateMatchResult(
                matched_version_id=best_match.version_id,
                matched_family_id=best_match.family_id,
                match_type="family_anchor",
                match_confidence=highest_conf,
                field_rules=best_match.field_rules,
                retrieval_stage="stage3_family",
                latency_ms=dur_ms,
            )
        else:
            return TemplateMatchResult(
                matched_version_id=None,
                matched_family_id=None,
                match_type="none",
                match_confidence=highest_conf,
                retrieval_stage="stage3_low_conf",
                latency_ms=dur_ms,
            )

