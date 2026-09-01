"""
understanding/template_retriever.py

Multi-Stage Indexed Template Retrieval Engine.

Solves the 100k template scale problem using indexed pruning & real mathematical similarity:
1. Instant Exact Hash ($O(1)$):
   Exact structural fingerprint match via in-memory hash index (< 1 ms).
2. Stage 1 (Inverted Index Candidate Narrowing, 100k → ~50):
   Candidate retrieval via GSTIN index, aspect bucket index, and inverted anchor index.
3. Stage 2 (Mathematical Similarity & Spatial Alignment, ~50 → ~5):
   - Anchor Set Jaccard similarity: |A ∩ B| / |A ∪ B|
   - Spatial Centroid alignment across shared anchors in normalized coordinate space
   - Visual region topology IoU / layout alignment
4. Stage 3 (Calibrated Decision Gating):
   - score >= 0.90: exact_version (deterministic fast-path rules)
   - 0.75 <= score < 0.90: family_anchor (adaptive anchor extraction + validation)
   - score < 0.75: none (AI understanding fallback)
"""

from __future__ import annotations
import time
import math
import hashlib
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, Any, Sequence, Set
from loguru import logger

from preprocessing.document_profile import DocumentProfile


@dataclass
class TemplateMatchResult:
    matched_version_id: Optional[str]
    matched_family_id: Optional[str]
    match_type: str                  # "exact_version" | "family_anchor" | "none"
    match_confidence: float          # 0.0 to 1.0
    field_rules: list[dict[str, Any]] = field(default_factory=list)
    retrieval_stage: str = "none"    # "exact_hash" | "indexed_stage2" | "indexed_stage3"
    latency_ms: float = 0.0


@dataclass
class CachedTemplateVersion:
    version_id: str
    family_id: str
    version_num: int
    exact_fingerprint: str
    family_fingerprint: str
    aspect_bucket: int
    aspect_ratio: float
    page_count: int
    vendor_gstin: Optional[str] = None
    vendor_name: Optional[str] = None
    anchor_set: set[str] = field(default_factory=set)
    anchor_positions: dict[str, tuple[float, float]] = field(default_factory=dict)
    region_topology: list[dict[str, Any]] = field(default_factory=list)
    field_rules: list[dict[str, Any]] = field(default_factory=list)
    sample_count: int = 1
    success_rate: float = 1.0
    version_fingerprint: str = ""
    anchor_signature: str = ""
    layout_signature: str = ""

    def __post_init__(self):
        if not self.version_fingerprint:
            self.version_fingerprint = self.exact_fingerprint
        if not self.aspect_bucket:
            self.aspect_bucket = int(round(float(self.aspect_ratio) * 10))


class TemplateRetriever:
    """
    High-speed, indexed template retrieval service using real mathematical similarity metrics.
    """

    def __init__(self, db_manager: Optional[Any] = None):
        self.db = db_manager
        self._in_memory_index: list[CachedTemplateVersion] = []
        self._versions_by_id: dict[str, CachedTemplateVersion] = {}
        self._exact_index: dict[str, CachedTemplateVersion] = {}
        self._gstin_index: dict[str, list[CachedTemplateVersion]] = defaultdict(list)
        self._aspect_index: dict[int, list[CachedTemplateVersion]] = defaultdict(list)
        self._anchor_inverted_index: dict[str, set[str]] = defaultdict(set)
        self._last_loaded_time: float = 0.0

    def register_in_memory_template(self, tpl: CachedTemplateVersion):
        """Registers a template directly in memory and updates all fast indexes."""
        self._index_template(tpl)

    def _index_template(self, tpl: CachedTemplateVersion):
        """Adds a template version to the in-memory inverted indexes."""
        self._in_memory_index.append(tpl)
        self._versions_by_id[tpl.version_id] = tpl

        if tpl.exact_fingerprint:
            self._exact_index[tpl.exact_fingerprint] = tpl
        if tpl.version_fingerprint:
            self._exact_index[tpl.version_fingerprint] = tpl
        if tpl.layout_signature:
            self._exact_index[tpl.layout_signature] = tpl

        if tpl.vendor_gstin:
            self._gstin_index[tpl.vendor_gstin.upper()].append(tpl)

        self._aspect_index[tpl.aspect_bucket].append(tpl)

        # Inverted index for anchor keywords
        for anc in tpl.anchor_set:
            self._anchor_inverted_index[anc].add(tpl.version_id)

    async def load_templates_from_db(self):
        """Loads all active template versions and builds fast multi-level retrieval indexes."""
        if not self.db:
            return

        try:
            versions = await self.db.get_all_active_template_versions()
            
            # Reset indexes
            self._in_memory_index = []
            self._versions_by_id = {}
            self._exact_index = {}
            self._gstin_index = defaultdict(list)
            self._aspect_index = defaultdict(list)
            self._anchor_inverted_index = defaultdict(set)

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

                # Get family info
                fam = await self.db.get_template_family(ver.family_id)
                v_gstin = fam.vendor_gstin if fam else None
                v_name = fam.vendor_name if fam else None

                # Extract topology features
                topo = ver.topology_spec or {}
                raw_anchor_set = set(topo.get("anchor_set", []))
                if not raw_anchor_set and rule_dicts:
                    for rd in rule_dicts:
                        for a in rd.get("anchors", []):
                            clean_a = a.replace(" ", "_").replace(":", "").strip("_").lower()
                            if clean_a:
                                raw_anchor_set.add(clean_a)

                anchor_positions = topo.get("anchor_positions", {})
                region_topology = topo.get("region_topology", [])

                cached = CachedTemplateVersion(
                    version_id=ver.id,
                    family_id=ver.family_id,
                    version_num=ver.version_num,
                    exact_fingerprint=ver.version_fingerprint or "",
                    family_fingerprint=fam.family_fingerprint if fam else "",
                    aspect_bucket=int(round(float(ver.aspect_ratio or 1.41) * 10)),
                    aspect_ratio=ver.aspect_ratio or 1.41,
                    page_count=ver.page_count,
                    vendor_gstin=v_gstin,
                    vendor_name=v_name,
                    anchor_set=raw_anchor_set,
                    anchor_positions=anchor_positions,
                    region_topology=region_topology,
                    field_rules=rule_dicts,
                    sample_count=ver.sample_count,
                    success_rate=ver.success_rate,
                    version_fingerprint=ver.version_fingerprint or "",
                    anchor_signature=ver.anchor_signature or "",
                    layout_signature=ver.layout_signature or "",
                )
                self._index_template(cached)

            self._last_loaded_time = time.time()
            logger.info(f"Loaded {len(self._in_memory_index)} templates into indexed retrieval engine")
        except Exception as e:
            logger.debug(f"Could not load templates from DB: {e}")

    # ── Mathematical Similarity Metrics ───────────────────────────────────────

    @staticmethod
    def compute_anchor_jaccard(doc_anchors: set[str], tpl_anchors: set[str]) -> float:
        """Calculates exact Jaccard similarity between document and template anchor sets."""
        if not doc_anchors and not tpl_anchors:
            return 1.0
        if not doc_anchors or not tpl_anchors:
            return 0.0
        intersection = len(doc_anchors & tpl_anchors)
        union = len(doc_anchors | tpl_anchors)
        return intersection / float(union) if union > 0 else 0.0

    @staticmethod
    def compute_spatial_alignment(
        doc_pos: dict[str, tuple[float, float]],
        tpl_pos: dict[str, tuple[float, float]],
        shared_anchors: set[str],
    ) -> float:
        """
        Calculates normalized spatial Euclidean alignment across shared anchors.
        Returns a score in [0.0, 1.0].
        """
        if not shared_anchors:
            return 0.50

        scores = []
        for anc in shared_anchors:
            if anc in doc_pos and anc in tpl_pos:
                p1 = doc_pos[anc]
                p2 = tpl_pos[anc]
                dx = p1[0] - p2[0]
                dy = p1[1] - p2[1]
                dist = math.sqrt(dx * dx + dy * dy)
                # Maximum acceptable anchor displacement = 300 normalized px
                score = max(0.0, 1.0 - (dist / 300.0))
                scores.append(score)

        return (sum(scores) / len(scores)) if scores else 0.50

    @staticmethod
    def compute_region_topology_similarity(
        doc_regions: list[dict[str, Any]],
        tpl_regions: list[dict[str, Any]],
    ) -> float:
        """Compares visual region structure (headers, tables, totals)."""
        if not doc_regions and not tpl_regions:
            return 1.0
        if not doc_regions or not tpl_regions:
            return 0.50

        doc_labels = set(r.get("label", "") for r in doc_regions if r.get("label"))
        tpl_labels = set(r.get("label", "") for r in tpl_regions if r.get("label"))
        if not doc_labels and not tpl_labels:
            return 1.0
        if not doc_labels or not tpl_labels:
            return 0.50

        jaccard = len(doc_labels & tpl_labels) / float(len(doc_labels | tpl_labels))
        return jaccard

    def retrieve(self, profile: DocumentProfile) -> TemplateMatchResult:
        """
        Executes indexed candidate retrieval and mathematical similarity evaluation.
        """
        start_t = time.perf_counter()

        # ── Fast-Path 1: Instant Exact Hash Match (O(1)) ─────────────────────
        if profile.exact_fingerprint and profile.exact_fingerprint in self._exact_index:
            tpl = self._exact_index[profile.exact_fingerprint]
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

        if profile.layout_signature and profile.layout_signature in self._exact_index:
            tpl = self._exact_index[profile.layout_signature]
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

        # ── Stage 1: Indexed Candidate Pruning (100k -> ~50) ─────────────────
        candidate_ids: set[str] = set()

        # 1. Vendor GSTIN exact lookup
        if profile.vendor_gstin and profile.vendor_gstin.upper() in self._gstin_index:
            for t in self._gstin_index[profile.vendor_gstin.upper()]:
                candidate_ids.add(t.version_id)

        # 2. Inverted anchor index lookup
        anchor_hits_per_version: dict[str, int] = defaultdict(int)
        for anc in profile.anchor_set:
            if anc in self._anchor_inverted_index:
                for vid in self._anchor_inverted_index[anc]:
                    anchor_hits_per_version[vid] += 1

        # Select top candidates by anchor overlap
        sorted_anchor_candidates = sorted(
            anchor_hits_per_version.items(), key=lambda x: x[1], reverse=True
        )
        for vid, _ in sorted_anchor_candidates[:40]:
            candidate_ids.add(vid)

        # 3. Geometry bucket candidates (if candidate pool is small)
        if len(candidate_ids) < 10:
            for bucket in (profile.aspect_bucket - 1, profile.aspect_bucket, profile.aspect_bucket + 1):
                if bucket in self._aspect_index:
                    for t in self._aspect_index[bucket][:10]:
                        candidate_ids.add(t.version_id)

        # Retrieve Candidate Objects
        candidates: list[CachedTemplateVersion] = [
            self._versions_by_id[vid] for vid in candidate_ids if vid in self._versions_by_id
        ]
        if not candidates and self._in_memory_index:
            candidates = self._in_memory_index[:20]

        # ── Stage 2: Mathematical Similarity & Multi-Factor Scoring ──────────
        scored_candidates: list[tuple[float, CachedTemplateVersion]] = []

        for tpl in candidates:
            # 1. Page count & geometry compatibility penalty
            if abs(tpl.page_count - profile.page_count) > 1 and tpl.page_count != 1 and profile.page_count != 1:
                continue

            # 2. Anchor Jaccard Similarity
            jaccard = self.compute_anchor_jaccard(profile.anchor_set, tpl.anchor_set)

            # 3. Spatial Alignment
            shared_anchors = profile.anchor_set & tpl.anchor_set
            spatial = self.compute_spatial_alignment(profile.anchor_positions, tpl.anchor_positions, shared_anchors)

            # 4. Region Topology Similarity
            reg_sim = self.compute_region_topology_similarity(profile.region_topology, tpl.region_topology)

            # 5. Vendor GSTIN Match
            gstin_match = 1.0 if (profile.vendor_gstin and tpl.vendor_gstin and profile.vendor_gstin.upper() == tpl.vendor_gstin.upper()) else 0.0

            # Composite Calibrated Score
            composite_score = (
                0.35 * jaccard +
                0.25 * spatial +
                0.20 * reg_sim +
                0.20 * gstin_match
            )

            # Empirical success rate calibration
            empirical_weight = 0.90 + 0.10 * (tpl.success_rate if tpl.sample_count >= 3 else 1.0)
            final_score = round(composite_score * empirical_weight, 3)

            scored_candidates.append((final_score, tpl))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)

        dur_ms = (time.perf_counter() - start_t) * 1000.0

        if scored_candidates:
            best_score, best_tpl = scored_candidates[0]

            # ── Stage 3: Calibrated Decision Gating ───────────────────────────
            if best_score >= 0.90 or (best_score >= 0.85 and best_tpl.vendor_gstin and profile.vendor_gstin == best_tpl.vendor_gstin):
                return TemplateMatchResult(
                    matched_version_id=best_tpl.version_id,
                    matched_family_id=best_tpl.family_id,
                    match_type="exact_version",
                    match_confidence=min(0.99, best_score),
                    field_rules=best_tpl.field_rules,
                    retrieval_stage="indexed_stage3_exact",
                    latency_ms=dur_ms,
                )
            elif best_score >= 0.75:
                return TemplateMatchResult(
                    matched_version_id=best_tpl.version_id,
                    matched_family_id=best_tpl.family_id,
                    match_type="family_anchor",
                    match_confidence=best_score,
                    field_rules=best_tpl.field_rules,
                    retrieval_stage="indexed_stage3_family",
                    latency_ms=dur_ms,
                )

        return TemplateMatchResult(
            matched_version_id=None,
            matched_family_id=None,
            match_type="none",
            match_confidence=scored_candidates[0][0] if scored_candidates else 0.0,
            retrieval_stage="indexed_stage3_none",
            latency_ms=dur_ms,
        )


