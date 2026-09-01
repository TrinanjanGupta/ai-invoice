"""
understanding/template_learner.py

Automated Template Learning & Rule Synthesis Engine.

Transforms human-verified invoices (Ground Truth) into deterministic TemplateVersions
and field extraction rules in PostgreSQL, closing the continuous learning loop.

When a reviewer approves or corrects an invoice:
1. Aligns verified field values with DocumentProfile word tokens.
2. Identifies optimal spatial anchors and calculates precise relative bounding offsets.
3. Creates or updates TemplateFamily, TemplateVersion, and TemplateFieldRules in PostgreSQL.
4. Registers the new template into TemplateRetriever for instant <50ms fast-path extraction.
"""

from __future__ import annotations
import re
import difflib
from dataclasses import dataclass
from typing import Optional, Any
from loguru import logger

from preprocessing.document_profile import DocumentProfile, WordToken, CANONICAL_ANCHORS
from understanding.template_retriever import TemplateRetriever, CachedTemplateVersion


FIELD_ANCHOR_CANDIDATES = {
    "invoice_number": ["invoice no", "inv no", "bill no", "invoice #", "voucher no", "bill number", "sl no"],
    "invoice_date": ["invoice date", "dated", "bill date", "date of issue", "date:"],
    "due_date": ["due date", "payment due", "pay by"],
    "po_number": ["po no", "order no", "purchase order"],
    "place_of_supply": ["place of supply", "state/ut code", "supply state"],
    "buyer_name": ["bill to", "billed to", "consignee", "buyer", "customer"],
    "vendor_name": ["tax invoice", "invoice", "vendor"],
    "vendor_gstin": ["gstin", "gst no", "vendor gstin"],
    "buyer_gstin": ["buyer gstin", "gstin"],
    "subtotal": ["sub total", "subtotal", "taxable value", "taxable amount", "total before tax"],
    "cgst": ["+cgst :", "+cgst", "cgst amt", "cgst amount", "central tax", "cgst"],
    "sgst": ["+sgst :", "+sgst", "sgst amt", "sgst amount", "state tax", "sgst", "utgst"],
    "igst": ["+igst :", "+igst", "igst amt", "igst amount", "integrated tax", "igst"],
    "tax_amount": ["gst total", "total tax", "tax amount", "gst amount", "tax total"],
    "discount": ["discount", "less discount", "trade discount", "global discount"],
    "round_off": ["round off", "roundoff", "rounding"],
    "grand_total": ["grand total", "total payable", "net amount", "total amount", "total:", "net payable"],
    "account_number": ["account no", "ac no", "a/c no", "bank a/c"],
    "ifsc_code": ["ifsc", "ifsc code"],
}


class TemplateLearner:
    """
    Synthesizes deterministic template extraction rules from ground truth reviews.
    """

    def __init__(self, db_manager: Optional[Any] = None, retriever: Optional[TemplateRetriever] = None):
        self.db = db_manager
        self.retriever = retriever

    async def learn_from_verified_invoice(
        self,
        profile: DocumentProfile,
        verified_data: dict[str, Any],
        vendor_name: Optional[str] = None,
        vendor_gstin: Optional[str] = None,
    ) -> Optional[str]:
        """
        Learns and persists template rules from a human-verified invoice.
        Returns the created/updated template_version_id.
        """
        if not profile or not profile.words:
            return None

        # 1. Synthesize field rules from token alignment
        rules = self.synthesize_field_rules(profile, verified_data)
        if not rules:
            return None

        v_name = vendor_name or verified_data.get("vendor_name")
        v_gstin = vendor_gstin or verified_data.get("vendor_gstin") or profile.vendor_gstin

        fam_fp = profile.family_fingerprint if profile.family_fingerprint else profile.layout_signature[:12]
        ver_fp = profile.exact_fingerprint if profile.exact_fingerprint else profile.layout_signature

        topology_spec = {
            "anchor_set": list(profile.anchor_set),
            "anchor_positions": profile.anchor_positions,
            "region_topology": profile.region_topology,
        }

        # 2. Persist to PostgreSQL if DB manager is available
        version_id = f"ver_{ver_fp[:10]}"
        family_id = f"fam_{fam_fp[:10]}"

        if self.db:
            try:
                family = await self.db.get_or_create_template_family(
                    family_fingerprint=fam_fp,
                    vendor_name=str(v_name) if v_name else None,
                    vendor_gstin=str(v_gstin) if v_gstin else None,
                )
                family_id = family.id

                ver = await self.db.get_or_create_template_version(
                    family_id=family.id,
                    version_fingerprint=ver_fp,
                    aspect_ratio=profile.aspect_ratio,
                    page_count=profile.page_count,
                    anchor_signature=profile.anchor_signature,
                    layout_signature=profile.layout_signature,
                    topology_spec=topology_spec,
                )
                version_id = ver.id

                await self.db.save_field_rules(version_id=ver.id, rules=rules)
                logger.info(f"Learned & saved {len(rules)} field rules for template version '{ver.id}' in DB")
            except Exception as ex:
                logger.warning(f"Failed to persist learned template to DB: {ex}")

        # 3. Register into in-memory TemplateRetriever index for immediate lookup
        if self.retriever:
            cached_ver = CachedTemplateVersion(
                version_id=version_id,
                family_id=family_id,
                version_num=1,
                exact_fingerprint=ver_fp,
                family_fingerprint=fam_fp,
                aspect_bucket=profile.aspect_bucket,
                aspect_ratio=profile.aspect_ratio,
                page_count=profile.page_count,
                vendor_gstin=v_gstin,
                vendor_name=str(v_name) if v_name else None,
                anchor_set=set(profile.anchor_set),
                anchor_positions=dict(profile.anchor_positions),
                region_topology=list(profile.region_topology),
                field_rules=rules,
                sample_count=1,
                success_rate=1.0,
                version_fingerprint=ver_fp,
                anchor_signature=profile.anchor_signature,
                layout_signature=profile.layout_signature,
            )
            self.retriever.register_in_memory_template(cached_ver)
            logger.info(f"Registered learned template version '{version_id}' into live TIE retriever index ✓")

        return version_id

    def synthesize_field_rules(
        self,
        profile: DocumentProfile,
        verified_data: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Calculates field-specific anchor offsets and bounding rules from ground truth.
        """
        rules: list[dict[str, Any]] = []

        for field_name, candidate_anchors in FIELD_ANCHOR_CANDIDATES.items():
            gt_val = verified_data.get(field_name)
            if gt_val is None or str(gt_val).strip() == "":
                continue

            clean_gt = str(gt_val).strip().lower().replace(",", "").replace(" ", "").replace("-", "").replace("/", "")
            if not clean_gt:
                continue

            # Special Rule Strategies
            if field_name in ("vendor_gstin", "buyer_gstin"):
                rules.append({
                    "field_name": field_name,
                    "strategy": "regex_pattern",
                    "anchors": candidate_anchors,
                    "confidence_score": 0.99,
                })
                continue
            elif field_name == "ifsc_code":
                rules.append({
                    "field_name": field_name,
                    "strategy": "regex_pattern",
                    "anchors": candidate_anchors,
                    "confidence_score": 0.98,
                })
                continue

            # Locate anchor token sequence in profile
            matched_anchor = None
            matched_anchor_phrase = None
            for anc in candidate_anchors:
                matches = profile.find_anchor_tokens(anc)
                if matches:
                    matched_anchor = matches[0]
                    matched_anchor_phrase = anc
                    break

            # ── Contiguous Ground Truth Sequence Alignment ──
            target_clean_tokens = [re.sub(r"[^a-z0-9]", "", p) for p in str(gt_val).lower().split() if len(p) > 0]
            target_clean_tokens = [p for p in target_clean_tokens if p]

            candidate_sequences: list[list[WordToken]] = []
            
            # 1. Multi-word contiguous sequence match
            if len(target_clean_tokens) > 1:
                n_tokens = len(target_clean_tokens)
                for i in range(len(profile.words) - n_tokens + 1):
                    window = profile.words[i : i + n_tokens]
                    # Ensure on same page and same vertical line
                    if len(set(w.page for w in window)) > 1:
                        continue
                    if max(w.bbox_norm[3] for w in window) - min(w.bbox_norm[1] for w in window) > 60:
                        continue
                    window_clean = [re.sub(r"[^a-z0-9]", "", w.text.lower()) for w in window]
                    if window_clean == target_clean_tokens:
                        candidate_sequences.append(window)

            # 2. Single token exact match or concatenated token match
            if not candidate_sequences:
                for i, w in enumerate(profile.words):
                    w_clean = re.sub(r"[^a-z0-9]", "", w.text.lower())
                    if w_clean == clean_gt:
                        candidate_sequences.append([w])
                    elif clean_gt in w_clean and len(clean_gt) >= 3:
                        candidate_sequences.append([w])

            # 3. Select best occurrence based on spatial proximity to anchor
            gt_words: list[WordToken] = []
            if candidate_sequences:
                if matched_anchor:
                    anc_cy = sum(w.center_norm[1] for w in matched_anchor) / len(matched_anchor)
                    anc_cx = sum(w.center_norm[0] for w in matched_anchor) / len(matched_anchor)
                    candidate_sequences.sort(
                        key=lambda seq: (
                            (sum(w.center_norm[0] for w in seq) / len(seq) - anc_cx) ** 2 +
                            (sum(w.center_norm[1] for w in seq) / len(seq) - anc_cy) ** 2
                        )
                    )
                gt_words = candidate_sequences[0]

            if matched_anchor and gt_words:
                # Calculate relative offset from anchor to value
                anc_x1 = min(w.bbox_norm[0] for w in matched_anchor)
                anc_y1 = min(w.bbox_norm[1] for w in matched_anchor)
                anc_x2 = max(w.bbox_norm[2] for w in matched_anchor)
                anc_y2 = max(w.bbox_norm[3] for w in matched_anchor)

                val_x1 = min(w.bbox_norm[0] for w in gt_words)
                val_y1 = min(w.bbox_norm[1] for w in gt_words)
                val_x2 = max(w.bbox_norm[2] for w in gt_words)
                val_y2 = max(w.bbox_norm[3] for w in gt_words)

                # Determine if inline (right of anchor) or below anchor
                if val_x1 >= anc_x1 and val_y1 <= anc_y2 + 20:
                    # Inline offset relative to anc_x2
                    dx1 = max(0, val_x1 - anc_x2)
                    dy1 = max(-10, val_y1 - anc_y1 - 5)
                    dx2 = min(400, max(dx1 + 40, val_x2 - anc_x2 + 10))
                    dy2 = min(20, max(dy1 + 10, val_y2 - anc_y2 + 5))
                    rel_box = [dx1, dy1, dx2, dy2]
                else:
                    # Below offset relative to anc_x1
                    dx1 = val_x1 - anc_x1 - 5
                    dy1 = max(0, val_y1 - anc_y2)
                    dx2 = min(450, max(dx1 + 60, val_x2 - anc_x1 + 15))
                    dy2 = min(40, max(dy1 + 15, val_y2 - anc_y2 + 10))
                    rel_box = [dx1, dy1, dx2, dy2]

                strat = "semantic_numeric" if field_name in ("grand_total", "subtotal", "tax_amount", "cgst", "sgst", "igst") else "anchor_relative"

                rules.append({
                    "field_name": field_name,
                    "strategy": strat,
                    "anchors": candidate_anchors,
                    "relative_box": rel_box,
                    "confidence_score": 0.98,
                })
            elif gt_words:
                # No clear anchor found, use normalized search region
                val_x1 = min(w.bbox_norm[0] for w in gt_words)
                val_y1 = min(w.bbox_norm[1] for w in gt_words)
                val_x2 = max(w.bbox_norm[2] for w in gt_words)
                val_y2 = max(w.bbox_norm[3] for w in gt_words)
                search_reg = [max(0, val_x1 - 20), max(0, val_y1 - 10), min(1000, val_x2 + 20), min(1000, val_y2 + 10)]

                strat = "text_region" if "name" in field_name or "address" in field_name else "anchor_relative"
                rules.append({
                    "field_name": field_name,
                    "strategy": strat,
                    "anchors": candidate_anchors,
                    "search_region": search_reg,
                    "confidence_score": 0.90,
                })

        return rules

