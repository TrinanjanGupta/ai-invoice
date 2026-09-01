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
    "invoice_number": [
        "invoice no", "inv no", "bill no", "invoice #", "inv #", "bill #", "invoice id", "bill id",
        "voucher no", "bill number", "sl no", "serial no", "ref no", "reference no", "memo no", "challan no"
    ],
    "invoice_date": [
        "invoice date", "dated", "bill date", "date of issue", "date:", "date", "inv date", "bill dt", "inv dt", "dt:"
    ],
    "due_date": [
        "due date", "payment due", "pay by", "due dt"
    ],
    "po_number": [
        "po no", "order no", "purchase order", "po number", "wo no", "work order"
    ],
    "place_of_supply": [
        "place of supply", "state/ut code", "supply state", "pos", "state code"
    ],
    "buyer_name": [
        "bill to", "billed to", "consignee", "buyer", "customer", "party name", "client", "m/s", "to:"
    ],
    "buyer_address": [
        "bill to", "billed to", "consignee address", "buyer address", "customer address", "address:"
    ],
    "buyer_address_line1": [
        "bill to", "billed to", "consignee", "buyer", "customer"
    ],
    "buyer_address_line2": [
        "bill to", "billed to", "consignee", "buyer"
    ],
    "buyer_phone": [
        "phone", "mobile", "ph:", "mob:", "tel:"
    ],
    "buyer_gstin": [
        "buyer gstin", "buyer gst", "party gstin", "gstin/uin", "gstin", "gst no"
    ],
    "vendor_name": [
        "tax invoice", "retail invoice", "bill of supply", "cash memo", "invoice", "vendor", "from:", "m/s"
    ],
    "vendor_address": [
        "tax invoice", "retail invoice", "invoice", "address"
    ],
    "vendor_address_line1": [
        "tax invoice", "retail invoice", "invoice"
    ],
    "vendor_address_line2": [
        "tax invoice", "retail invoice", "invoice"
    ],
    "vendor_gstin": [
        "vendor gstin", "gstin/uin", "gstin", "gst no", "gstn"
    ],
    "vendor_pan": [
        "pan", "pan no", "vendor pan"
    ],
    "vendor_phone": [
        "phone", "mobile", "tel:", "ph:"
    ],
    "vendor_email": [
        "email", "e-mail", "mail"
    ],
    "subtotal": [
        "sub total", "sub-total", "subtotal", "taxable value", "taxable amount", "basic amount", "total before tax"
    ],
    "cgst": [
        "+cgst :", "+cgst", "cgst amt", "cgst amount", "central tax", "cgst", "cgst @", "cgst rate"
    ],
    "sgst": [
        "+sgst :", "+sgst", "sgst amt", "sgst amount", "state tax", "sgst", "utgst", "sgst @", "sgst rate"
    ],
    "igst": [
        "+igst :", "+igst", "igst amt", "igst amount", "integrated tax", "igst", "igst @", "igst rate"
    ],
    "tax_amount": [
        "gst total", "total tax", "tax amount", "gst amount", "tax total", "total gst"
    ],
    "discount": [
        "discount", "less discount", "trade discount", "global discount", "disc.", "disc"
    ],
    "round_off": [
        "round off", "roundoff", "rounding", "round-off"
    ],
    "grand_total": [
        "grand total", "total payable", "net amount", "total amount", "total:", "net payable", "total", "invoice total", "bill total", "total (inr)", "total rs", "final amount", "amount payable", "balance due"
    ],
    "amount_in_words": [
        "amount in words", "in words", "rupees in words", "total in words"
    ],
    "bank_name": [
        "bank name", "bank:", "bank details", "bank"
    ],
    "branch_name": [
        "branch", "branch name", "branch:"
    ],
    "account_name": [
        "account name", "a/c name", "beneficiary name", "account holder"
    ],
    "account_number": [
        "account no", "ac no", "a/c no", "account number", "bank a/c", "bank account"
    ],
    "ifsc_code": [
        "ifsc", "ifsc code", "ifsc:"
    ],
    "payment_terms": [
        "payment terms", "terms:", "terms and conditions", "terms & conditions", "payment mode"
    ],
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

        all_field_keys = list(FIELD_ANCHOR_CANDIDATES.keys())
        for k in verified_data.keys():
            if k not in all_field_keys and not k.startswith("_") and k not in ("line_items", "field_confidences", "spatial_candidates", "review_reasons"):
                all_field_keys.append(k)

        for field_name in all_field_keys:
            candidate_anchors = FIELD_ANCHOR_CANDIDATES.get(field_name, [])
            gt_val = verified_data.get(field_name)
            if gt_val is None or str(gt_val).strip() == "" or str(gt_val).lower() in ("null", "none", "unknown"):
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

            # ── 1. Locate all candidate anchor occurrences ──
            all_anchor_seqs: list[tuple[str, list[WordToken]]] = []
            for anc in candidate_anchors:
                matches = profile.find_anchor_tokens(anc)
                for seq in matches:
                    all_anchor_seqs.append((anc, seq))

            # ── 2. Locate all candidate ground truth token sequences ──
            target_clean_tokens = [re.sub(r"[^a-z0-9]", "", p) for p in str(gt_val).lower().split() if len(p) > 0]
            target_clean_tokens = [p for p in target_clean_tokens if p]

            candidate_sequences: list[list[WordToken]] = []

            # Check if gt_val is a float/number
            num_val = None
            try:
                num_val = float(str(gt_val).replace(",", "").replace("₹", "").replace("Rs.", "").strip())
            except (ValueError, TypeError):
                pass
            
            # A. Multi-word contiguous sequence match
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

            # B. Single token exact match, clean substring match, or numeric float match
            if not candidate_sequences:
                for i, w in enumerate(profile.words):
                    w_clean = re.sub(r"[^a-z0-9]", "", w.text.lower())
                    if w_clean == clean_gt:
                        candidate_sequences.append([w])
                    elif clean_gt in w_clean and len(clean_gt) >= 3:
                        candidate_sequences.append([w])
                    elif num_val is not None:
                        try:
                            w_num = float(re.sub(r"[^\d.]", "", w.text))
                            if abs(w_num - num_val) <= 0.05:
                                candidate_sequences.append([w])
                        except Exception:
                            pass

            # ── 3. Score all (anchor, ground_truth) candidate pairs ──
            scored_pairs: list[tuple[float, list[WordToken], list[WordToken]]] = []

            for anc_phrase, anc_seq in all_anchor_seqs:
                anc_p = anc_seq[0].page
                anc_x1 = min(w.bbox_norm[0] for w in anc_seq)
                anc_y1 = min(w.bbox_norm[1] for w in anc_seq)
                anc_x2 = max(w.bbox_norm[2] for w in anc_seq)
                anc_y2 = max(w.bbox_norm[3] for w in anc_seq)
                anc_cx = sum(w.center_norm[0] for w in anc_seq) / len(anc_seq)
                anc_cy = sum(w.center_norm[1] for w in anc_seq) / len(anc_seq)

                for gt_seq in candidate_sequences:
                    gt_p = gt_seq[0].page
                    if anc_p != gt_p:
                        continue  # Must be on same page
                    
                    val_x1 = min(w.bbox_norm[0] for w in gt_seq)
                    val_y1 = min(w.bbox_norm[1] for w in gt_seq)
                    val_x2 = max(w.bbox_norm[2] for w in gt_seq)
                    val_y2 = max(w.bbox_norm[3] for w in gt_seq)
                    val_cx = sum(w.center_norm[0] for w in gt_seq) / len(gt_seq)
                    val_cy = sum(w.center_norm[1] for w in gt_seq) / len(gt_seq)

                    pair_score = 1.0

                    # A. Semantic anchor specificity & suitability
                    if len(anc_phrase.split()) > 1:
                        pair_score += 0.30

                    anc_lower = anc_phrase.lower()
                    if field_name == "invoice_date":
                        if any(k in anc_lower for k in ("due", "po", "delivery", "lr", "valid")):
                            pair_score -= 0.80
                        if any(k in anc_lower for k in ("invoice date", "bill date", "inv date", "dated")):
                            pair_score += 0.40
                    elif field_name == "grand_total":
                        if any(k in anc_lower for k in ("sub", "taxable", "item", "qty", "rate")):
                            pair_score -= 0.70
                        if any(k in anc_lower for k in ("grand total", "net total", "invoice total", "total amount")):
                            pair_score += 0.40

                    # B. Spatial arrangement
                    is_inline = (val_x1 >= anc_x1 - 10 and abs(val_cy - anc_cy) <= 25 and val_x1 - anc_x2 <= 450)
                    is_below = (abs(val_cx - anc_cx) <= 150 and val_y1 >= anc_y1 and val_y1 - anc_y2 <= 80)

                    if is_inline:
                        dist_x = max(0, val_x1 - anc_x2)
                        dist_y = abs(val_cy - anc_cy)
                        pair_score += max(0.0, 0.50 - (dist_x / 500.0) - (dist_y / 50.0))
                    elif is_below:
                        dist_y = max(0, val_y1 - anc_y2)
                        dist_x = abs(val_cx - anc_cx)
                        pair_score += max(0.0, 0.40 - (dist_y / 150.0) - (dist_x / 200.0))
                    else:
                        euc_dist = ((val_cx - anc_cx)**2 + (val_cy - anc_cy)**2) ** 0.5
                        pair_score -= min(0.80, euc_dist / 600.0)

                    scored_pairs.append((pair_score, anc_seq, gt_seq))

            matched_anchor = None
            gt_words: list[WordToken] = []

            if scored_pairs:
                scored_pairs.sort(key=lambda p: p[0], reverse=True)
                best_pair_score, best_anc, best_gt = scored_pairs[0]
                if best_pair_score >= 0.40:
                    matched_anchor = best_anc
                    gt_words = best_gt
            elif candidate_sequences:
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

