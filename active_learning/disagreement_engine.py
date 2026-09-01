"""
active_learning/disagreement_engine.py

Multi-Model Disagreement & Consensus Engine.

Compares candidate predictions across multiple independent inference streams:
1. Spatial Transformer (LayoutLMv3)
2. Geometric Heuristics & Deterministic Regex
3. Generative LLM Fallback (Ollama Llama 3.1 / Mistral)

Calculates a Disagreement Score [0.0, 100.0]. High disagreement indicates
maximum learning value and elevates the invoice in the Active Learning Review Queue.
"""

import difflib
import re
from typing import Any, Optional
from loguru import logger


def clean_val(val: Any) -> str:
    if val is None:
        return ""
    return re.sub(r"[^a-zA-Z0-9]", "", str(val).lower())


def numeric_match(val1: Any, val2: Any, tolerance: float = 0.02) -> bool:
    if val1 is None or val2 is None:
        return False
    try:
        n1 = float(re.sub(r"[^\d.]", "", str(val1)))
        n2 = float(re.sub(r"[^\d.]", "", str(val2)))
        return abs(n1 - n2) <= max(1.0, n1 * tolerance)
    except Exception:
        return False


def text_match(val1: Any, val2: Any, threshold: float = 0.80) -> bool:
    c1 = clean_val(val1)
    c2 = clean_val(val2)
    if not c1 and not c2:
        return True
    if not c1 or not c2:
        return False
    if c1 == c2 or c1 in c2 or c2 in c1:
        return True
    return difflib.SequenceMatcher(None, c1, c2).ratio() >= threshold


def evaluate_model_disagreement(
    layoutlm_preds: Optional[dict] = None,
    heuristic_preds: Optional[dict] = None,
    llm_preds: Optional[dict] = None,
    tie_preds: Optional[dict] = None,
) -> dict[str, Any]:
    """
    Computes cross-model discrepancy across key semantic invoice fields (TIE, LayoutLM, Heuristics, LLM).
    """
    layoutlm = layoutlm_preds or {}
    heuristic = heuristic_preds or {}
    llm = llm_preds or {}
    tie = tie_preds or {}

    disagreements = []
    disagreement_score = 0.0

    # 1. Grand Total Consensus (Critical financial metric - highest weight)
    tt = tie.get("grand_total")
    lt = layoutlm.get("grand_total")
    ht = heuristic.get("grand_total")
    llmt = llm.get("grand_total")

    # If TIE produced a prediction, check against heuristic/LayoutLM
    if tt is not None:
        if ht is not None and not numeric_match(tt, ht):
            disagreements.append({
                "field": "grand_total",
                "engine_1": "tie",
                "val_1": tt,
                "engine_2": "heuristic",
                "val_2": ht,
                "reason": f"TIE ({tt}) != Heuristic ({ht})",
            })
            disagreement_score += 35.0
        elif lt is not None and not numeric_match(tt, lt):
            disagreements.append({
                "field": "grand_total",
                "engine_1": "tie",
                "val_1": tt,
                "engine_2": "layoutlm",
                "val_2": lt,
                "reason": f"TIE ({tt}) != LayoutLM ({lt})",
            })
            disagreement_score += 30.0
    elif lt is not None and ht is not None and not numeric_match(lt, ht):
        disagreements.append({
            "field": "grand_total",
            "engine_1": "layoutlm",
            "val_1": lt,
            "engine_2": "heuristic",
            "val_2": ht,
            "reason": f"LayoutLM ({lt}) != Heuristic ({ht})",
        })
        disagreement_score += 40.0

    # 2. Vendor GSTIN Consensus (Legal tax entity)
    tg = tie.get("vendor_gstin")
    lg = layoutlm.get("vendor_gstin")
    hg = heuristic.get("vendor_gstin")

    if tg and hg and not text_match(tg, hg, threshold=0.90):
        disagreements.append({
            "field": "vendor_gstin",
            "engine_1": "tie",
            "val_1": tg,
            "engine_2": "heuristic",
            "val_2": hg,
            "reason": f"GSTIN mismatch: TIE ({tg}) vs Heuristic ({hg})",
        })
        disagreement_score += 25.0
    elif lg and hg and not text_match(lg, hg, threshold=0.90):
        disagreements.append({
            "field": "vendor_gstin",
            "engine_1": "layoutlm",
            "val_1": lg,
            "engine_2": "heuristic",
            "val_2": hg,
            "reason": f"GSTIN mismatch: LayoutLM ({lg}) vs Heuristic ({hg})",
        })
        disagreement_score += 30.0

    # 3. Invoice Number Consensus
    ti = tie.get("invoice_number")
    li = layoutlm.get("invoice_number")
    hi = heuristic.get("invoice_number")

    if ti and hi and not text_match(ti, hi, threshold=0.85):
        disagreements.append({
            "field": "invoice_number",
            "engine_1": "tie",
            "val_1": ti,
            "engine_2": "heuristic",
            "val_2": hi,
            "reason": f"Invoice No mismatch: TIE ({ti}) vs Heuristic ({hi})",
        })
        disagreement_score += 20.0
    elif li and hi and not text_match(li, hi, threshold=0.85):
        disagreements.append({
            "field": "invoice_number",
            "engine_1": "layoutlm",
            "val_1": li,
            "engine_2": "heuristic",
            "val_2": hi,
            "reason": f"Invoice No mismatch: LayoutLM ({li}) vs Heuristic ({hi})",
        })
        disagreement_score += 20.0

    # 4. Vendor Name Consensus
    tv = tie.get("vendor_name")
    lv = layoutlm.get("vendor_name")
    hv = heuristic.get("vendor_name")

    if tv and hv and not text_match(tv, hv, threshold=0.75):
        disagreements.append({
            "field": "vendor_name",
            "engine_1": "tie",
            "val_1": tv,
            "engine_2": "heuristic",
            "val_2": hv,
            "reason": f"Vendor name divergence: TIE ({tv}) vs Heuristic ({hv})",
        })
        disagreement_score += 15.0
    elif lv and hv and not text_match(lv, hv, threshold=0.75):
        disagreements.append({
            "field": "vendor_name",
            "engine_1": "layoutlm",
            "val_1": lv,
            "engine_2": "heuristic",
            "val_2": hv,
            "reason": f"Vendor name divergence: LayoutLM ({lv}) vs Heuristic ({hv})",
        })
        disagreement_score += 15.0

    final_score = min(100.0, disagreement_score)
    return {
        "disagreement_score": round(final_score, 2),
        "has_contradiction": len(disagreements) > 0,
        "disagreement_count": len(disagreements),
        "disagreements": disagreements,
    }
