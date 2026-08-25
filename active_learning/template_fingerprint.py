"""
active_learning/template_fingerprint.py

Invoice Template Fingerprinting & Novelty Detection Engine.

Analyzes geometric visual structure (YOLO macro regions, aspect ratio, anchor centroids)
to assign a deterministic template ID (e.g., 'tpl_a8f9c2d1') to each incoming invoice.

Tracks:
- Known Templates vs Novel / Unknown Templates
- Template frequency and historical correction rate
- Alerts active learning queue when an unknown layout appears (+25 informativeness boost).
"""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Any
from loguru import logger


@dataclass
class TemplateInfo:
    template_id: str
    first_seen: str
    total_invoices: int = 1
    reviewed_count: int = 0
    correction_count: int = 0
    known_vendor: Optional[str] = None
    is_novel: bool = True

    @property
    def accuracy_rate(self) -> float:
        if self.reviewed_count == 0:
            return 1.0
        return max(0.0, 1.0 - (self.correction_count / self.reviewed_count))


def compute_template_fingerprint(
    width: int,
    height: int,
    regions: list[dict[str, Any]],
    vendor_gstin: Optional[str] = None,
) -> str:
    """
    Generates a deterministic layout signature hash from page geometry & YOLO macro-regions.
    """
    w_safe = max(1, width)
    h_safe = max(1, height)
    aspect_ratio = round(h_safe / w_safe, 2)

    # Extract normalized centroids and region labels
    sig_parts = [f"ar:{aspect_ratio}"]

    if vendor_gstin and len(vendor_gstin.strip()) == 15:
        # Vendor-specific layout prefix
        sig_parts.append(f"v:{vendor_gstin.strip().upper()}")

    sorted_regions = sorted(
        regions,
        key=lambda r: (r.get("bbox", [0, 0, 0, 0])[1], r.get("bbox", [0, 0, 0, 0])[0])
    )

    for r in sorted_regions[:12]:
        lbl = r.get("label", "region")
        b = r.get("bbox", [0, 0, 0, 0])
        # Grid cell normalization (10x10 grid)
        gx = int(b[0] * 10 / w_safe)
        gy = int(b[1] * 10 / h_safe)
        sig_parts.append(f"{lbl}_{gx}_{gy}")

    raw_signature = "|".join(sig_parts)
    hash_digest = hashlib.sha256(raw_signature.encode("utf-8")).hexdigest()[:12]
    return f"tpl_{hash_digest}"


REGISTRY_FILE = Path("data/templates_registry.json")


class TemplateManager:
    """
    Maintains persistent template registry and historical learning stats.
    Saves state to disk (data/templates_registry.json) so learned templates survive restarts.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(TemplateManager, cls).__new__(cls)
            cls._instance._registry = {}
            cls._instance._load_registry()
        return cls._instance

    def _load_registry(self):
        try:
            if REGISTRY_FILE.exists():
                with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
                    self._registry = json.load(f)
                logger.info(f"Loaded {len(self._registry)} layout templates from {REGISTRY_FILE}")
        except Exception as e:
            logger.warning(f"Could not load templates registry: {e}")
            self._registry = {}

    def _save_registry(self):
        try:
            REGISTRY_FILE.parent.mkdir(parents=True, exist_ok=True)
            with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
                json.dump(self._registry, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save templates registry: {e}")

    def register_invoice(
        self,
        template_id: str,
        vendor_name: Optional[str] = None,
    ) -> dict[str, Any]:
        if template_id not in self._registry:
            self._registry[template_id] = {
                "template_id": template_id,
                "total_invoices": 1,
                "reviewed_count": 0,
                "correction_count": 0,
                "known_vendor": vendor_name,
                "is_novel": True,
            }
            is_novel = True
        else:
            self._registry[template_id]["total_invoices"] += 1
            if vendor_name and not self._registry[template_id].get("known_vendor"):
                self._registry[template_id]["known_vendor"] = vendor_name
            is_novel = self._registry[template_id]["total_invoices"] < 3
            self._registry[template_id]["is_novel"] = is_novel

        self._save_registry()

        return {
            "template_id": template_id,
            "is_novel": is_novel,
            "total_seen": self._registry[template_id]["total_invoices"],
            "known_vendor": self._registry[template_id]["known_vendor"],
        }

