"""
Stage 3a: YOLOv8 invoice region detector.

Detects the following invoice regions:
  0: header          - invoice title, number, date
  1: vendor_block    - seller info (name, address, GSTIN)
  2: buyer_block     - buyer info
  3: line_items      - the table of goods/services
  4: totals_block    - subtotal, tax, grand total
  5: tax_block       - GST / VAT breakdown
  6: payment_terms   - bank details, due date
  7: qr_barcode      - QR code or barcode

Falls back to a heuristic grid split if the YOLO model is not yet trained.
"""

import cv2
import numpy as np
from pathlib import Path
from loguru import logger
from dataclasses import dataclass, field
from typing import Optional


REGION_LABELS = {
    0: "header",
    1: "vendor_block",
    2: "buyer_block",
    3: "line_items",
    4: "totals_block",
    5: "tax_block",
    6: "payment_terms",
    7: "qr_barcode",
}

REGION_IDS = {v: k for k, v in REGION_LABELS.items()}


@dataclass
class DetectedRegion:
    label: str
    class_id: int
    confidence: float
    bbox: tuple           # (x1, y1, x2, y2) in pixels
    crop: np.ndarray      # cropped image of the region


@dataclass
class DetectionResult:
    regions: list[DetectedRegion]
    image_size: tuple     # (width, height)
    model_used: str       # "yolo" or "heuristic"


class InvoiceDetector:
    """
    YOLOv8-based invoice region detector.
    Loads a fine-tuned model if available; otherwise uses heuristic fallback.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.model_path = model_path
        self._try_load_model()

    def _try_load_model(self):
        if not self.model_path:
            logger.warning("No YOLO model path configured — using heuristic fallback")
            return
        path = Path(self.model_path)
        if not path.exists():
            logger.warning(f"YOLO model not found at {path} — using heuristic fallback")
            return
        try:
            from ultralytics import YOLO
            self.model = YOLO(str(path))
            logger.info(f"YOLO model loaded: {path}")
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            self.model = None

    def detect(self, image: np.ndarray, conf_threshold: float = 0.35) -> DetectionResult:
        """
        Run region detection on a pre-processed invoice image.
        Returns DetectionResult with all detected regions and their crops.
        """
        h, w = image.shape[:2]

        if self.model is not None:
            return self._detect_yolo(image, conf_threshold)
        else:
            return self._detect_heuristic(image)

    def _detect_yolo(self, image: np.ndarray, conf_threshold: float) -> DetectionResult:
        results = self.model(image, conf=conf_threshold, verbose=False)
        regions = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = REGION_LABELS.get(cls_id, f"class_{cls_id}")
                crop = image[y1:y2, x1:x2]
                regions.append(DetectedRegion(
                    label=label,
                    class_id=cls_id,
                    confidence=conf,
                    bbox=(x1, y1, x2, y2),
                    crop=crop,
                ))

        regions.sort(key=lambda r: r.bbox[1])  # sort top-to-bottom
        h, w = image.shape[:2]
        return DetectionResult(regions=regions, image_size=(w, h), model_used="yolo")

    def _detect_heuristic(self, image: np.ndarray) -> DetectionResult:
        """
        Heuristic region splitter when no YOLO model is available.
        Splits the invoice into logical zones based on position.
        This is a reasonable fallback for standard A4 invoices.
        """
        h, w = image.shape[:2]
        regions = []

        # Define approximate zones as (label, class_id, y_start_pct, y_end_pct)
        zones = [
            ("header",       0, 0.00, 0.15),
            ("vendor_block", 1, 0.10, 0.30),
            ("buyer_block",  2, 0.25, 0.40),
            ("line_items",   3, 0.38, 0.72),
            ("totals_block", 4, 0.70, 0.88),
            ("tax_block",    5, 0.75, 0.92),
            ("payment_terms",6, 0.88, 1.00),
        ]

        for label, cls_id, y_start_pct, y_end_pct in zones:
            y1 = int(h * y_start_pct)
            y2 = int(h * y_end_pct)
            x1, x2 = 0, w
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            regions.append(DetectedRegion(
                label=label,
                class_id=cls_id,
                confidence=0.50,  # heuristic confidence is always 0.50
                bbox=(x1, y1, x2, y2),
                crop=crop,
            ))

        logger.debug(f"Heuristic detection: {len(regions)} zones")
        return DetectionResult(regions=regions, image_size=(w, h), model_used="heuristic")

    def visualise(self, image: np.ndarray, result: DetectionResult) -> np.ndarray:
        """Draw bounding boxes and labels on the image for debugging."""
        vis = image.copy()
        colors = {
            "header": (255, 100, 0),
            "vendor_block": (0, 200, 100),
            "buyer_block": (0, 100, 255),
            "line_items": (200, 0, 200),
            "totals_block": (255, 200, 0),
            "tax_block": (0, 200, 255),
            "payment_terms": (150, 150, 0),
            "qr_barcode": (200, 100, 200),
        }
        for region in result.regions:
            x1, y1, x2, y2 = region.bbox
            color = colors.get(region.label, (128, 128, 128))
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            label_text = f"{region.label} {region.confidence:.2f}"
            cv2.putText(vis, label_text, (x1, max(y1 - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return vis
