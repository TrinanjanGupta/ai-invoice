"""
data/handwriting/synthetic_generator.py

Synthetic Generator for Indian Handwritten Invoices, Pads, and Mixed Forms.
Generates realistic training and evaluation samples with:
- Ruled notebook pad lines (blue, gray, red margin)
- Printed invoice templates with cursive/digit handwritten entries
- Optical variations (ink degradation, phone shadows, skew)
- Bounding box annotations for YOLO (12 classes) and line-level transcription pairs for TrOCR
"""

from __future__ import annotations
import os
import cv2
import json
import random
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class SyntheticSample:
    image: np.ndarray
    yolo_boxes: list[dict]       # list of {"class_id": int, "bbox": [x1, y1, x2, y2], "label": str}
    line_transcriptions: list[dict] # list of {"crop": np.ndarray, "text": str, "field_name": str}
    doc_type: str               # "HANDWRITTEN_PAD" | "MIXED_FORM" | "PRINTED_FORM"


class SyntheticHandwritingGenerator:
    """
    Generates synthetic invoice images simulating Indian handwritten pads and mixed bills.
    """

    SAMPLE_VENDORS = [
        "SHREE GANESH ENTERPRISES",
        "MAA TARA HARDWARE & PAINTS",
        "KOLKATA ELECTRICAL WORKS",
        "BALAJI TRADING CO.",
        "GUPTA GENERAL STORES",
    ]

    SAMPLE_ITEMS = [
        ("CEMENT 50KG BAG", "10", "380.00", "3800.00"),
        ("STEEL ROD 12MM", "5", "550.00", "2750.00"),
        ("PAINT 20L DRUM", "2", "3200.00", "6400.00"),
        ("PVC PIPE 4 INCH", "12", "180.00", "2160.00"),
        ("SWITCH BOARD 8 MOD", "4", "450.00", "1800.00"),
    ]

    def generate_pad_sample(self, width: int = 800, height: int = 1100) -> SyntheticSample:
        """Generates a realistic handwritten notebook invoice pad."""
        img = np.ones((height, width, 3), dtype=np.uint8) * random.randint(240, 255)

        # 1. Background paper texture and faint yellow/cream tint
        img[:, :, 0] = np.clip(img[:, :, 0] - random.randint(5, 20), 0, 255) # slightly less blue -> warm paper
        img[:, :, 1] = np.clip(img[:, :, 1] - random.randint(0, 10), 0, 255)

        # 2. Ruled horizontal lines
        line_spacing = random.randint(45, 55)
        ruled_color = (random.randint(180, 210), random.randint(180, 200), random.randint(220, 240)) # faint blue/gray
        for y in range(120, height - 100, line_spacing):
            cv2.line(img, (40, y), (width - 40, y), ruled_color, 1)

        # 3. Vertical left red margin line
        margin_x = random.randint(90, 120)
        cv2.line(img, (margin_x, 40), (margin_x, height - 40), (140, 140, 220), 1)

        # 4. Handwritten text entries and YOLO bounding box collection
        yolo_boxes = []
        line_transcriptions = []

        vendor = random.choice(self.SAMPLE_VENDORS)
        inv_no = f"INV/{random.randint(10, 99)}/{random.randint(100, 999)}"
        inv_date = f"{random.randint(1, 28):02d}/{random.randint(1, 12):02d}/2026"
        gstin = f"27AABCU{random.randint(1000, 9999)}R1Z{random.choice('123456789ABC')}"

        ink_color = (random.randint(80, 130), random.randint(20, 40), random.randint(20, 40)) # Blue ballpoint ink (BGR)

        # Header Block
        cv2.putText(img, vendor, (margin_x + 20, 90), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 0.9, ink_color, 2)
        yolo_boxes.append({"class_id": 8, "label": "handwriting", "bbox": [margin_x + 10, 60, width - 60, 110]})

        # Meta details: Invoice No & Date
        cv2.putText(img, f"Bill No: {inv_no}", (margin_x + 20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ink_color, 2)
        cv2.putText(img, f"Date: {inv_date}", (width - 250, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ink_color, 2)
        yolo_boxes.append({"class_id": 8, "label": "handwriting", "bbox": [margin_x + 15, 125, width - 50, 165]})

        # Line items
        curr_y = 210
        total_sum = 0.0
        for item_name, qty, rate, amt in self.SAMPLE_ITEMS[:3]:
            row_txt = f"{item_name}  {qty} x {rate} = {amt}"
            cv2.putText(img, row_txt, (margin_x + 20, curr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, ink_color, 2)
            total_sum += float(amt)
            curr_y += line_spacing

        # Totals block
        tot_y = curr_y + 20
        cv2.putText(img, f"Grand Total: Rs. {total_sum:.2f}", (margin_x + 20, tot_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, ink_color, 2)
        yolo_boxes.append({"class_id": 4, "label": "totals_block", "bbox": [margin_x + 10, tot_y - 30, width - 60, tot_y + 15]})
        line_transcriptions.append({
            "text": f"{total_sum:.2f}",
            "field_name": "grand_total",
            "crop": img[tot_y - 30:tot_y + 15, margin_x + 10:width - 60],
        })

        # Signature Block
        sig_center = (width - 150, height - 120)
        cv2.ellipse(img, sig_center, (60, 20), 15, 0, 360, ink_color, 2)
        cv2.putText(img, "For Shree Ganesh", (width - 220, height - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50, 50, 50), 1)
        yolo_boxes.append({"class_id": 10, "label": "signature", "bbox": [width - 230, height - 150, width - 50, height - 60]})

        # Rubber Stamp
        stamp_center = (margin_x + 80, height - 120)
        cv2.circle(img, stamp_center, 45, (40, 40, 180), 2) # Red/violet seal
        cv2.putText(img, "PAID", (stamp_center[0] - 25, stamp_center[1] + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 180), 2)
        yolo_boxes.append({"class_id": 11, "label": "stamp", "bbox": [stamp_center[0] - 50, stamp_center[1] - 50, stamp_center[0] + 50, stamp_center[1] + 50]})

        return SyntheticSample(
            image=img,
            yolo_boxes=yolo_boxes,
            line_transcriptions=line_transcriptions,
            doc_type="HANDWRITTEN_PAD",
        )
