"""
train/train_handwriting_yolo.py

Training script to fine-tune YOLOv8 / DocLayout-YOLO with the 12-class schema
(including handwriting, table, signature, stamp) on invoice datasets.
"""

from __future__ import annotations
import os
import yaml
from pathlib import Path
from loguru import logger


def generate_dataset_yaml(data_root: str = "data/handwriting") -> str:
    """Creates YOLO dataset.yaml configuration file."""
    yaml_content = {
        "path": str(Path(data_root).resolve()),
        "train": "images/train",
        "val": "images/val",
        "names": {
            0: "header",
            1: "vendor_block",
            2: "buyer_block",
            3: "line_items",
            4: "totals_block",
            5: "tax_block",
            6: "payment_terms",
            7: "qr_barcode",
            8: "handwriting",
            9: "table",
            10: "signature",
            11: "stamp",
        }
    }
    yaml_path = Path(data_root) / "dataset.yaml"
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    return str(yaml_path)


def train_yolo(
    base_model: str = "yolov8s.pt",
    data_yaml: Optional[str] = None,
    epochs: int = 50,
    imgsz: int = 1024,
    batch_size: int = 8,
    output_dir: str = "data/models/handwriting_yolo",
):
    """Fine-tunes YOLO model on handwriting dataset."""
    try:
        from ultralytics import YOLO
        if not data_yaml:
            data_yaml = generate_dataset_yaml()

        logger.info(f"Starting YOLO fine-tuning: base={base_model}, epochs={epochs}, imgsz={imgsz}")
        model = YOLO(base_model)
        results = model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch_size,
            project=output_dir,
            name="train_run",
            exist_ok=True,
            verbose=True,
        )
        logger.info(f"YOLO training completed successfully. Weights saved to {output_dir}")
        return results
    except Exception as e:
        logger.warning(f"YOLO training skipped or failed: {e}")
        return None


if __name__ == "__main__":
    train_yolo()
