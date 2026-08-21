"""
scripts/train_layoutlm.py

Fine-tune LayoutLMv3 on your invoice field extraction dataset.
Aligned with the latest Invoice Builder & FastAPI schema.
Uses HuggingFace Trainer — works on CPU (slow) or GPU (recommended).

Usage:
    python scripts/train_layoutlm.py \
        --data_dir data/layoutlm_dataset \
        --output_dir data/models/layoutlmv3-finetuned \
        --epochs 15 \
        --batch_size 2 \
        --lr 2e-5

BIO Label scheme:
    O                         - Other (not an invoice field)
    B-INVOICE_NUMBER, I-INVOICE_NUMBER
    B-PO_NUMBER, I-PO_NUMBER
    B-INVOICE_DATE, I-INVOICE_DATE
    B-DUE_DATE, I-DUE_DATE
    B-PLACE_OF_SUPPLY, I-PLACE_OF_SUPPLY
    B-CATEGORY, I-CATEGORY
    B-SUBCATEGORY, I-SUBCATEGORY
    B-VENDOR_NAME, I-VENDOR_NAME
    B-VENDOR_ADDRESS, I-VENDOR_ADDRESS
    B-VENDOR_GSTIN, B-VENDOR_PAN
    B-VENDOR_EMAIL, I-VENDOR_EMAIL
    B-VENDOR_PHONE, I-VENDOR_PHONE
    B-BUYER_NAME, I-BUYER_NAME
    B-BUYER_ADDRESS, I-BUYER_ADDRESS
    B-BUYER_GSTIN
    B-BUYER_PHONE, I-BUYER_PHONE
    B-SLS_CODE, I-SLS_CODE
    B-SUBTOTAL
    B-DISCOUNT, B-GLOBAL_DISCOUNT
    B-TAX_AMOUNT
    B-CGST, B-SGST, B-IGST
    B-GLOBAL_CGST_RATE, B-GLOBAL_SGST_RATE, B-GLOBAL_IGST_RATE
    B-ROUND_OFF
    B-GRAND_TOTAL
    B-AMOUNT_IN_WORDS, I-AMOUNT_IN_WORDS
    B-CURRENCY
    B-BANK_NAME, I-BANK_NAME
    B-BRANCH_NAME, I-BRANCH_NAME
    B-ACCOUNT_NAME, I-ACCOUNT_NAME
    B-ACCOUNT_NUMBER, I-ACCOUNT_NUMBER
    B-IFSC_CODE
    B-PAYMENT_TERMS, I-PAYMENT_TERMS
    B-REMARKS, I-REMARKS
    B-CERTIFIED_REMARKS, I-CERTIFIED_REMARKS
    B-LINE_ITEM_DESC, I-LINE_ITEM_DESC
    B-LINE_ITEM_HSN, I-LINE_ITEM_HSN
    B-LINE_ITEM_QTY, B-LINE_ITEM_UNIT
    B-LINE_ITEM_RATE, B-LINE_ITEM_DISCOUNT
    B-LINE_ITEM_TAXABLE_VALUE
    B-LINE_ITEM_CGST_RATE, B-LINE_ITEM_CGST_AMOUNT
    B-LINE_ITEM_SGST_RATE, B-LINE_ITEM_SGST_AMOUNT
    B-LINE_ITEM_IGST_RATE, B-LINE_ITEM_IGST_AMOUNT
    B-LINE_ITEM_AMOUNT
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Canonical BIO Labels list for LayoutLMv3
LABEL_LIST = [
    "O",
    "B-INVOICE_NUMBER", "I-INVOICE_NUMBER",
    "B-PO_NUMBER", "I-PO_NUMBER",
    "B-INVOICE_DATE", "I-INVOICE_DATE",
    "B-DUE_DATE", "I-DUE_DATE",
    "B-PLACE_OF_SUPPLY", "I-PLACE_OF_SUPPLY",
    "B-CATEGORY", "I-CATEGORY",
    "B-SUBCATEGORY", "I-SUBCATEGORY",
    "B-VENDOR_NAME", "I-VENDOR_NAME",
    "B-VENDOR_ADDRESS", "I-VENDOR_ADDRESS",
    "B-VENDOR_GSTIN",
    "B-VENDOR_PAN",
    "B-VENDOR_EMAIL", "I-VENDOR_EMAIL",
    "B-VENDOR_PHONE", "I-VENDOR_PHONE",
    "B-BUYER_NAME", "I-BUYER_NAME",
    "B-BUYER_ADDRESS", "I-BUYER_ADDRESS",
    "B-BUYER_GSTIN",
    "B-BUYER_PHONE", "I-BUYER_PHONE",
    "B-SLS_CODE", "I-SLS_CODE",
    "B-SUBTOTAL",
    "B-DISCOUNT",
    "B-GLOBAL_DISCOUNT",
    "B-TAX_AMOUNT",
    "B-CGST",
    "B-SGST",
    "B-IGST",
    "B-GLOBAL_CGST_RATE",
    "B-GLOBAL_SGST_RATE",
    "B-GLOBAL_IGST_RATE",
    "B-ROUND_OFF",
    "B-GRAND_TOTAL",
    "B-AMOUNT_IN_WORDS", "I-AMOUNT_IN_WORDS",
    "B-CURRENCY",
    "B-BANK_NAME", "I-BANK_NAME",
    "B-BRANCH_NAME", "I-BRANCH_NAME",
    "B-ACCOUNT_NAME", "I-ACCOUNT_NAME",
    "B-ACCOUNT_NUMBER", "I-ACCOUNT_NUMBER",
    "B-IFSC_CODE",
    "B-PAYMENT_TERMS", "I-PAYMENT_TERMS",
    "B-REMARKS", "I-REMARKS",
    "B-CERTIFIED_REMARKS", "I-CERTIFIED_REMARKS",
    "B-LINE_ITEM_DESC", "I-LINE_ITEM_DESC",
    "B-LINE_ITEM_HSN", "I-LINE_ITEM_HSN",
    "B-LINE_ITEM_QTY",
    "B-LINE_ITEM_UNIT",
    "B-LINE_ITEM_RATE",
    "B-LINE_ITEM_DISCOUNT",
    "B-LINE_ITEM_TAXABLE_VALUE",
    "B-LINE_ITEM_CGST_RATE", "B-LINE_ITEM_CGST_AMOUNT",
    "B-LINE_ITEM_SGST_RATE", "B-LINE_ITEM_SGST_AMOUNT",
    "B-LINE_ITEM_IGST_RATE", "B-LINE_ITEM_IGST_AMOUNT",
    "B-LINE_ITEM_AMOUNT",
]

LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for i, l in enumerate(LABEL_LIST)}


def parse_args():
    p = argparse.ArgumentParser(description="Train LayoutLMv3 for invoice field extraction")
    p.add_argument("--data_dir", "--data-dir", dest="data_dir", default="data/layoutlm_dataset", help="Directory with train/ and val/ JSON samples")
    p.add_argument("--output_dir", "--output-dir", dest="output_dir", default="data/models/layoutlmv3-finetuned", help="Where to save trained model")
    p.add_argument("--base_model", "--base-model", dest="base_model", default="microsoft/layoutlmv3-base", help="HuggingFace base model")
    p.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    p.add_argument("--batch_size", "--batch-size", dest="batch_size", type=int, default=2, help="Batch size per device")
    p.add_argument("--grad_accum", "--grad-accum", dest="grad_accum", type=int, default=4, help="Gradient accumulation steps")
    p.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    p.add_argument("--max_length", "--max-length", dest="max_length", type=int, default=512, help="Max sequence length")
    p.add_argument("--auto_export", "--auto-export", dest="auto_export", action="store_true", help="Auto export from DB if dataset missing")
    return p.parse_args()


def load_dataset_from_dir(data_dir: str, processor, split: str = "train"):
    import torch
    from torch.utils.data import Dataset
    from PIL import Image

    split_dir = Path(data_dir) / split
    samples = sorted(split_dir.glob("*.json"))

    if not samples:
        # Fallback to searching all json files in data_dir if no split folder
        all_samples = sorted(Path(data_dir).glob("*.json"))
        if all_samples:
            samples = all_samples
        else:
            raise FileNotFoundError(f"No samples found in {split_dir} or {data_dir}")

    print(f"Loading {len(samples)} {split} samples from {split_dir}...")

    class InvoiceDataset(Dataset):
        def __init__(self):
            self.data = []
            for p in samples:
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        s = json.load(f)
                        if s.get("words") and s.get("boxes"):
                            self.data.append(s)
                except Exception as ex:
                    print(f"Warning: skipped corrupted sample {p}: {ex}")

        def __len__(self):
            return max(1, len(self.data))

        def __getitem__(self, idx):
            if not self.data:
                # Return dummy blank sample if empty
                dummy_img = Image.new("RGB", (224, 224), color="white")
                enc = processor(dummy_img, ["INVOICE"], boxes=[[0, 0, 100, 100]], word_labels=[0], truncation=True, padding="max_length", max_length=512, return_tensors="pt")
                return {k: v.squeeze(0) for k, v in enc.items()}

            sample = self.data[idx % len(self.data)]
            img_path = sample["image_path"]

            try:
                image = Image.open(img_path).convert("RGB")
            except Exception:
                image = Image.new("RGB", (800, 1000), color="white")

            words = sample["words"]
            boxes = sample["boxes"]
            label_strs = sample["labels"]
            labels = [LABEL2ID.get(l, 0) for l in label_strs]

            # Ensure non-empty
            if not words:
                words = ["INVOICE"]
                boxes = [[0, 0, 100, 100]]
                labels = [0]

            # Clamp bbox values
            clamped_boxes = []
            for b in boxes:
                clamped_boxes.append([
                    max(0, min(1000, int(b[0]))),
                    max(0, min(1000, int(b[1]))),
                    max(0, min(1000, int(b[2]))),
                    max(0, min(1000, int(b[3]))),
                ])

            encoding = processor(
                image,
                words,
                boxes=clamped_boxes,
                word_labels=labels,
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt",
            )
            return {k: v.squeeze(0) for k, v in encoding.items()}

    return InvoiceDataset()


def compute_metrics(eval_pred):
    import numpy as np

    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=2)

    # Filter out -100 (ignored tokens)
    true_predictions = [
        [ID2LABEL[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(preds, labels)
    ]
    true_labels = [
        [ID2LABEL[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(preds, labels)
    ]

    total_tokens = 0
    correct_tokens = 0
    non_o_true = 0
    non_o_correct = 0

    for p_seq, l_seq in zip(true_predictions, true_labels):
        for p, l in zip(p_seq, l_seq):
            total_tokens += 1
            if p == l:
                correct_tokens += 1
            if l != "O":
                non_o_true += 1
                if p == l:
                    non_o_correct += 1

    acc = correct_tokens / max(1, total_tokens)
    entity_acc = non_o_correct / max(1, non_o_true)

    return {
        "accuracy": round(acc, 4),
        "entity_accuracy": round(entity_acc, 4),
    }


def main():
    args = parse_args()

    try:
        from transformers import (
            LayoutLMv3ForTokenClassification,
            LayoutLMv3Processor,
            TrainingArguments,
            Trainer,
        )
        import torch
    except ImportError:
        print("ERROR: transformers or torch not installed. Run: pip install transformers torch")
        sys.exit(1)

    # Check if dataset exists, else auto-export
    data_dir = Path(args.data_dir)
    if not data_dir.exists() or not list(data_dir.rglob("*.json")):
        print(f"Dataset not found at {data_dir}. Running export_reviewed_to_layoutlm.py...")
        import subprocess
        subprocess.run([sys.executable, "scripts/export_reviewed_to_layoutlm.py", "--output-dir", str(data_dir)], check=True)

    print(f"\n========================================================")
    print(f"Loading Base LayoutLMv3 Model & Processor: {args.base_model}")
    print(f"Number of schema labels: {len(LABEL_LIST)}")
    print(f"========================================================\n")

    processor = LayoutLMv3Processor.from_pretrained(args.base_model, apply_ocr=False)
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        args.base_model,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    train_dataset = load_dataset_from_dir(args.data_dir, processor, "train")
    try:
        eval_dataset = load_dataset_from_dir(args.data_dir, processor, "val")
    except FileNotFoundError:
        eval_dataset = train_dataset

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_fp16 = torch.cuda.is_available()

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=10,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        logging_steps=5,
        report_to="none",
        dataloader_num_workers=0,
        fp16=use_fp16,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    print(f"\nStarting LayoutLMv3 Fine-Tuning:")
    print(f"  Training samples:   {len(train_dataset)}")
    print(f"  Validation samples: {len(eval_dataset)}")
    print(f"  Epochs:             {args.epochs}")
    print(f"  Batch size:         {args.batch_size} (Grad Accum: {args.grad_accum})")
    print(f"  Learning rate:      {args.lr}")
    print(f"  Device:             {'CUDA GPU (fp16)' if use_fp16 else 'CPU'}")
    print(f"  Output directory:   {output_dir}\n")

    train_result = trainer.train()

    # Save fine-tuned model and processor
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))

    # Save id2label and metadata
    with open(output_dir / "id2label.json", "w", encoding="utf-8") as f:
        json.dump(ID2LABEL, f, indent=2)

    metadata = {
        "model_type": "layoutlmv3",
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "epochs": args.epochs,
        "labels": LABEL_LIST,
        "num_labels": len(LABEL_LIST),
        "train_loss": getattr(train_result, "training_loss", None),
    }
    with open(output_dir / "layoutlm_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[OK] LayoutLMv3 fine-tuning successfully completed!")
    print(f"  Model saved to: {output_dir}")
    print(f"  Set LAYOUTLM_MODEL_PATH={output_dir} in your .env or API config\n")


if __name__ == "__main__":
    main()
