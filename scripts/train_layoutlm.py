"""
scripts/train_layoutlm.py

Fine-tune LayoutLMv3 on your invoice field extraction dataset.
Uses HuggingFace Trainer — works on CPU (slow) or GPU (recommended).

Recommended: run on Google Colab free GPU tier for training,
then copy the saved model to data/models/layoutlmv3-finetuned/

Usage:
    python scripts/train_layoutlm.py \
        --data_dir data/layoutlm_dataset \
        --output_dir data/models/layoutlmv3-finetuned \
        --epochs 10 \
        --batch_size 2

Dataset format (data/layoutlm_dataset/):
    Each sample is a JSON file with:
    {
      "image_path": "path/to/invoice.png",
      "words": ["INVOICE", "NO:", "INV-001", ...],
      "boxes": [[x1, y1, x2, y2], ...],  # normalised 0-1000
      "labels": ["O", "B-INVOICE_NUMBER", "I-INVOICE_NUMBER", ...]
    }

BIO Label scheme:
    O                 - Other (not a field)
    B-INVOICE_NUMBER  - Beginning of invoice number
    I-INVOICE_NUMBER  - Inside invoice number
    B-INVOICE_DATE
    B-VENDOR_NAME
    B-VENDOR_ADDRESS
    B-VENDOR_GSTIN
    B-BUYER_NAME
    B-BUYER_ADDRESS
    B-BUYER_GSTIN
    B-SUBTOTAL
    B-TAX_AMOUNT
    B-GRAND_TOTAL
    B-LINE_ITEM_DESC
    B-LINE_ITEM_QTY
    B-LINE_ITEM_RATE
    B-LINE_ITEM_AMOUNT
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

LABEL_LIST = [
    "O",
    "B-INVOICE_NUMBER", "I-INVOICE_NUMBER",
    "B-INVOICE_DATE",   "I-INVOICE_DATE",
    "B-DUE_DATE",       "I-DUE_DATE",
    "B-VENDOR_NAME",    "I-VENDOR_NAME",
    "B-VENDOR_ADDRESS", "I-VENDOR_ADDRESS",
    "B-VENDOR_GSTIN",
    "B-BUYER_NAME",     "I-BUYER_NAME",
    "B-BUYER_ADDRESS",  "I-BUYER_ADDRESS",
    "B-BUYER_GSTIN",
    "B-SUBTOTAL",
    "B-TAX_AMOUNT",
    "B-GRAND_TOTAL",
    "B-LINE_ITEM_DESC",  "I-LINE_ITEM_DESC",
    "B-LINE_ITEM_QTY",
    "B-LINE_ITEM_RATE",
    "B-LINE_ITEM_AMOUNT",
]
LABEL2ID = {l: i for i, l in enumerate(LABEL_LIST)}
ID2LABEL = {i: l for i, l in enumerate(LABEL_LIST)}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/layoutlm_dataset")
    p.add_argument("--output_dir", default="data/models/layoutlmv3-finetuned")
    p.add_argument("--base_model", default="microsoft/layoutlmv3-base")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max_length", type=int, default=512)
    return p.parse_args()


def load_dataset_from_dir(data_dir: str, processor, split: str = "train"):
    from torch.utils.data import Dataset
    from PIL import Image
    import torch

    samples = sorted(Path(data_dir).glob(f"{split}/*.json"))
    if not samples:
        raise FileNotFoundError(f"No {split} samples in {data_dir}/{split}/")

    print(f"Loading {len(samples)} {split} samples...")

    class InvoiceDataset(Dataset):
        def __init__(self):
            self.data = []
            for p in samples:
                with open(p) as f:
                    self.data.append(json.load(f))

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            sample = self.data[idx]
            image = Image.open(sample["image_path"]).convert("RGB")
            words = sample["words"]
            boxes = sample["boxes"]
            label_strs = sample["labels"]
            labels = [LABEL2ID.get(l, 0) for l in label_strs]

            encoding = processor(
                image,
                words,
                boxes=boxes,
                word_labels=labels,
                truncation=True,
                padding="max_length",
                max_length=512,
                return_tensors="pt",
            )
            return {k: v.squeeze(0) for k, v in encoding.items()}

    return InvoiceDataset()


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
        print("ERROR: transformers not installed. Run: pip install transformers torch")
        sys.exit(1)

    print(f"Loading processor from: {args.base_model}")
    processor = LayoutLMv3Processor.from_pretrained(args.base_model, apply_ocr=False)

    print(f"Loading base model: {args.base_model}")
    model = LayoutLMv3ForTokenClassification.from_pretrained(
        args.base_model,
        num_labels=len(LABEL_LIST),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    train_dataset = load_dataset_from_dir(args.data_dir, processor, "train")
    eval_dataset  = load_dataset_from_dir(args.data_dir, processor, "val")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=5,
        report_to="none",   # no wandb required
        dataloader_num_workers=0,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    print(f"\nStarting LayoutLMv3 fine-tuning:")
    print(f"  Samples:    {len(train_dataset)} train / {len(eval_dataset)} val")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Output dir: {args.output_dir}\n")

    trainer.train()
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
    print(f"\n[OK] LayoutLMv3 Model saved to: {output_dir}")
    print(f"  Update LAYOUTLM_MODEL_PATH={output_dir} in your .env")


if __name__ == "__main__":
    main()
