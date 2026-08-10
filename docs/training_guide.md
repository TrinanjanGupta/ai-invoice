# Training Guide

## Overview

The pipeline works in two modes:
- **Heuristic mode** (no training needed) — works immediately, ~65–75% accuracy
- **Trained mode** (after fine-tuning) — 90%+ accuracy

Follow the phases below to progressively improve accuracy.

---

## Phase 1: Collect invoice samples (no training needed)

Start using the system immediately in heuristic mode.
Every invoice processed generates OCR text you can use for training.

Target: collect **500 diverse invoices** before fine-tuning.

Diversity checklist:
- [ ] Indian GST invoices (CGST+SGST, IGST)
- [ ] E-commerce receipts (Amazon, Flipkart, Meesho)
- [ ] Utility bills (electricity, telecom)
- [ ] Service invoices (consulting, software)
- [ ] Scanned invoices (photocopied, mobile camera)
- [ ] Digital PDFs (exported from Tally, Zoho, QuickBooks)
- [ ] Handwritten elements (amounts, signatures)
- [ ] Multi-page invoices

---

## Phase 2: Annotate for YOLO (region detection)

### Setup Label Studio

```bash
pip install label-studio
label-studio start
```

Open http://localhost:8080, create a project, and choose **Object Detection with Bounding Boxes**.

### Annotation labels

Create exactly these 8 labels (names must match):
```
header
vendor_block
buyer_block
line_items
totals_block
tax_block
payment_terms
qr_barcode
```

### Export format

Export → YOLO format → download ZIP.
Extract to `data/annotations/`:
```
data/annotations/
├── dataset.yaml          ← create manually (see below)
├── images/
│   ├── train/
│   └── val/
└── labels/
    ├── train/
    └── val/
```

### Create dataset.yaml

```yaml
path: /absolute/path/to/invoice-digitizer/data/annotations
train: images/train
val: images/val
nc: 8
names:
  0: header
  1: vendor_block
  2: buyer_block
  3: line_items
  4: totals_block
  5: tax_block
  6: payment_terms
  7: qr_barcode
```

### Train YOLO

```bash
# CPU (slow but works)
python scripts/train_yolo.py --data data/annotations/dataset.yaml --epochs 100

# GPU (much faster — use Google Colab if no local GPU)
python scripts/train_yolo.py --data data/annotations/dataset.yaml --epochs 100 --device 0
```

Target: **mAP50 > 0.85** before using in production.

### Update config

```env
# .env
YOLO_MODEL_PATH=data/models/invoice_yolo.pt
```

---

## Phase 3: Annotate for LayoutLMv3 (field extraction)

LayoutLMv3 needs token-level BIO annotations.
This is more detailed than YOLO but dramatically improves field extraction.

### Annotation format

Each sample is a JSON file saved to `data/layoutlm_dataset/train/` or `/val/`:

```json
{
  "image_path": "data/raw/invoice_001.png",
  "words": ["INVOICE", "NO:", "INV-2024-001", "DATE:", "15/06/2024", ...],
  "boxes": [[10, 20, 80, 35], [82, 20, 110, 35], ...],
  "labels": ["O", "O", "B-INVOICE_NUMBER", "O", "B-INVOICE_DATE", ...]
}
```

Boxes must be normalised to 0–1000 range:
```python
norm_x1 = int(1000 * x1 / image_width)
norm_y1 = int(1000 * y1 / image_height)
```

Use PaddleOCR to auto-generate words+boxes, then add labels manually.

### BIO label scheme

```
O                   — Not a field (most tokens)
B-INVOICE_NUMBER    — First token of invoice number
I-INVOICE_NUMBER    — Continuation token
B-INVOICE_DATE
B-DUE_DATE
B-VENDOR_NAME       — First token of vendor name
I-VENDOR_NAME
B-VENDOR_ADDRESS
I-VENDOR_ADDRESS
B-VENDOR_GSTIN
B-BUYER_NAME
I-BUYER_NAME
B-BUYER_ADDRESS
I-BUYER_ADDRESS
B-BUYER_GSTIN
B-SUBTOTAL
B-TAX_AMOUNT
B-GRAND_TOTAL
B-LINE_ITEM_DESC
I-LINE_ITEM_DESC
B-LINE_ITEM_QTY
B-LINE_ITEM_RATE
B-LINE_ITEM_AMOUNT
```

### Train LayoutLMv3

```bash
# Recommended: run on Google Colab T4 GPU
python scripts/train_layoutlm.py \
  --data_dir data/layoutlm_dataset \
  --output_dir data/models/layoutlmv3-finetuned \
  --epochs 15 \
  --batch_size 2
```

### Update config

```env
LAYOUTLM_MODEL_PATH=data/models/layoutlmv3-finetuned
```

---

## Phase 4: Tune Ollama fallback

Edit `.env` to choose the best local model for your hardware:

| Model | RAM needed | Quality |
|---|---|---|
| `mistral` | 6 GB | Good |
| `llama3.1:8b` | 6 GB | Better |
| `llama3.1:70b` | 40 GB | Best (needs GPU) |
| `phi3:mini` | 2 GB | Fast, lower quality |

```bash
ollama pull mistral          # recommended default
ollama pull llama3.1:8b     # if you have 8+ GB RAM
```

---

## Phase 5: Tune confidence thresholds

After collecting data on real invoices, adjust thresholds in `.env`:

```env
CONFIDENCE_THRESHOLD=0.80    # below this → needs_review = true
LLM_FALLBACK_THRESHOLD=0.60  # below this → call Ollama for that field
```

Start conservative (0.80 / 0.60) and lower as you validate the system's accuracy on your invoice types.

---

## Google Colab training template

To train on Colab's free GPU:

1. Upload your `data/annotations/` folder to Google Drive
2. Open a new Colab notebook
3. Run:

```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Install deps
!pip install ultralytics transformers datasets

# Clone your repo or upload code
!git clone https://github.com/yourname/invoice-digitizer

# Train YOLO
!python invoice-digitizer/scripts/train_yolo.py \
  --data /content/drive/MyDrive/invoice_data/dataset.yaml \
  --epochs 100 \
  --device 0

# Download the model
from google.colab import files
files.download('invoice-digitizer/data/models/invoice_yolo.pt')
```
