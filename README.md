# AI Invoice Digitizer — Intelligent Multi-Page Extraction Pipeline

A production-grade, 100% free & open-source hybrid AI pipeline that converts multi-page invoice PDFs and scanned images into structured, canonicalized digital records. Runs fully offline with zero API costs.

---

## 🏗️ Architecture Overview

```mermaid
flowchart TD
    A[Input: Multi-Page PDF / Scanned Image] --> B[PDF Converter & OpenCV Preprocessing]
    B --> C[YOLOv8 Macro Region Detection]
    C --> D[PaddleOCR / EasyOCR Regional & Full Extraction]
    D --> E{Multi-Engine Ensemble Fusion}
    
    subgraph AI Extraction Ensemble
        E --> F[LayoutLMv3 Spatial Token Classifier]
        E --> G[Heuristic Geometric & Regex Engine]
        E --> H[Ollama Llama 3.1 / Mistral LLM Fallback]
    end

    F --> I[Confidence-Weighted Field Merger]
    G --> I
    H --> I

    I --> J[Multi-Page Table & Totals Aggregator]
    J --> K[Pydantic Business Rules & Tax Validator]
    K --> L[PostgreSQL Storage & MinIO Object Store]
    L --> M[Human Review UI & In-App Retraining Loop]
```

---

## ✨ Key Capabilities & Recent Enhancements

### 1. 📄 Multi-Page PDF Extraction
* Iterates through **all pages** of multi-page invoices.
* Performs YOLO macro region detection and OCR across all pages.
* Automatically merges multi-page item tables and extracts grand totals, taxes, and bank details located on later pages.

### 2. 🛡️ Multi-Engine Ensemble Fusion (`_merge_invoices`)
* Combines **LayoutLMv3** (spatial transformer) + **PaddleOCR** + **Deterministic Heuristic Regex** + **Local LLM Fallback (Ollama)**.
* **Confidence-Weighted Winner Selection**: Prioritizes the highest-confidence extraction for every individual field to eliminate blank/null values.
* Subword BPE alignment (`encoding.word_ids(0)`) prevents token desynchronization.

### 3. 🖥️ Modern React Review UI (Split-View & Wizard)
* **Guided Wizard & Standard Form Modes**: Fast verification and step-by-step review.
* **High-Precision Document Viewer**: Smooth bidirectional panning and zoom (40% to 350%) without cropping or overflow issues.
* **Live Document Paging**: Multi-page selector for original PDFs and rendered HTML previews.
* **One-Click Export**: Copy JSON for form integration (`invoiceForm.patchValue`) and instant PDF downloads.

### 4. 🔁 Continuous Learning & Model Retraining Loop
* **LayoutLMv3 Fine-Tuning**: Automatically exports verified invoice corrections into BIO token datasets to improve field accuracy toward **90%+**.
* **YOLOv8 Region Retraining**: Retrains visual bounding boxes on newly introduced invoice layouts.

---

## 🧰 Tech Stack

| Layer | Technology | License |
|---|---|---|
| **API Backend** | FastAPI + Uvicorn (Python 3.12 / 3.13) | MIT |
| **Task Queue** | Celery + Redis | BSD |
| **Image Preprocessing** | OpenCV + Pillow | BSD / HPND |
| **PDF Rasterization** | PyMuPDF (`pymupdf`) | AGPL |
| **Macro Region Detection** | YOLOv8 (`ultralytics`) | AGPL |
| **OCR Engine** | PaddleOCR v3 / PaddleX + EasyOCR Fallback | Apache 2.0 |
| **Spatial Layout AI** | LayoutLMv3 (`transformers`, HuggingFace) | MIT |
| **LLM Fallback** | Ollama (Llama 3.1 / Mistral) | MIT |
| **Data Validation** | Pydantic v2 | MIT |
| **Database** | PostgreSQL (SQLAlchemy ORM) | PostgreSQL |
| **Object Storage** | MinIO | AGPL |
| **Frontend UI** | React + Vite + Tailwind CSS + Lucide Icons | MIT |

---

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.12+ (or Python 3.13 / 3.14)
- Node.js 18+ & npm
- Redis Server (for background tasks)
- PostgreSQL Server
- Ollama (Optional, for zero-shot LLM fallback) — [https://ollama.com](https://ollama.com)

### 2. Ollama Setup (Optional Fallback)

```bash
# Pull model for local extraction fallback
ollama pull llama3.1
# or
ollama pull mistral
```

### 3. Python Environment Setup

```bash
# Clone or navigate to the directory
cd ai-invoice

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Environment Configuration

```bash
cp .env.example .env
# Edit .env to set your DB credentials, Redis URL, and MinIO endpoints
```

### 5. Launch Services

#### Terminal 1: FastAPI Server
```bash
uvicorn api.main:app --reload --reload-dir api --host 0.0.0.0
```

#### Terminal 2: Celery Worker
```bash
# Windows:
celery -A worker.celery_app worker --loglevel=info --pool=solo

# Linux/macOS:
celery -A worker.celery_app worker --loglevel=info
```

#### Terminal 3: React Review UI
```bash
cd review_ui
npm install
npm run dev
```

### 6. Access Web Interfaces

* **Review UI**: [http://localhost:5173](http://localhost:5173)
* **Interactive API Documentation (Swagger)**: [http://localhost:8000/docs](http://localhost:8000/docs)
* **ReDoc API Documentation**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 🎯 Active Learning & Model Training Guide

The system uses a **Human-in-the-Loop Active Learning Flywheel** designed to scale from 30 verified invoices to thousands with minimal human review effort.

```text
                 NEW INVOICE
                      ↓
                  Current AI
                      ↓
       Multi-Evidence Confidence Engine
         (OCR + Checksums + Arithmetic)
                ↙            ↘
        HIGH CONFIDENCE     UNCERTAIN / MISMATCH
             ↓                       ↓
     AUTO ACCEPT (≥85%)    PRIORITIZED REVIEW QUEUE
             ↓                       ↓
             └───────┬───────────────┘
                     ↓
      Track Human Corrections (Diffs)
                     ↓
     Self-Contained Dataset Generation
                     ↓
         LayoutLMv3 Fine-Tuning
                     ↓
       Trained Model → Less Review Needed
```

---

### ⚡ Step-by-Step Training Workflow

| Step | Action | Command | Estimated Time |
|---|---|---|---|
| **1. Auto-Accept** | Batch approve high-confidence ($\ge 0.85$) invoices | `python scripts/auto_accept_high_confidence.py --min-confidence 0.85` | **~0.05 seconds** |
| **2. Export Dataset** | Generate self-contained LayoutLM dataset + QA validation | `python scripts/export_reviewed_to_layoutlm.py --output-dir data/layoutlm_dataset --val-ratio 0.2` | **~1.5 minutes** (for 31 samples) |
| **3. Visual Inspection** | Verify token bounding boxes on generated image samples | `python scripts/visualize_layoutlm_sample.py --sample data/layoutlm_dataset/train/<job_id>.json` | **~0.5 seconds** |
| **4. Benchmark Baseline** | Measure pre-training extraction accuracy scorecard | `python scripts/benchmark_accuracy.py --limit 31` | **~1 minute** |
| **5. Train LayoutLMv3** | Fine-tune spatial token classification transformer | `python scripts/train_layoutlm.py --data_dir data/layoutlm_dataset --epochs 10` | **~3 to 5 minutes** |

---

### Detailed Step Instructions

#### Step 1: Batch Auto-Accept High-Confidence Invoices
Eliminates manual review of standard invoices with valid math and high confidence:
```powershell
python scripts/auto_accept_high_confidence.py --min-confidence 0.85
```

#### Step 2: Export Ground Truth to LayoutLM Format
Extracts pixel-grounded word coordinates, aligns them to verified semantic fields using the focused 24-field header/financial taxonomy, bundles images into `data/layoutlm_dataset/images/`, and runs automated QA validation:
```powershell
python scripts/export_reviewed_to_layoutlm.py --output-dir data/layoutlm_dataset --val-ratio 0.2
```
*Creates:*
```text
data/layoutlm_dataset/
├── train/          # JSON files with words, boxes, BIO labels
├── val/            # 20% validation split JSON files
├── images/         # Bundled page PNG images (self-contained & portable)
└── metadata.json   # Dataset metrics & alignment statistics
```

#### Step 3: Visually Verify Token Labels (Optional)
Renders colored bounding boxes for labeled tokens on top of the original invoice image:
```powershell
python scripts/visualize_layoutlm_sample.py --sample data/layoutlm_dataset/train/ceddc6c6-9705-4cb1-99bb-f05dc4d0d094.json
```
*(Outputs annotated PNG to `output/debug_viz/`)*.

#### Step 4: Establish Accuracy Benchmark
Computes exact field-by-field precision, recall, and accuracy against ground truth:
```powershell
python scripts/benchmark_accuracy.py --limit 31
```

#### Step 5: Fine-Tune LayoutLMv3
Trains the spatial transformer on the exported dataset:
```powershell
python scripts/train_layoutlm.py --data_dir data/layoutlm_dataset --epochs 10
```

---

### Active Learning APIs

The FastAPI backend exposes endpoints for active learning queue integration:
- `GET /api/active-learning/queue?limit=50`: Returns pending invoices ranked by **Informativeness Score** (uncertainty + arithmetic errors + model divergence).
- `POST /api/active-learning/auto-accept`: Batch auto-approves all invoices with $\ge 0.85$ confidence.
- `GET /api/active-learning/stats`: Real-time tracking of auto-accepted, human-confirmed, and human-corrected dataset pools.

---

## 📁 Repository Structure

```
ai-invoice/
├── api/                          # FastAPI application & pipeline orchestrator
│   ├── main.py                   # REST endpoints, background tasks & model retraining
│   ├── pipeline_runner.py        # Multi-page extraction pipeline & ensemble runner
│   ├── models.py                 # Pydantic schemas & response models
│   └── dependencies.py           # Shared dependency injectors
├── preprocessing/                # OpenCV image processing & deskewing
│   ├── pipeline.py               # CLAHE ink enhancement, binarization, noise reduction, deskew
│   └── pdf_converter.py          # PyMuPDF rasterization & image conversion
├── detection/                    # Pretrained DocLayout-YOLO macro region detector
│   └── detector.py               # Table, Header, Footer, Text layout bounding boxes
├── ocr/                          # Optical character recognition
│   └── extractor.py              # PaddleOCR wrapper with EasyOCR fallback
├── understanding/                # Spatial entity extraction
│   └── layoutlm.py               # LayoutLMv3 token classifier + Ensemble Merger
├── llm_fallback/                 # Local LLM batch entity extractor
│   └── ollama_client.py          # Dynamic Few-Shot In-Context learning & invoice-expert client
├── validation/                   # Pydantic rules engine & format canonicalizer
│   └── validator.py              # GSTIN format, tax math, dates & item validation
├── worker/                       # Celery asynchronous task definitions
│   └── tasks.py                  # Background job queue processing
├── output/                       # Output generators
│   └── renderer.py               # HTML template & WeasyPrint PDF renderer
├── storage/                      # Persistence layer
│   └── db.py                     # PostgreSQL models & SQLAlchemy async session
├── review_ui/                    # React + Vite Human-in-the-Loop Review UI
│   ├── src/pages/ReviewPage.jsx  # Split-view & Guided Wizard editor
│   ├── src/pages/InvoiceListPage.jsx # Invoices dashboard & upload manager
│   └── src/index.css             # Tailored styling & design system
├── scripts/                      # Training & migration CLI utilities
│   ├── build_ollama_model.py     # Custom Ollama Modelfile compiler
│   ├── export_reviewed_to_llm.py # Ground-truth exporter for LLM fine-tuning
│   ├── train_llm_lora.py         # LoRA / PEFT fine-tuning script
│   ├── export_reviewed_to_layoutlm.py # Ground-truth exporter for BIO tagging
│   ├── train_layoutlm.py         # LayoutLMv3 fine-tuning script
│   └── train_yolo.py             # DocLayout-YOLO fine-tuning script
├── data/                         # Local models, annotations & datasets
│   ├── models/                   # Active YOLO (.pt) & LayoutLM model weights
│   └── raw/                      # Sample input documents
└── tests/                        # Pytest suite
```

---

## 📜 License

This project is licensed under the **MIT License**. All bundled and integrated dependencies (PaddleOCR, YOLOv8, LayoutLMv3, PyMuPDF, OpenCV, Ollama) are open-source and free for development and deployment.
