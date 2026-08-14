# Invoice Digitizer — 100% Free & Open Source

A production-grade hybrid pipeline that converts any invoice image or PDF into a structured digital invoice. Zero API costs. Runs fully offline after setup.

## Architecture

```
Input (PDF/Image)
  → Pre-processing (OpenCV)
  → Region Detection (YOLOv8)
  → OCR (PaddleOCR)
  → Layout Understanding (LayoutLMv3 / HuggingFace)
  → LLM Fallback (Ollama + Mistral 7B, for low-confidence fields only)
  → Validation (Pydantic rules engine)
  → Output (JSON / PDF / HTML)
```

## Tech Stack (all free & open source)

| Layer | Tool | License |
|---|---|---|
| API | FastAPI + Uvicorn | MIT |
| Task queue | Celery + Redis | BSD |
| Pre-processing | OpenCV + Pillow | BSD / HPND |
| PDF rasterising | PyMuPDF | AGPL |
| Region detection | YOLOv8 (Ultralytics) | AGPL |
| OCR | PaddleOCR v3 | Apache 2.0 |
| Layout AI | LayoutLMv3 (HuggingFace) | MIT |
| LLM fallback | Ollama + Mistral 7B | MIT |
| Validation | Pydantic v2 | MIT |
| Database | PostgreSQL | PostgreSQL |
| Object storage | MinIO | AGPL |
| PDF output | WeasyPrint | BSD |
| Frontend | React + Vite + Tailwind | MIT |

## Quick Start

### 1. Prerequisites

- Python 3.14+
- Node.js 18+
- Redis (for task queue)
- PostgreSQL (for storage)
- Ollama (for LLM fallback) — https://ollama.com/install.sh
- (Optional) GPU for faster inference

### 2. Install Ollama and pull model

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull mistral
```

### 3. Backend setup

```bash
cd invoice-digitizer
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Configure environment

```bash
cp .env.example .env
# Edit .env with your DB credentials
```

### 5. Start services

```bash
# Terminal 1: API server
uvicorn api.main:app --reload --port 8000

# Terminal 2: Celery worker (Linux/macOS)
celery -A worker.celery_app worker --loglevel=info
# On Windows PowerShell:
celery -A worker.celery_app worker --loglevel=info --pool=solo

# Terminal 3: Frontend
cd review_ui && npm install && npm run dev
```

### 6. Open browser

- API docs: http://localhost:8000/docs
- Review UI: http://localhost:5173

## Training YOLO on your invoices

```bash
python scripts/train_yolo.py --data data/annotations/dataset.yaml --epochs 100
```

See `docs/training_guide.md` for full annotation and training instructions.

## Project Structure

```
invoice-digitizer/
├── api/                  FastAPI application
│   ├── main.py           App entry point + routes
│   ├── models.py         Request/response schemas
│   └── dependencies.py   Shared dependencies
├── preprocessing/        OpenCV pre-processing
│   └── pipeline.py
├── detection/            YOLOv8 region detection
│   └── detector.py
├── ocr/                  PaddleOCR wrapper
│   └── extractor.py
├── understanding/        LayoutLMv3 field extraction
│   └── layoutlm.py
├── llm_fallback/         Ollama integration
│   └── ollama_client.py
├── validation/           Pydantic schemas + rules
│   └── validator.py
├── worker/               Celery task definitions
│   └── tasks.py
├── output/               PDF + JSON exporters
│   └── renderer.py
├── storage/              DB + MinIO clients
│   └── db.py
├── review_ui/            React frontend
│   └── src/
├── scripts/              Training + utility scripts
├── data/                 Datasets and model weights
├── config/               App configuration
└── tests/                Test suite
```
