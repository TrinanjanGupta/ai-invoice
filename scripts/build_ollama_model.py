"""
Build & Register Custom Ollama Model: invoice-expert

Reads data/models/Modelfile.invoice, verifies local Ollama daemon,
and compiles the domain-specialized 'invoice-expert' model into Ollama.
"""

import sys
import subprocess
import httpx
from pathlib import Path
from loguru import logger

OLLAMA_BASE_URL = "http://localhost:11434"
MODELFILE_PATH = Path("data/models/Modelfile.invoice")
MODEL_NAME = "invoice-expert"


def check_ollama_status() -> list[str]:
    """Checks if Ollama is running and lists installed models."""
    try:
        resp = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5.0)
        resp.raise_for_status()
        models = [m.get("name", "") for m in resp.json().get("models", [])]
        logger.info(f"Ollama is running. Installed models: {', '.join(models) if models else 'None'}")
        return models
    except Exception as e:
        logger.error(f"Cannot connect to Ollama at {OLLAMA_BASE_URL}: {e}")
        logger.error("Please ensure Ollama is running (run 'ollama serve' or open Ollama app).")
        return []


def select_best_base_model(installed_models: list[str]) -> str:
    """Finds the best base model installed locally in Ollama."""
    preference_order = [
        "qwen2.5:7b", "qwen2.5:latest", "qwen2.5",
        "mistral:7b", "mistral:latest", "mistral",
        "llama3.1:8b", "llama3:latest", "llama3",
        "phi3:medium", "gemma2:9b"
    ]
    for pref in preference_order:
        for m in installed_models:
            if m.startswith(pref) or pref in m:
                return m
    if installed_models:
        return installed_models[0]
    return "qwen2.5:7b"


def build_custom_model(base_model: str):
    """Compiles the custom Modelfile into Ollama."""
    if not MODELFILE_PATH.exists():
        logger.error(f"Modelfile not found at {MODELFILE_PATH}")
        sys.exit(1)

    # Update FROM line in Modelfile if needed
    content = MODELFILE_PATH.read_text(encoding="utf-8")
    lines = content.splitlines()
    updated_lines = []
    for line in lines:
        if line.strip().startswith("FROM "):
            updated_lines.append(f"FROM {base_model}")
        else:
            updated_lines.append(line)
    MODELFILE_PATH.write_text("\n".join(updated_lines), encoding="utf-8")
    logger.info(f"Updated {MODELFILE_PATH} to use base model: {base_model}")

    logger.info(f"Creating custom model '{MODEL_NAME}' in Ollama...")
    cmd = ["ollama", "create", MODEL_NAME, "-f", str(MODELFILE_PATH)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info(f"Model creation successful!\n{proc.stdout}")
        logger.info(f"'{MODEL_NAME}' is now active and ready for inference in the pipeline.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to create model: {e.stderr}")
        sys.exit(1)
    except FileNotFoundError:
        logger.error("'ollama' command not found in PATH. Make sure Ollama CLI is installed.")
        sys.exit(1)


def main():
    logger.info("=== Ollama Custom Invoice Model Builder ===")
    installed = check_ollama_status()
    if not installed:
        logger.warning("No models found or Ollama is offline. Ensure Ollama is running.")
        sys.exit(1)

    base = select_best_base_model(installed)
    logger.info(f"Selected base model for specialization: {base}")
    build_custom_model(base)


if __name__ == "__main__":
    main()
