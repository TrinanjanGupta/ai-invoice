"""
Fine-Tune LLM with LoRA on Verified Invoice Datasets (Method 4)

Uses Hugging Face PEFT (LoRA) to fine-tune an instruction model (e.g. Qwen2.5-7B, Mistral-7B, Llama-3-8B)
on your verified invoice JSONL datasets.
"""

import argparse
import sys
from pathlib import Path
from loguru import logger


def train_lora(
    data_dir: str,
    base_model: str,
    output_dir: str,
    epochs: int = 3,
    batch_size: int = 2,
    learning_rate: float = 2e-4,
    lora_r: int = 16,
    lora_alpha: int = 32,
):
    import torch
    from datasets import load_dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        DataCollatorForSeq2Seq,
    )
    from peft import LoraConfig, get_peft_model, TaskType

    data_path = Path(data_dir)
    train_file = data_path / "train.jsonl"
    val_file = data_path / "val.jsonl"

    if not train_file.exists():
        logger.error(f"Training dataset not found at {train_file}. Run scripts/export_reviewed_to_llm.py first.")
        sys.exit(1)

    logger.info(f"Loading base tokenizer and model: {base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    data_files = {"train": str(train_file)}
    if val_file.exists():
        data_files["validation"] = str(val_file)

    dataset = load_dataset("json", data_files=data_files)
    logger.info(f"Loaded dataset: {dataset}")

    def formatting_prompts_func(example):
        prompt = (
            f"<|im_start|>system\n{example['instruction']}<|im_end|>\n"
            f"<|im_start|>user\n{example['input']}<|im_end|>\n"
            f"<|im_start|>assistant\n{example['output']}<|im_end|>"
        )
        enc = tokenizer(prompt, truncation=True, max_length=2048)
        enc["labels"] = enc["input_ids"].copy()
        return enc

    tokenized_dataset = dataset.map(formatting_prompts_func, remove_columns=dataset["train"].column_names)

    device_map = "auto" if torch.cuda.is_available() else None
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    logger.info("Instantiating base model...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    # Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        logging_steps=5,
        save_strategy="epoch",
        evaluation_strategy="epoch" if "validation" in tokenized_dataset else "no",
        fp16=torch.cuda.is_available(),
        save_total_limit=2,
        report_to="none",
    )

    from transformers import Trainer

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset.get("validation"),
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True),
    )

    logger.info("Starting LoRA fine-tuning...")
    trainer.train()

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_path))
    tokenizer.save_pretrained(str(out_path))
    logger.info(f"LoRA fine-tuning complete! Weights saved to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune LLM on verified invoices using LoRA.")
    parser.add_argument("--data-dir", default="data/llm_dataset", help="Path to jsonl datasets")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct", help="Hugging Face base model")
    parser.add_argument("--output-dir", default="data/models/invoice_llm_lora", help="Output directory for LoRA weights")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size per device")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")

    args = parser.parse_args()
    train_lora(
        data_dir=args.data_dir,
        base_model=args.base_model,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )


if __name__ == "__main__":
    main()
