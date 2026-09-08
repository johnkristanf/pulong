"""
02_train_lora.py

LoRA fine-tune OpenAI Whisper using QLoRA (4-bit quantization).
All parameters are optimized for RTX 3050 Laptop GPU (4GB VRAM).

Usage:
  uv run python scripts/02_train_lora.py
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import evaluate
import torch
from datasets import DatasetDict, load_from_disk
from transformers import (
    BitsAndBytesConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Model
MODEL_NAME = "openai/whisper-small"
LANGUAGE = "tl"  # Tagalog
TASK = "transcribe"

# Dataset
HF_DATASET_DIR = "dataset/hf_dataset"

# LoRA hyperparameters
LORA_R = 8             # LoRA rank -- low to save VRAM
LORA_ALPHA = 16        # Scaling factor (standard 2x rank)
LORA_DROPOUT = 0.05    # Light regularization
LORA_TARGET_MODULES = ["q_proj", "v_proj"]  # Attention projections only

# Training hyperparameters (optimized for 4GB VRAM)
OUTPUT_DIR = "models/whisper-small-lora"
PER_DEVICE_BATCH_SIZE = 1       # Minimum batch size
GRADIENT_ACCUMULATION = 8       # Effective batch size = 8
LEARNING_RATE = 1e-4
WARMUP_STEPS = 50
MAX_STEPS = 500                 # Adjust based on dataset size
EVAL_STEPS = 50
SAVE_STEPS = 100
LOGGING_STEPS = 10


# ---------------------------------------------------------------------------
# Data collator for Whisper
# ---------------------------------------------------------------------------
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """
    Custom data collator that pads input features and labels
    for Whisper sequence-to-sequence training.
    """
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: list[dict]) -> dict:
        # Split input features and labels
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        label_features = [
            {"input_ids": feature["labels"]} for feature in features
        ]

        # Pad input features
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # Pad labels
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )

        # Replace padding token id with -100 so it's ignored by loss
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # Remove BOS token if it was appended during tokenization
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------
def prepare_dataset(batch: dict, processor: WhisperProcessor) -> dict:
    """Process a single example: extract features and tokenize labels."""
    audio = batch["audio"]

    # Extract log-Mel spectrogram features
    batch["input_features"] = processor.feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
    ).input_features[0]

    # Tokenize the transcription text
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids

    return batch


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(pred, tokenizer, metric):
    """Compute Word Error Rate (WER) on evaluation predictions."""
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # Replace -100 with pad token id for decoding
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    wer = 100 * metric.compute(predictions=pred_str, references=label_str)
    return {"wer": wer}


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 60)
    print("Whisper LoRA Fine-Tuning (QLoRA -- 4-bit)")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Load processor
    # ------------------------------------------------------------------
    print("\n[1/6] Loading processor...")
    processor = WhisperProcessor.from_pretrained(
        MODEL_NAME, language=LANGUAGE, task=TASK
    )

    # ------------------------------------------------------------------
    # 2. Load model with 4-bit quantization (QLoRA)
    # ------------------------------------------------------------------
    print("\n[2/6] Loading model with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )

    # Disable cache (incompatible with gradient checkpointing)
    model.config.use_cache = False

    # Set forced decoder IDs for language and task
    model.generation_config.language = LANGUAGE
    model.generation_config.task = TASK
    model.generation_config.forced_decoder_ids = None

    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)

    # ------------------------------------------------------------------
    # 3. Apply LoRA
    # ------------------------------------------------------------------
    print("\n[3/6] Applying LoRA adapters...")
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ------------------------------------------------------------------
    # 4. Load and preprocess dataset
    # ------------------------------------------------------------------
    print("\n[4/6] Loading dataset...")
    dataset_path = Path(HF_DATASET_DIR)
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {HF_DATASET_DIR}. "
            "Run 01_prepare_dataset.py first."
        )

    dataset = load_from_disk(str(dataset_path))
    assert isinstance(dataset, DatasetDict), "Expected a DatasetDict with train/test splits"

    print(f"  Train: {len(dataset['train'])} samples")
    print(f"  Eval:  {len(dataset['test'])} samples")

    # Preprocess: extract features and tokenize labels
    print("  Preprocessing dataset...")
    dataset = dataset.map(
        lambda batch: prepare_dataset(batch, processor),
        remove_columns=dataset["train"].column_names,
        num_proc=1,  # Use 1 to avoid memory spikes
    )

    # ------------------------------------------------------------------
    # 5. Set up training
    # ------------------------------------------------------------------
    print("\n[5/6] Setting up trainer...")

    # Data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    # WER metric
    wer_metric = evaluate.load("wer")

    # Training arguments (optimized for 4GB VRAM)
    training_args = Seq2SeqTrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        max_steps=MAX_STEPS,
        fp16=True,
        gradient_checkpointing=True,
        optim="adamw_bnb_8bit",
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        logging_steps=LOGGING_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=225,
        report_to="none",
        push_to_hub=False,
        save_total_limit=3,
        dataloader_num_workers=0,  # Avoid multiprocessing memory overhead
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        processing_class=processor.feature_extractor,
        compute_metrics=lambda pred: compute_metrics(
            pred, processor.tokenizer, wer_metric
        ),
    )

    # ------------------------------------------------------------------
    # 6. Train
    # ------------------------------------------------------------------
    print("\n[6/6] Starting training...")
    print(f"  Effective batch size: {PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION}")
    print(f"  Max steps: {MAX_STEPS}")
    print(f"  Learning rate: {LEARNING_RATE}")

    # Monitor VRAM
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        print(f"  VRAM allocated: {allocated:.2f} GB")
        print(f"  VRAM reserved:  {reserved:.2f} GB")

    trainer.train()

    # Save final adapter
    print(f"\n  Saving LoRA adapter to {OUTPUT_DIR}...")
    model.save_pretrained(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"LoRA adapter saved to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

