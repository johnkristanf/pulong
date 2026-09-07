"""
03_merge_and_convert.py

Merge LoRA weights back into the base Whisper model, then convert
to CTranslate2 format for faster-whisper inference.

Steps:
  1. Load base whisper-small model (CPU, full precision)
  2. Load LoRA adapter from training checkpoint
  3. Merge weights with merge_and_unload()
  4. Save merged model
  5. Convert to CTranslate2 format via ct2-transformers-converter

Usage:
  uv run python scripts/03_merge_and_convert.py
"""

import shutil
import subprocess
import sys
from pathlib import Path

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from peft import PeftModel


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_NAME = "openai/whisper-small"
LORA_ADAPTER_DIR = "models/whisper-small-lora"
MERGED_MODEL_DIR = "models/whisper-small-merged"
CT2_MODEL_DIR = "models/whisper-small-ct2"
CT2_QUANTIZATION = "float16"  # Options: float16, int8, int8_float16


def merge_lora() -> None:
    """Load base model + LoRA adapter, merge weights, and save."""
    print("\n[1/2] Merging LoRA weights into base model...")

    # Verify adapter exists
    adapter_path = Path(LORA_ADAPTER_DIR)
    if not adapter_path.exists():
        print(f"  ERROR: LoRA adapter not found at {LORA_ADAPTER_DIR}")
        print("  Run 02_train_lora.py first.")
        sys.exit(1)

    # Load base model on CPU (full precision for clean merge)
    print(f"  Loading base model: {MODEL_NAME}")
    base_model = WhisperForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="cpu",
    )

    # Load processor
    print(f"  Loading processor...")
    processor = WhisperProcessor.from_pretrained(MODEL_NAME)

    # Load and apply LoRA adapter
    print(f"  Loading LoRA adapter from: {LORA_ADAPTER_DIR}")
    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_DIR)

    # Merge LoRA weights into base model
    print("  Merging weights...")
    merged_model = model.merge_and_unload()

    # Save merged model
    merged_path = Path(MERGED_MODEL_DIR)
    merged_path.mkdir(parents=True, exist_ok=True)

    print(f"  Saving merged model to: {MERGED_MODEL_DIR}")
    merged_model.save_pretrained(MERGED_MODEL_DIR)
    processor.save_pretrained(MERGED_MODEL_DIR)

    print("  Merge complete!")


def convert_to_ct2() -> None:
    """Convert merged Whisper model to CTranslate2 format."""
    print("\n[2/2] Converting to CTranslate2 format...")

    merged_path = Path(MERGED_MODEL_DIR)
    if not merged_path.exists():
        print(f"  ERROR: Merged model not found at {MERGED_MODEL_DIR}")
        sys.exit(1)

    ct2_path = Path(CT2_MODEL_DIR)
    ct2_path.mkdir(parents=True, exist_ok=True)

    # Build the ct2-transformers-converter command
    cmd = [
        sys.executable, "-m", "ctranslate2.converters.transformers",
        "--model", MERGED_MODEL_DIR,
        "--output_dir", CT2_MODEL_DIR,
        "--quantization", CT2_QUANTIZATION,
        "--force",
    ]

    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        # Fallback: try the CLI tool directly
        print("  Python module approach failed, trying CLI...")
        cmd_cli = [
            "ct2-transformers-converter",
            "--model", MERGED_MODEL_DIR,
            "--output_dir", CT2_MODEL_DIR,
            "--quantization", CT2_QUANTIZATION,
            "--force",
        ]
        print(f"  Running: {' '.join(cmd_cli)}")
        result = subprocess.run(cmd_cli, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"  STDERR: {result.stderr}")
            print("  ERROR: CTranslate2 conversion failed.")
            sys.exit(1)

    print(f"  Conversion output: {result.stdout}")

    # Copy necessary config files that faster-whisper needs
    files_to_copy = [
        "tokenizer.json",
        "preprocessor_config.json",
        "added_tokens.json",
        "special_tokens_map.json",
        "normalizer.json",
        "vocab.json",
        "merges.txt",
    ]

    for filename in files_to_copy:
        src = merged_path / filename
        if src.exists():
            dst = ct2_path / filename
            shutil.copy2(str(src), str(dst))
            print(f"  Copied: {filename}")

    print(f"\n  CTranslate2 model saved to: {CT2_MODEL_DIR}")

    # List output files
    print("\n  Output files:")
    for f in sorted(ct2_path.iterdir()):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"    {f.name} ({size_mb:.1f} MB)")


def main() -> None:
    print("=" * 60)
    print("Whisper LoRA Merge + CTranslate2 Conversion")
    print("=" * 60)

    merge_lora()
    convert_to_ct2()

    print("\n" + "=" * 60)
    print("Done! Your model is ready for faster-whisper inference.")
    print(f"Model path: {CT2_MODEL_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()

