"""
04_test_inference.py

Test the fine-tuned CTranslate2 Whisper model with faster-whisper.
Optionally compares output against the base (non-fine-tuned) model.

Usage:
  uv run python scripts/04_test_inference.py
  uv run python scripts/04_test_inference.py --audio path/to/audio.m4a
  uv run python scripts/04_test_inference.py --compare
"""

import argparse
import sys
import time
from pathlib import Path

from faster_whisper import WhisperModel


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CT2_MODEL_DIR = "models/whisper-small-ct2"
BASE_MODEL_SIZE = "small"
DEVICE = "cuda"
COMPUTE_TYPE = "int8_float16"
BEAM_SIZE = 5
LANGUAGE = "tl"  # Tagalog

# Default test audio (first chunk from dataset if available)
DEFAULT_TEST_AUDIO = "dataset/audio"


def find_test_audio() -> str:
    """Find a test audio file to use for inference."""
    # Try original audio directory at project root first
    audio_dir = Path("audio")
    if audio_dir.exists():
        for ext in ["*.wav", "*.m4a", "*.mp3", "*.flac", "*.ogg"]:
            files = sorted(audio_dir.glob(ext))
            if files:
                return str(files[0])

    # Fall back to dataset chunks
    dataset_audio = Path(DEFAULT_TEST_AUDIO)
    if dataset_audio.exists():
        wav_files = sorted(dataset_audio.glob("*.wav"))
        if wav_files:
            return str(wav_files[0])

    return ""


def transcribe_with_model(
    model_path: str, audio_path: str, label: str
) -> tuple[str, float]:
    """Transcribe audio and return (text, elapsed_seconds)."""
    print(f"\n{'─' * 50}")
    print(f"  Model: {label}")
    print(f"  Audio: {audio_path}")
    print(f"{'─' * 50}")

    model = WhisperModel(model_path, device=DEVICE, compute_type=COMPUTE_TYPE)

    start = time.perf_counter()
    segments, info = model.transcribe(
        audio_path,
        beam_size=BEAM_SIZE,
        language=LANGUAGE,
    )

    texts = []
    for segment in segments:
        text = segment.text.strip()
        print(f"  [{segment.start:6.1f}s -> {segment.end:6.1f}s]  {text}")
        if text:
            texts.append(text)

    elapsed = time.perf_counter() - start
    full_text = " ".join(texts)

    print(f"\n  Language: {info.language} (p={info.language_probability:.2f})")
    print(f"  Time: {elapsed:.2f}s")

    return full_text, elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Test fine-tuned Whisper model")
    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Path to audio file to transcribe",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare fine-tuned model against base model",
    )
    args = parser.parse_args()

    # Find audio file
    audio_path = args.audio or find_test_audio()
    if not audio_path or not Path(audio_path).exists():
        print("ERROR: No test audio file found.")
        print("  Provide one with: --audio path/to/file.wav")
        sys.exit(1)

    # Check that CT2 model exists
    ct2_path = Path(CT2_MODEL_DIR)
    if not ct2_path.exists():
        print(f"ERROR: Fine-tuned model not found at {CT2_MODEL_DIR}")
        print("  Run 03_merge_and_convert.py first.")
        sys.exit(1)

    print("=" * 60)
    print("Whisper Inference Test")
    print("=" * 60)

    # Transcribe with fine-tuned model
    finetuned_text, finetuned_time = transcribe_with_model(
        CT2_MODEL_DIR, audio_path, f"Fine-tuned ({CT2_MODEL_DIR})"
    )

    # Optionally compare with base model
    if args.compare:
        base_text, base_time = transcribe_with_model(
            BASE_MODEL_SIZE, audio_path, f"Base ({BASE_MODEL_SIZE})"
        )

        print("\n" + "=" * 60)
        print("Comparison Summary")
        print("=" * 60)
        print(f"\n  Base model time:       {base_time:.2f}s")
        print(f"  Fine-tuned model time: {finetuned_time:.2f}s")
        print(f"  Speedup: {base_time / finetuned_time:.2f}x")
        print(f"\n  Base output ({len(base_text)} chars):")
        print(f"    {base_text[:200]}...")
        print(f"\n  Fine-tuned output ({len(finetuned_text)} chars):")
        print(f"    {finetuned_text[:200]}...")

    print("\n" + "=" * 60)
    print("Inference test complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

