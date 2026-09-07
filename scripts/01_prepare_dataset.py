"""
01_prepare_dataset.py

Prepare a Hugging Face dataset from long-form audio files for Whisper fine-tuning.

Strategy:
  1. Read metadata.jsonl (audio path + full transcription)
  2. Split each audio file into ~30-second non-overlapping chunks
  3. Use base Whisper (via faster-whisper) to pseudo-label each chunk
     (since original metadata has only full-file transcriptions)
  4. Save chunked audio + labels as a HF Dataset

Usage:
  uv run python scripts/01_prepare_dataset.py
"""

import json
import math
import sys
from pathlib import Path

from pydub import AudioSegment


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
METADATA_FILE = "metadata.jsonl"
OUTPUT_DIR = Path("dataset")
AUDIO_OUTPUT_DIR = OUTPUT_DIR / "audio"
HF_DATASET_DIR = OUTPUT_DIR / "hf_dataset"
CHUNK_DURATION_MS = 30_000  # 30 seconds per chunk
SAMPLE_RATE = 16_000  # Whisper expects 16kHz
WHISPER_MODEL_SIZE = "small"  # Model used for pseudo-labeling
LANGUAGE = "tl"  # Tagalog


def load_metadata(metadata_path: str) -> list[dict]:
    """Load audio paths and transcriptions from metadata.jsonl."""
    entries = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def chunk_audio(audio_path: str, chunk_ms: int = CHUNK_DURATION_MS) -> list[Path]:
    """
    Split an audio file into fixed-length chunks.

    Returns a list of paths to the saved chunk files (WAV, 16kHz mono).
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        print(f"  [SKIP] Audio file not found: {audio_path}")
        return []

    print(f"  Loading: {audio_path}")
    audio = AudioSegment.from_file(str(audio_path))

    # Convert to 16kHz mono for Whisper
    audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1)

    total_ms = len(audio)
    num_chunks = math.ceil(total_ms / chunk_ms)
    stem = audio_path.stem

    chunk_paths = []
    for i in range(num_chunks):
        start = i * chunk_ms
        end = min((i + 1) * chunk_ms, total_ms)
        chunk = audio[start:end]

        # Skip very short chunks (less than 1 second)
        if len(chunk) < 1000:
            continue

        chunk_filename = f"{stem}_chunk_{i:04d}.wav"
        chunk_path = AUDIO_OUTPUT_DIR / chunk_filename
        chunk.export(str(chunk_path), format="wav")
        chunk_paths.append(chunk_path)

    print(f"  Created {len(chunk_paths)} chunks from {audio_path.name}")
    return chunk_paths


def pseudo_label_chunks(chunk_paths: list[Path]) -> list[dict]:
    """
    Use faster-whisper to transcribe each audio chunk.

    This provides per-chunk labels for fine-tuning since the original
    metadata only has full-file transcriptions.
    """
    from faster_whisper import WhisperModel

    print(f"\n  Loading Whisper ({WHISPER_MODEL_SIZE}) for pseudo-labeling...")
    model = WhisperModel(WHISPER_MODEL_SIZE, device="cuda", compute_type="int8_float16")

    labeled_data = []
    for i, chunk_path in enumerate(chunk_paths):
        segments, info = model.transcribe(
            str(chunk_path),
            beam_size=5,
            language=LANGUAGE,
        )

        text = " ".join(seg.text.strip() for seg in segments).strip()

        if text:
            labeled_data.append({
                "audio": str(chunk_path),
                "text": text,
                "language": LANGUAGE,
            })

        if (i + 1) % 10 == 0 or (i + 1) == len(chunk_paths):
            print(f"  Pseudo-labeled {i + 1}/{len(chunk_paths)} chunks")

    return labeled_data


def save_dataset(labeled_data: list[dict]) -> None:
    """Save labeled data as JSONL and as a HF Dataset."""
    # Save JSONL
    jsonl_path = OUTPUT_DIR / "train.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for entry in labeled_data:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"\n  Saved {len(labeled_data)} entries to {jsonl_path}")

    # Build HF Dataset
    from datasets import Audio, Dataset

    dataset = Dataset.from_list(labeled_data)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))

    # Train/eval split (90/10)
    split = dataset.train_test_split(test_size=0.1, seed=42)
    split.save_to_disk(str(HF_DATASET_DIR))
    print(f"  Saved HF Dataset to {HF_DATASET_DIR}/")
    print(f"    Train: {len(split['train'])} samples")
    print(f"    Eval:  {len(split['test'])} samples")


def main() -> None:
    print("=" * 60)
    print("Whisper Dataset Preparation")
    print("=" * 60)

    # Create output directories
    AUDIO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load metadata
    print("\n[1/3] Loading metadata...")
    entries = load_metadata(METADATA_FILE)
    print(f"  Found {len(entries)} audio files")

    if not entries:
        print("  No entries found. Exiting.")
        sys.exit(1)

    # Step 2: Chunk audio files
    print("\n[2/3] Chunking audio files into ~30s segments...")
    all_chunk_paths = []
    for entry in entries:
        chunk_paths = chunk_audio(entry["audio"])
        all_chunk_paths.extend(chunk_paths)

    print(f"\n  Total chunks: {len(all_chunk_paths)}")

    if not all_chunk_paths:
        print("  No chunks created. Check that audio files exist. Exiting.")
        sys.exit(1)

    # Step 3: Pseudo-label each chunk
    print("\n[3/3] Pseudo-labeling chunks with Whisper...")
    labeled_data = pseudo_label_chunks(all_chunk_paths)

    # Save dataset
    print("\n  Saving dataset...")
    save_dataset(labeled_data)

    print("\n" + "=" * 60)
    print("Dataset preparation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

