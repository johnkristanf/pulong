import json
from pathlib import Path
from faster_whisper import WhisperModel


def update_metadata(audio_path: str, text: str, file_path: str = "metadata.jsonl") -> None:
    entries = {}
    path = Path(file_path)

    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                entries[item["audio"]] = item["text"]

    entries[audio_path] = text

    with open(path, "w", encoding="utf-8") as f:
        for audio, txt in entries.items():
            f.write(json.dumps({"audio": audio, "text": txt}, ensure_ascii=False) + "\n")


def transcribe(audio_path: str, model_size: str, device: str) -> None:
    model = WhisperModel(model_size, device=device, compute_type="int8_float16")
    segments, info = model.transcribe(audio_path, beam_size=5)

    print(f"Detected language: {info.language!r} (p={info.language_probability:.2f})\n")
    texts = []
    for segment in segments:
        text = segment.text.strip()
        print(f"[{segment.start:.1f}s → {segment.end:.1f}s]  {text}")
        if text:
            texts.append(text)

    full_text = " ".join(texts)
    update_metadata(audio_path, full_text)


def main() -> None:
    audio_path = 'audio/TEAM MEETING (JAN 7, 2026).m4a'
    model_size = "medium"
    device = "cuda"

    transcribe(audio_path, model_size, device)


if __name__ == "__main__":
    main()

