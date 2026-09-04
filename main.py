from faster_whisper import WhisperModel


def transcribe(audio_path: str, model_size: str, device: str) -> None:
    model = WhisperModel(model_size, device=device, compute_type="int8_float16")
    segments, info = model.transcribe(audio_path, beam_size=5)

    print(f"Detected language: {info.language!r} (p={info.language_probability:.2f})\n")
    for segment in segments:
        print(f"[{segment.start:.1f}s → {segment.end:.1f}s]  {segment.text.strip()}")


def main() -> None:
    audio_path = 'audio/WISHLIST FOR CUSTOMER ORDER MONITORING.m4a'
    model_size = "medium"
    device = "cuda"

    transcribe(audio_path, model_size, device)


if __name__ == "__main__":
    main()
