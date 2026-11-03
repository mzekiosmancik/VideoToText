"""Simple MP4-to-text transcriber using faster-whisper (CTranslate2, CPU-optimized).

By default, transcribes every .mp4 in the mp4/ folder and writes a matching
.txt file into the outputs/ folder.

Usage:
    python transcribe.py
    python transcribe.py --model small
    python transcribe.py "video.mp4" --output "video.txt"
"""

import argparse
import os

from faster_whisper import WhisperModel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT_DIR = os.path.join(SCRIPT_DIR, "mp4")
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")


def format_timestamp(seconds: float) -> str:
    minutes, secs = divmod(int(seconds), 60)
    return f"{minutes:02d}:{secs:02d}"


def transcribe_file(model: WhisperModel, video_path: str, output_path: str) -> None:
    print(f"Transcribing '{video_path}'...")
    # vad_filter skips silence, which speeds up long recordings with gaps/pauses.
    segments, _ = model.transcribe(video_path, vad_filter=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for segment in segments:
            f.write(f"[{format_timestamp(segment.start)}] {segment.text.strip()}\n")

    print(f"Done. Transcript saved to: {output_path}")


def load_model(model_name: str) -> WhisperModel:
    print(f"Loading Whisper model '{model_name}'...")
    # int8 quantization on CPU gives a large speedup with minimal accuracy loss.
    return WhisperModel(model_name, device="cpu", compute_type="int8")


def transcribe_folder(input_dir: str, output_dir: str, model_name: str) -> None:
    videos = sorted(
        f for f in os.listdir(input_dir) if f.lower().endswith(".mp4")
    )
    if not videos:
        print(f"No .mp4 files found in: {input_dir}")
        return

    model = load_model(model_name)

    for video in videos:
        video_path = os.path.join(input_dir, video)
        output_path = os.path.join(output_dir, os.path.splitext(video)[0] + ".txt")
        transcribe_file(model, video_path, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe MP4 file(s) to text using Whisper.")
    parser.add_argument(
        "video",
        nargs="?",
        default=None,
        help="Path to a single .mp4 file. If omitted, transcribes all .mp4 files in the mp4/ folder.",
    )
    parser.add_argument(
        "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base). Bigger = more accurate but slower.",
    )
    parser.add_argument("--output", default=None, help="Path to the output .txt file (single-file mode only)")
    args = parser.parse_args()

    if args.video:
        if not os.path.isfile(args.video):
            raise FileNotFoundError(f"File not found: {args.video}")
        out_path = args.output or os.path.join(
            DEFAULT_OUTPUT_DIR, os.path.splitext(os.path.basename(args.video))[0] + ".txt"
        )
        loaded_model = load_model(args.model)
        transcribe_file(loaded_model, args.video, out_path)
    else:
        transcribe_folder(DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_DIR, args.model)
