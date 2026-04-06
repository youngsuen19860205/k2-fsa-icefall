#!/usr/bin/env python3
# Copyright 2024 (Author: Hotword FSA Experiment)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Generate a TTS-based test set for hotword evaluation.

This script synthesises speech for a fixed set of Chinese sentences that
either contain hotwords (positive examples) or do not (negative / false-alarm
examples).  The audio files are written as 16 kHz mono WAV, along with a text
file and a JSONL manifest.

Two TTS backends are supported:
  - ``edge-tts``  (default; no GPU required; uses Microsoft cloud TTS)
  - ``pyttsx3``   (local fallback; lower quality)

Usage::

    # With edge-tts (recommended):
    python generate_tts_testset.py --output-dir data/tts_testset

    # With pyttsx3 fallback:
    python generate_tts_testset.py --output-dir data/tts_testset \\
        --tts-backend pyttsx3

    # Dry run (print sentences only):
    python generate_tts_testset.py --dry-run

Requirements (pip):
    edge-tts        (optional but recommended)
    pydub           (for MP3 → WAV conversion when using edge-tts)
    pyttsx3         (optional fallback)
    ffmpeg          (CLI tool, required by pydub for MP3 decoding)
"""

import argparse
import asyncio
import json
import logging
import os
import random
import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sentence lists
# ---------------------------------------------------------------------------

# Positive examples (contain at least one hotword)
HOTWORD_SENTENCES: List[Tuple[str, str]] = [
    ("hw_001", "请帮我买一瓶礼冥酒"),
    ("hw_002", "这家店的礼冥酒很有名"),
    ("hw_003", "樟烘烘的气味很特别"),
    ("hw_004", "我闻到一股樟烘烘的味道"),
    ("hw_005", "棋贸恙是个古老的游戏"),
    ("hw_006", "他突然得了棋贸恙"),
    ("hw_007", "膏痞岔在哪里"),
    ("hw_008", "这条路通向膏痞岔"),
    ("hw_009", "洪西效应非常明显"),
    ("hw_010", "科学家发现了洪西效应"),
    ("hw_011", "西门污痕很难清除"),
    ("hw_012", "这面墙上有西门污痕"),
    ("hw_013", "打给他让他来"),
    ("hw_014", "打电话给我妈妈"),
    ("hw_015", "打电话给客服"),
    ("hw_016", "扣他电话"),
    ("hw_017", "你去扣一下门"),
    ("hw_018", "礼冥酒和樟烘烘都是特产"),
]

# Negative examples (no hotwords – used for false alarm evaluation)
NORMAL_SENTENCES: List[Tuple[str, str]] = [
    ("normal_001", "今天天气真好"),
    ("normal_002", "我们去公园散步吧"),
    ("normal_003", "这家餐厅的菜很好吃"),
    ("normal_004", "请问现在几点了"),
    ("normal_005", "明天有一个重要的会议"),
    ("normal_006", "我需要买一些日用品"),
    ("normal_007", "这本书非常有意思"),
    ("normal_008", "北京的故宫很壮观"),
    ("normal_009", "他是一个很优秀的工程师"),
    ("normal_010", "我喜欢听音乐和看电影"),
]

ALL_SENTENCES: List[Tuple[str, str]] = HOTWORD_SENTENCES + NORMAL_SENTENCES


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate TTS test set for hotword evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/tts_testset",
        help="Root directory for the generated test set.",
    )
    parser.add_argument(
        "--tts-backend",
        type=str,
        default="edge-tts",
        choices=["edge-tts", "pyttsx3"],
        help="TTS backend to use.",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default="zh-CN-XiaoxiaoNeural",
        help="Voice name (used by edge-tts).",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Target sample rate (Hz) for output WAV files.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print sentences without generating audio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser


# ---------------------------------------------------------------------------
# Edge-TTS backend
# ---------------------------------------------------------------------------

async def _edge_tts_generate(text: str, out_mp3: str, voice: str) -> None:
    """Async helper: generate TTS MP3 via edge-tts."""
    try:
        import edge_tts
    except ImportError as exc:
        raise ImportError(
            "edge-tts is not installed.  Run: pip install edge-tts"
        ) from exc

    communicate = edge_tts.Communicate(text, voice=voice)
    await communicate.save(out_mp3)


def _mp3_to_wav(mp3_path: str, wav_path: str, sample_rate: int) -> None:
    """Convert MP3 to WAV at the specified sample rate using ffmpeg."""
    cmd = [
        "ffmpeg",
        "-y",
        "-i", mp3_path,
        "-ac", "1",
        "-ar", str(sample_rate),
        wav_path,
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        # Fallback: try pydub
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_mp3(mp3_path)
            audio = audio.set_channels(1).set_frame_rate(sample_rate)
            audio.export(wav_path, format="wav")
        except Exception as e2:
            raise RuntimeError(
                f"ffmpeg failed and pydub fallback also failed: {e2}\n"
                f"ffmpeg stderr: {result.stderr.decode()}"
            ) from e2


def generate_with_edge_tts(
    sentences: List[Tuple[str, str]],
    output_dir: Path,
    voice: str,
    sample_rate: int,
) -> None:
    """Generate WAV files for *sentences* using edge-tts."""
    wav_dir = output_dir / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)

    async def _run_all():
        for utt_id, text in sentences:
            out_wav = str(wav_dir / f"{utt_id}.wav")
            if Path(out_wav).exists():
                logger.info("Skipping %s (already exists).", utt_id)
                continue
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp_mp3 = tmp.name
            try:
                logger.info("Generating TTS for '%s': %s", utt_id, text)
                await _edge_tts_generate(text, tmp_mp3, voice)
                _mp3_to_wav(tmp_mp3, out_wav, sample_rate)
                logger.info("  → saved %s", out_wav)
            except Exception as exc:
                logger.error("Failed to generate %s: %s", utt_id, exc)
            finally:
                if os.path.exists(tmp_mp3):
                    os.remove(tmp_mp3)

    asyncio.run(_run_all())


# ---------------------------------------------------------------------------
# pyttsx3 backend
# ---------------------------------------------------------------------------

def generate_with_pyttsx3(
    sentences: List[Tuple[str, str]],
    output_dir: Path,
    sample_rate: int,
) -> None:
    """Generate WAV files for *sentences* using pyttsx3 (local TTS)."""
    try:
        import pyttsx3
    except ImportError as exc:
        raise ImportError(
            "pyttsx3 is not installed.  Run: pip install pyttsx3"
        ) from exc

    wav_dir = output_dir / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)

    engine = pyttsx3.init()
    for utt_id, text in sentences:
        out_wav = str(wav_dir / f"{utt_id}.wav")
        if Path(out_wav).exists():
            logger.info("Skipping %s (already exists).", utt_id)
            continue
        logger.info("Generating TTS (pyttsx3) for '%s': %s", utt_id, text)
        engine.save_to_file(text, out_wav)
        engine.runAndWait()
        logger.info("  → saved %s", out_wav)


# ---------------------------------------------------------------------------
# Manifest / text file writers
# ---------------------------------------------------------------------------

def write_text_file(sentences: List[Tuple[str, str]], output_dir: Path) -> None:
    """Write a Kaldi-style text file (utt_id<TAB>text)."""
    text_path = output_dir / "text"
    with open(text_path, "w", encoding="utf-8") as fh:
        for utt_id, text in sentences:
            fh.write(f"{utt_id}\t{text}\n")
    logger.info("Text file written to: %s", text_path)


def write_manifest(
    sentences: List[Tuple[str, str]],
    output_dir: Path,
    sample_rate: int,
) -> None:
    """Write a JSONL manifest file."""
    manifest_path = output_dir / "manifest.jsonl"
    wav_dir = output_dir / "wav"
    with open(manifest_path, "w", encoding="utf-8") as fh:
        for utt_id, text in sentences:
            entry = {
                "id": utt_id,
                "text": text,
                "audio_filepath": str(wav_dir / f"{utt_id}.wav"),
                "sample_rate": sample_rate,
                "is_hotword": utt_id.startswith("hw_"),
            }
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
    logger.info("Manifest written to: %s", manifest_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = get_parser()
    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
    )

    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sentences = ALL_SENTENCES

    if args.dry_run:
        logger.info("Dry run – printing %d sentences:", len(sentences))
        for utt_id, text in sentences:
            print(f"{utt_id}\t{text}")
        return

    # Write text + manifest regardless of backend
    write_text_file(sentences, output_dir)
    write_manifest(sentences, output_dir, args.sample_rate)

    # Generate audio
    if args.tts_backend == "edge-tts":
        logger.info("Using edge-tts backend (voice=%s) …", args.voice)
        generate_with_edge_tts(sentences, output_dir, args.voice, args.sample_rate)
    elif args.tts_backend == "pyttsx3":
        logger.info("Using pyttsx3 backend …")
        generate_with_pyttsx3(sentences, output_dir, args.sample_rate)
    else:
        raise ValueError(f"Unknown TTS backend: {args.tts_backend}")

    logger.info("TTS test set generation complete.  Output: %s", output_dir)


if __name__ == "__main__":
    main()
