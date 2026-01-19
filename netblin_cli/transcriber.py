"""Whisper транскрипция через whisper.cpp (быстрая нативная версия)."""

import json
import subprocess
from pathlib import Path
from dataclasses import dataclass
from rich.console import Console

console = Console()

# whisper.cpp пути
WHISPER_CPP_PATH = Path.home() / "whisper.cpp"
WHISPER_BIN = WHISPER_CPP_PATH / "build" / "bin" / "whisper-cli"
WHISPER_MODEL = WHISPER_CPP_PATH / "models" / "ggml-large-v3.bin"


@dataclass
class Segment:
    """Сегмент транскрипции с таймкодами."""
    text: str
    start: float
    end: float

    def to_dict(self) -> dict:
        return {"text": self.text, "start": self.start, "end": self.end}


@dataclass
class TranscriptResult:
    """Результат транскрипции."""
    text: str
    language: str
    segments: list[Segment]

    def to_json(self) -> str:
        """Возвращает JSON для LLM."""
        return json.dumps(
            [s.to_dict() for s in self.segments],
            ensure_ascii=False,
            indent=2
        )


class Transcriber:
    """Whisper транскрибер через whisper.cpp."""

    def __init__(self, model: str = "large-v3"):
        self.model = model

        if not WHISPER_BIN.exists():
            raise FileNotFoundError(
                f"whisper.cpp не найден: {WHISPER_BIN}\n"
                "Установи: https://github.com/ggerganov/whisper.cpp"
            )

        if not WHISPER_MODEL.exists():
            raise FileNotFoundError(
                f"Модель не найдена: {WHISPER_MODEL}\n"
                "Скачай: cd ~/whisper.cpp && ./models/download-ggml-model.sh large-v3"
            )

    def transcribe(self, audio_path: Path, language: str | None = None) -> TranscriptResult:
        """Транскрибирует аудио и возвращает сегменты."""
        audio_path = Path(audio_path)
        output_base = audio_path.with_suffix("")
        output_json = audio_path.with_suffix(".json")

        console.print(f"[cyan]Модель:[/cyan] {self.model} (whisper.cpp)")

        cmd = [
            str(WHISPER_BIN),
            "-m", str(WHISPER_MODEL),
            "-f", str(audio_path),
            "-l", language or "auto",
            "-ml", "80",  # Короткие сегменты (макс 80 символов ~10-12 слов)
            "-sow",  # Разбивать по словам
            "-oj",  # output JSON
            "-of", str(output_base),
        ]

        with console.status("[bold green]Транскрибирую..."):
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600
            )

        if result.returncode != 0:
            raise RuntimeError(f"Whisper ошибка: {result.stderr}")

        if not output_json.exists():
            raise FileNotFoundError(f"Whisper не создал JSON: {output_json}")

        with open(output_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Извлекаем сегменты как есть от whisper
        segments = []
        full_text = ""

        for seg_data in data.get("transcription", []):
            text = seg_data.get("text", "").strip()
            if not text:
                continue

            full_text += text + " "

            # whisper.cpp даёт offsets в миллисекундах
            start_ms = seg_data.get("offsets", {}).get("from", 0)
            end_ms = seg_data.get("offsets", {}).get("to", 0)

            segments.append(Segment(
                text=text,
                start=start_ms / 1000,  # в секунды
                end=end_ms / 1000,
            ))

        # Удаляем временный JSON
        output_json.unlink(missing_ok=True)

        return TranscriptResult(
            text=full_text.strip(),
            language="auto",
            segments=segments,
        )
