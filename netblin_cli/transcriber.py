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
class Word:
    """Слово с таймкодами."""
    text: str
    start: float
    end: float

    def to_dict(self) -> dict:
        return {"word": self.text, "start": self.start, "end": self.end}


@dataclass
class TranscriptResult:
    """Результат транскрипции."""
    text: str
    language: str
    words: list[Word]

    def to_json(self) -> str:
        """Возвращает JSON для LLM."""
        return json.dumps(
            [w.to_dict() for w in self.words],
            ensure_ascii=False,
            indent=2
        )


class Transcriber:
    """Whisper транскрибер через whisper.cpp."""

    def __init__(self, model: str = "large-v3"):
        self.model = model

        # Проверяем наличие whisper.cpp
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
        """Транскрибирует аудио с word-level timestamps."""
        audio_path = Path(audio_path)
        output_base = audio_path.with_suffix("")
        output_json = audio_path.with_suffix(".json")

        console.print(f"[cyan]Модель:[/cyan] {self.model} (whisper.cpp)")

        cmd = [
            str(WHISPER_BIN),
            "-m", str(WHISPER_MODEL),
            "-f", str(audio_path),
            "-l", language or "auto",
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

        # Парсим JSON вывод
        if not output_json.exists():
            raise FileNotFoundError(f"Whisper не создал JSON: {output_json}")

        with open(output_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Извлекаем слова из сегментов
        words = []
        full_text = ""
        detected_lang = "auto"

        for segment in data.get("transcription", []):
            text = segment.get("text", "").strip()
            if not text:
                continue

            full_text += text + " "

            # whisper.cpp даёт offsets в миллисекундах
            start_ms = segment.get("offsets", {}).get("from", 0)
            end_ms = segment.get("offsets", {}).get("to", 0)

            # Разбиваем на слова, распределяя время равномерно
            segment_words = text.split()
            if segment_words:
                duration = end_ms - start_ms
                word_duration = duration / len(segment_words)

                for i, word_text in enumerate(segment_words):
                    word_start = start_ms + i * word_duration
                    word_end = start_ms + (i + 1) * word_duration
                    words.append(Word(
                        text=word_text,
                        start=word_start / 1000,  # в секунды
                        end=word_end / 1000,
                    ))

        # Удаляем временный JSON
        output_json.unlink(missing_ok=True)

        return TranscriptResult(
            text=full_text.strip(),
            language=detected_lang,
            words=words,
        )
