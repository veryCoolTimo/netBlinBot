"""Whisper транскрипция с word-level timestamps."""

import json
from pathlib import Path
from dataclasses import dataclass
from rich.console import Console

console = Console()


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
    """Whisper транскрибер с word-level timestamps."""

    def __init__(self, model: str = "large-v3"):
        self.model = model
        self._whisper = None

    def transcribe(self, audio_path: Path, language: str | None = None) -> TranscriptResult:
        """Транскрибирует аудио с word-level timestamps."""
        from lightning_whisper_mlx.transcribe import transcribe_audio

        model_mapping = {
            "tiny": "mlx-community/whisper-tiny-mlx",
            "small": "mlx-community/whisper-small-mlx",
            "base": "mlx-community/whisper-base-mlx",
            "medium": "mlx-community/whisper-medium-mlx",
            "large": "mlx-community/whisper-large-mlx",
            "large-v2": "mlx-community/whisper-large-v2-mlx",
            "large-v3": "mlx-community/whisper-large-v3-mlx",
        }

        hf_repo = model_mapping.get(self.model, f"mlx-community/whisper-{self.model}-mlx")

        kwargs = {
            "audio": str(audio_path),
            "path_or_hf_repo": hf_repo,
            "word_timestamps": True,
            "batch_size": 12,
        }
        if language:
            kwargs["language"] = language

        console.print(f"[cyan]Модель:[/cyan] {self.model}")

        with console.status("[bold green]Транскрибирую..."):
            result = transcribe_audio(**kwargs)

        # Извлекаем слова с таймкодами
        words = []
        for segment in result.get("segments", []):
            for word_info in segment.get("words", []):
                # word_info может быть dict или tuple
                if isinstance(word_info, dict):
                    words.append(Word(
                        text=word_info.get("word", "").strip(),
                        start=word_info.get("start", 0),
                        end=word_info.get("end", 0),
                    ))
                elif isinstance(word_info, (list, tuple)) and len(word_info) >= 3:
                    words.append(Word(
                        text=str(word_info[2]).strip(),
                        start=float(word_info[0]),
                        end=float(word_info[1]),
                    ))

        # Фильтруем пустые слова
        words = [w for w in words if w.text]

        return TranscriptResult(
            text=result.get("text", ""),
            language=result.get("language", "unknown"),
            words=words,
        )
