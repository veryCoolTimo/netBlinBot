#!/usr/bin/env python3
"""NetBlinBot CLI - автоматический генератор reaction-видео."""

import argparse
import sys
from pathlib import Path

from rich.console import Console

from .config import DEFAULT_IMAGE, WHISPER_MODEL, OLLAMA_MODEL, DEFAULT_VOICE
from .pipeline import Pipeline

console = Console()


def main():
    parser = argparse.ArgumentParser(
        description="NetBlinBot - автоматический генератор reaction-видео",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  %(prog)s video.mp4                          # Базовое использование
  %(prog)s video.mp4 --voice jessie           # Другой голос
  %(prog)s video.mp4 --dry-run                # Только превью
  %(prog)s video.mp4 -o result.mp4            # Указать выходной файл
        """,
    )

    parser.add_argument(
        "video",
        nargs="?",  # Опционально для --voices
        help="Путь к входному видео",
    )

    parser.add_argument(
        "-o", "--output",
        help="Путь к выходному видео (по умолчанию: <video>_reaction.mp4)",
    )

    parser.add_argument(
        "-i", "--image",
        default=str(DEFAULT_IMAGE),
        help=f"Картинка для реакции (по умолчанию: {DEFAULT_IMAGE.name})",
    )

    parser.add_argument(
        "-v", "--voice",
        default=DEFAULT_VOICE,
        help=f"Голос TTS (по умолчанию: {DEFAULT_VOICE})",
    )

    parser.add_argument(
        "-m", "--model",
        default=OLLAMA_MODEL,
        help=f"Ollama модель (по умолчанию: {OLLAMA_MODEL})",
    )

    parser.add_argument(
        "--whisper-model",
        default=WHISPER_MODEL,
        help=f"Whisper модель (по умолчанию: {WHISPER_MODEL})",
    )

    parser.add_argument(
        "-l", "--lang",
        help="Язык видео (ru, en, etc). Автодетекция если не указан",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Только показать план без рендера",
    )

    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Не удалять временные файлы",
    )

    parser.add_argument(
        "--voices",
        action="store_true",
        help="Показать доступные голоса",
    )

    args = parser.parse_args()

    # Показать голоса
    if args.voices:
        from .tts import VOICES
        console.print("[bold]Доступные голоса:[/bold]")
        for alias, full_name in VOICES.items():
            console.print(f"  {alias:12} → {full_name}")
        return

    # Проверяем что видео указано
    if not args.video:
        parser.print_help()
        sys.exit(1)

    # Проверяем видео
    video_path = Path(args.video)
    if not video_path.exists():
        console.print(f"[red]Файл не найден:[/red] {video_path}")
        sys.exit(1)

    # Проверяем картинку
    image_path = Path(args.image)
    if not image_path.exists():
        console.print(f"[red]Картинка не найдена:[/red] {image_path}")
        sys.exit(1)

    # Запускаем пайплайн
    pipeline = Pipeline(
        image_path=image_path,
        voice=args.voice,
        whisper_model=args.whisper_model,
        ollama_model=args.model,
        keep_temp=args.keep_temp,
    )

    output_path = Path(args.output) if args.output else None

    try:
        result = pipeline.run(
            video_path=video_path,
            output_path=output_path,
            language=args.lang,
            dry_run=args.dry_run,
        )

        if result:
            sys.exit(0)
        else:
            sys.exit(1)

    except KeyboardInterrupt:
        console.print("\n[yellow]Прервано[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"[red]Ошибка:[/red] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
