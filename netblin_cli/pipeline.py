"""Главный пайплайн обработки видео."""

from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from .config import DEFAULT_IMAGE, TEMP_DIR, WHISPER_MODEL, OLLAMA_MODEL, DEFAULT_VOICE
from .transcriber import Transcriber
from .ollama_client import OllamaClient
from .tts import TTSEngine
from .video_processor import VideoProcessor

console = Console()


class Pipeline:
    """Главный пайплайн для создания reaction-видео."""

    def __init__(
        self,
        image_path: Path = DEFAULT_IMAGE,
        voice: str = DEFAULT_VOICE,
        whisper_model: str = WHISPER_MODEL,
        ollama_model: str = OLLAMA_MODEL,
        keep_temp: bool = False,
    ):
        self.image_path = image_path
        self.voice = voice
        self.whisper_model = whisper_model
        self.ollama_model = ollama_model
        self.keep_temp = keep_temp

        # Инициализируем компоненты
        self.transcriber = Transcriber(model=whisper_model)
        self.ollama = OllamaClient(model=ollama_model)
        self.tts = TTSEngine(voice=voice)
        self.video_processor = VideoProcessor(TEMP_DIR)

    def run(
        self,
        video_path: Path,
        output_path: Path | None = None,
        language: str | None = None,
        dry_run: bool = False,
    ) -> Path | None:
        """Запускает полный пайплайн."""
        video_path = Path(video_path)

        if not video_path.exists():
            console.print(f"[red]Файл не найден:[/red] {video_path}")
            return None

        if output_path is None:
            output_path = video_path.with_name(f"{video_path.stem}_reaction.mp4")

        console.print()
        console.print("[bold cyan]═══ NetBlinBot ═══[/bold cyan]")
        console.print()
        console.print(f"[cyan]Видео:[/cyan] {video_path}")
        console.print(f"[cyan]Картинка:[/cyan] {self.image_path}")
        console.print(f"[cyan]Голос:[/cyan] {self.voice}")
        console.print()

        # Проверяем Ollama
        if not self.ollama.check_connection():
            console.print("[red]Ollama недоступен![/red]")
            console.print("[yellow]Запусти: ollama serve[/yellow]")
            return None

        try:
            # 1. Получаем информацию о видео
            console.print("[bold]1. Анализ видео[/bold]")
            video_info = self.video_processor.get_video_info(video_path)
            console.print(f"   Размер: {video_info.width}x{video_info.height}")
            console.print(f"   Длительность: {video_info.duration:.1f}s")
            console.print()

            # 2. Извлекаем аудио
            console.print("[bold]2. Извлечение аудио[/bold]")
            audio_path = TEMP_DIR / "audio.wav"
            self.video_processor.extract_audio(video_path, audio_path)
            console.print(f"   [green]✓[/green] {audio_path}")
            console.print()

            # 3. Транскрипция
            console.print("[bold]3. Транскрипция[/bold]")
            transcript = self.transcriber.transcribe(audio_path, language=language)
            console.print(f"   [green]✓[/green] Язык: {transcript.language}")
            console.print(f"   [green]✓[/green] Сегментов: {len(transcript.segments)}")
            console.print()

            # 4. LLM обработка
            console.print("[bold]4. Генерация антонимов[/bold]")
            segments = self.ollama.process_segments(transcript.segments)
            console.print(f"   [green]✓[/green] Сегментов: {len(segments)}")
            console.print()

            # Показываем превью
            console.print("[bold]Превью сегментов:[/bold]")
            for i, seg in enumerate(segments[:5]):
                console.print(f"   {i+1}. \"{seg.original}\" → \"{seg.antonym}\"")
            if len(segments) > 5:
                console.print(f"   ... и ещё {len(segments) - 5}")
            console.print()

            if dry_run:
                console.print("[yellow]Dry run — остановка[/yellow]")
                return None

            # 5. Генерация TTS
            console.print("[bold]5. Генерация голоса[/bold]")
            audio_files = []
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("TTS", total=len(segments))
                for i, seg in enumerate(segments):
                    audio_file = TEMP_DIR / f"tts_{i:03d}.mp3"
                    # generate возвращает реальный путь (может быть .wav для Silero)
                    actual_path = self.tts.generate(seg.antonym, audio_file)
                    audio_files.append(actual_path)
                    progress.advance(task)
            console.print(f"   [green]✓[/green] Создано {len(audio_files)} аудио")
            console.print()

            # 6. Сборка видео
            console.print("[bold]6. Сборка видео[/bold]")
            all_clips = []

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("Сборка", total=len(segments))

                for i, seg in enumerate(segments):
                    # Пропускаем сегменты с невалидными таймкодами
                    if seg.start <= 0 and seg.end <= 0:
                        continue

                    # Вырезаем оригинальный клип
                    original_clip = TEMP_DIR / f"orig_{i:03d}.mp4"
                    self.video_processor.cut_video(
                        video_path, seg.start, seg.end, original_clip
                    )
                    all_clips.append(original_clip)

                    # Создаём reaction клип
                    reaction_clip = TEMP_DIR / f"react_{i:03d}.mp4"
                    self.video_processor.create_reaction_clip(
                        self.image_path,
                        audio_files[i],
                        seg.antonym,
                        video_info,
                        reaction_clip,
                    )
                    all_clips.append(reaction_clip)

                    progress.advance(task)

            console.print(f"   [green]✓[/green] Создано {len(all_clips)} клипов")
            console.print()

            # 7. Финальная склейка
            console.print("[bold]7. Финальная склейка[/bold]")
            with console.status("[green]Склеиваю..."):
                self.video_processor.concat_videos(all_clips, output_path)
            console.print(f"   [green]✓[/green] {output_path}")
            console.print()

            # Очистка
            if not self.keep_temp:
                self.video_processor.cleanup()

            console.print("[bold green]═══ Готово! ═══[/bold green]")
            console.print(f"[cyan]Результат:[/cyan] {output_path}")

            return output_path

        except Exception as e:
            console.print(f"[red]Ошибка:[/red] {e}")
            if not self.keep_temp:
                self.video_processor.cleanup()
            raise
