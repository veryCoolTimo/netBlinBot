"""Обработка видео через FFmpeg."""

import subprocess
import json
from pathlib import Path
from dataclasses import dataclass
from rich.console import Console

from .config import FONT_PATH

console = Console()


@dataclass
class VideoInfo:
    """Информация о видео."""
    width: int
    height: int
    duration: float
    fps: float

    @property
    def is_vertical(self) -> bool:
        return self.height > self.width

    @property
    def is_horizontal(self) -> bool:
        return self.width > self.height


class VideoProcessor:
    """Обработчик видео через FFmpeg."""

    def __init__(self, temp_dir: Path):
        self.temp_dir = temp_dir
        self.temp_dir.mkdir(exist_ok=True)

    def get_video_info(self, video_path: Path) -> VideoInfo:
        """Получает информацию о видео."""
        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_streams",
            "-show_format",
            str(video_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(result.stdout)

        # Ищем видео поток
        video_stream = None
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                video_stream = stream
                break

        if not video_stream:
            raise ValueError("Видео поток не найден")

        # Парсим fps
        fps_str = video_stream.get("r_frame_rate", "30/1")
        if "/" in fps_str:
            num, den = fps_str.split("/")
            fps = float(num) / float(den)
        else:
            fps = float(fps_str)

        return VideoInfo(
            width=int(video_stream.get("width", 1920)),
            height=int(video_stream.get("height", 1080)),
            duration=float(data.get("format", {}).get("duration", 0)),
            fps=fps,
        )

    def extract_audio(self, video_path: Path, output_path: Path) -> Path:
        """Извлекает аудио из видео."""
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vn",  # без видео
            "-acodec", "pcm_s16le",
            "-ar", "16000",  # 16kHz для Whisper
            "-ac", "1",  # моно
            str(output_path)
        ]

        subprocess.run(cmd, capture_output=True, check=True)
        return output_path

    def cut_video(self, video_path: Path, start: float, end: float, output_path: Path) -> Path:
        """Вырезает фрагмент видео."""
        duration = end - start

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", str(video_path),
            "-t", str(duration),
            "-c:v", "libx264",
            "-c:a", "aac",
            "-avoid_negative_ts", "make_zero",
            str(output_path)
        ]

        subprocess.run(cmd, capture_output=True, check=True)
        return output_path

    def create_reaction_clip(
        self,
        image_path: Path,
        audio_path: Path,
        text: str,
        video_info: VideoInfo,
        output_path: Path,
    ) -> Path:
        """Создаёт reaction клип: картинка + текст + аудио."""
        # Получаем длительность аудио
        probe_cmd = [
            "ffprobe", "-v", "quiet",
            "-show_entries", "format=duration",
            "-of", "csv=p=0",
            str(audio_path)
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True)
        duration_str = result.stdout.strip()
        if not duration_str:
            raise ValueError(f"Не удалось получить длительность аудио: {audio_path}")
        audio_duration = float(duration_str)

        # Масштабирование картинки под размер видео
        w, h = video_info.width, video_info.height

        if video_info.is_horizontal:
            # Горизонтальное видео: картинка по высоте, pillarbox
            scale_filter = f"scale=-1:{h}"
            pad_filter = f"pad={w}:{h}:(ow-iw)/2:0:black"
        else:
            # Вертикальное видео: картинка по ширине, letterbox
            scale_filter = f"scale={w}:-1"
            pad_filter = f"pad={w}:{h}:0:(oh-ih)/2:black"

        # Экранируем текст для FFmpeg
        escaped_text = text.replace("'", "'\\''").replace(":", "\\:")

        # Формируем фильтр
        filter_complex = (
            f"[0:v]{scale_filter},{pad_filter},"
            f"drawtext=text='{escaped_text}':"
            f"fontfile='{FONT_PATH}':"
            f"fontsize={min(w, h) // 10}:"  # Размер шрифта относительно видео
            f"fontcolor=white:"
            f"borderw=4:"
            f"bordercolor=black:"
            f"x=(w-text_w)/2:"
            f"y=h-h/6"
            f"[v]"
        )

        cmd = [
            "ffmpeg", "-y",
            "-loop", "1",
            "-i", str(image_path),
            "-i", str(audio_path),
            "-filter_complex", filter_complex,
            "-map", "[v]",
            "-map", "1:a",
            "-t", str(audio_duration),
            "-r", str(int(video_info.fps)),
            "-c:v", "libx264",
            "-c:a", "aac",
            "-ar", "44100",
            "-ac", "2",  # Стерео
            "-pix_fmt", "yuv420p",
            "-shortest",
            str(output_path)
        ]

        subprocess.run(cmd, capture_output=True, check=True)
        return output_path

    def concat_videos(self, video_paths: list[Path], output_path: Path) -> Path:
        """Склеивает видео в одно."""
        # Создаём файл со списком
        list_file = self.temp_dir / "concat_list.txt"
        with open(list_file, "w") as f:
            for video_path in video_paths:
                f.write(f"file '{video_path}'\n")

        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_file),
            "-c:v", "libx264",
            "-c:a", "aac",
            "-ar", "44100",  # Унифицируем sample rate
            str(output_path)
        ]

        subprocess.run(cmd, capture_output=True, check=True)
        return output_path

    def cleanup(self):
        """Удаляет временные файлы."""
        import shutil
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
