"""Конфигурация NetBlinBot."""

from pathlib import Path

# Пути
PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_IMAGE = PROJECT_ROOT / "image.png"
TEMP_DIR = PROJECT_ROOT / "temp"

# Ollama
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "gpt-oss:20b"

# Whisper
WHISPER_MODEL = "large-v3"

# TTS
DEFAULT_VOICE = "dmitry"

# FFmpeg
FONT_PATH = "/System/Library/Fonts/Supplemental/Arial.ttf"  # macOS default

# Промпт для LLM
LLM_PROMPT = """Ты получаешь транскрипцию видео с таймкодами по словам в формате JSON.

Твоя задача:
1. Разбить текст на смысловые фразы (1-5 слов)
2. Для каждой фразы придумать антоним или противоположность по смыслу
3. Антонимы должны быть короткими, смешными и ироничными

Правила разбивки:
- Разбивай по смыслу, не механически
- Учитывай паузы (gap > 0.3 сек между словами = новая фраза)
- Не разрывай устойчивые выражения
- Каждая фраза должна иметь start (начало первого слова) и end (конец последнего слова)

ВАЖНО: Верни ТОЛЬКО валидный JSON массив, без markdown, без ```json, без пояснений.

Формат ответа:
[
  {{"original": "Всем привет", "antonym": "Никому пока", "start": 0.0, "end": 0.7}},
  {{"original": "друзья", "antonym": "Враги", "start": 0.75, "end": 1.1}}
]

Транскрипция:
{transcription}"""
