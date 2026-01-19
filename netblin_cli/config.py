"""Конфигурация NetBlinBot."""

from pathlib import Path

# Пути
PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_IMAGE = PROJECT_ROOT / "image.png"
TEMP_DIR = PROJECT_ROOT / "temp"

# Ollama
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "dolphin3"

# Whisper
WHISPER_MODEL = "large-v3"

# TTS
DEFAULT_VOICE = "aidar"

# FFmpeg
FONT_PATH = "/System/Library/Fonts/Supplemental/Arial.ttf"  # macOS default

# Промпт для LLM
LLM_PROMPT = """Задача: разбей транскрипцию на фразы и придумай антонимы.

Правила:
1. Разбей на фразы по 2-5 слов по смыслу
2. Для каждой фразы придумай короткий смешной антоним
3. start = время начала первого слова фразы
4. end = время конца последнего слова фразы

КРИТИЧНО: Ответ ТОЛЬКО JSON массив. Никакого текста до или после.

Пример ответа:
[{{"original":"привет друзья","antonym":"пока враги","start":0.0,"end":0.8}},{{"original":"сегодня","antonym":"вчера","start":0.9,"end":1.2}}]

Транскрипция:
{transcription}

JSON:"""
