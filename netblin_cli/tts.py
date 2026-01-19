"""TTS генерация через Edge-TTS."""

import asyncio
from pathlib import Path
from rich.console import Console

console = Console()

# Маппинг голосов на Edge-TTS идентификаторы
VOICES = {
    # Русские
    "dmitry": "ru-RU-DmitryNeural",
    "svetlana": "ru-RU-SvetlanaNeural",
    # Английские (TikTok-style)
    "jessie": "en-US-JennyNeural",
    "brian": "en-GB-RyanNeural",
    "aria": "en-US-AriaNeural",
    "emma": "en-US-EmmaNeural",
}


class TTSEngine:
    """TTS движок на базе Edge-TTS."""

    def __init__(self, voice: str = "dmitry"):
        self.voice = VOICES.get(voice.lower(), voice)
        self.voice_alias = voice

    async def _generate_async(self, text: str, output_path: Path) -> Path:
        """Асинхронная генерация."""
        import edge_tts

        communicate = edge_tts.Communicate(text, self.voice)
        await communicate.save(str(output_path))
        return output_path

    def generate(self, text: str, output_path: Path) -> Path:
        """Генерирует аудио из текста."""
        return asyncio.run(self._generate_async(text, output_path))

    def generate_batch(self, items: list[tuple[str, Path]]) -> list[Path]:
        """Генерирует несколько аудио файлов."""
        results = []
        for text, output_path in items:
            result = self.generate(text, output_path)
            results.append(result)
        return results
