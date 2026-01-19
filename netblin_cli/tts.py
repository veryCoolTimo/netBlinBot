"""TTS генерация через Edge-TTS и Silero."""

import asyncio
from pathlib import Path
from rich.console import Console

console = Console()

# Edge-TTS голоса
EDGE_VOICES = {
    "dmitry": "ru-RU-DmitryNeural",
    "svetlana": "ru-RU-SvetlanaNeural",
    "jessie": "en-US-JennyNeural",
    "brian": "en-GB-RyanNeural",
    "aria": "en-US-AriaNeural",
    "emma": "en-US-EmmaNeural",
}

# Silero голоса (русские)
SILERO_VOICES = {"aidar", "baya", "kseniya", "xenia", "eugene", "random"}

# Все голоса для --voices
VOICES = {
    **EDGE_VOICES,
    "aidar": "silero (мужской RU)",
    "baya": "silero (женский RU)",
    "kseniya": "silero (женский RU)",
    "xenia": "silero (женский RU)",
    "eugene": "silero (мужской RU)",
}


class TTSEngine:
    """TTS движок с поддержкой Edge-TTS и Silero."""

    def __init__(self, voice: str = "dmitry"):
        self.voice_alias = voice.lower()
        self.use_silero = self.voice_alias in SILERO_VOICES

        if self.use_silero:
            self._init_silero()
        else:
            self.voice = EDGE_VOICES.get(self.voice_alias, self.voice_alias)

    def _init_silero(self):
        """Инициализация Silero модели."""
        import torch

        self.device = torch.device("cpu")
        self.model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-models",
            model="silero_tts",
            language="ru",
            speaker="v4_ru",
            trust_repo=True,
        )
        self.model.to(self.device)
        self.sample_rate = 48000

    async def _generate_edge_async(self, text: str, output_path: Path) -> Path:
        """Edge-TTS генерация."""
        import edge_tts

        communicate = edge_tts.Communicate(text, self.voice)
        await communicate.save(str(output_path))
        return output_path

    def _generate_silero(self, text: str, output_path: Path) -> Path:
        """Silero генерация."""
        audio = self.model.apply_tts(
            text=text,
            speaker=self.voice_alias,
            sample_rate=self.sample_rate,
        )

        # Сохраняем в wav через scipy
        import numpy as np
        from scipy.io import wavfile

        audio_np = audio.numpy()
        # Нормализуем к int16
        audio_int16 = (audio_np * 32767).astype(np.int16)
        wavfile.write(str(output_path), self.sample_rate, audio_int16)
        return output_path

    def generate(self, text: str, output_path: Path) -> Path:
        """Генерирует аудио из текста."""
        if self.use_silero:
            # Silero сохраняет в wav
            wav_path = output_path.with_suffix(".wav")
            return self._generate_silero(text, wav_path)
        else:
            return asyncio.run(self._generate_edge_async(text, output_path))

    def generate_batch(self, items: list[tuple[str, Path]]) -> list[Path]:
        """Генерирует несколько аудио файлов."""
        results = []
        for text, output_path in items:
            result = self.generate(text, output_path)
            results.append(result)
        return results
