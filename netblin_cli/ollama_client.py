"""Клиент для Ollama API."""

import json
import requests
from dataclasses import dataclass
from rich.console import Console

from .config import OLLAMA_URL, OLLAMA_MODEL

console = Console()


@dataclass
class ReactionSegment:
    """Сегмент с оригиналом и антонимом."""
    original: str
    antonym: str
    start: float
    end: float


class OllamaClient:
    """Клиент для работы с Ollama."""

    def __init__(self, model: str = OLLAMA_MODEL, url: str = OLLAMA_URL):
        self.model = model
        self.url = url
        self.api_generate = f"{url}/api/generate"

    def check_connection(self) -> bool:
        """Проверяет доступность Ollama."""
        try:
            resp = requests.get(f"{self.url}/api/tags", timeout=5)
            return resp.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def generate(self, prompt: str) -> str:
        """Генерирует ответ от LLM."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.8,
                "num_predict": 100,  # Короткий ответ
            }
        }

        resp = requests.post(self.api_generate, json=payload, timeout=60)
        resp.raise_for_status()

        return resp.json().get("response", "").strip()

    def generate_antonym(self, text: str) -> str:
        """Генерирует антоним/противоположность для фразы."""
        prompt = f"""Переверни смысл фразы на противоположный. Сохрани структуру но замени ключевые слова на антонимы.

Примеры:
"Я влюбилась в гея" → "я разлюбилась в натурала"
"что делать?" → "что не делать?"
"не знаю как быть" → "знаю как не быть"
"расскажу вам сказку" → "промолчу про правду"
"далеко за горами" → "близко перед долинами"

Фраза: "{text}"
Противоположность (одна фраза, без пояснений):"""

        response = self.generate(prompt)

        # Очищаем ответ
        response = response.strip().strip('"').strip("'")
        # Убираем возможные префиксы
        for prefix in ["Противоположность:", "Антоним:", "Ответ:"]:
            if response.lower().startswith(prefix.lower()):
                response = response[len(prefix):].strip()

        # Если ответ слишком длинный или пустой
        if not response or len(response) > len(text) * 3:
            return "совсем наоборот"

        return response

    def process_segments(self, segments: list) -> list[ReactionSegment]:
        """Обрабатывает сегменты и генерирует антонимы для каждого."""
        console.print(f"[cyan]Модель:[/cyan] {self.model}")

        results = []

        from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Генерация антонимов", total=len(segments))

            for seg in segments:
                antonym = self.generate_antonym(seg.text)

                results.append(ReactionSegment(
                    original=seg.text,
                    antonym=f"Нет блин, {antonym.lower()}",
                    start=seg.start,
                    end=seg.end,
                ))

                progress.advance(task)

        return results
