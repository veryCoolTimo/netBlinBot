"""Клиент для Ollama API."""

import json
import requests
from dataclasses import dataclass
from rich.console import Console

from .config import OLLAMA_URL, OLLAMA_MODEL, LLM_PROMPT

console = Console()


@dataclass
class Segment:
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
                "temperature": 0.7,
                "num_predict": 4096,
            }
        }

        resp = requests.post(self.api_generate, json=payload, timeout=120)
        resp.raise_for_status()

        return resp.json().get("response", "")

    def process_transcript(self, transcript_json: str) -> list[Segment]:
        """Обрабатывает транскрипцию и возвращает сегменты с антонимами."""
        prompt = LLM_PROMPT.format(transcription=transcript_json)

        console.print(f"[cyan]Модель:[/cyan] {self.model}")

        with console.status("[bold green]Генерирую антонимы..."):
            response = self.generate(prompt)

        # Парсим JSON из ответа
        # LLM может вернуть markdown блок, очищаем
        response = response.strip()
        if response.startswith("```"):
            # Убираем ```json и ```
            lines = response.split("\n")
            response = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

        try:
            data = json.loads(response)
        except json.JSONDecodeError as e:
            console.print(f"[red]Ошибка парсинга JSON:[/red] {e}")
            console.print(f"[dim]Ответ LLM:[/dim]\n{response[:500]}")
            raise ValueError("LLM вернул невалидный JSON")

        segments = []
        for item in data:
            segments.append(Segment(
                original=item.get("original", ""),
                antonym=item.get("antonym", ""),
                start=float(item.get("start", 0)),
                end=float(item.get("end", 0)),
            ))

        return segments
