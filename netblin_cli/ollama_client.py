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

        resp = requests.post(self.api_generate, json=payload, timeout=300)
        resp.raise_for_status()

        return resp.json().get("response", "")

    def _parse_response(self, response: str) -> list[dict]:
        """Парсит JSON из ответа LLM."""
        response = response.strip()

        # Убираем markdown блоки
        if response.startswith("```"):
            lines = response.split("\n")
            response = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])

        # Ищем JSON массив в ответе
        start_idx = response.find("[")
        end_idx = response.rfind("]")

        if start_idx != -1 and end_idx != -1:
            response = response[start_idx:end_idx + 1]

        return json.loads(response)

    def process_transcript(self, transcript_json: str) -> list[Segment]:
        """Обрабатывает транскрипцию и возвращает сегменты с антонимами."""
        words = json.loads(transcript_json)

        console.print(f"[cyan]Модель:[/cyan] {self.model}")

        # Разбиваем на чанки по ~50 слов
        chunk_size = 50
        all_segments = []

        for i in range(0, len(words), chunk_size):
            chunk = words[i:i + chunk_size]
            chunk_json = json.dumps(chunk, ensure_ascii=False)

            prompt = LLM_PROMPT.format(transcription=chunk_json)

            with console.status(f"[bold green]Обрабатываю слова {i+1}-{min(i+chunk_size, len(words))}..."):
                response = self.generate(prompt)

            try:
                data = self._parse_response(response)
                for item in data:
                    # Пропускаем сегменты без таймкодов
                    start_val = item.get("start", "")
                    end_val = item.get("end", "")
                    if start_val == "" or end_val == "":
                        continue

                    all_segments.append(Segment(
                        original=item.get("original", ""),
                        antonym=item.get("antonym", ""),
                        start=float(start_val),
                        end=float(end_val),
                    ))
            except (json.JSONDecodeError, ValueError) as e:
                console.print(f"[yellow]Чанк {i//chunk_size + 1} не распарсился, пропускаю[/yellow]")
                console.print(f"[dim]{response[:200]}[/dim]")
                continue

        if not all_segments:
            raise ValueError("Не удалось получить ни одного сегмента от LLM")

        return all_segments
