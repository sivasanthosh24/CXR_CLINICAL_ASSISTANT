from typing import Any, Dict, Optional

import requests

from .schemas import OllamaResponse


class OllamaClient:
    def __init__(
        self,
        host: str = "http://localhost:11434",
        model: str = "phi3:latest",
        timeout: int = 120,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ):
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.temperature = temperature
        self.max_tokens = max_tokens

    def is_available(self) -> bool:
        try:
            response = requests.get(
                f"{self.host}/api/tags",
                timeout=10,
            )
            return response.ok
        except requests.RequestException:
            return False

    def list_models(self) -> Dict[str, Any]:
        try:
            response = requests.get(
                f"{self.host}/api/tags",
                timeout=10,
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as error:
            raise RuntimeError(
                f"Unable to connect to Ollama at {self.host}: {error}"
            ) from error

    def model_available(self) -> bool:
        try:
            data = self.list_models()
            models = data.get("models", [])

            available_names = set()

            for item in models:
                name = item.get("name")
                if name:
                    available_names.add(name)

            return self.model in available_names

        except RuntimeError:
            return False

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
    ) -> OllamaResponse:
        if not prompt or not prompt.strip():
            return OllamaResponse(
                text="",
                model=self.model,
                success=False,
                error="The prompt is empty.",
            )

        payload: Dict[str, Any] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        if system_prompt:
            payload["system"] = system_prompt

        try:
            response = requests.post(
                f"{self.host}/api/generate",
                json=payload,
                timeout=self.timeout,
            )

            response.raise_for_status()

            data = response.json()

            generated_text = str(
                data.get("response", "")
            ).strip()

            if not generated_text:
                return OllamaResponse(
                    text="",
                    model=self.model,
                    success=False,
                    error="Ollama returned an empty response.",
                    raw_response=data,
                )

            return OllamaResponse(
                text=generated_text,
                model=self.model,
                success=True,
                raw_response=data,
            )

        except requests.Timeout as error:
            return OllamaResponse(
                text="",
                model=self.model,
                success=False,
                error=f"Ollama request timed out: {error}",
            )

        except requests.ConnectionError:
            return OllamaResponse(
                text="",
                model=self.model,
                success=False,
                error=(
                    "Unable to connect to the local Ollama service. "
                    "Make sure Ollama is running."
                ),
            )

        except requests.HTTPError as error:
            response_text = ""

            if error.response is not None:
                response_text = error.response.text[:500]

            return OllamaResponse(
                text="",
                model=self.model,
                success=False,
                error=(
                    f"Ollama returned an HTTP error: {error}. "
                    f"{response_text}"
                ).strip(),
            )

        except requests.RequestException as error:
            return OllamaResponse(
                text="",
                model=self.model,
                success=False,
                error=f"Ollama request failed: {error}",
            )

        except Exception as error:
            return OllamaResponse(
                text="",
                model=self.model,
                success=False,
                error=f"Unexpected Ollama error: {error}",
            )