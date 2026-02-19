from typing import Any

import requests

from src.config import config, logger


class YandexNativeClient:
    """
    Client for interacting with YandexGPT via the native REST API.
    """

    def __init__(self) -> None:
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Api-Key {config.api_key}",
                "x-folder-id": config.folder_id,
                "Content-Type": "application/json",
            },
        )

    def _build_payload(
        self,
        prompt: str,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        """Constructs the request payload."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "text": system_prompt})
        messages.append({"role": "user", "text": prompt})

        return {
            "modelUri": config.model_uri,
            "completionOptions": {
                "stream": False,
                "temperature": config.temperature,
                "maxTokens": str(config.max_tokens),
            },
            "messages": messages,
        }

    def generate_text(self, prompt: str, system_prompt: str = "You are a helpful assistant.") -> str:
        """
        Sends a synchronous request to YandexGPT.

        Args:
            prompt: User input text.
            system_prompt: Context for the AI.

        Returns:
            Generated text string or empty string on failure.

        """
        payload = self._build_payload(prompt, system_prompt)
        logger.info("Sending native request. Model: %s", config.model_name)

        try:
            response = self.session.post(
                config.native_api_url,
                json=payload,
                timeout=30,
            )
            response.raise_for_status()

            result = response.json()
            # Navigate safety through dictionary
            alternatives = result.get("result", {}).get("alternatives", [])

            if not alternatives:
                logger.warning("API returned no alternatives.")
                return ""

            text_content = alternatives[0].get("message", {}).get("text", "")
            logger.info("Request successful. Received %d chars.", len(text_content))
            return text_content

        except requests.exceptions.RequestException as e:
            logger.error("Native API Request failed: %s", e)
            if e.response is not None:
                logger.error("Error details: %s", e.response.text)
            return ""
