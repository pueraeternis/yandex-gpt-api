import logging
import os
import sys
from typing import Any

import requests
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Constants for Native API
# Note: Native API uses 'llm.api.cloud.yandex.net' and 'foundationModels' path
API_BASE_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
DEFAULT_MODEL_NAME = "yandexgpt-lite"  # Use "yandexgpt" for the Pro version
DEFAULT_TEMPERATURE = 0.6
DEFAULT_MAX_TOKENS = 2000


def load_config() -> tuple[str, str]:
    """
    Load and validate FOLDER_ID and API_KEY from environment.

    Returns:
        tuple[str, str]: (folder_id, api_key)

    """
    load_dotenv()
    folder_id = os.getenv("ID", "").strip()
    api_key = os.getenv("API_KEY", "").strip()

    if not folder_id or not api_key:
        logger.error("Missing ID or API_KEY in environment variables.")
        sys.exit(1)

    return folder_id, api_key


def build_payload(
    folder_id: str,
    user_prompt: str,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """
    Construct the JSON payload for the Native YandexGPT API.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "text": system_prompt})

    messages.append({"role": "user", "text": user_prompt})

    # URI Format: gpt://<folder_id>/<model_name>/<version>
    model_uri = f"gpt://{folder_id}/{DEFAULT_MODEL_NAME}/latest"

    payload = {
        "modelUri": model_uri,
        "completionOptions": {
            "stream": False,
            "temperature": DEFAULT_TEMPERATURE,
            "maxTokens": str(DEFAULT_MAX_TOKENS),
        },
        "messages": messages,
    }
    return payload


def generate_text_requests(
    folder_id: str,
    api_key: str,
    prompt: str,
) -> str:
    """
    Generate text using YandexGPT via direct REST API calls.
    """
    headers = {
        "Authorization": f"Api-Key {api_key}",
        "x-folder-id": folder_id,
        "Content-Type": "application/json",
    }

    payload = build_payload(folder_id, prompt, system_prompt="You are a helpful assistant.")

    logger.info("Sending native request to YandexGPT (Model: %s)", DEFAULT_MODEL_NAME)

    try:
        response = requests.post(API_BASE_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()

        result_json = response.json()
        alternatives = result_json.get("result", {}).get("alternatives", [])

        if not alternatives:
            logger.warning("No alternatives returned from API.")
            return ""

        text_content = alternatives[0].get("message", {}).get("text", "")
        logger.info("Successfully received response (%d chars)", len(text_content))
        return text_content

    except requests.exceptions.RequestException as e:
        logger.error("HTTP Request failed: %s", e)
        if e.response is not None:
            logger.error("API Error details: %s", e.response.text)
        return ""


def main():
    """Main execution entry point."""
    folder_id, api_key = load_config()

    test_prompt = "Explain why the sky is blue in one short sentence."

    response = generate_text_requests(folder_id, api_key, test_prompt)

    if response:
        print("\n--- YandexGPT (Requests) Response ---")
        print(response)


if __name__ == "__main__":
    main()
