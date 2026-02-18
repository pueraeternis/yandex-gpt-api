import logging
import os
import sys

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
# UPDATED: Based on latest Yandex AI Studio documentation
YANDEX_OPENAI_BASE_URL = "https://ai.api.cloud.yandex.net/v1"
DEFAULT_MODEL_NAME = "yandexgpt-lite"  # Use 'yandexgpt' for Pro
DEFAULT_TEMPERATURE = 0.6


def load_config() -> tuple[str, str]:
    """
    Load and validate FOLDER_ID and API_KEY from environment.
    """
    load_dotenv()
    folder_id = os.getenv("ID", "").strip()
    api_key = os.getenv("API_KEY", "").strip()

    if not folder_id or not api_key:
        logger.error("Missing ID or API_KEY in environment variables.")
        sys.exit(1)

    return folder_id, api_key


def generate_text_openai_sdk(
    client: OpenAI,
    folder_id: str,
    prompt: str,
) -> str:
    """
    Generate text using the OpenAI SDK adapted for YandexGPT.
    """
    # Yandex requires the full model URI in the 'model' parameter
    model_uri = f"gpt://{folder_id}/{DEFAULT_MODEL_NAME}/latest"

    logger.info("Sending request via OpenAI SDK (Model URI: %s)", model_uri)

    try:
        response = client.chat.completions.create(
            model=model_uri,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=DEFAULT_TEMPERATURE,
        )

        content = response.choices[0].message.content
        if content:
            logger.info("Successfully received response (%d chars)", len(content))
            return content
        logger.warning("Received empty content from API.")
        return ""

    except OpenAIError as e:
        logger.error("OpenAI SDK error: %s", e)
        return ""


def main():
    """Main execution entry point."""
    folder_id, api_key = load_config()

    # Initialize OpenAI client with Yandex-specific parameters
    # 'project' parameter maps to the X-Folder-Id header required by Yandex
    client = OpenAI(
        base_url=YANDEX_OPENAI_BASE_URL,
        api_key=api_key,
        project=folder_id,
    )

    test_prompt = "Write a Python function to check if a number is prime."

    result = generate_text_openai_sdk(client, folder_id, test_prompt)

    if result:
        print("\n--- YandexGPT (OpenAI SDK) Response ---")
        print(result)


if __name__ == "__main__":
    main()
