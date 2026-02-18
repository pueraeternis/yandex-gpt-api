import asyncio
import logging
import os
import sys

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAIError

# Configure logging with lazy formatting
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Constants
YANDEX_OPENAI_BASE_URL = "https://ai.api.cloud.yandex.net/v1"
DEFAULT_MODEL_NAME = "yandexgpt-lite"


def load_config() -> tuple[str, str]:
    """Load config from env."""
    load_dotenv()
    f_id, a_key = os.getenv("ID", ""), os.getenv("API_KEY", "")
    if not f_id or not a_key:
        logger.error("Environment variables ID and API_KEY are required.")
        sys.exit(1)
    return f_id, a_key


async def fetch_completion(client: AsyncOpenAI, folder_id: str, prompt: str) -> str:
    """
    Asynchronously fetch completion from YandexGPT.
    """
    model_uri = f"gpt://{folder_id}/{DEFAULT_MODEL_NAME}/latest"
    try:
        response = await client.chat.completions.create(
            model=model_uri,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1000,
        )
        return response.choices[0].message.content or ""
    except OpenAIError as e:
        logger.error("Async API call failed: %s", e)
        return ""


async def main():
    """Main async entry point for batch processing."""
    folder_id, api_key = load_config()

    # Initialize Async Client
    client = AsyncOpenAI(
        base_url=YANDEX_OPENAI_BASE_URL,
        api_key=api_key,
        project=folder_id,
    )

    # Example: Processing a batch of prompts
    prompts = [
        "Explain A100 GPU architecture.",
        "What is bfloat16 and why is it used in DL?",
        "Write a clean python decorator example.",
    ]

    logger.info("Starting batch processing of %d prompts", len(prompts))

    # Run tasks in parallel
    tasks = [fetch_completion(client, folder_id, p) for p in prompts]
    results = await asyncio.gather(*tasks)

    for i, res in enumerate(results):
        logger.info("Result %d: %s...", i, res[:50].replace("\n", " "))

    await client.close()


if __name__ == "__main__":
    asyncio.run(main())
