import asyncio
import sys
from pathlib import Path

# --- CONFIGURATION & PATH SETUP ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from openai import AsyncOpenAI, OpenAIError

from src.clients.wrapper import get_async_openai_client
from src.config import config, logger

# Constants
MAX_CONCURRENT_REQUESTS = 5


async def fetch_completion_safe(
    client: AsyncOpenAI,
    prompt: str,
    sem: asyncio.Semaphore,
) -> str:
    """
    Fetch completion with semaphore protection and error handling.

    Args:
        client: The active AsyncOpenAI client instance.
        prompt: The user prompt to send.
        sem: Semaphore to control concurrency.

    Returns:
        Generated text or empty string on failure.

    """
    async with sem:  # Acquire lock to limit concurrency
        try:
            logger.debug("Processing prompt: %.30s...", prompt)

            response = await client.chat.completions.create(
                model=config.model_uri,
                messages=[
                    {"role": "system", "content": "You are a concise technical expert."},
                    {"role": "user", "content": prompt},
                ],
                temperature=config.temperature,
                max_tokens=1000,
            )

            # Check for content existence safely
            if not response.choices:
                logger.warning("No choices returned for prompt: %.20s", prompt)
                return ""

            return response.choices[0].message.content or ""

        except OpenAIError as e:
            # Log the error but don't crash the whole batch
            logger.error("Async request failed for prompt '%.20s': %s", prompt, e)
            return ""
        except Exception:
            # Catch unexpected errors (e.g., parsing)
            logger.exception("Unexpected error processing prompt '%.20s'", prompt)
            return ""


async def process_batch(prompts: list[str]) -> list[str]:
    """
    Orchestrate the batch processing of prompts using a connection pool.

    Uses 'async with' to ensure the client is properly closed even if
    interruptions occur.
    """
    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    # Initialize client within a context manager for safety
    async with get_async_openai_client() as client:
        logger.info("Starting batch processing of %d items...", len(prompts))

        # Create tasks
        tasks = [fetch_completion_safe(client, p, sem) for p in prompts]

        # Execute tasks concurrently
        results = await asyncio.gather(*tasks)

    return results


def main():
    """Main entry point for the async demo."""
    # Simulating a dataset related to high-performance computing
    test_prompts = [
        "Explain NVIDIA A100 Tensor Cores.",
        "What is the difference between FP32 and BF16?",
        "How does Gradient Accumulation work?",
        "Explain Data Parallelism vs Model Parallelism.",
        "What is NCCL in distributed training?",
        "Optimize a matrix multiplication in Python.",
    ]

    try:
        results = asyncio.run(process_batch(test_prompts))
    except KeyboardInterrupt:
        logger.info("Batch processing interrupted by user.")
        sys.exit(0)

    print("\n--- Batch Processing Results ---")
    # 'strict=True' ensures input and output lists have the same length
    for prompt, res in zip(test_prompts, results, strict=True):
        print(f"\nQ: {prompt}")
        # Clean up newlines for cleaner console output
        clean_res = res.replace("\n", " ")
        print(f"A: {clean_res[:100]}... (truncated)")


if __name__ == "__main__":
    main()
