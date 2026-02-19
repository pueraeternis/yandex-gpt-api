import asyncio

from openai import AsyncOpenAI, OpenAIError

from src.clients.wrapper import get_async_openai_client
from src.config import config, logger

# Limit concurrent requests to avoid Rate Limiting (HTTP 429)
MAX_CONCURRENT_REQUESTS = 5


async def fetch_completion_safe(
    client: AsyncOpenAI,
    prompt: str,
    sem: asyncio.Semaphore,
) -> str:
    """
    Fetches completion with semaphore protection and error handling.
    """
    async with sem:  # Acquire lock
        try:
            logger.debug("Processing prompt: %s...", prompt[:30])
            response = await client.chat.completions.create(
                model=config.model_uri,
                messages=[
                    {"role": "system", "content": "You are a concise technical expert."},
                    {"role": "user", "content": prompt},
                ],
                temperature=config.temperature,
                max_tokens=1000,
            )
            content = response.choices[0].message.content or ""
            return content
        except OpenAIError as e:
            logger.error("Async request failed for prompt '%s': %s", prompt[:20], e)
            return ""


async def process_batch(prompts: list[str]):
    """
    Orchestrates the batch processing of prompts.
    """
    client = get_async_openai_client()
    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    tasks = [fetch_completion_safe(client, p, sem) for p in prompts]

    logger.info("Starting batch processing of %d items...", len(prompts))
    results = await asyncio.gather(*tasks)

    # Close client gracefully
    await client.close()

    return results


def run_demo():
    """Entry point for the async demo."""
    # Simulating a dataset related to high-performance computing
    test_prompts = [
        "Explain NVIDIA A100 Tensor Cores.",
        "What is the difference between FP32 and BF16?",
        "How does Gradient Accumulation work?",
        "Explain Data Parallelism vs Model Parallelism.",
        "What is NCCL in distributed training?",
        "Optimize a matrix multiplication in Python.",
    ]

    results = asyncio.run(process_batch(test_prompts))

    print("\n--- Batch Processing Results ---")
    for prompt, res in zip(test_prompts, results, strict=True):
        print(f"\nQ: {prompt}")
        print(f"A: {res[:100]}... (truncated)")


if __name__ == "__main__":
    run_demo()
