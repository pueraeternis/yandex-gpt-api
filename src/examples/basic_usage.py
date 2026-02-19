import logging
import sys
from pathlib import Path

# Standard boilerplate to ensure 'src' is importable when running from the 'examples' folder
# This allows students to run the file directly via IDE or CLI without setting PYTHONPATH manually.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from openai import OpenAIError

from src.clients.wrapper import get_openai_client
from src.config import config

# Configure local logger for this script
logger = logging.getLogger("basic_usage")
logging.basicConfig(level=logging.INFO)


def demonstrate_simple_completion():
    """
    Demonstrates a single synchronous request to YandexGPT using the OpenAI SDK wrapper.
    """
    # 1. Initialize the client using our helper factory
    # This hides the complexity of setting base_url and headers
    client = get_openai_client()

    # 2. Define the prompt
    user_prompt = "Explain the difference between List and Tuple in Python."

    # 3. Define messages structure (Standard Chat format)
    messages = [
        {
            "role": "system",
            "content": "You are a senior Python instructor. Be concise and use code examples.",
        },
        {
            "role": "user",
            "content": user_prompt,
        },
    ]

    logger.info("Sending request to model: %s", config.model_name)

    try:
        # 4. Make the API call
        # We use config.model_uri to get the full "gpt://folder_id/..." string
        response = client.chat.completions.create(
            model=config.model_uri,
            messages=messages,  # pyright: ignore[reportArgumentType]
            temperature=0.3,  # Lower temperature for more factual answers
            max_tokens=1000,
        )

        # 5. Extract the content
        # Always check if choices exist to avoid IndexError
        if not response.choices:
            logger.warning("No choices returned from the model.")
            return

        answer = response.choices[0].message.content

        # 6. Display the result
        print("\n" + "=" * 40)
        print(f"Question: {user_prompt}")
        print("-" * 40)
        print(f"Answer:\n{answer}")
        print("=" * 40 + "\n")

        # Optional: Log token usage for cost estimation
        usage = response.usage
        if usage:
            logger.info(
                "Token usage - Prompt: %d, Completion: %d, Total: %d",
                usage.prompt_tokens,
                usage.completion_tokens,
                usage.total_tokens,
            )

    except OpenAIError as e:
        logger.error("API request failed: %s", e)
    except Exception:
        logger.exception("An unexpected error occurred:")


def main():
    """Entry point."""
    print("Starting Basic Usage Demo...")
    demonstrate_simple_completion()
    print("Demo finished.")


if __name__ == "__main__":
    main()
