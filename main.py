from openai import OpenAIError

from src.clients.native import YandexNativeClient
from src.clients.wrapper import get_openai_client
from src.config import config, logger


def run_native_demo(prompt: str):
    """
    Demonstrates usage of the custom Native Client (based on 'requests').
    """
    print("\n>>> [1] Running Native API Request...")

    try:
        client = YandexNativeClient()
        response = client.generate_text(prompt)

        if response:
            print(f"Result:\n{response}")
        else:
            print("No response received.")

    except Exception as e:
        logger.error("Native demo failed: %s", e)


def run_sdk_demo(prompt: str):
    """
    Demonstrates usage of the Standard OpenAI SDK.
    """
    print("\n>>> [2] Running OpenAI SDK Request...")

    try:
        # Get configured client
        client = get_openai_client()

        # Make request
        response = client.chat.completions.create(
            model=config.model_uri,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
        )

        content = response.choices[0].message.content
        print(f"Result:\n{content}")

    except OpenAIError as e:
        logger.error("SDK request failed: %s", e)
    except Exception:
        logger.exception("Unexpected error in SDK demo")


def main():
    """
    Main entry point.
    Executes both approaches sequentially for demonstration.
    """
    print(f"--- YandexGPT Demo (Model: {config.model_name}) ---")

    # Common test prompt
    test_prompt = "Explain the difference between 'Process' and 'Thread' in 3 bullet points."

    print(f"Prompt: '{test_prompt}'")

    # 1. Run Native implementation
    run_native_demo(test_prompt)

    # 2. Run SDK implementation
    run_sdk_demo(test_prompt)

    print("\n--- Demo Finished ---")


if __name__ == "__main__":
    main()
