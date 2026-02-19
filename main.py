import sys

from src.clients.native import YandexNativeClient
from src.clients.wrapper import get_openai_client
from src.config import config, logger


def run_native_demo():
    client = YandexNativeClient()
    prompt = "Explain why Clean Code is important in Python."
    result = client.generate_text(prompt)
    print(f"\n[Native API] Result:\n{result}\n")


def run_openai_sdk_demo():
    client = get_openai_client()
    prompt = "Write a hello world in Rust."

    logger.info("Sending request via OpenAI SDK...")
    try:
        response = client.chat.completions.create(
            model=config.model_uri,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content
        print(f"\n[OpenAI SDK] Result:\n{content}\n")
    except Exception as e:
        logger.error("OpenAI SDK demo failed: %s", e)


def main():
    """Main CLI entry point."""
    print(f"Running YandexGPT Demo (Model: {config.model_name})")
    print("-" * 50)

    if len(sys.argv) > 1 and sys.argv[1] == "native":
        run_native_demo()
    else:
        # Default to SDK as it's the modern standard
        run_openai_sdk_demo()


if __name__ == "__main__":
    main()
