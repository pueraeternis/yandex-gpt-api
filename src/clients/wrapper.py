from openai import AsyncOpenAI, OpenAI

from src.config import config


def get_openai_client() -> OpenAI:
    """
    Factory to create a synchronous OpenAI client configured for YandexGPT.
    """
    return OpenAI(
        base_url=config.openai_base_url,
        api_key=config.api_key,
        # In Yandex OpenAI-compatible API, 'project' maps to X-Folder-Id
        project=config.folder_id,
    )


def get_async_openai_client() -> AsyncOpenAI:
    """
    Factory to create an asynchronous OpenAI client configured for YandexGPT.
    """
    return AsyncOpenAI(
        base_url=config.openai_base_url,
        api_key=config.api_key,
        project=config.folder_id,
    )
