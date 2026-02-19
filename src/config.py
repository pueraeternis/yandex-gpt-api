import logging
import os
import sys
from dataclasses import dataclass

from dotenv import load_dotenv

# Configure logging centrally
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("yandex_gpt")

load_dotenv()


@dataclass(frozen=True)
class AppConfig:
    """
    Application configuration.
    """

    folder_id: str
    api_key: str
    model_name: str = "yandexgpt-lite"
    temperature: float = 0.6
    max_tokens: int = 2000

    native_api_url: str = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
    openai_base_url: str = "https://ai.api.cloud.yandex.net/v1"

    @classmethod
    def from_env(cls) -> "AppConfig":
        folder_id = os.getenv("YC_FOLDER_ID", "").strip()
        api_key = os.getenv("YC_API_KEY", "").strip()

        if not folder_id or not api_key:
            logger.critical("Missing YC_FOLDER_ID or YC_API_KEY.")
            sys.exit(1)

        return cls(folder_id=folder_id, api_key=api_key)

    @property
    def model_uri(self) -> str:
        """URI for text generation model."""
        return f"gpt://{self.folder_id}/{self.model_name}/latest"

    @property
    def embedding_doc_uri(self) -> str:
        """Model for indexing documents (Database)."""
        return f"emb://{self.folder_id}/text-search-doc/latest"

    @property
    def embedding_query_uri(self) -> str:
        """Model for processing user queries."""
        return f"emb://{self.folder_id}/text-search-query/latest"


config = AppConfig.from_env()
