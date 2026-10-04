"""Настройки приложения: читаются из переменных окружения и файла .env."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Секреты YandexGPT
    yc_api_key: SecretStr
    yc_folder_id: str

    # Языковая модель
    llm_provider: Literal["yandexgpt"] = "yandexgpt"
    llm_model: str = "yandexgpt/latest"
    llm_temperature: float = Field(default=0.1, ge=0.0, le=1.0)

    # Векторная база
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "kb_docs"

    # Поиск и нарезка
    embedding_model: str = "intfloat/multilingual-e5-base"
    chunk_size: int = Field(default=800, gt=0)
    chunk_overlap: int = Field(default=100, ge=0)
    top_k: int = Field(default=4, ge=1, le=20)
    min_score: float = Field(default=0.0, ge=0.0, le=1.0)

    # Пути
    raw_data_dir: Path = Path("data/raw")


@lru_cache
def get_settings() -> Settings:
    return Settings()