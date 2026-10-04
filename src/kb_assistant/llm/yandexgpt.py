"""Клиент YandexGPT через REST API Yandex Foundation Models."""

import logging
import time

import httpx

from kb_assistant.llm.base import LLMClient, LLMResponse

logger = logging.getLogger(__name__)

COMPLETION_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
FINAL_STATUS = "ALTERNATIVE_STATUS_FINAL"
RETRY_STATUS_CODES = {429, 500, 502, 503, 504}  # временные сбои: стоит повторить
MAX_ATTEMPTS = 3


class YandexGPTClient(LLMClient):
    """YandexGPT по контракту LLMClient."""

    def __init__(
        self,
        api_key: str,
        folder_id: str,
        model: str,
        temperature: float,
        max_tokens: int = 2000,
        timeout: float = 60.0,
    ) -> None:
        self._model_uri = f"gpt://{folder_id}/{model}"
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._http = httpx.Client(
            timeout=timeout,
            headers={"Authorization": f"Api-Key {api_key}", "x-folder-id": folder_id},
        )

    def generate(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        body = {
            "modelUri": self._model_uri,
            "completionOptions": {
                "stream": False,
                "temperature": self._temperature,
                "maxTokens": str(self._max_tokens),
            },
            "messages": [
                {"role": "system", "text": system_prompt},
                {"role": "user", "text": user_prompt},
            ],
        }
        result = self._post(body)["result"]
        alternative = result["alternatives"][0]
        # проверяем что в ответе пришел имеено ответ на запрос, а не сообщение об ошибке, например из-за ограничений безопасности
        if alternative["status"] != FINAL_STATUS:
            logger.warning("YandexGPT вернул статус %s", alternative["status"])
        usage = result.get("usage", {})
        return LLMResponse(
            text=alternative["message"]["text"].strip(),
            input_tokens=int(usage.get("inputTextTokens", 0)),
            output_tokens=int(usage.get("completionTokens", 0)),
            model_version=result.get("modelVersion"),
        )

    def _post(self, body: dict) -> dict:
        """Отправить запрос; при временном сбое повторить с паузой."""
        for attempt in range(1, MAX_ATTEMPTS + 1):
            response = self._http.post(COMPLETION_URL, json=body)
            if response.status_code in RETRY_STATUS_CODES and attempt < MAX_ATTEMPTS:
                delay = 2 ** attempt
                logger.warning("YandexGPT ответил %d, повтор через %d с", response.status_code, delay)
                time.sleep(delay)
                continue
            response.raise_for_status()
            return response.json()
        raise RuntimeError("YandexGPT: исчерпаны попытки выполнить запрос к модели")