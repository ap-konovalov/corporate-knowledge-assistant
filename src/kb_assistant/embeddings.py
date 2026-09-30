"""Превращение текста в векторы (эмбеддинги) моделью семейства E5."""

from sentence_transformers import SentenceTransformer

# Модели E5 обучены с приписками: так они понимают, где вопрос, а где документ
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "


class Embedder:
    """Одна модель эмбеддингов — и для документов, и для вопросов."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model = SentenceTransformer(model_name)

    @property
    def dimension(self) -> int:
        """Сколько чисел в одном векторе."""
        return self._model.get_embedding_dimension()

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        """Векторы для кусочков документов."""
        vectors = self._model.encode(
            [PASSAGE_PREFIX + text for text in texts],
            normalize_embeddings=True,
            # считать пачками по 32 текста: так быстрее, чем по одному
            batch_size=32,
            show_progress_bar=True,
        )
        # модель отдаёт результат в формате библиотеки numpy (математика с массивами чисел), а Qdrant удобнее принимает обычные списки Python
        return vectors.tolist()

    def embed_query(self, text: str) -> list[float]:
        """Вектор для вопроса пользователя."""
        vector = self._model.encode(QUERY_PREFIX + text, normalize_embeddings=True)
        return vector.tolist()