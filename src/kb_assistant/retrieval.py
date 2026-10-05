"""Отбор найденных чанков из разных файлов для разнообразия источников в контексте."""

from kb_assistant.models import SearchResult


def diversify(results: list[SearchResult], top_k: int, max_per_source: int) -> list[SearchResult]:
    """Взять top_k чанков, по возможности не больше max_per_source из одного файла.

    results должны быть отсортированы от самого похожего к наименее похожему.
    """
    picked: list[SearchResult] = []
    deferred: list[SearchResult] = []
    per_source: dict[str, int] = {}

    for result in results:
        # если набрали нужное количество чанков - выходим из цикла
        if len(picked) == top_k:
            break
        source = result.chunk.meta.source
        # если из файла взято меньше лимита - добавляем в picked еще 1 чанк
        if per_source.get(source, 0) < max_per_source:
            picked.append(result)
            per_source[source] = per_source.get(source, 0) + 1
        else:
            deferred.append(result)

    # набираем чанки из отложенных в deferred если не добрали до количества чанков top_k
    picked += deferred[: top_k - len(picked)]
    # сортируем чанки по "похожести", сначала будут те, у которых выше score. Они первыми пойдут в контекст
    return sorted(picked, key=lambda result: result.score, reverse=True)