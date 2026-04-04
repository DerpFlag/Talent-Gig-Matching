def precision_at_k(predicted_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be > 0")
    top = predicted_ids[:k]
    if not top:
        return 0.0
    hits = sum(1 for rid in top if rid in relevant_ids)
    return hits / k


def recall_at_k(predicted_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be > 0")
    if not relevant_ids:
        return 0.0
    top = predicted_ids[:k]
    hits = sum(1 for rid in top if rid in relevant_ids)
    return hits / len(relevant_ids)


def reciprocal_rank(predicted_ids: list[str], relevant_ids: set[str]) -> float:
    if not relevant_ids:
        return 0.0
    for idx, rid in enumerate(predicted_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / idx
    return 0.0
