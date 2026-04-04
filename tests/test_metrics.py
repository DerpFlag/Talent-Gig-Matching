from src.eval.metrics import precision_at_k, recall_at_k, reciprocal_rank


def test_metrics_basic_values() -> None:
    predicted = ["a", "b", "c", "d"]
    relevant = {"b", "x"}
    assert precision_at_k(predicted, relevant, 2) == 0.5
    assert recall_at_k(predicted, relevant, 2) == 0.5
    assert reciprocal_rank(predicted, relevant) == 0.5
