from src.metrics import classification_metrics


def test_perfect_classification_metrics():
    metrics = classification_metrics([0, 1, 0, 1], [0.1, 0.9, 0.2, 0.8])
    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
