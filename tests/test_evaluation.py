import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from kpi_sentinel.evaluation import evaluate_model


def test_evaluate_model_returns_metrics():
    X_train = pd.DataFrame(
        {
            "age": [25, 32, 45, 28],
            "income": [50000, 75000, 120000, 62000],
        }
    )

    y_train = pd.Series([0, 1, 1, 0])

    X_test = pd.DataFrame(
        {
            "age": [26, 40],
            "income": [52000, 105000],
        }
    )

    y_test = pd.Series([0, 1])

    model = RandomForestClassifier(
        random_state=42
    )

    model.fit(X_train, y_train)

    metrics = evaluate_model(
        model,
        X_test,
        y_test,
    )

    assert "accuracy" in metrics
    assert "confusion_matrix" in metrics
    assert "classification_report" in metrics

    assert metrics["accuracy"] == 1.0