import pandas as pd
from sklearn.model_selection import StratifiedKFold

from kpi_sentinel.config import RANDOM_STATE
from kpi_sentinel.preprocessing import prepare_data, split_data
from kpi_sentinel.train import train_model


def test_train_model_returns_fitted_estimator():
    df = pd.DataFrame(
        {
            "age": [
                25, 32, 45, 28, 39,
                22, 35, 41, 29, 50,
                24, 31, 37, 43, 27,
                48, 34, 26, 40, 30,
            ],
            "income": [
                50000, 75000, 120000, 62000, 98000,
                42000, 85000, 110000, 58000, 130000,
                47000, 70000, 92000, 115000, 54000,
                125000, 80000, 52000, 105000, 65000,
            ],
            "purchased": [
                0, 1, 1, 0, 1,
                0, 1, 1, 0, 1,
                0, 0, 1, 1, 0,
                1, 1, 0, 1, 0,
            ],
        }
    )

    X, y = prepare_data(df)
    X_train, _, y_train, _ = split_data(X, y)

    cv = StratifiedKFold(
        n_splits=4,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    model = train_model(X_train, y_train, cv)

    assert hasattr(model, "predict")
    assert hasattr(model, "estimators_")