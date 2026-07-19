import pandas as pd
import pytest

from kpi_sentinel.preprocessing import prepare_data, split_data

def test_prepare_data():
    df = pd.DataFrame(
        {
            "age": [25, 32],
            "income": [50000, 75000],
            "purchased": [0, 1],
        }
    )

    X, y = prepare_data(df)

    assert list(X.columns) == ["age", "income"]
    assert list(y) == [0, 1]

def test_split_data_is_reproducible():
    df = pd.DataFrame(
        {
            "age": range(20),
            "income": range(50000, 70000, 1000),
            "purchased": [0, 1] * 10,
        }
    )

    X, y = prepare_data(df)

    split_1 = split_data(X, y)
    split_2 = split_data(X, y)

    X_train_1, X_test_1, y_train_1, y_test_1 = split_1
    X_train_2, X_test_2, y_train_2, y_test_2 = split_2

    assert X_train_1.equals(X_train_2)
    assert X_test_1.equals(X_test_2)
    assert y_train_1.equals(y_train_2)
    assert y_test_1.equals(y_test_2)

    assert len(X_train_1) == 16
    assert len(X_test_1) == 4

    assert y_train_1.value_counts(normalize=True).to_dict() == {0: 0.5, 1: 0.5}
    assert y_test_1.value_counts(normalize=True).to_dict() == {0: 0.5, 1: 0.5}


def test_prepare_data_missing_target():
    df = pd.DataFrame(
        {
            "age": [25, 32],
            "income": [50000, 75000],
        }
    )

    with pytest.raises(KeyError):
        prepare_data(df)