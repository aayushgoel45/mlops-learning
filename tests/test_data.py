import pandas as pd
import pytest

from kpi_sentinel.data import load_data

def test_load_data_file_not_found(tmp_path):
    missing_file = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError):
        load_data(missing_file)

def test_load_data(tmp_path):
    test_file = tmp_path / "test.csv"

    expected_df = pd.DataFrame(
        {
            "age": [25, 32],
            "income": [50000, 75000],
            "purchased": [0, 1],
        }
    )

    expected_df.to_csv(test_file, index=False)

    actual_df = load_data(test_file)

    pd.testing.assert_frame_equal(actual_df, expected_df)