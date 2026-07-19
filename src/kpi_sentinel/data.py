import pandas as pd

def load_data(file_path):
    """Load the dataset for model training."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Training data not found: {file_path}")

    df = pd.read_csv(file_path)

    return df