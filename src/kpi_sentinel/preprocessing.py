from sklearn.model_selection import train_test_split
from kpi_sentinel.config import TEST_SIZE, RANDOM_STATE

def prepare_data(df):
    """Split the dataset into features and target."""

    X = df.drop(columns=["purchased"])
    y = df["purchased"]
    return X,y

def split_data(X, y):
    """Split features and target into training and test sets."""

    return train_test_split(
        X, 
        y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE,
        stratify=y,
    )
