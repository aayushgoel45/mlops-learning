from kpi_sentinel.data import load_data
from kpi_sentinel.config import DATA_DIR, RANDOM_STATE, TEST_SIZE
from kpi_sentinel.preprocessing import prepare_data, split_data
from kpi_sentinel.evaluation import evaluate_model
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def main():
    train_path = DATA_DIR / "train.csv"

    df = load_data(train_path)

    X, y  = prepare_data(df)

    X_train, X_test, y_train, y_test = split_data(X,y)
    
    
    cv = StratifiedKFold(n_splits=4, shuffle = True, random_state=RANDOM_STATE)

    best_model = train_model(X_train, y_train, cv)

    metrics = evaluate_model(
        best_model,
        X_test,
        y_test,
        )
    
    print(f"Test Accuracy: {metrics['accuracy']:.2f}")
    print("Confusion Matrix:")
    print(metrics["confusion_matrix"])
    print("Classification Report:")
    print(metrics["classification_report"])


def train_model(X_train, y_train, cv):
    """Train a Random Forest model using GridSearchCV."""
    
    model = RandomForestClassifier(
        random_state=RANDOM_STATE
    )

    param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [None, 5],
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring="accuracy",
        )
    
    grid_search.fit(X_train, y_train)

    print("Best Parameters:", grid_search.best_params_)
    print("Best CV Score:", grid_search.best_score_)

    return grid_search.best_estimator_



if __name__ == "__main__":
    main()