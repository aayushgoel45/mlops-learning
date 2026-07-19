from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model on test data."""

    predictions = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "confusion_matrix": confusion_matrix(y_test, predictions),
        "classification_report": classification_report(
            y_test,
            predictions,
            output_dict=True,
            )
    }

    return metrics
