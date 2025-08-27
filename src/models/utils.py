"""Model evaluation utilities.

This module provides functions for evaluating model performance
using standard classification metrics.
"""

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


def model_metrics(y_true, y_pred):
    """Calculate and display comprehensive classification metrics.
    
    Args:
        y_true (array-like): True labels.
        y_pred (array-like): Predicted labels.
        
    Prints:
        Accuracy, precision, recall, F1-score, and confusion matrix.
    """
    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted")
    rec = recall_score(y_true, y_pred, average="weighted")
    f1 = f1_score(y_true, y_pred, average="weighted")
    confusion = confusion_matrix(y_true, y_pred)

    # Display results
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"Precision: {prec*100:.2f}%")
    print(f"Recall: {rec*100:.2f}%")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion matrix:\n", confusion)
