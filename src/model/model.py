"""LightGBM model creation and hyperparameter tuning utilities.

This module provides functions to create and configure LightGBM classifiers
for sentiment analysis, including hyperparameter search capabilities.
"""

from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV


def create_lgbm_model():
    """
    Creates and returns a LightGBM classifier configured for sentiment analysis.
    
    The model is configured for multiclass classification with 3 classes:
    0 (Negative), 1 (Positive), 2 (Neutral).
    
    Returns:
        LGBMClassifier: Configured LightGBM classifier ready for training.
    """
    return LGBMClassifier(
        objective="multiclass",
        metric="multi_logloss",
        num_class=3,
        random_state=8,
        n_jobs=-1,  # Use all available CPU cores for efficiency
    )


def create_hyperparameter_search(estimator, n_iter=15):
    """
    Creates a RandomizedSearchCV object for hyperparameter optimization.
    
    Args:
        estimator: The base estimator to tune (typically LGBMClassifier).
        n_iter (int, optional): Number of parameter combinations to try. Defaults to 15.
        
    Returns:
        RandomizedSearchCV: Configured search object for hyperparameter tuning.
    """
    # Parameter grid with corrected parameter names and sensible ranges
    param_grid = {
        "learning_rate": [0.01, 0.05, 0.1],
        "n_estimators": [500, 1000, 1500],
        "num_leaves": [20, 31, 40],
        "max_depth": [10, 20, -1],
        "min_child_samples": [10, 20, 30],
        "subsample": [0.8, 0.9, 1.0],
        "colsample_bytree": [0.8, 0.9, 1.0],
        "reg_alpha": [0.01, 0.1, 1.0],
        "reg_lambda": [0.01, 0.1, 1.0],
    }

    random_search = RandomizedSearchCV(
        estimator=estimator,
        param_distributions=param_grid,
        n_iter=n_iter,
        cv=3,
        scoring="f1_weighted",  # A more robust metric than accuracy
        verbose=2,
        random_state=8,
        n_jobs=-1,  # Run cross-validation folds in parallel
    )
    return random_search
