"""training module"""

import pickle
from typing import Any, Dict
import numpy as np
from sklearn.metrics import (precision_score, recall_score,
                             accuracy_score, f1_score,
                             confusion_matrix)
import pandas as pd
from sklearn.model_selection import GridSearchCV


def train_model(X_train: np.array, y_train: np.array, model: Any) -> None:
    """train the model"""

    model.fit(X_train, y_train)


def test_model(model: Any, X_test: np.array) -> np.array:
    """test models"""

    return model.predict(X_test)


def get_scores(y_true: np.array, y_test: np.array) -> Dict:
    """get all scores"""

    return {'accuracy':accuracy_score(y_true, y_test), 'precision':precision_score(y_true, y_test),
            'recall':recall_score(y_true, y_test), 'f1': f1_score(y_true, y_test)}


def get_confusion_matrix(y_true: np.array, y_test: np.array) -> np.array:
    """get confusion matrix"""

    return confusion_matrix(y_true, y_test)


def grid_search(model, param_grid, X_train, y_train,
                scoring="accuracy", cv=5, n_jobs=-1, verbose=1):
    """Generic hyperparameter tuning using GridSearchCV."""

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring=scoring,
                               cv=cv, n_jobs=n_jobs, verbose=verbose, refit=True,
                               return_train_score=True)

    grid_search.fit(X_train, y_train)
    results = pd.DataFrame(grid_search.cv_results_)
    results = results.sort_values(by="rank_test_score").reset_index(drop=True)

    return {
        "best_model": grid_search.best_estimator_,
        "best_params": grid_search.best_params_,
        "best_score": grid_search.best_score_,
        "results": results,
        "grid_search": grid_search
    }


def save_model(model: Any) -> None:
    """save model"""

    with open('model.pkl','wb') as f:
        pickle.dump(model, f)


def load_model(pickle_model_path: str) -> Any:
    """load model"""

    with open(pickle_model_path, 'rb') as f:
        model = pickle.load(f)
    return model
