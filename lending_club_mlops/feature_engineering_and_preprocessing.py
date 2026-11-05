"""data and feature engineering module"""

from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.model_selection import train_test_split


def feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    """feature engineering"""

    data = pd.concat([pd.get_dummies(data['purpose']).drop('all_other', axis=1), data.drop('purpose', axis=1)], axis=1)
    return data


def split_data(data: pd.DataFrame, dependent_feature: str) -> Tuple[np.array]:
    """perform train test split"""

    X = data.drop(dependent_feature, axis=1).values
    y = data[dependent_feature].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                        stratify=y, random_state=52)
    return (X_train, X_test, y_train, y_test)


def scale_data(splits: Tuple, scaler: Normalizer|StandardScaler) -> Tuple:
    """scales the data"""

    scale = scaler
    X_train, X_test = scale.fit_transform(splits[0]) , scale.transform(splits[1])
    return (X_train, X_test)
