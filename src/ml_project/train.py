from __future__ import annotations

from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


@dataclass
class TrainedModels:
    logistic_regression: LogisticRegression
    random_forest: RandomForestClassifier


def train_models(X_train, y_train, random_state: int = 42) -> TrainedModels:
    """
    Train two baseline models:
    1) Logistic Regression
    2) Random Forest
    """
    # Logistic Regression: simple linear baseline
    lr = LogisticRegression(max_iter=5000, random_state=random_state)
    lr.fit(X_train, y_train)

    # Random Forest: non-linear model using many decision trees
    rf = RandomForestClassifier(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    return TrainedModels(logistic_regression=lr, random_forest=rf)
