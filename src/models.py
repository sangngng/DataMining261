import numpy as np
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from typing import Optional
import lightgbm as lgb

class TwoStageRecommender:
    """
    Two-Stage Recommendation Architecture:
      - Stage 1 (Filter): Binary classification to predict P(label > 1).
                          If P < threshold, paper is assigned priority 0 (filtered out).
      - Stage 2 (Rank): Regression model to predict continuous priority score for worthwhile papers.
    """
    def __init__(
        self,
        classifier=None,
        regressor=None,
        threshold: float = 0.40,
        random_state: int = 42
    ):
        self.random_state = random_state
        self.threshold = threshold
        
        self.clf = classifier if classifier is not None else lgb.LGBMClassifier(
            n_estimators=150,
            learning_rate=0.05,
            random_state=self.random_state,
            verbose=-1
        )
        self.reg = regressor if regressor is not None else lgb.LGBMRegressor(
            n_estimators=150,
            learning_rate=0.05,
            random_state=self.random_state,
            verbose=-1
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        # Stage 1: Binary label: 1 if worthwhile (y > 1), 0 if not (y == 1)
        y_binary = (y > 1.0).astype(int)
        self.clf.fit(X, y_binary)

        # Stage 2: Fit regressor on all data (or on y > 1 subset)
        self.reg.fit(X, y)
        return self

    def predict_priority(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        t = threshold if threshold is not None else self.threshold
        
        # Stage 1 probabilities: P(worthwhile)
        proba_worthwhile = self.clf.predict_proba(X)[:, 1]
        
        # Stage 2 raw predicted ratings
        raw_ratings = self.reg.predict(X)
        
        # Apply filtering threshold
        priorities = np.where(proba_worthwhile >= t, raw_ratings, 0.0)
        # Ensure non-negative
        priorities = np.clip(priorities, 0.0, None)
        return priorities
