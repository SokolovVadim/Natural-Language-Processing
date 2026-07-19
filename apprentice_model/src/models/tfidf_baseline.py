"""Classical TF-IDF + Logistic Regression baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


class TfidfLogisticRegressionBaseline:
    """ TF-IDF + Logistic Regression text classifier."""

    def __init__(
        self,
        max_features: int = 20_000,
        ngram_range: tuple[int, int] = (1, 2),
        lowercase: bool = True,
        stop_words: str | list[str] | None = None,
        class_weight: str | dict[int, float] | None = "balanced",
        max_iter: int = 1000,
        solver: str = "liblinear",
        random_state: int = 1234,
    ) -> None:
        """Create the baseline classifier.

        Args:
            max_features: Maximum number of TF-IDF features.
            ngram_range: Lower and upper n-gram range for TF-IDF features.
            lowercase: Whether TF-IDF should lowercase text.
            stop_words: Stop words passed to TF-IDF.
            class_weight: Logistic Regression class weighting strategy.
            max_iter: Maximum optimizer iterations for Logistic Regression.
            solver: Logistic Regression solver.
            random_state: Random seed for Logistic Regression.
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.lowercase = lowercase
        self.stop_words = stop_words
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.solver = solver
        self.random_state = random_state
        self.pipeline = self._build_pipeline()

    def _build_pipeline(self) -> Pipeline:
        """Create the sklearn pipeline."""
        vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            lowercase=self.lowercase,
            stop_words=self.stop_words,
        )
        classifier = LogisticRegression(
            class_weight=self.class_weight,
            max_iter=self.max_iter,
            solver=self.solver,
            random_state=self.random_state,
        )
        return Pipeline(
            steps=[
                ("tfidf", vectorizer),
                ("classifier", classifier),
            ]
        )

    def fit(
        self,
        texts: Iterable[str],
        labels: Iterable[int],
    ) -> TfidfLogisticRegressionBaseline:
        """Train the TF-IDF + Logistic Regression baseline."""
        self.pipeline.fit(texts, labels)
        return self

    def predict(self, texts: Iterable[str]) -> list[int]:
        """Predict class labels for input texts."""
        predictions = self.pipeline.predict(texts)
        return predictions.tolist()

    def predict_proba(self, texts: Iterable[str]) -> list[list[float]]:
        """Predict class probabilities for input texts."""
        probabilities = self.pipeline.predict_proba(texts)
        return probabilities.tolist()

    def save(self, path: str) -> None:
        """Save the trained baseline model to disk."""
        model_path = Path(path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, model_path)

    @classmethod
    def load(cls, path: str) -> TfidfLogisticRegressionBaseline:
        """Load a saved baseline model from disk."""
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError(f"Loaded object is not a {cls.__name__}: {type(model)!r}")
        return model


def create_tfidf_vectorizer(
    max_features: int = 20_000,
    ngram_range: tuple[int, int] = (1, 2),
    lowercase: bool = True,
    stop_words: str | list[str] | None = None,
) -> TfidfVectorizer:
    """Create a TF-IDF vectorizer with project defaults."""
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        lowercase=lowercase,
        stop_words=stop_words,
    )


def create_logistic_regression_classifier(
    class_weight: str | dict[int, float] | None = "balanced",
    max_iter: int = 1000,
    solver: str = "liblinear",
    random_state: int = 1234,
) -> LogisticRegression:
    """Create a Logistic Regression classifier with project defaults."""
    return LogisticRegression(
        class_weight=class_weight,
        max_iter=max_iter,
        solver=solver,
        random_state=random_state,
    )
