import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class HandCentering(BaseEstimator, TransformerMixin):
    """Translate all landmarks so the wrist (landmark 0) is at the origin."""

    def __sklearn_is_fitted__(self):
        return True

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        data_array = X.values if hasattr(X, 'values') else X
        landmarks = data_array.reshape(-1, 21, 3)
        wrist = landmarks[:, 0:1, :]
        centered = landmarks - wrist
        return centered.reshape(-1, 63)


class HandNormalization(BaseEstimator, TransformerMixin):
    """Scale landmarks so the vector from wrist to middle-finger MCP has unit length."""

    def __sklearn_is_fitted__(self):
        return True

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        data_array = X.values if hasattr(X, 'values') else X
        landmarks = data_array.reshape(-1, 21, 3)
        middle = landmarks[:, 12, :]
        magnitude = np.linalg.norm(middle, axis=1, keepdims=True).reshape(-1, 1, 1)
        magnitude = np.where(magnitude == 0, 1.0, magnitude)   # avoid division by zero
        normalized = landmarks / magnitude
        return normalized.reshape(-1, 63)
