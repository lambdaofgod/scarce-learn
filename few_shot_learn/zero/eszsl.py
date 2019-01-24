import numpy as np
from sklearn.preprocessing import OneHotEncoder


class ESZSLearner:
    """
        Embarassingly Simple Zero Shot Learning
        see http://proceedings.mlr.press/v37/romera-paredes15.pdf
        for the paper
    """

    def __init__(self, lmbda, gamma):
        self.lmbda = lmbda
        self.gamma = gamma

    def fit(self, X, y_features, y_labels):
        ohe = OneHotEncoder()
        y_labels_ohe = ohe.fit_transform(y_labels.reshape(-1, 1)).toarray()
        X_correlation_term_inv = np.linalg.pinv(X.T @ X + self.gamma * np.eye(X.shape[1]))
        y_features_correlation_term_inv = np.linalg.pinv(y_features.T @ y_features + self.lmbda * np.eye(y_features.shape[1]))
        X_times_ohe_times_features = X.T @ y_labels_ohe @ y_features
        self.features_to_attributes = X_correlation_term_inv @ X_times_ohe_times_features @ y_features_correlation_term_inv

    def predict(self, X, y_features):
        scores = X @ self.features_to_attributes @ y_features.T
        return np.argmax(scores, axis=1)
