import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


class ESZSLearner:
    """
        Embarassingly Simple Zero Shot Learning
        see http://proceedings.mlr.press/v37/romera-paredes15.pdf
        for the paper
    """
    def __init__(self, lmbda=1e-2, gamma=1e-2):
        self.lmbda = lmbda
        self.gamma = gamma

    def fit(self, X, y, attributes):
        le = LabelEncoder()
        ohe = OneHotEncoder()
        y_labels_encoded = le.fit_transform(y)

        y_labels_ohe = ohe.fit_transform(y_labels_encoded.reshape(-1, 1)).toarray()
        X_correlation_term_inv = np.linalg.pinv(X.T @ X + self.gamma * np.eye(X.shape[1]))
        attributes_correlation_term_inv = np.linalg.pinv(
            attributes.T @ attributes + self.lmbda * np.eye(attributes.shape[1]))
        X_times_ohe_times_attributes = X.T @ y_labels_ohe @ attributes
        self.attributes_to_attributes = X_correlation_term_inv @ X_times_ohe_times_attributes @ attributes_correlation_term_inv

    def predict(self, X, attributes, attributes_to_labels=None):
        if attributes_to_labels is None:
            attributes_to_labels = np.arange(attributes.shape[0])
        scores = X @ self.attributes_to_attributes @ attributes.T
        return attributes_to_labels[np.argmax(scores, axis=1)]
