from scarce_shot_learn.zero_shot import zsl_base
import attr
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

@attr.s
class ESZSLearner(zsl_base.ZeroShotClassifier):
    """
        Embarassingly Simple Zero Shot Learning
        see http://proceedings.mlr.press/v37/romera-paredes15.pdf
        for the paper
    """
    lmbda = attr.ib(default=1e-2) 
    gamma = attr.ib(default=1e-2) 

    def fit(self, X, y, class_attributes):
        le = LabelEncoder()
        ohe = OneHotEncoder()
        y_labels_encoded = le.fit_transform(y)

        y_labels_ohe = ohe.fit_transform(y_labels_encoded.reshape(-1, 1)).toarray()
        X_correlation_term_inv = np.linalg.pinv(X.T @ X + self.gamma * np.eye(X.shape[1]))
        attributes_correlation_term_inv = np.linalg.pinv(
            class_attributes.T @ class_attributes + self.lmbda * np.eye(class_attributes.shape[1]))
        X_times_ohe_times_attributes = X.T @ y_labels_ohe @ class_attributes
        self.attributes_to_attributes = X_correlation_term_inv @ X_times_ohe_times_attributes @ attributes_correlation_term_inv

    def predict(self, X, class_attributes, labels_to_attributes=None):
        if labels_to_attributes is None:
            labels_to_attributes = np.arange(class_attributes.shape[0])
        scores = X @ self.attributes_to_attributes @ class_attributes.T
        return labels_to_attributes[np.argmax(scores, axis=1)]

    fit.__doc__ = zsl_base.ZeroShotClassifier.fit.__doc__
    predict.__doc__ = zsl_base.ZeroShotClassifier.predict.__doc__
