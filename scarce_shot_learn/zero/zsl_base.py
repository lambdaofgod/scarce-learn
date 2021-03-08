class ZeroShotClassifier:

    def fit(self, X, y, class_attributes):
        """
        X: {array-like, sparse matrix} of shape (n_samples, n_features)

        Training vector, where n_samples is the number of samples and n_features is the number of features.
yarray-like of shape (n_samples,)

        y: array-like of shape (n_samples,)
        Target vector. n_classes = len(set(y))

        class_attributes: array-like of shape (training_n_classes, class_n_features)
        where training_n_classes >= n_classes
        classes from y map to attributes alphabetically
        """
        raise NotImplementedError()

    def predict(self, X, class_attributes, labels_to_attributes=None):
        """
        X: {array-like, sparse matrix} of shape (n_samples, n_features)

        Feature vector, where n_samples is the number of samples and n_features is the number of features.
yarray-like of shape (n_samples,)

        class_attributes: array-like of shape (prediction_n_classes, class_n_features)

        labels_to_attributes: array-like of shape (prediction_n_classes,)
        mapping between classes from class_attributes to prediction classes
        (classes that will be in y_pred)
        """
        raise NotImplementedError()
