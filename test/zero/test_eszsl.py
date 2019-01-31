import numpy as np
from few_shot_learn.zero import ESZSLearner


def test_attribute_to_labels():
    X = np.eye(4)
    y = np.array(['even', 'odd', 'even', 'odd'])
    attributes = np.eye(2)

    eszslearner = ESZSLearner()
    eszslearner.fit(X, y, attributes)

    prediction = eszslearner.predict(X, attributes, np.array(['even', 'odd']))
    assert np.all(prediction == np.array(['even', 'odd', 'even', 'odd']))