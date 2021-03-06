import numpy as np
from scarce_shot_learn.scarce import ProtoNetTF


def test_model():
    proto_net = ProtoNetTF(input_shape=(28, 28, 1))

    n_classes = 10
    n_examples_per_class = 2
    data_shape = (28, 28, 1)
    data = np.random.randn(n_classes, n_examples_per_class, *data_shape)

    proto_net.train(data, 1, 1, 1, 1, 1)

    proto_net.test(data, 10, 1, 1, 1, 1)


def test_embedding():
    z_dim = 2
    proto_net = ProtoNetTF((28, 28, 1), z_dim=2)

    n_classes = 10
    n_examples_per_class = 2
    data_shape = (28, 28, 1)
    data = np.random.randn(n_classes, n_examples_per_class, *data_shape)

    proto_net.train(data, 1, 1, 1, 1, 1)

    embedded_batch = proto_net.embed(data[0])
    assert embedded_batch.shape == (n_examples_per_class, z_dim)