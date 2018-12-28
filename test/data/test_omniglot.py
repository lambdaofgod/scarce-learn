from few_shot_learn.data.omniglot import load_omniglot


def test_omniglot_data():
    train, test = load_omniglot()

    assert train.shape == (4112, 20, 28, 28)
    assert test.shape == (1692, 20, 28, 28)
